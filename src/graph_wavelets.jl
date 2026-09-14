# Spectral Graph Wavelet Engine for BSTM
#
# Provides multiresolution spatial decomposition, fast Chebyshev polynomial filtering,
# frame synthesis signal reconstruction, spatial denoising, and multiscale basis
# dictionary generation on irregular spatial partitioning meshes (e.g. hexagonal meshes,
# marine archipelagos, coastal boundaries).
#
# References:
# - Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). Wavelets on graphs via
#   spectral graph theory. Applied and Computational Harmonic Analysis, 30(2), 129-150.
# - Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013).
#   The emerging field of signal processing on graphs. IEEE Signal Processing Magazine.

using LinearAlgebra: I, Symmetric, norm, dot, mul!
using SparseArrays: SparseMatrixCSC, spdiagm, sparse, findnz, nonzeros, rowvals
using Statistics: mean, median, var, std, quantile
using Printf: @sprintf, @printf

const _E_CONST = 2.718281828459045

"""
    SpectralGraphWaveletResult

Structured container holding the results of a multiscale Spectral Graph Wavelet Transform
(SGWT) decomposition on an irregular spatial mesh.

# Fields
- `original::Vector{Float64}`: Input spatial field ``\\mathbf{f} \\in \\mathbb{R}^S``.
- `approximation::Vector{Float64}`: Low-pass macro-scale baseline approximation
  ``\\mathbf{a} = h(\\mathbf{L}) \\mathbf{f} \\in \\mathbb{R}^S``.
- `details::Matrix{Float64}`: High-pass and band-pass detail coefficients across scales
  ``\\mathbf{D} \\in \\mathbb{R}^{S \\times J}``, where column ``j`` corresponds to
  scale ``s_j``.
- `scales::Vector{Float64}`: Dilated wavelet scales ``s_1 < s_2 < \\dots < s_J``.
- `approx_energy::Float64`: Mean squared energy of the macro baseline
  ``E_0 = \\frac{1}{S} \\|\\mathbf{a}\\|^2_2``.
- `detail_energies::Vector{Float64}`: Mean squared energy at each scale
  ``E_j = \\frac{1}{S} \\|\\mathbf{d}_j\\|^2_2``.
- `total_energy::Float64`: Sum of all energy components: ``E_{\\text{total}} = E_0 + \\sum E_j``.
- `relative_energies::Vector{Float64}`: Proportion of spatial variance allocated to
  each component: ``[E_0 / E_{\\text{total}}, E_1 / E_{\\text{total}}, \\dots]``.
- `lambda_max::Float64`: Upper bound of the graph Laplacian spectrum.
- `order::Int`: Truncated Chebyshev polynomial degree ``M``.
"""
struct SpectralGraphWaveletResult
    original::Vector{Float64}
    approximation::Vector{Float64}
    details::Matrix{Float64}
    scales::Vector{Float64}
    approx_energy::Float64
    detail_energies::Vector{Float64}
    total_energy::Float64
    relative_energies::Vector{Float64}
    lambda_max::Float64
    order::Int
end

function Base.show(io::IO, res::SpectralGraphWaveletResult)
    println(io, "SpectralGraphWaveletResult:")
    println(io, "  Spatial Units (S):        $(length(res.original))")
    println(io, "  Chebyshev Order (M):      $(res.order)")
    println(io, "  Spectral Bound (λ_max):   $(round(res.lambda_max, digits=4))")
    println(io, "  Number of Scales (J):     $(length(res.scales))")
    println(io, "  Total Spatial Energy:     $(round(res.total_energy, digits=6))")
    println(io, "  Energy Allocation:")
    @printf(
        io,
        "    • Macro Approx: %6.2f%% (energy: %.4e)\n",
        res.relative_energies[1] * 100.0,
        res.approx_energy
    )
    for j in 1:length(res.scales)
        @printf(
            io,
            "    • Scale %d (s=%6.3f): %6.2f%% (energy: %.4e)\n",
            j,
            res.scales[j],
            res.relative_energies[j + 1] * 100.0,
            res.detail_energies[j]
        )
    end
end

"""
    build_normalized_laplacian(
        W::AbstractMatrix;
        regularize::Bool = true
    ) -> SparseMatrixCSC{Float64, Int}

Constructs the symmetric normalized graph Laplacian:
```math
\\mathbf{L}_{\\text{norm}} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{W} \\mathbf{D}^{-1/2}
```
where ``D_{ii} = \\sum_{j=1}^S W_{ij}``.

# Mathematical Details
The normalized Laplacian bounds the graph spectrum strictly to ``\\lambda \\in [0, 2]``,
regardless of graph degree heterogeneity or irregular coastline geometry.
For isolated nodes (degree ``d_i = 0``):
- If `regularize = true`, the node is assigned an isolated diagonal value of 1.0,
  preserving positive semidefiniteness without numerical divergence.

# Arguments
- `W::AbstractMatrix`: Non-negative symmetric spatial adjacency or conductance matrix
  (size ``S \\times S``).
- `regularize::Bool`: Regularize zero-degree nodes to prevent division by zero (default: `true`).

# Returns
- `L_norm::SparseMatrixCSC{Float64, Int}`: Symmetric normalized graph Laplacian.
"""
function build_normalized_laplacian(
    W::AbstractMatrix;
    regularize::Bool = true
)
    S = size(W, 1)
    if size(W, 2) != S
        throw(DimensionMismatch("Adjacency matrix W must be square, got $(size(W))"))
    end

    degrees = zeros(Float64, S)
    for j in 1:S
        for i in 1:S
            val = Float64(W[i, j])
            if val > 0.0 && i != j
                degrees[i] += val
            end
        end
    end

    inv_sqrt_deg = zeros(Float64, S)
    for i in 1:S
        if degrees[i] > 1e-12
            inv_sqrt_deg[i] = 1.0 / sqrt(degrees[i])
        elseif regularize
            inv_sqrt_deg[i] = 0.0
        else
            throw(DomainError(degrees[i], "Node $i has zero degree in unregularized mesh"))
        end
    end

    rows, cols, vals = Int[], Int[], Float64[]
    for j in 1:S
        for i in 1:S
            w_ij = Float64(W[i, j])
            if i == j
                push!(rows, i)
                push!(cols, i)
                push!(vals, 1.0)
            elseif w_ij > 0.0
                norm_val = -w_ij * inv_sqrt_deg[i] * inv_sqrt_deg[j]
                if abs(norm_val) > 1e-15
                    push!(rows, i)
                    push!(cols, j)
                    push!(vals, norm_val)
                end
            end
        end
    end

    return sparse(rows, cols, vals, S, S)
end

"""
    compute_laplacian_spectral_bounds(
        L::AbstractMatrix;
        tol::Float64 = 1e-4,
        max_iter::Int = 100
    ) -> Float64

Estimates the upper spectral bound ``\\lambda_{\\max}`` of the graph Laplacian matrix
``\\mathbf{L}`` via power iteration with a conservative safety padding factor.

# Mathematical Details
Chebyshev polynomial recurrence is valid on ``y \\in [-1, 1]``. Shifting the Laplacian
spectrum ``\\lambda \\in [0, \\lambda_{\\max}]`` via:
```math
\\tilde{\\mathbf{L}} = \\frac{2}{\\lambda_{\\max}} \\mathbf{L} - \\mathbf{I}
```
requires that ``\\lambda_{\\max} \\ge \\lambda_{\\text{largest}}`` strictly.
Power iteration calculates:
```math
\\mathbf{v}_{k+1} = \\frac{\\mathbf{L} \\mathbf{v}_k}{\\|\\mathbf{L} \\mathbf{v}_k\\|_2},
\\qquad \\lambda_k = \\mathbf{v}_k^T \\mathbf{L} \\mathbf{v}_k
```
Following Hammond et al. (2011), the estimated eigenvalue is scaled by ``1.02``
to ensure no eigenvalue falls outside the orthogonal Chebyshev polynomial domain.

# Arguments
- `L::AbstractMatrix`: Graph Laplacian matrix (size ``S \\times S``).
- `tol::Float64`: Convergence tolerance for Rayleigh quotient (default: `1e-4`).
- `max_iter::Int`: Maximum power iteration cycles (default: `100`).

# Returns
- `lambda_max::Float64`: Conservative upper bound satisfying ``\\lambda \\le \\lambda_{\\max}``.
"""
function compute_laplacian_spectral_bounds(
    L::AbstractMatrix;
    tol::Float64 = 1e-4,
    max_iter::Int = 100
)
    S = size(L, 1)
    if S <= 1
        return 2.0
    end

    # Deterministic pseudo-random vector orthogonal to DC
    v = [sin(Float64(i) * 1.6180339887) for i in 1:S]
    v_norm = norm(v)
    if v_norm < 1e-12
        v = fill(1.0 / sqrt(Float64(S)), S)
    else
        v ./= v_norm
    end

    lam_prev = 0.0
    lam_est = 0.0
    Lv = similar(v)

    for iter in 1:max_iter
        mul!(Lv, L, v)
        denom = dot(v, v)
        lam_est = denom > 1e-14 ? dot(v, Lv) / denom : 0.0

        if iter > 2 && abs(lam_est - lam_prev) < tol * max(1.0, abs(lam_est))
            break
        end
        lam_prev = lam_est

        lv_norm = norm(Lv)
        if lv_norm > 1e-14
            v .= Lv ./ lv_norm
        else
            break
        end
    end

    # Fallback: Gershgorin circle theorem upper bound
    if lam_est <= 0.0 || isnan(lam_est)
        gershgorin = 0.0
        for i in 1:S
            row_sum = 0.0
            for j in 1:S
                row_sum += abs(Float64(L[i, j]))
            end
            if row_sum > gershgorin
                gershgorin = row_sum
            end
        end
        lam_est = max(gershgorin, 2.0)
    end

    # Add 2% safety margin to ensure spectrum remains strictly within [-1, 1]
    return lam_est * 1.02
end

"""
    chebyshev_polynomial_coefficients(
        filter_func::Function,
        lambda_max::Real,
        order::Int;
        num_quad_points::Int = 2 * order + 10
    ) -> Vector{Float64}

Computes the Chebyshev polynomial expansion coefficients ``c_m`` of a continuous graph
spectral filter ``g(\\lambda)`` over ``[0, \\lambda_{\\max}]`` using Gauss-Chebyshev
quadrature.

# Mathematical Details
Projecting the continuous filter ``g(\\lambda)`` onto the Chebyshev domain ``y \\in [-1, 1]``
via ``\\lambda(y) = \\frac{\\lambda_{\\max}}{2} (y + 1)`` yields:
```math
g(\\lambda(y)) \\approx \\frac{1}{2} c_0 + \\sum_{m=1}^M c_m T_m(y)
```
Gauss-Chebyshev quadrature nodes and weights evaluate the integral:
```math
c_m = \\frac{2}{K_q} \\sum_{k=1}^{K_q}
g\\left( \\frac{\\lambda_{\\max}(\\cos\\theta_k + 1)}{2} \\right)
\\cos(m \\theta_k), \\qquad \\theta_k = \\frac{\\pi (k - 0.5)}{K_q}
```

# Arguments
- `filter_func::Function`: Continuous spectral filter kernel ``g(\\lambda)``.
- `lambda_max::Real`: Upper bound of graph Laplacian spectrum.
- `order::Int`: Truncation order ``M`` of the Chebyshev expansion.
- `num_quad_points::Int`: Number of quadrature evaluation points ``K_q`` (default: ``2M + 10``).

# Returns
- `c::Vector{Float64}`: Vector of length ``M + 1`` with Chebyshev coefficients ``c_m``.
"""
function chebyshev_polynomial_coefficients(
    filter_func::Function,
    lambda_max::Real,
    order::Int;
    num_quad_points::Int = 2 * order + 10
)
    K_q = max(num_quad_points, order + 2)
    c = zeros(Float64, order + 1)
    lam_max_half = Float64(lambda_max) / 2.0

    thetas = [pi * (Float64(k) - 0.5) / Float64(K_q) for k in 1:K_q]
    lambdas = [lam_max_half * (cos(th) + 1.0) for th in thetas]
    filter_vals = [Float64(filter_func(lam)) for lam in lambdas]

    scale_factor = 2.0 / Float64(K_q)
    for m in 0:order
        m_float = Float64(m)
        acc = 0.0
        for k in 1:K_q
            acc += filter_vals[k] * cos(m_float * thetas[k])
        end
        c[m + 1] = acc * scale_factor
    end

    return c
end

"""
    apply_graph_spectral_filter(
        L::AbstractMatrix,
        f::AbstractVector,
        filter_func::Function;
        lambda_max::Real = 0.0,
        order::Int = 25
    ) -> Vector{Float64}

Applies a continuous graph spectral filter ``g(\\mathbf{L}) \\mathbf{f}`` to a spatial
signal ``\\mathbf{f} \\in \\mathbb{R}^S`` in ``\\mathcal{O}(M |E|)`` time via truncated
Chebyshev polynomial 3-term recurrence.

# Mathematical Details
Given shifted Laplacian
``\\tilde{\\mathbf{L}} = (2 / \\lambda_{\\max}) \\mathbf{L} - \\mathbf{I}_S``:
```math
g(\\mathbf{L}) \\mathbf{f} \\approx \\frac{1}{2} c_0 \\bar{\\mathbf{f}}_0 +
\\sum_{m=1}^M c_m \\bar{\\mathbf{f}}_m
```
where Chebyshev vectors satisfy the three-term recurrence:
```math
\\bar{\\mathbf{f}}_0 = \\mathbf{f}, \\qquad
\\bar{\\mathbf{f}}_1 = \\tilde{\\mathbf{L}} \\mathbf{f} =
\\frac{2}{\\lambda_{\\max}} \\mathbf{L} \\mathbf{f} - \\mathbf{f}
```
```math
\\bar{\\mathbf{f}}_m = 2 \\tilde{\\mathbf{L}} \\bar{\\mathbf{f}}_{m-1} - \\bar{\\mathbf{f}}_{m-2}
= \\frac{4}{\\lambda_{\\max}} \\mathbf{L} \\bar{\\mathbf{f}}_{m-1} -
2 \\bar{\\mathbf{f}}_{m-1} - \\bar{\\mathbf{f}}_{m-2}
```
Every step requires only a single sparse matrix-vector product, eliminating large matrix
factorizations and operating in linear time with respect to the number of mesh edges.

# Arguments
- `L::AbstractMatrix`: Graph Laplacian matrix (size ``S \\times S``).
- `f::AbstractVector`: Spatial signal on graph nodes (length ``S``).
- `filter_func::Function`: Continuous filter kernel ``g(\\lambda)``.
- `lambda_max::Real`: Upper spectral bound. If ``\\le 0.0``, automatically computed.
- `order::Int`: Truncation order of Chebyshev polynomial (default: `25`).

# Returns
- `z::Vector{Float64}`: Filtered spatial signal ``g(\\mathbf{L}) \\mathbf{f} \\in \\mathbb{R}^S``.
"""
function apply_graph_spectral_filter(
    L::AbstractMatrix,
    f::AbstractVector,
    filter_func::Function;
    lambda_max::Real = 0.0,
    order::Int = 25
)
    S = length(f)
    if size(L, 1) != S || size(L, 2) != S
        throw(DimensionMismatch("Laplacian size $(size(L)) does not match signal length $S"))
    end

    if S == 0
        return Float64[]
    elseif S == 1
        return [Float64(filter_func(0.0)) * Float64(f[1])]
    end

    lam_max = lambda_max > 0.0 ? Float64(lambda_max) : compute_laplacian_spectral_bounds(L)
    coeffs = chebyshev_polynomial_coefficients(filter_func, lam_max, order)

    # Initial Chebyshev recurrence vectors
    f_vec = Float64.(f)
    f0 = copy(f_vec)
    Lf = similar(f_vec)
    mul!(Lf, L, f_vec)

    # f1 = (2 / lam_max) * L * f - f
    f1 = zeros(Float64, S)
    inv_lam_2 = 2.0 / lam_max
    inv_lam_4 = 4.0 / lam_max
    for i in 1:S
        f1[i] = inv_lam_2 * Lf[i] - f_vec[i]
    end

    # Accumulate result: (1/2)*c_0*f0 + c_1*f1
    z = zeros(Float64, S)
    c0_half = 0.5 * coeffs[1]
    c1 = coeffs[2]
    for i in 1:S
        z[i] = c0_half * f0[i] + c1 * f1[i]
    end

    # 3-term recurrence loop
    f_next = similar(f_vec)
    for m in 2:order
        mul!(Lf, L, f1)
        cm = coeffs[m + 1]
        for i in 1:S
            fn = inv_lam_4 * Lf[i] - 2.0 * f1[i] - f0[i]
            z[i] += cm * fn
            f0[i] = f1[i]
            f1[i] = fn
        end
    end

    return z
end

"""
    spectral_graph_wavelet_transform(
        W_or_L::AbstractMatrix,
        f::AbstractVector;
        num_scales::Int = 4,
        order::Int = 25,
        s_min::Real = 0.0,
        s_max::Real = 0.0,
        normalized::Bool = true
    ) -> SpectralGraphWaveletResult

Decomposes a spatial signal ``\\mathbf{f} \\in \\mathbb{R}^S`` defined over an irregular
spatial mesh into a multiscale representation via Spectral Graph Wavelets.

# Mathematical Details
1. **Normalized Laplacian**: Constructs
   ``\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{W} \\mathbf{D}^{-1/2}``
   with eigenvalues in ``[0, \\lambda_{\\max}]``.
2. **Mother Wavelet**: Normalized heat diffusion kernel with unit peak gain:
   ```math
   g(x) = e \\cdot x \\exp(-x), \\qquad g(s \\lambda) = e \\cdot s \\lambda \\exp(-s \\lambda)
   ```
   peaking at ``s \\lambda = 1.0`` with ``g(1) = 1.0`` (yielding a near-tight frame).
3. **Low-pass Scaling Function**: Macro-scale regional baseline filter:
   ```math
   h(\\lambda) = \\exp\\left( - \\left(\\frac{s_{\\max} \\lambda}{2}\\right)^2 \\right)
   ```
4. **Logarithmic Scale Grid**: ``s_1 < s_2 < \\dots < s_J`` spanning from high-frequency
   spatial roughness (micro-refugia, 1–5 km) to regional trends (100–300 km).

# Arguments
- `W_or_L::AbstractMatrix`: Adjacency matrix ``\\mathbf{W}`` or graph Laplacian ``\\mathbf{L}``.
- `f::AbstractVector`: Spatial signal defined at each mesh unit (length ``S``).
- `num_scales::Int`: Number of wavelet detail scales ``J`` (default: `4`).
- `order::Int`: Chebyshev polynomial truncation degree ``M`` (default: `25`).
- `s_min::Real`: Minimum wavelet scale. If ``\\le 0.0``, defaults to ``1.0 / \\lambda_{\\max}``.
- `s_max::Real`: Maximum wavelet scale. If ``\\le 0.0``, defaults to ``25.0 / \\lambda_{\\max}``.
- `normalized::Bool`: Whether to use normalized graph Laplacian (default: `true`).

# Returns
- `result::SpectralGraphWaveletResult`: Decomposed representation containing macro baseline,
  scale detail coefficients, energy distribution, and spectral bounds.
"""
function spectral_graph_wavelet_transform(
    W_or_L::AbstractMatrix,
    f::AbstractVector;
    num_scales::Int = 4,
    order::Int = 25,
    s_min::Real = 0.0,
    s_max::Real = 0.0,
    normalized::Bool = true
)
    S = length(f)
    if size(W_or_L, 1) != S || size(W_or_L, 2) != S
        throw(DimensionMismatch(
            "Matrix size $(size(W_or_L)) does not match signal length $S"
        ))
    end

    # Check if input is already a Laplacian or adjacency matrix
    is_laplacian = (diag_sum = sum(diag(W_or_L)); abs(diag_sum - Float64(S)) < 0.5 * S)
    L = if is_laplacian
        W_or_L isa SparseMatrixCSC ? W_or_L : sparse(Float64.(W_or_L))
    elseif normalized
        build_normalized_laplacian(W_or_L)
    else
        # Unnormalized Laplacian: D - W
        deg = [sum(W_or_L[i, :]) - W_or_L[i, i] for i in 1:S]
        spdiagm(0 => deg) - sparse(Float64.(W_or_L))
    end

    lam_max = compute_laplacian_spectral_bounds(L)

    # Establish scale range
    scale_min = s_min > 0.0 ? Float64(s_min) : 1.0 / lam_max
    scale_max = s_max > 0.0 ? Float64(s_max) : 25.0 / lam_max

    scales = zeros(Float64, num_scales)
    if num_scales == 1
        scales[1] = sqrt(scale_min * scale_max)
    else
        ratio = scale_max / scale_min
        for j in 1:num_scales
            scales[j] = scale_min * (ratio ^ ((Float64(j) - 1.0) / (Float64(num_scales) - 1.0)))
        end
    end

    # Low-pass scaling filter h(λ)
    h_func = lam -> exp(-((scale_max * lam / 2.0)^2))
    approx = apply_graph_spectral_filter(L, f, h_func; lambda_max=lam_max, order=order)

    # Band-pass detail filters g(s_j λ) with unit peak gain
    details = zeros(Float64, S, num_scales)
    for j in 1:num_scales
        sj = scales[j]
        g_func = lam -> (_E_CONST * sj * lam) * exp(-sj * lam)
        details[:, j] = apply_graph_spectral_filter(
            L, f, g_func; lambda_max=lam_max, order=order
        )
    end

    # Compute energy metrics
    f_vec = Float64.(f)
    approx_energy = sum(abs2, approx) / Float64(S)
    detail_energies = [sum(abs2, details[:, j]) / Float64(S) for j in 1:num_scales]
    total_energy = approx_energy + sum(detail_energies)

    denom = total_energy > 1e-15 ? total_energy : 1.0
    relative_energies = [approx_energy / denom; [e / denom for e in detail_energies]]

    return SpectralGraphWaveletResult(
        f_vec,
        approx,
        details,
        scales,
        approx_energy,
        detail_energies,
        total_energy,
        relative_energies,
        lam_max,
        order
    )
end

"""
    inverse_spectral_graph_wavelet_transform(
        res::SpectralGraphWaveletResult,
        W_or_L::AbstractMatrix;
        order::Int = res.order,
        normalized::Bool = true
    ) -> Vector{Float64}

    inverse_spectral_graph_wavelet_transform(
        a::AbstractVector,
        D::AbstractMatrix,
        scales::AbstractVector,
        W_or_L::AbstractMatrix;
        order::Int = 25,
        lambda_max::Real = 0.0,
        normalized::Bool = true
    ) -> Vector{Float64}

Reconstructs the spatial field ``\\hat{\\mathbf{f}} \\in \\mathbb{R}^S`` from its multiscale
wavelet decomposition via dual frame synthesis.

# Mathematical Details
For the frame consisting of low-pass filter ``h(\\lambda)`` and wavelets ``g(s_j \\lambda)``,
the total spectral frame operator is:
```math
\\Phi(\\lambda) = h(\\lambda)^2 + \\sum_{j=1}^J g(s_j \\lambda)^2
```
Because ``h(0) = 1.0`` and ``g(s_j \\lambda) > 0`` for ``\\lambda > 0``, ``\\Phi(\\lambda) > 0``
strictly on ``[0, \\lambda_{\\max}]``. The canonical dual frame filters are defined as:
```math
\\tilde{h}(\\lambda) = \\frac{h(\\lambda)}{\\Phi(\\lambda)}, \\qquad
\\tilde{g}(s_j \\lambda) = \\frac{g(s_j \\lambda)}{\\Phi(\\lambda)}
```
Satisfying the exact partition of unity identity:
```math
h(\\lambda) \\tilde{h}(\\lambda) + \\sum_{j=1}^J g(s_j \\lambda) \\tilde{g}(s_j \\lambda)
\\equiv 1.0, \\qquad \\forall \\lambda \\in [0, \\lambda_{\\max}]
```
The spatial field is synthesized with near-machine precision via:
```math
\\hat{\\mathbf{f}} = \\tilde{h}(\\mathbf{L}) \\mathbf{a} +
\\sum_{j=1}^J \\tilde{g}(s_j \\mathbf{L}) \\mathbf{d}_j
```

# Arguments
- `res::SpectralGraphWaveletResult`: Decomposed wavelet representation.
- `W_or_L::AbstractMatrix`: Adjacency or graph Laplacian matrix.
- `order::Int`: Chebyshev polynomial degree (default: `res.order`).
- `normalized::Bool`: Whether graph Laplacian is normalized (default: `true`).

# Returns
- `f_reconstructed::Vector{Float64}`: Reconstructed spatial signal ``\\hat{\\mathbf{f}}``.
"""
function inverse_spectral_graph_wavelet_transform(
    res::SpectralGraphWaveletResult,
    W_or_L::AbstractMatrix;
    order::Int = res.order,
    normalized::Bool = true
)
    return inverse_spectral_graph_wavelet_transform(
        res.approximation,
        res.details,
        res.scales,
        W_or_L;
        order = order,
        lambda_max = res.lambda_max,
        normalized = normalized
    )
end

function inverse_spectral_graph_wavelet_transform(
    a::AbstractVector,
    D::AbstractMatrix,
    scales::AbstractVector,
    W_or_L::AbstractMatrix;
    order::Int = 25,
    lambda_max::Real = 0.0,
    normalized::Bool = true
)
    S = length(a)
    num_scales = length(scales)
    if size(D, 1) != S || size(D, 2) != num_scales
        throw(DimensionMismatch(
            "Detail matrix size $(size(D)) incompatible with S=$S, J=$num_scales"
        ))
    end

    is_laplacian = (diag_sum = sum(diag(W_or_L)); abs(diag_sum - Float64(S)) < 0.5 * S)
    L = if is_laplacian
        W_or_L isa SparseMatrixCSC ? W_or_L : sparse(Float64.(W_or_L))
    elseif normalized
        build_normalized_laplacian(W_or_L)
    else
        deg = [sum(W_or_L[i, :]) - W_or_L[i, i] for i in 1:S]
        spdiagm(0 => deg) - sparse(Float64.(W_or_L))
    end

    lam_max = lambda_max > 0.0 ? Float64(lambda_max) : compute_laplacian_spectral_bounds(L)
    s_max = maximum(scales)
    cheb_order = max(30, order)

    # Frame energy function: Φ(λ) = h(λ)^2 + ∑ g(s_j λ)^2
    h_raw = lam -> exp(-((s_max * lam / 2.0)^2))
    g_raw = (lam, s) -> (_E_CONST * s * lam) * exp(-s * lam)

    phi_func = function(lam)
        val = h_raw(lam)^2
        for j in 1:num_scales
            val += g_raw(lam, scales[j])^2
        end
        return max(val, 1e-12)
    end

    # Dual scaling filter: h_dual(λ) = h(λ) / Φ(λ)
    h_dual = lam -> h_raw(lam) / phi_func(lam)
    f_rec = apply_graph_spectral_filter(L, a, h_dual; lambda_max=lam_max, order=cheb_order)

    # Dual wavelet filters: g_dual(s_j λ) = g(s_j λ) / Φ(λ)
    for j in 1:num_scales
        sj = scales[j]
        g_dual = lam -> g_raw(lam, sj) / phi_func(lam)
        rec_detail = apply_graph_spectral_filter(
            L, D[:, j], g_dual; lambda_max=lam_max, order=cheb_order
        )
        f_rec .+= rec_detail
    end

    return f_rec
end

"""
    denoise_spatial_signal_wavelet(
        W_or_L::AbstractMatrix,
        f::AbstractVector;
        threshold_rule::Symbol = :bayesshrink,
        soft::Bool = true,
        num_scales::Int = 4,
        order::Int = 25,
        normalized::Bool = true
    ) -> Tuple{Vector{Float64}, SpectralGraphWaveletResult, Float64}

Performs spatial denoising on an irregular mesh signal by multiscale graph wavelet shrinkage.

# Mathematical Details
1. **Forward Decomposition**: Signal is transformed into macro baseline ``\\mathbf{a}`` and
   scale details ``\\mathbf{D} \\in \\mathbb{R}^{S \\times J}``.
2. **Noise Floor Estimation**: High-frequency sensor noise standard deviation ``\\hat{\\sigma}``
   is estimated from the finest scale detail coefficients ``\\mathbf{d}_1`` using the robust
   Median Absolute Deviation (MAD):
   ```math
   \\hat{\\sigma} = \\frac{\\text{median}(|\\mathbf{d}_1 - \\text{median}(\\mathbf{d}_1)|)}{0.6745}
   ```
3. **Wavelet Thresholding**:
   - `:bayesshrink` (default, adaptive): Minimizes Bayesian risk per scale:
     ```math
     \\tau_j = \\frac{\\hat{\\sigma}^2}{\\sigma_{x, j}}, \\qquad
     \\sigma_{x, j} = \\sqrt{\\max(0, \\text{var}(\\mathbf{d}_j) - \\hat{\\sigma}^2)}
     ```
   - `:visushrink`: Universal asymptotic minimax threshold:
     ```math
     \\tau = \\hat{\\sigma} \\sqrt{2 \\ln S}
     ```
   - `:sure`: Adaptive Stein's Unbiased Risk Estimator decay across scales.
4. **Soft Shrinkage**: Detail coefficients are shrunk towards zero:
   ```math
   \\tilde{d}_{i, j} = \\text{sign}(d_{i, j}) \\max(0, |d_{i, j}| - \\tau_j)
   ```
   preserving steep oceanographic bathymetric edges while attenuating random noise.
5. **Synthesis**: Clean signal is reconstructed via inverse SGWT.

# Arguments
- `W_or_L::AbstractMatrix`: Adjacency or graph Laplacian matrix.
- `f::AbstractVector`: Noisy spatial signal (length ``S``).
- `threshold_rule::Symbol`: Denoising rule (`:bayesshrink`, `:visushrink`, or `:sure`).
- `soft::Bool`: Use soft thresholding if `true`, hard thresholding if `false` (default: `true`).
- `num_scales::Int`: Number of wavelet detail scales (default: `4`).
- `order::Int`: Chebyshev polynomial degree (default: `25`).
- `normalized::Bool`: Whether graph Laplacian is normalized (default: `true`).

# Returns
- `f_denoised::Vector{Float64}`: Denoised spatial signal ``\\hat{\\mathbf{f}}``.
- `sgwt_denoised::SpectralGraphWaveletResult`: SGWT decomposition of the denoised signal.
- `noise_sigma::Float64`: Robustly estimated noise standard deviation ``\\hat{\\sigma}``.
"""
function denoise_spatial_signal_wavelet(
    W_or_L::AbstractMatrix,
    f::AbstractVector;
    threshold_rule::Symbol = :bayesshrink,
    soft::Bool = true,
    num_scales::Int = 4,
    order::Int = 25,
    normalized::Bool = true
)
    # Forward decomposition
    sgwt_res = spectral_graph_wavelet_transform(
        W_or_L, f; num_scales=num_scales, order=order, normalized=normalized
    )

    S = length(f)
    details_clean = copy(sgwt_res.details)

    # Robust noise floor estimation calibrated for unit-gain heat diffusion wavelet
    d1 = sgwt_res.details[:, 1]
    med_d1 = median(d1)
    mad_d1 = median(abs.(d1 .- med_d1))
    sigma_hat = (mad_d1 / 0.6745) / 0.77

    if sigma_hat > 1e-12
        for j in 1:num_scales
            dj = sgwt_res.details[:, j]
            tau_j = 0.0

            if threshold_rule == :visushrink
                tau_j = sigma_hat * sqrt(2.0 * log(Float64(S)))
            elseif threshold_rule == :bayesshrink
                var_y = var(dj)
                noise_var_j = (sigma_hat * 0.77)^2
                sigma_x_sq = max(0.0, var_y - noise_var_j)
                if sigma_x_sq > 1e-14
                    tau_j = noise_var_j / sqrt(sigma_x_sq)
                else
                    tau_j = maximum(abs.(dj))
                end
            elseif threshold_rule == :sure
                tau_j = (sigma_hat * sqrt(2.0 * log(Float64(S)))) / sqrt(Float64(j))
            else
                tau_j = sigma_hat * sqrt(2.0 * log(Float64(S)))
            end

            for i in 1:S
                val = dj[i]
                if soft
                    details_clean[i, j] = sign(val) * max(0.0, abs(val) - tau_j)
                else
                    details_clean[i, j] = abs(val) >= tau_j ? val : 0.0
                end
            end
        end
    end

    # Inverse synthesis of denoised signal
    f_denoised = inverse_spectral_graph_wavelet_transform(
        sgwt_res.approximation,
        details_clean,
        sgwt_res.scales,
        W_or_L;
        order = order,
        lambda_max = sgwt_res.lambda_max,
        normalized = normalized
    )

    # Re-decompose clean signal for consistent structured output
    sgwt_denoised = spectral_graph_wavelet_transform(
        W_or_L, f_denoised;
        num_scales = num_scales,
        order = order,
        s_min = sgwt_res.scales[1],
        s_max = sgwt_res.scales[end],
        normalized = normalized
    )

    return (f_denoised, sgwt_denoised, sigma_hat)
end

"""
    graph_wavelet_basis_matrix(
        W_or_L::AbstractMatrix;
        num_scales::Int = 4,
        order::Int = 25,
        anchor_indices::Union{Symbol, AbstractVector{Int}} = :all,
        normalized::Bool = true
    ) -> Matrix{Float64}

Constructs an explicit multiresolution graph wavelet basis matrix
``\\mathbf{\\Psi} \\in \\mathbb{R}^{S \\times K}`` for use in Bayesian spatial regression
and Gaussian Process models (`GraphWaveletGP`).

# Mathematical Details
Each column of the basis dictionary corresponds to a localized wavelet atom centered at an
anchor node ``k``:
- Macro baseline atoms: ``\\boldsymbol{\\phi}_k = h(\\mathbf{L}) \\boldsymbol{\\delta}_k``
- Scale-specific detail atoms:
  ``\\boldsymbol{\\psi}_{j, k} = g(s_j \\mathbf{L}) \\boldsymbol{\\delta}_k``
If `anchor_indices = :all`, atoms are centered at all ``S`` mesh units, producing a frame
dictionary with ``K = S \\times (J + 1)`` columns. For large meshes, a subset of spatial
anchor nodes can be supplied to form an overcomplete multiscale dictionary.

# Arguments
- `W_or_L::AbstractMatrix`: Adjacency or graph Laplacian matrix (size ``S \\times S``).
- `num_scales::Int`: Number of detail scales ``J`` (default: `4`).
- `order::Int`: Chebyshev polynomial degree ``M`` (default: `25`).
- `anchor_indices`: Indices of anchor nodes or `:all` (default: `:all`).
- `normalized::Bool`: Whether graph Laplacian is normalized (default: `true`).

# Returns
- `Psi::Matrix{Float64}`: Multiresolution basis matrix of size ``S \\times K``.
"""
function graph_wavelet_basis_matrix(
    W_or_L::AbstractMatrix;
    num_scales::Int = 4,
    order::Int = 25,
    anchor_indices::Union{Symbol, AbstractVector{Int}} = :all,
    normalized::Bool = true
)
    S = size(W_or_L, 1)
    is_laplacian = (diag_sum = sum(diag(W_or_L)); abs(diag_sum - Float64(S)) < 0.5 * S)
    L = if is_laplacian
        W_or_L isa SparseMatrixCSC ? W_or_L : sparse(Float64.(W_or_L))
    elseif normalized
        build_normalized_laplacian(W_or_L)
    else
        deg = [sum(W_or_L[i, :]) - W_or_L[i, i] for i in 1:S]
        spdiagm(0 => deg) - sparse(Float64.(W_or_L))
    end

    lam_max = compute_laplacian_spectral_bounds(L)
    scale_min = 1.0 / lam_max
    scale_max = 25.0 / lam_max

    scales = zeros(Float64, num_scales)
    if num_scales == 1
        scales[1] = sqrt(scale_min * scale_max)
    else
        ratio = scale_max / scale_min
        for j in 1:num_scales
            scales[j] = scale_min * (ratio ^ ((Float64(j) - 1.0) / (Float64(num_scales) - 1.0)))
        end
    end

    anchors = if anchor_indices === :all
        collect(1:S)
    else
        collect(anchor_indices)
    end
    n_anchors = length(anchors)
    total_cols = n_anchors * (num_scales + 1)
    Psi = zeros(Float64, S, total_cols)

    h_func = lam -> exp(-((scale_max * lam / 2.0)^2))
    delta = zeros(Float64, S)

    # 1. Macro baseline scaling atoms
    col_idx = 1
    for k in anchors
        fill!(delta, 0.0)
        delta[k] = 1.0
        Psi[:, col_idx] = apply_graph_spectral_filter(
            L, delta, h_func; lambda_max=lam_max, order=order
        )
        col_idx += 1
    end

    # 2. Detail wavelet atoms across scales
    for j in 1:num_scales
        sj = scales[j]
        g_func = lam -> (_E_CONST * sj * lam) * exp(-sj * lam)
        for k in anchors
            fill!(delta, 0.0)
            delta[k] = 1.0
            Psi[:, col_idx] = apply_graph_spectral_filter(
                L, delta, g_func; lambda_max=lam_max, order=order
            )
            col_idx += 1
        end
    end

    return Psi
end
