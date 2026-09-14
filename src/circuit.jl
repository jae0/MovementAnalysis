# Circuit Theory & Resistance Distance Engine for BSTM
#
# Provides all-paths ecological connectivity, migratory current density mapping,
# electrical circuit network modeling, and barrier-aware Gaussian Process spatial
# covariance matrices based on the symmetric graph Laplacian.

using LinearAlgebra: I, Symmetric, cholesky, norm
using SparseArrays: SparseMatrixCSC, spdiagm, sparse, findnz, nzrange, nonzeros, rowvals
using Statistics: mean, quantile
using Graphs: SimpleGraph, connected_components, add_edge!

"""
    build_circuit_laplacian(
        W::AbstractMatrix;
        conductance::Union{Nothing, AbstractVector{<:Real}, AbstractMatrix{<:Real}} = nothing,
        resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
    ) -> (L::SparseMatrixCSC{Float64, Int}, C::SparseMatrixCSC{Float64, Int})

Constructs the symmetric conductance matrix ``\\mathbf{C}`` and graph Laplacian
``\\mathbf{L} = \\mathbf{D} - \\mathbf{C}`` for circuit theory analysis across the
spatial mesh.

# Mathematical Formulation
Given spatial adjacency matrix ``\\mathbf{W} \\in \\{0, 1\\}^{S \\times S}``, edge conductance
``C_{ij} = C_{ji} \\ge 0`` represents the ease of animal movement between adjacent cells:
- **Habitat Suitability Index (HSI)**:
  ```math
  C_{ij} = W_{ij} \\cdot \\exp\\left( \\frac{\\text{HSI}_i + \\text{HSI}_j}{2} \\right)
  ```
- **Environmental Resistance / Friction**:
  ```math
  C_{ij} = W_{ij} \\cdot \\frac{2}{R_i + R_j}
  ```
- **Uniform Adjacency**: ``C_{ij} = W_{ij}`` if no habitat weights are supplied.

The symmetric graph Laplacian is defined as:
```math
\\mathbf{L} = \\mathbf{D} - \\mathbf{C}, \\qquad D_{ii} = \\sum_{j=1}^S C_{ij}
```
where ``\\mathbf{L}`` is positive semidefinite with null space spanned by ``\\mathbf{1}_S``
for connected components.

# Arguments
- `W::AbstractMatrix`: Adjacency matrix (size ``S \\times S``).
- `conductance`: Optional custom edge or nodal conductance.
- `resistance`: Optional nodal resistance/friction vector (length ``S``).
- `hsi`: Optional Habitat Suitability Index vector (length ``S``).
- `land_mask`: Optional boolean vector (length ``S``, `true` for land units). Land
  nodes and their incident edges are zeroed out.

# Returns
- `L::SparseMatrixCSC{Float64, Int}`: Symmetric graph Laplacian (size ``S \\times S``).
- `C::SparseMatrixCSC{Float64, Int}`: Symmetric conductance matrix (size ``S \\times S``).
"""
function build_circuit_laplacian(
    W::AbstractMatrix;
    conductance::Union{Nothing, AbstractVector{<:Real}, AbstractMatrix{<:Real}} = nothing,
    resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)
    S = size(W, 1)
    if size(W, 2) != S
        throw(DimensionMismatch("Adjacency matrix W must be square, got $(size(W))"))
    end

    rows, cols, vals = Int[], Int[], Float64[]

    # Pre-calculate node conductivity factors if provided
    for j in 1:S
        for i in 1:S
            w_ij = Float64(W[i, j])
            w_ij <= 0.0 && continue

            # Zero out land units if land_mask provided
            if !isnothing(land_mask) && (land_mask[i] || land_mask[j])
                continue
            end

            c_ij = w_ij
            if !isnothing(conductance)
                if conductance isa AbstractMatrix
                    c_ij = Float64(conductance[i, j])
                else
                    c_ij = w_ij * sqrt(max(1e-6, Float64(conductance[i])) *
                                       max(1e-6, Float64(conductance[j])))
                end
            elseif !isnothing(resistance)
                r_i = max(1e-6, Float64(resistance[i]))
                r_j = max(1e-6, Float64(resistance[j]))
                c_ij = w_ij * (2.0 / (r_i + r_j))
            elseif !isnothing(hsi)
                h_i = clamp(Float64(hsi[i]), 0.0, 1.0)
                h_j = clamp(Float64(hsi[j]), 0.0, 1.0)
                c_ij = w_ij * exp(0.5 * (h_i + h_j))
            end

            push!(rows, i)
            push!(cols, j)
            push!(vals, c_ij)
        end
    end

    C_raw = sparse(rows, cols, vals, S, S)
    # Symmetrize conductance to ensure exact numerical symmetry
    C = 0.5 .* (C_raw .+ C_raw')

    # Degree matrix D_ii = sum_j C_ij
    deg = vec(sum(C, dims = 2))
    L = spdiagm(0 => deg) - C

    return L, C
end

"""
    effective_resistance_matrix(
        L::AbstractMatrix;
        W::Union{Nothing, AbstractMatrix} = nothing,
        centroids::Union{Nothing, AbstractVector} = nothing,
        disconnected_dist::Real = 1e6
    ) -> Matrix{Float64}

Computes the pairwise Effective Resistance Distance matrix ``\\boldsymbol{\\Omega}``
across all nodes of the spatial graph.

# Mathematical Formulation
Effective resistance (Klein & Randić, 1993) is the potential difference between node ``u``
and node ``v`` when a unit current is injected at ``u`` and drained at ``v``:
```math
\\Omega(u, v) = (\\mathbf{e}_u - \\mathbf{e}_v)^T \\mathbf{L}^\\dagger
                (\\mathbf{e}_u - \\mathbf{e}_v)
= L^\\dagger_{uu} + L^\\dagger_{vv} - 2 L^\\dagger_{uv}
```
where ``\\mathbf{L}^\\dagger`` is the Moore-Penrose pseudoinverse. For a connected graph
of size ``S``, ``\\mathbf{L}^\\dagger = (\\mathbf{L} + \\frac{1}{S}\\mathbf{1}\\mathbf{1}^T)^{-1}
- \\frac{1}{S}\\mathbf{1}\\mathbf{1}^T``. Because uniform shift
``\\frac{1}{S}\\mathbf{1}\\mathbf{1}^T`` cancels in ``u + v - 2uv``, we compute:
```math
\\tilde{\\mathbf{L}} = \\left( \\mathbf{L} + \\frac{1}{S} \\mathbf{1}_S
                             \\mathbf{1}_S^T \\right)^{-1},
\\qquad \\Omega(u, v) = \\tilde{L}_{uu} + \\tilde{L}_{vv} - 2 \\tilde{L}_{uv}
```
For disconnected components (e.g. islands without water connections), resistance between
separate components is set to `disconnected_dist`.

If `centroids` are provided, resistance is calibrated to effective kilometers by scaling
by the mean physical distance between neighboring graph nodes.

# Arguments
- `L::AbstractMatrix`: Graph Laplacian matrix (size ``S \\times S``).
- `W`: Optional adjacency matrix for detecting connected components via graph traversal.
- `centroids`: Optional vector of node coordinates `[lon, lat]` or `[x, y]` to scale
  resistance into physical distance units.
- `disconnected_dist`: Assigned resistance distance between disconnected components
  (default: 1e6).

# Returns
- `Matrix{Float64}`: Symmetric ``S \\times S`` pairwise effective resistance distance matrix.
"""
function effective_resistance_matrix(
    L::AbstractMatrix;
    W::Union{Nothing, AbstractMatrix} = nothing,
    centroids::Union{Nothing, AbstractVector} = nothing,
    disconnected_dist::Real = 1e6
)
    S = size(L, 1)
    Omega = fill(Float64(disconnected_dist), S, S)
    for i in 1:S
        Omega[i, i] = 0.0
    end

    # Identify connected components using adjacency
    adj = if !isnothing(W)
        W
    else
        # Extract adjacency structure from off-diagonals of L
        sparse(abs.(L) .> 1e-12) - spdiagm(0 => diag(abs.(L) .> 1e-12))
    end

    # Build SimpleGraph for component labeling
    g = SimpleGraph(S)
    for j in 1:S
        for i in (j + 1):S
            if adj[i, j] > 0
                add_edge!(g, i, j)
            end
        end
    end

    comps = connected_components(g)

    # Compute effective resistance per connected component
    for comp in comps
        n_comp = length(comp)
        if n_comp <= 1
            continue
        elseif n_comp == 2
            u, v = comp[1], comp[2]
            c_val = abs(L[u, v])
            r_val = c_val > 0.0 ? 1.0 / c_val : Float64(disconnected_dist)
            Omega[u, v] = r_val
            Omega[v, u] = r_val
            continue
        end

        # Sub-Laplacian for component
        L_sub = Matrix(L[comp, comp])
        J_sub = fill(1.0 / n_comp, n_comp, n_comp)
        L_reg = L_sub .+ J_sub

        # Invert regularized Laplacian
        L_tilde = try
            inv(Symmetric(L_reg))
        catch
            pinv(L_reg)
        end

        # Compute pairwise resistance distances within component
        diag_tilde = diag(L_tilde)
        for j in 1:n_comp
            u_idx = comp[j]
            d_u = diag_tilde[j]
            for i in 1:(j - 1)
                v_idx = comp[i]
                d_v = diag_tilde[i]
                r_uv = max(0.0, d_u + d_v - 2.0 * L_tilde[i, j])
                Omega[u_idx, v_idx] = r_uv
                Omega[v_idx, u_idx] = r_uv
            end
        end
    end

    # Physical spatial distance calibration if centroids provided
    if !isnothing(centroids) && length(centroids) == S
        edge_dists = Float64[]
        for j in 1:S
            for i in (j + 1):S
                if adj[i, j] > 0 && isfinite(Omega[i, j]) && Omega[i, j] > 0.0
                    c1, c2 = centroids[i], centroids[j]
                    d_phys = _spatial_node_distance(c1, c2)
                    if d_phys > 0.0
                        push!(edge_dists, d_phys / Omega[i, j])
                    end
                end
            end
        end
        if !isempty(edge_dists)
            scale_factor = mean(edge_dists)
            for j in 1:S
                for i in 1:S
                    if i != j && isfinite(Omega[i, j]) && Omega[i, j] < disconnected_dist
                        Omega[i, j] *= scale_factor
                    end
                end
            end
        end
    end

    return Omega
end

"""
    solve_circuit_voltage(
        L::AbstractMatrix,
        C::AbstractMatrix,
        source::Int,
        sink::Int;
        I_total::Real = 1.0
    ) -> (
        V::Vector{Float64},
        R_eff::Float64,
        edge_currents::SparseMatrixCSC{Float64, Int}
    )

Solves the point-to-point electrical potential field and branch currents between a
specified release (source) and recapture (sink) node.

# Mathematical Formulation
For injected current ``I_{\\text{total}}`` at source node ``u`` and extracted at sink
node ``v``, setting the ground reference potential ``V_v = 0`` reduces the singular
Laplacian ``\\mathbf{L} \\mathbf{V} = \\mathbf{I}`` to a strictly positive-definite
``(S-1) \\times (S-1)`` system:
```math
\\mathbf{L}_{-v, -v} \\mathbf{V}_{-v} = I_{\\text{total}} \\mathbf{e}_u
```
The effective resistance is the resulting voltage drop:
```math
\\Omega(u, v) = \\frac{V_u - V_v}{I_{\\text{total}}} = \\frac{V_u}{I_{\\text{total}}}
```
Branch currents along each directed edge ``i \\to j`` satisfy Ohm's law:
```math
I_{ij} = C_{ij} (V_i - V_j)
```
Kirchhoff's current law ensures exact mass/current conservation:
``\\sum_{j \\sim i} I_{ij} = 0`` for all intermediate nodes ``i \\notin \\{u, v\\}``.

# Arguments
- `L::AbstractMatrix`: Graph Laplacian matrix (size ``S \\times S``).
- `C::AbstractMatrix`: Conductance matrix (size ``S \\times S``).
- `source::Int`: Source / release node index.
- `sink::Int`: Sink / recapture node index.
- `I_total::Real`: Injected electrical current (default: 1.0).

# Returns
- `V::Vector{Float64}`: Node electric potentials (length ``S``, with ``V_{\\text{sink}} = 0``).
- `R_eff::Float64`: Point-to-point effective resistance distance ``\\Omega(u, v)``.
- `edge_currents::SparseMatrixCSC{Float64, Int}`: Directed edge currents ``I_{ij}``.
"""
function solve_circuit_voltage(
    L::AbstractMatrix,
    C::AbstractMatrix,
    source::Int,
    sink::Int;
    I_total::Real = 1.0
)
    S = size(L, 1)
    if source < 1 || source > S || sink < 1 || sink > S
        throw(BoundsError(L, (source, sink)))
    end

    V = zeros(Float64, S)
    if source == sink
        return V, 0.0, spzeros(Float64, S, S)
    end

    # Find connected component of sink using BFS on conductance matrix C
    C_sp = sparse(C)
    visited = falses(S)
    queue = Int[sink]
    visited[sink] = true
    head = 1
    while head <= length(queue)
        curr = queue[head]
        head += 1
        for ptr in nzrange(C_sp, curr)
            nbr = rowvals(C_sp)[ptr]
            if nonzeros(C_sp)[ptr] > 0.0 && !visited[nbr]
                visited[nbr] = true
                push!(queue, nbr)
            end
        end
    end

    # If source is disconnected from sink, no current can flow
    if !visited[source]
        return V, Inf, spzeros(Float64, S, S)
    end

    # Reduce system by grounding sink node within the connected component
    active_indices = sort([i for i in queue if i != sink])
    n_act = length(active_indices)

    # Sub-Laplacian for active component nodes (strictly positive-definite)
    L_act = L[active_indices, active_indices]

    # Current injection vector b
    b_act = zeros(Float64, n_act)
    source_act_idx = searchsortedfirst(active_indices, source)
    b_act[source_act_idx] = Float64(I_total)

    # Solve grounded positive-definite Laplacian system
    L_act_sym = Symmetric(sparse(L_act))
    V_act = try
        cholesky(L_act_sym) \ b_act
    catch
        (Matrix(L_act) + 1e-10 * I) \ b_act
    end

    for (k, idx) in enumerate(active_indices)
        V[idx] = V_act[k]
    end
    V[sink] = 0.0

    R_eff = max(0.0, V[source] / Float64(I_total))

    # Compute directed edge currents I_ij = C_ij * (V_i - V_j)
    rows, cols, vals = Int[], Int[], Float64[]
    C_sp = sparse(C)
    for col in 1:S
        for ptr in nzrange(C_sp, col)
            row = rowvals(C_sp)[ptr]
            c_val = nonzeros(C_sp)[ptr]
            if c_val > 0.0
                i_val = c_val * (V[row] - V[col])
                push!(rows, row)
                push!(cols, col)
                push!(vals, i_val)
            end
        end
    end
    edge_currents = sparse(rows, cols, vals, S, S)

    return V, R_eff, edge_currents
end

"""
    pairwise_effective_resistance(
        L::AbstractMatrix,
        source::Int,
        sink::Int
    ) -> Float64

Computes the effective resistance distance between a single source and sink pair in
``\\mathcal{O}(|E| \\sqrt{\\kappa})`` time without inverting the full Laplacian.
"""
function pairwise_effective_resistance(
    L::AbstractMatrix,
    source::Int,
    sink::Int
)
    if source == sink
        return 0.0
    end
    S = size(L, 1)
    C = -L + spdiagm(0 => diag(L))
    _, r_eff, _ = solve_circuit_voltage(L, C, source, sink; I_total = 1.0)
    return r_eff
end

"""
    current_density_map(
        W::AbstractMatrix,
        sources::AbstractVector{Int},
        sinks::AbstractVector{Int};
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        conductance::Union{Nothing, AbstractVector{<:Real}, AbstractMatrix{<:Real}} = nothing,
        resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
    ) -> (
        current_density::Vector{Float64},
        total_edge_currents::SparseMatrixCSC{Float64, Int},
        L::SparseMatrixCSC{Float64, Int},
        C::SparseMatrixCSC{Float64, Int}
    )

Calculates the population-level migratory current density map across all spatial units
for a collection of release and recapture events.

# Mathematical Formulation
For each release-recapture event ``k`` with source ``u_k`` and sink ``v_k``, the branch
current along edge ``(i, j)`` is ``I^{(k)}_{ij} = C_{ij}(V^{(k)}_i - V^{(k)}_j)``.
The total flux passing through node ``i`` for event ``k`` is:
```math
J^{(k)}_i = \\begin{cases}
I_{\\text{total}} & \\text{if } i \\in \\{u_k, v_k\\} \\\\
\\frac{1}{2} \\sum_{j \\sim i} |I^{(k)}_{ij}| & \\text{if } i \\notin \\{u_k, v_k\\}
\\end{cases}
```
The aggregated population current density is the weighted sum across all ``K`` events:
```math
J_{\\text{pop}} = \\sum_{k=1}^K w_k J^{(k)}
```
Nodes with high current density highlight primary migration corridors and ecological
pinch-points.

# Arguments
- `W::AbstractMatrix`: Spatial graph adjacency matrix (size ``S \\times S``).
- `sources::AbstractVector{Int}`: Vector of release node indices.
- `sinks::AbstractVector{Int}`: Vector of recapture node indices.
- `weights`: Optional vector of event weights (e.g. tag observation counts).
- `conductance`: Optional custom conductance.
- `resistance`: Optional nodal resistance/friction vector.
- `hsi`: Optional Habitat Suitability Index vector.
- `land_mask`: Optional boolean mask (`true` for land units).

# Returns
- `current_density::Vector{Float64}`: Total current flux passing through each node (length ``S``).
- `total_edge_currents::SparseMatrixCSC{Float64, Int}`: Absolute branch current across each edge.
- `L`: System graph Laplacian.
- `C`: System conductance matrix.
"""
function current_density_map(
    W::AbstractMatrix,
    sources::AbstractVector{Int},
    sinks::AbstractVector{Int};
    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
    conductance::Union{Nothing, AbstractVector{<:Real}, AbstractMatrix{<:Real}} = nothing,
    resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)
    n_events = length(sources)
    if length(sinks) != n_events
        throw(DimensionMismatch("sources and sinks must have equal length"))
    end

    S = size(W, 1)
    L, C = build_circuit_laplacian(
        W;
        conductance = conductance,
        resistance = resistance,
        hsi = hsi,
        land_mask = land_mask
    )

    ev_weights = if isnothing(weights)
        fill(1.0 / max(1, n_events), n_events)
    else
        w = Float64.(weights)
        s = sum(w)
        s > 0 ? w ./ s : fill(1.0 / n_events, n_events)
    end

    current_density = zeros(Float64, S)
    total_edge_currents = spzeros(Float64, S, S)

    # Detect connected components to skip unreachable pairs
    g = SimpleGraph(S)
    for j in 1:S
        for i in (j + 1):S
            if C[i, j] > 0.0
                add_edge!(g, i, j)
            end
        end
    end
    comps = connected_components(g)
    comp_id = zeros(Int, S)
    for (cid, comp) in enumerate(comps)
        comp_id[comp] .= cid
    end

    for k in 1:n_events
        u = sources[k]
        v = sinks[k]
        w_k = ev_weights[k]
        (w_k <= 0.0 || u == v) && continue

        # Verify nodes are in valid range and water units
        if u < 1 || u > S || v < 1 || v > S
            continue
        end
        if !isnothing(land_mask) && (land_mask[u] || land_mask[v])
            continue
        end

        # Verify both nodes lie in the same connected water component
        if comp_id[u] == 0 || comp_id[u] != comp_id[v]
            continue
        end

        # Solve circuit voltages and currents for this transition
        V_k, _, edge_I_k = solve_circuit_voltage(L, C, u, v; I_total = 1.0)

        # Accumulate edge currents
        total_edge_currents .+= (w_k .* abs.(edge_I_k))

        # Accumulate nodal throughput current density
        # For intermediate nodes: J_i = 0.5 * sum_j |I_ij|
        # For source and sink: J_u = J_v = 1.0
        for i in 1:S
            if i == u || i == v
                current_density[i] += w_k * 1.0
            else
                row_flow = sum(abs.(edge_I_k[i, :]))
                if row_flow > 0.0
                    current_density[i] += w_k * (0.5 * row_flow)
                end
            end
        end
    end

    return current_density, total_edge_currents, L, C
end

"""
    identify_ecological_pinchpoints(
        current_density::AbstractVector{<:Real};
        resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
        top_quantile::Real = 0.90
    ) -> (
        pinch_mask::Vector{Bool},
        pinch_score::Vector{Float64},
        threshold::Float64
    )

Identifies critical migration pinch-points (chokepoints) where migratory current density
is disproportionately concentrated through constrained habitat corridors.

# Arguments
- `current_density::AbstractVector{<:Real}`: Nodal current density vector (length ``S``).
- `resistance`: Optional resistance/friction surface vector. If provided, pinch score is
  ``\\text{Score}_i = J_i \\times R_i``; otherwise ``\\text{Score}_i = J_i``.
- `top_quantile`: Quantile threshold for flagging pinch-points (default: 0.90).

# Returns
- `pinch_mask::Vector{Bool}`: Boolean indicator vector for identified pinch-points.
- `pinch_score::Vector{Float64}`: Quantitative bottleneck pinch scores across all units.
- `threshold::Float64`: Cutoff score value corresponding to `top_quantile`.
"""
function identify_ecological_pinchpoints(
    current_density::AbstractVector{<:Real};
    resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
    top_quantile::Real = 0.90
)
    S = length(current_density)
    pinch_score = zeros(Float64, S)

    for i in 1:S
        j_val = max(0.0, Float64(current_density[i]))
        r_val = !isnothing(resistance) ? max(1e-3, Float64(resistance[i])) : 1.0
        pinch_score[i] = j_val * r_val
    end

    active_scores = filter(s -> s > 1e-9, pinch_score)
    threshold = if isempty(active_scores)
        0.0
    else
        quantile(active_scores, clamp(Float64(top_quantile), 0.0, 1.0))
    end

    pinch_mask = [pinch_score[i] >= threshold && pinch_score[i] > 1e-9 for i in 1:S]

    return pinch_mask, pinch_score, threshold
end

"""
    PosteriorCircuitResult

Container holding posterior inference results for circuit-theoretic current density,
credible intervals, and bottleneck pinch-point probabilities propagated across
uncertain habitat suitability draws or observation error terms.

# Fields
- `mean_density::Vector{Float64}`: Posterior expected current density.
- `sd_density::Vector{Float64}`: Posterior standard deviation of current density.
- `ci_lower::Vector{Float64}`: Lower credible limit (e.g. 2.5% quantile).
- `ci_upper::Vector{Float64}`: Upper credible limit (e.g. 97.5% quantile).
- `median_density::Vector{Float64}`: Posterior median current density (50% quantile).
- `pinchpoint_prob::Vector{Float64}`: Posterior probability of bottleneck designation.
- `cv_density::Vector{Float64}`: Coefficient of variation `SD / (mean + eps)`.
- `draws_density::Matrix{Float64}`: Realizations across all posterior draws (size `S x M`).
- `mean_edge_currents::SparseMatrixCSC{Float64, Int}`: Posterior expected edge branch flux.
- `sources::Vector{Int}`: Evaluated source nodes.
- `sinks::Vector{Int}`: Evaluated sink nodes.
- `top_quantile::Float64`: Threshold quantile used for pinch-point designation.
"""
struct PosteriorCircuitResult
    mean_density::Vector{Float64}
    sd_density::Vector{Float64}
    ci_lower::Vector{Float64}
    ci_upper::Vector{Float64}
    median_density::Vector{Float64}
    pinchpoint_prob::Vector{Float64}
    cv_density::Vector{Float64}
    draws_density::Matrix{Float64}
    mean_edge_currents::SparseMatrixCSC{Float64, Int}
    sources::Vector{Int}
    sinks::Vector{Int}
    top_quantile::Float64
end

function Base.show(io::IO, res::PosteriorCircuitResult)
    S = length(res.mean_density)
    M = size(res.draws_density, 2)
    n_pinch = count(p -> p >= 0.80, res.pinchpoint_prob)
    println(io, "PosteriorCircuitResult:")
    println(io, "  Spatial Units (S):           $S")
    println(io, "  Posterior Draws (M):         $M")
    println(io, "  Max Mean Current Density:    $(round(maximum(res.mean_density), digits=4))")
    println(io, "  Robust Pinch-Points (P>=0.8): $n_pinch")
    print(io,   "  Top Quantile Threshold:      $(res.top_quantile)")
end

"""
    identify_stochastic_pinchpoints(
        res::PosteriorCircuitResult;
        prob_threshold::Real = 0.80
    ) -> (
        robust_mask::Vector{Bool},
        prob_threshold::Float64,
        n_robust::Int
    )

Identifies robust migration bottleneck pinch-points whose posterior probability of
being in the top `top_quantile` of current density meets or exceeds `prob_threshold`.

# Arguments
- `res::PosteriorCircuitResult`: Stochastic circuit analysis result.
- `prob_threshold::Real`: Posterior probability cutoff (default: 0.80, i.e., 80% certainty).

# Returns
- `robust_mask::Vector{Bool}`: Indicator vector for robust pinch-points.
- `prob_threshold::Float64`: Applied probability threshold.
- `n_robust::Int`: Number of identified robust pinch-points.
"""
function identify_stochastic_pinchpoints(
    res::PosteriorCircuitResult;
    prob_threshold::Real = 0.80
)
    thresh = clamp(Float64(prob_threshold), 0.0, 1.0)
    robust_mask = [res.pinchpoint_prob[i] >= thresh for i in eachindex(res.pinchpoint_prob)]
    return robust_mask, thresh, count(robust_mask)
end

"""
    posterior_circuit_inference(
        W::AbstractMatrix,
        sources::AbstractVector{Int},
        sinks::AbstractVector{Int};
        hsi_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        hsi_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hsi_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
        n_draws::Int = 50,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        top_quantile::Real = 0.90,
        ci_alpha::Real = 0.05,
        conductance_power::Real = 1.0,
        seed::Union{Nothing, Int} = nothing
    ) -> PosteriorCircuitResult

Performs Bayesian stochastic circuit-theoretic path and pinch-point inference by
propagating posterior uncertainty in habitat suitability index (HSI) or observation
error terms through the random walk electrical network.

# Mathematical Formulation
Deterministic circuit analysis computes current density ``\\mathbf{J} = f(\\hat{\\mathbf{H}})``
at a single point estimate ``\\hat{\\mathbf{H}}``, ignoring parameter and measurement
uncertainty. Under Bayesian modeling, the true habitat field has posterior distribution:
```math
\\mathbf{H}^{(m)} \\sim \\pi(\\mathbf{H} \\mid \\mathbf{y}), \\quad m = 1, \\dots, M
```
Alternatively, with observation errors ``\\sigma_{H, i}``, realizations are drawn via:
```math
H_i^{(m)} = \\text{clamp}(H_i^{\\text{obs}} + \\sigma_{H, i} Z_i^{(m)}, 0, 1)
```
with standard normal perturbation ``Z_i^{(m)} \\sim \\mathcal{N}(0, 1)``.
For each draw ``m``, the conductance matrix is updated:
```math
C_{uv}^{(m)} = \\left( \\frac{H_u^{(m)} + H_v^{(m)}}{2} \\right)^\\gamma
```
The Kirchhoff potential ``\\mathbf{V}^{(m)}`` and branch current fluxes are solved.
Across all ``M`` draws, the posterior expectation, uncertainty intervals, and bottleneck
probability are derived:
```math
\\mathbb{E}[J_i \\mid \\text{data}] = \\frac{1}{M} \\sum_{m=1}^M J_i^{(m)}
```
```math
P(\\text{PinchPoint}_i \\mid \\text{data}) =
\\frac{1}{M} \\sum_{m=1}^M \\mathbb{I}(J_i^{(m)} \\ge q_{0.90}^{(m)})
```

# Arguments
- `W::AbstractMatrix`: Spatial graph adjacency matrix (size ``S \\times S``).
- `sources::AbstractVector{Int}`: Release node indices.
- `sinks::AbstractVector{Int}`: Recapture / destination node indices.
- `hsi_samples`: Optional ``S \\times M`` matrix of MCMC posterior draws of HSI.
- `hsi_mean`: Optional posterior mean / estimate vector (length ``S``).
- `hsi_se`: Optional standard error / observation error vector (length ``S``).
- `n_draws`: Number of Monte Carlo samples if generating from `(hsi_mean, hsi_se)`.
- `weights`: Optional weights for source-sink events.
- `land_mask`: Optional boolean mask (`true` for land units).
- `top_quantile`: Quantile threshold for bottleneck pinch-points (default: 0.90).
- `ci_alpha`: Significance level for credible intervals (default: 0.05 for 95% CI).
- `conductance_power`: Exponent ``\\gamma`` for conductance scaling (default: 1.0).
- `seed`: Optional random seed for reproducibility.

# Returns
- `PosteriorCircuitResult`: Comprehensive summary of posterior current density,
  credible intervals, bottleneck probabilities, and edge branch fluxes.

# References
- McRae, B. H., Dickson, B. G., Keitt, T. H., & Shah, V. B. (2008). Using circuit theory to
  model connectivity in ecology, evolution, and conservation. Ecology, 89(10), 2712-2724.
- Hanks, E. M., & Hooten, M. B. (2013). Circuit theory and model-based inference for
  landscape connectivity. Journal of the American Statistical Association, 108(501), 22-33.
"""
function posterior_circuit_inference(
    W::AbstractMatrix,
    sources::AbstractVector{Int},
    sinks::AbstractVector{Int};
    hsi_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    hsi_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
    hsi_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
    n_draws::Int = 50,
    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    top_quantile::Real = 0.90,
    ci_alpha::Real = 0.05,
    conductance_power::Real = 1.0,
    seed::Union{Nothing, Int} = nothing
)
    S = size(W, 1)
    if length(sources) != length(sinks)
        throw(DimensionMismatch("sources and sinks must have identical length"))
    end

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # 1. Assemble posterior realizations of habitat suitability
    h_draws::Matrix{Float64} = if !isnothing(hsi_samples)
        if size(hsi_samples, 1) != S
            throw(DimensionMismatch(
                "hsi_samples rows ($(size(hsi_samples, 1))) must equal units S ($S)"
            ))
        end
        Float64.(hsi_samples)
    elseif !isnothing(hsi_mean)
        if length(hsi_mean) != S
            throw(DimensionMismatch(
                "hsi_mean length ($(length(hsi_mean))) must equal units S ($S)"
            ))
        end
        mean_vec = Float64.(hsi_mean)
        M_sim = max(1, n_draws)
        draws = zeros(Float64, S, M_sim)
        if !isnothing(hsi_se)
            if length(hsi_se) != S
                throw(DimensionMismatch(
                    "hsi_se length ($(length(hsi_se))) must equal units S ($S)"
                ))
            end
            se_vec = Float64.(hsi_se)
            for m in 1:M_sim
                z = randn(rng, S)
                draws[:, m] = clamp.(mean_vec .+ se_vec .* z, 0.0, 1.0)
            end
        else
            for m in 1:M_sim
                draws[:, m] = clamp.(mean_vec, 0.0, 1.0)
            end
        end
        draws
    else
        throw(ArgumentError(
            "Either hsi_samples or hsi_mean (with optional hsi_se) must be provided."
        ))
    end

    M_total = size(h_draws, 2)
    dens_mat = zeros(Float64, S, M_total)
    edge_accum = spzeros(Float64, S, S)
    pinch_counts = zeros(Int, S)
    t_quant = clamp(Float64(top_quantile), 0.0, 1.0)
    c_pow = Float64(conductance_power)

    # 2. Iterate across stochastic draws
    for m in 1:M_total
        h_m = h_draws[:, m]
        if c_pow != 1.0
            h_m = clamp.(h_m, 0.0, 1.0) .^ c_pow
        end

        cur_dens, edge_I, _, _ = current_density_map(
            W,
            sources,
            sinks;
            weights = weights,
            hsi = h_m,
            land_mask = land_mask
        )

        dens_mat[:, m] = cur_dens
        edge_accum .+= abs.(edge_I)

        # Flag bottleneck pinch-points for draw m
        act_scores = filter(s -> s > 1e-9, cur_dens)
        thresh_m = isempty(act_scores) ? 0.0 : quantile(act_scores, t_quant)
        for i in 1:S
            if cur_dens[i] >= thresh_m && cur_dens[i] > 1e-9
                pinch_counts[i] += 1
            end
        end
    end

    # 3. Compute posterior expectations and credible intervals
    mean_dens = vec(mean(dens_mat, dims=2))
    sd_dens = M_total > 1 ? vec(std(dens_mat, dims=2)) : zeros(Float64, S)
    alpha_lo = clamp(Float64(ci_alpha) / 2.0, 0.0, 0.5)
    alpha_hi = 1.0 - alpha_lo

    ci_lo = [quantile(dens_mat[i, :], alpha_lo) for i in 1:S]
    ci_hi = [quantile(dens_mat[i, :], alpha_hi) for i in 1:S]
    med_dens = [median(dens_mat[i, :]) for i in 1:S]
    pinch_prob = pinch_counts ./ Float64(M_total)
    cv_dens = [mean_dens[i] > 1e-9 ? sd_dens[i] / mean_dens[i] : 0.0 for i in 1:S]
    mean_edges = edge_accum ./ Float64(M_total)

    return PosteriorCircuitResult(
        mean_dens,
        sd_dens,
        ci_lo,
        ci_hi,
        med_dens,
        pinch_prob,
        cv_dens,
        dens_mat,
        mean_edges,
        Int.(sources),
        Int.(sinks),
        t_quant
    )
end

"""
    resistance_covariance_matrix(
        W::AbstractMatrix;
        length_scale::Real = 1.0,
        variance::Real = 1.0,
        noise::Real = 1e-6,
        kernel_type::Symbol = :exponential,
        centroids::Union{Nothing, AbstractVector} = nothing,
        L::Union{Nothing, AbstractMatrix} = nothing,
        R_mat::Union{Nothing, AbstractMatrix} = nothing,
        disconnected_dist::Real = 1e6
    ) -> Matrix{Float64}

Constructs a barrier-aware Gaussian Process spatial covariance matrix based on
effective resistance distances along the marine graph.

# Mathematical Formulation
Standard Euclidean covariance kernels
``k(u, v) = \\sigma^2 \\exp(-\\|\\mathbf{x}_u - \\mathbf{x}_v\\| / \\ell)``
erroneously leak spatial correlation across peninsulas, islands, and terrestrial barriers.
Because effective resistance distance ``\\Omega(u, v)`` is a conditionally negative-definite
metric on graphs (Klein & Randić, 1993), by Schoenberg's theorem the exponential kernel:
```math
K_{uv} = \\sigma^2 \\exp\\left( -\\frac{\\Omega(u, v)}{\\ell} \\right) + \\tau^2 \\delta_{uv}
```
is mathematically guaranteed to be positive semidefinite. Distance between two coastal
locations separated by an isthmus is measured strictly through contiguous water channels,
preventing spurious correlation leakage.

# Supported Kernel Types
- `:exponential`, `:matern12`: ``k(d) = \\sigma^2 \\exp(-d / \\ell)``
- `:matern32`: ``k(d) = \\sigma^2 (1 + \\sqrt{3}d/\\ell) \\exp(-\\sqrt{3}d/\\ell)``
- `:matern52`:
  ``k(d) = \\sigma^2 (1 + \\sqrt{5}d/\\ell + \\frac{5}{3}(d/\\ell)^2) \\exp(-\\sqrt{5}d/\\ell)``
- `:gaussian`, `:se`: ``k(d) = \\sigma^2 \\exp(-0.5 (d / \\ell)^2)``

# Arguments
- `W::AbstractMatrix`: Adjacency matrix with land edges severed (size ``S \\times S``).
- `length_scale::Real`: Spatial correlation range parameter ``\\ell > 0``.
- `variance::Real`: Marginal signal variance ``\\sigma^2 > 0`` (default: 1.0).
- `noise::Real`: Diagonal nugget / jitter parameter ``\\tau^2 \\ge 0`` (default: 1e-6).
- `kernel_type::Symbol`: Covariance function family (default: `:exponential`).
- `centroids`: Optional coordinates vector to scale resistance to physical distance.
- `L`: Optional precomputed graph Laplacian.
- `R_mat`: Optional precomputed pairwise effective resistance matrix.
- `disconnected_dist`: Assigned distance for disconnected components (default: 1e6).

# Returns
- `Matrix{Float64}`: Symmetric positive-definite ``S \\times S`` barrier-aware covariance matrix.
"""
function resistance_covariance_matrix(
    W::AbstractMatrix;
    length_scale::Real = 1.0,
    variance::Real = 1.0,
    noise::Real = 1e-6,
    kernel_type::Symbol = :exponential,
    centroids::Union{Nothing, AbstractVector} = nothing,
    L::Union{Nothing, AbstractMatrix} = nothing,
    R_mat::Union{Nothing, AbstractMatrix} = nothing,
    disconnected_dist::Real = 1e6
)
    S = size(W, 1)
    ls = max(1e-6, Float64(length_scale))
    sig2 = max(1e-6, Float64(variance))
    nugget = max(0.0, Float64(noise))

    # Compute or retrieve pairwise effective resistance distance matrix
    Omega = if !isnothing(R_mat)
        R_mat
    else
        L_use = !isnothing(L) ? L : build_circuit_laplacian(W)[1]
        effective_resistance_matrix(
            L_use;
            W = W,
            centroids = centroids,
            disconnected_dist = disconnected_dist
        )
    end

    K = zeros(Float64, S, S)

    for j in 1:S
        for i in 1:j
            r_val = Omega[i, j]
            k_val = 0.0

            if isfinite(r_val) && r_val < (0.9 * disconnected_dist)
                d = r_val / ls
                if kernel_type in (:exponential, :matern12)
                    k_val = sig2 * exp(-d)
                elseif kernel_type == :matern32
                    s3 = sqrt(3.0) * d
                    k_val = sig2 * (1.0 + s3) * exp(-s3)
                elseif kernel_type == :matern52
                    s5 = sqrt(5.0) * d
                    k_val = sig2 * (1.0 + s5 + (5.0 / 3.0) * (d^2)) * exp(-s5)
                elseif kernel_type in (:gaussian, :se, :rbf)
                    k_val = sig2 * exp(-0.5 * (d^2))
                else
                    k_val = sig2 * exp(-d)
                end
            end

            # Diagonal nugget
            if i == j
                k_val += nugget
            end

            K[i, j] = k_val
            K[j, i] = k_val
        end
    end

    return Symmetric(K)
end
