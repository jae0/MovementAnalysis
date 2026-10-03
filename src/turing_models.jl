"""
    turing_models.jl

Explicit Turing.jl probabilistic formulations for animal movement trajectory
estimation and joint survey count + mark-recapture telemetry modeling.
"""

using Turing
using Distributions
using LinearAlgebra
using SparseArrays

# =============================================================================
# Shared kernel parameter derivation
# =============================================================================

"""
    movement_alpha_rho(velocity, diffusion) -> (alpha, rho)

Derive the two composite kernel parameters from advection velocity and diffusion.

    alpha = v / (v + d)      advection weight: fraction of movement that is directed
    rho   = 1 / (1 + v + d)  residence: diagonal self-transition probability

Both are **derived, not fitted**: they are deterministic functions of the two free
movement parameters, so the model carries two free parameters rather than four.

This lives in one place because it used to exist in two, and they drifted apart.
`src/pipeline.jl` had `rho = 1 / (v + d)` while every model had
`rho = 1 / (1 + v + d)` — the missing `1.0 +` meant the reported and propagated
residence parameter was a different quantity from the one the likelihood was
fitted with, and it saturated the `0.95` clamp over a far wider region of
parameter space as a result. A model and the code that reports its posterior must
not each re-derive the same transform.

Arguments accept scalars or arrays and broadcast, so this serves both a single
model evaluation and a vector of posterior draws.
"""
function movement_alpha_rho(velocity::Real, diffusion::Real)
    tot = velocity + diffusion + 1e-6
    alpha = clamp(velocity / tot, 0.0, 1.0)
    rho   = clamp(1.0 / (1.0 + tot), 0.01, 0.95)
    return alpha, rho
end

function movement_alpha_rho(velocity::AbstractArray, diffusion::AbstractArray)
    tot = velocity .+ diffusion .+ 1e-6
    alpha = clamp.(velocity ./ tot, 0.0, 1.0)
    rho   = clamp.(1.0 ./ (1.0 .+ tot), 0.01, 0.95)
    return alpha, rho
end

"""
    build_sparse_transition_kernel(W, hsi, gamma, residence, advection, land_mask)

Memory-efficient stochastic transition kernel constructor for discrete-time telemetry models.
Generates a `SparseMatrixCSC` instead of a dense matrix to vastly reduce memory allocations
during Turing MCMC sampling.

# Mathematical Formulation
Constructs transition probabilities:
```math
T_{ij} = (1 - \\rho) \\left[ (1 - \\alpha) T^{\\text{diff}}_{ij} + \\alpha \\frac{\\exp(\\gamma(U_j - U_i))}{\\sum_k \\exp(\\gamma(U_k - U_i))} \\right]
```
where ``T^{\\text{diff}}`` is the unbiased random walk matrix.

# Arguments
- `W::SparseMatrixCSC{Float64, Int}`: Spatial adjacency matrix.
- `hsi::Vector{Float64}`: Habitat suitability index vector.
- `gamma`: Gradient responsiveness parameter.
- `residence`: Diagonal residence probability (``\\rho``). A scalar, or a
  length-`S` vector giving a per-unit residency, which is how habitat coupling
  enters: a vector built by `residency_from_advantage` makes an animal more
  likely to stay put where the habitat is locally favourable. A scalar makes
  every unit behave identically.
- `advection`: Advection weighting parameter (``\\alpha``).
- `land_mask`: Optional boolean mask denoting impermeable units.

# Returns
- `SparseMatrixCSC`: Row-stochastic transition probability matrix.
"""
function build_sparse_transition_kernel(
    W::SparseMatrixCSC{<:Real, <:Integer},
    hsi::AbstractVector{<:Real},
    gamma, residence, advection, land_mask
)
    S = size(W, 1)
    if length(hsi) != S
        throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), but hsi has length $(length(hsi))."
        ))
    end
    if !isnothing(land_mask) && length(land_mask) != S
        throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), but land_mask has length $(length(land_mask))."
        ))
    end
    # `residence` may be a scalar or a per-unit vector, so promote on its element
    # type: typeof() of a vector would widen T to Any and break every constructor.
    residence_eltype = residence isa AbstractVector ? eltype(residence) : typeof(residence)
    T = promote_type(Float64, eltype(W), typeof(gamma), residence_eltype, typeof(advection))
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    sizehint!(I_idx, nnz(W) + S)
    sizehint!(J_idx, nnz(W) + S)
    sizehint!(V_val, nnz(W) + S)

    rows = rowvals(W)
    vals = nonzeros(W)

    # Use T throughout so ForwardDiff.Dual partials are preserved
    alpha = clamp(T(advection), T(0), T(1))
    gam    = T(gamma)

    # `residence` may be a scalar or a per-unit vector. A scalar keeps the
    # historical behaviour exactly; a vector makes residency habitat-coupled,
    # which is what `residency_from_advantage` produces. Indexing the vector
    # inside the loop keeps each element on the AD tape.
    unit_residence = residence isa AbstractVector
    if unit_residence
        length(residence) == S || throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), but residence has length " *
            "$(length(residence))."
        ))
    end
    rho_shared = unit_residence ? zero(T) : clamp(T(residence), T(0), T(0.999))

    for i in 1:S
        rho = unit_residence ? clamp(T(residence[i]), T(0), T(0.999)) : rho_shared
        w_move = one(T) - rho
        w_adv  = w_move * alpha
        w_diff = w_move * (one(T) - alpha)

        if !isnothing(land_mask) && land_mask[i]
            push!(I_idx, i)
            push!(J_idx, i)
            push!(V_val, one(T))
            continue
        end

        h_i = T(hsi[i])

        # 1. Identify eligible marine neighbors (excluding self-loops)
        nbrs  = Int[]
        nbr_w = T[]
        for idx in nzrange(W, i)
            j = rows[idx]
            if j != i && (isnothing(land_mask) || !land_mask[j])
                push!(nbrs, j)
                push!(nbr_w, T(vals[idx]))
            end
        end

        if isempty(nbrs)
            push!(I_idx, i)
            push!(J_idx, i)
            push!(V_val, one(T))
            continue
        end

        # 2. Diffusive weights: normalized by total marine degree
        sum_deg      = sum(nbr_w)
        diff_weights = sum_deg > zero(T) ?
            nbr_w ./ sum_deg :
            fill(one(T) / length(nbrs), length(nbrs))

        # 3. Directed taxis weights: log-sum-exp numerically stable softmax
        dh = [clamp(gam * (T(hsi[j]) - h_i), T(-25), T(25)) for j in nbrs]
        max_dh  = maximum(dh)
        exp_dh  = [w * exp(d - max_dh) for (w, d) in zip(nbr_w, dh)]
        sum_exp = sum(exp_dh)
        tax_weights = sum_exp > zero(T) ? exp_dh ./ sum_exp : diff_weights

        # 4. Assemble off-diagonal transition probabilities
        row_sum = zero(T)
        for (k_idx, j) in enumerate(nbrs)
            val = w_adv * tax_weights[k_idx] + w_diff * diff_weights[k_idx]
            if val > T(1e-12)
                push!(I_idx, i)
                push!(J_idx, j)
                push!(V_val, val)
                row_sum += val
            end
        end

        # 5. Diagonal residence probability guarantees exact row-stochasticity
        diag_val = max(zero(T), one(T) - row_sum)
        push!(I_idx, i)
        push!(J_idx, i)
        push!(V_val, diag_val)
    end
    return sparse(I_idx, J_idx, V_val, S, S)
end

"""
    _rel_probs(probs) -> Vector{Float64}

Return a probability vector acceptable to `Categorical`.

The transition rows are non-negative by construction, but a row that underflows to
all zeros cannot be normalised. Such a row is replaced by a uniform distribution
over the mesh, so the observation contributes a constant rather than a `-Inf` that
would reject every proposal. Structurally impossible release/recapture pairs are
excluded upstream, so this is a numerical guard rather than a modelling choice.
"""
function _rel_probs(row::AbstractVector{<:Real})
    # `collect`, not `Float64.`: this is the whole gradient path. Casting to
    # Float64 here silently severs the parameters from the AD tape, which is why
    # `NUTS` failed with `Float64(::ForwardDiff.Dual)` and why `MH(cov)` could
    # not be used either -- a linked-space random walk needs the Jacobian, so it
    # also runs under ForwardDiff. The element type must survive.
    p = collect(row)
    isempty(p) && return p
    T = eltype(p)
    s = sum(p)
    if !isfinite(s) || s <= 0
        return fill(T(inv(length(p))), length(p))
    end
    p ./= s
    # Guard against an exact zero surviving normalisation, which Categorical cannot
    # represent. The floor is relative to the largest entry so it never displaces a
    # probability the model actually relies on.
    pmax = maximum(p)
    pmax > 0 || return fill(T(inv(length(p))), length(p))
    floor_ = pmax * 1e-12
    @inbounds for i in eachindex(p)
        p[i] < floor_ && (p[i] = floor_)
    end
    return p ./ sum(p)
end

"""
    kstep_transition_cache(P_T, releases, ks; normalize = true) -> Dict

Pre-compute the ``k``-step transition row ``(Pᵀ)^k e_rel`` for every
`(release, k)` pair the likelihood will ask for.

# Why this exists
The obvious construction is to loop over the distinct `(release, k)` pairs and,
for each, apply `P_T` `k` times. That is what this replaced, and it was the single
largest cost in the whole analysis — on the snow crab dataset, 969,597 sparse
mat-vecs per likelihood evaluation, about 63.6 GFLOP.

It is almost entirely redundant work. Only **123** distinct release units appear
in 5,294 observations, and `k` reaches 2,278. The iterates ``e_rel, Pᵀe_rel,
Pᵀ²e_rel, …`` are a *prefix*, so all the `k` values wanted for one release node
are obtained from a single sweep that walks the chain once and stores the rows it
passes through. That is 89,523 mat-vecs instead — measured at 43.3 s → 4.5 s on
this dataset, with the resulting rows **bit-identical** to the naive scheme
(max absolute difference 0.0 across all 2,491 cached rows).

The saving is roughly ``Σ k`` over distinct pairs divided by ``Σ max(k)`` over
distinct release nodes, so it grows as the number of distinct intervals per
release site grows.

# Arguments
- `P_T`: transposed transition matrix, so the sweep walks the chain forward.
- `releases`, `ks`: per-observation release index and elapsed step count.
- `normalize`: apply [`_rel_probs`](@ref) to each stored row. The kernel is exactly
  row-stochastic, so this is a no-op on well-formed input; it is kept because it
    also floors exact zeros, which `Categorical` cannot represent.

# Notes
`k = 0` is handled explicitly: it is the starting point of the sweep, not a step
within it, so it is stored before the loop. The result is indexed by the *clamped*
`k` so that callers which cap the step count (the posterior predictive check caps
at 10) get the same row for every `k` above the cap, as they did before.
"""
function kstep_transition_cache(
    P_T::AbstractMatrix{ET},
    releases::AbstractVector{<:Integer},
    ks::AbstractVector{<:Integer};
    normalize::Bool = true
)::Dict{Tuple{Int,Int},Vector{ET}} where {ET<:Real}
    S = size(P_T, 1)
    steps = max.(0, Int.(ks))

    # Which step counts are actually wanted from each release node.
    wanted = Dict{Int,Set{Int}}()
    for (r, k) in zip(releases, steps)
        1 <= r <= S || continue
        push!(get!(wanted, Int(r), Set{Int}()), k)
    end

    # The accumulator takes its element type from `P_T`, not from `Float64`. With
    # a Float64 kernel this is identical to before, and bit-for-bit so; with a
    # kernel carrying `ForwardDiff.Dual` entries the cache stays on the AD tape,
    # which is the only way a gradient sampler can see the k-step propagation at
    # all. A hard `Float64` here would discard the gradient of every cached row.
    T = eltype(P_T)
    store(v) = normalize ? _rel_probs(v) : collect(v)

    cache = Dict{Tuple{Int,Int},Vector{T}}()
    for (rel, needed) in wanted
        e = zeros(T, S)
        e[rel] = one(T)
        0 in needed && (cache[(rel, 0)] = store(e))
        kmax = maximum(needed)
        v = e
        for k in 1:kmax
            v = P_T * v
            k in needed && (cache[(rel, k)] = store(v))
        end
    end
    return cache
end

"""
    pure_telemetry_turing_model(releases, recaptures, ks, W, hsi, land_mask)

Mark-recapture telemetry model for a single, unstratified population.

Estimates advection velocity, diffusion rate, and habitat gradient responsiveness
(gamma), which drive the transition kernel

    T = (1 - rho) [(1 - alpha) T_diff + alpha A(eta)] + rho I

`alpha` and `rho` are derived from velocity and diffusion by
[`movement_alpha_rho`](@ref) rather than fitted separately, so the model carries
**three** free parameters: `mu_velocity`, `mu_diffusion`, `mu_gamma`.

# Arguments
- `releases::Vector{Int}`: Unit index at release (1-indexed).
- `recaptures::Vector{Int}`: Unit index at recapture (1-indexed).
- `ks::Vector{Int}`: Elapsed discrete time steps between release and recapture.
- `W::SparseMatrixCSC{Float64, Int}`: Spatial mesh adjacency matrix.
- `hsi::Vector{Float64}`: Habitat suitability index per spatial unit.
- `land_mask`: Optional boolean vector denoting impermeable barriers.

# Likelihood
    recapture ~ Categorical(P^k[release, :])

# Note on individual heterogeneity
This is a **population-level** model. An earlier revision carried per-individual
random effects (`z_velocity`, `z_diffusion`, `z_gamma`, one 3,766-element vector
each for this dataset), taking the sampled space from 3 dimensions to roughly
11,300. Those effects never reached the likelihood: the kernel was built from
their population mean, so all but one direction of each block was exactly the
prior, and the informative direction was pinned near zero by the prior on the
mean.

They were removed rather than repaired because there was nothing to repair — see
`todo.md` §1.5. If individual heterogeneity is scientifically required it has to
enter the likelihood per animal, which is a different model.

This does not make `MH()` faster. Measured on a 40-unit mesh with 2,000
observations, the old and new models cost the same per draw (5.0 ms), because
sampling cost here is dominated by rebuilding the transition kernel and the
k-step cache on every likelihood evaluation, not by the number of parameters. The
win is correctness, plus removing the obstacle to a gradient sampler later.
"""
@model function pure_telemetry_turing_model(
    releases::Vector{Int},
    recaptures::Vector{Int},
    ks::Vector{Int},
    W::SparseMatrixCSC{Float64, Int},
    hsi::Vector{Float64},
    land_mask::Union{Nothing, BitVector, Vector{Bool}}
)
    mu_velocity  ~ truncated(Distributions.Normal(0.3, 0.2), 0.0, 0.95)
    mu_diffusion ~ truncated(Distributions.Normal(0.1, 0.2), 0.0, Inf)
    mu_gamma     ~ Distributions.Normal(1.0, 1.0)

    velocity  = clamp(mu_velocity,  0.0, 0.95)
    diffusion = max(mu_diffusion,  0.0)
    gamma     = mu_gamma

    alpha, rho = movement_alpha_rho(velocity, diffusion)

    P = build_sparse_transition_kernel(W, hsi, gamma, rho, alpha, land_mask)
    P_T = sparse(P')

    # Pre-compute the k-step transition row for each (release, k) pair the
    # likelihood asks for, so the loop below is a table lookup rather than a
    # matrix power. One sweep per release node, not per pair -- see
    # `kstep_transition_cache`.
    cache = kstep_transition_cache(P_T, releases, ks)

    for n in eachindex(releases)
        recaptures[n] ~ Categorical(cache[(releases[n], ks[n])])
    end
end

"""
    joint_survey_telemetry_turing_model(counts, depths, releases, recaptures, ks, W, hsi, land_mask)

Joint scientific survey density and mark-recapture telemetry model for a single,
unstratified population.

# !! The survey density likelihood is not implemented
`counts` and `depths` are accepted but **not used**: there is no
`counts[n] ~ NegBin(...)` term, so this model currently fits the telemetry
component alone and is identical to
[`pure_telemetry_turing_model`](@ref) with two unused arguments.

This is recorded rather than papered over. The Negative Binomial dispersion
`inv_r` and the depth coefficients `beta0`/`beta_depth` are declared below only to
make the omission visible in the code; they are prior-only and do not reach the
likelihood. A warning is emitted on every construction for the same reason.

Until the survey term is written, the honest description of this function is "a
telemetry model that also accepts survey columns and ignores them". It is
retained, rather than deleted, because the survey-dependent approaches are
declared in `model_modes` and something must answer for them.
"""
@model function joint_survey_telemetry_turing_model(
    counts::Vector{Int},
    depths::Vector{Float64},
    releases::Vector{Int},
    recaptures::Vector{Int},
    ks::Vector{Int},
    W::SparseMatrixCSC{Float64, Int},
    hsi::Vector{Float64},
    land_mask::Union{Nothing, BitVector, Vector{Bool}}
)
    @warn "joint_survey_telemetry_turing_model has no survey density likelihood: " *
          "counts/depths are ignored and this reduces to pure_telemetry_turing_model. " *
          "See todo.md 1.6."

    # Declared but unused -- see the note above. Present so the gap is visible.
    beta0      ~ Distributions.Normal(0.0, 5.0)
    beta_depth ~ Distributions.Normal(0.0, 2.0)
    inv_r      ~ truncated(Distributions.Normal(0.0, 1.0), 0.0, Inf)

    mu_velocity  ~ truncated(Distributions.Normal(0.3, 0.2), 0.0, 0.95)
    mu_diffusion ~ truncated(Distributions.Normal(0.1, 0.2), 0.0, Inf)
    mu_gamma     ~ Distributions.Normal(1.0, 1.0)

    velocity  = clamp(mu_velocity,  0.0, 0.95)
    diffusion = max(mu_diffusion,  0.0)
    gamma     = mu_gamma

    alpha, rho = movement_alpha_rho(velocity, diffusion)

    P = build_sparse_transition_kernel(W, hsi, gamma, rho, alpha, land_mask)
    P_T = sparse(P')

    # One sweep per release node -- see `kstep_transition_cache`.
    cache = kstep_transition_cache(P_T, releases, ks)

    for n in eachindex(releases)
        recaptures[n] ~ Categorical(cache[(releases[n], ks[n])])
    end
end
