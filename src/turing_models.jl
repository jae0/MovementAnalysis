"""
    turing_models.jl

Explicit Turing.jl probabilistic formulations for animal movement trajectory
estimation and joint survey count + mark-recapture telemetry modeling.
"""

using Turing
using Distributions
using LinearAlgebra
using SparseArrays

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
- `residence`: Diagonal residence probability parameter (``\\rho``).
- `advection`: Advection weighting parameter (``\\alpha``).
- `land_mask`: Optional boolean mask denoting impermeable units.

# Returns
- `SparseMatrixCSC`: Row-stochastic transition probability matrix.
"""
function build_sparse_transition_kernel(
    W::SparseMatrixCSC{<:Real, <:Integer},
    hsi::AbstractVector{<:Real},
    gamma, residence, advection, land_mask;
    rest_coupling::Real = 0.0,
    rest_advantage_form::Symbol = :difference
)
    S = size(W, 1)
    T = promote_type(Float64, eltype(W), typeof(gamma), typeof(residence), typeof(advection))
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

    # Stay probability. With no coupling this is the single global `residence` for
    # every unit, so the habitat-aware path is a strict generalisation rather than
    # a change of default. The advantage is computed in plain Float64 away from the
    # dual-typed loop: it feeds only a clamp, so it needs no gradient.
    rho_vec = if rest_coupling == 0
        fill(clamp(Float64(residence), 0.0, 0.999), S)
    else
        residency_from_advantage(
            residence,
            local_hsi_advantage(hsi, W; form = rest_advantage_form),
            rest_coupling,
            rest_advantage_form,
        )
    end

    for i in 1:S
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
        rho_i   = clamp(T(rho_vec[i]), T(0), T(0.999))
        w_move  = one(T) - rho_i
        w_adv   = w_move * alpha
        w_diff  = w_move * (one(T) - alpha)
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
    pure_telemetry_turing_model(releases, recaptures, ks, W, hsi, land_mask)

Pure mark-recapture telemetry transition model evaluated over spatial graph nodes.
Estimates pooled advection velocity, diffusion rate, and habitat gradient
responsiveness (gamma) parameters for all telemetry observations.

# Arguments
- `releases::Vector{Int}`: Node index at release (1-indexed).
- `recaptures::Vector{Int}`: Node index at recapture (1-indexed).
- `ks::Vector{Int}`: Elapsed time steps between release and recapture.
- `W::SparseMatrixCSC{Float64, Int}`: Spatial mesh adjacency matrix.
- `hsi::Vector{Float64}`: Habitat suitability index per spatial unit.
- `land_mask`: Optional boolean vector denoting impermeable barriers.
"""
@model function pure_telemetry_turing_model(
  releases::Vector{Int},
  recaptures::Vector{Int},
  ks::Vector{Int},
  W::SparseMatrixCSC{Float64, Int},
  hsi::Vector{Float64},
  land_mask::Union{Nothing, BitVector, Vector{Bool}};
  rest_coupling::Real = 0.0,
  rest_advantage_form::Symbol = :difference
  )
    velocity  ~ truncated(Normal(0.3, 0.2), 0.0, 0.95)
    diffusion ~ truncated(Normal(0.1, 0.2), 0.0, Inf)
    gamma     ~ Normal(1.0, 1.0)

    tot = velocity + diffusion + 1e-6
    alpha = clamp(velocity / tot, 0.0, 1.0)
    rho   = clamp(1.0 / (1.0 + tot), 0.01, 0.95)

    P = build_sparse_transition_kernel(W, hsi, gamma, rho, alpha, land_mask;
        rest_coupling = rest_coupling, rest_advantage_form = rest_advantage_form)

    unique_rel_k = unique(zip(releases, ks))
    P_row_cache = Dict{Tuple{Int, Int}, Vector}()
    P_t = sparse(P')
    for (rel, k_n) in unique_rel_k
        v = zeros(eltype(P), size(W, 1))
        v[rel] = 1.0
        k_steps = max(0, k_n)
        for _ in 1:k_steps
            v = P_t * v
            s_v = sum(v)
            if s_v > 1e-12
                v ./= s_v
            end
        end
        P_row_cache[(rel, k_n)] = v
    end

    N = length(releases)
    for n in 1:N
        rel = releases[n]; rec = recaptures[n]; k_n = ks[n]
        p_row = P_row_cache[(rel, k_n)]
        p_sum = sum(p_row)
        rec_prob = (p_sum > 1e-12 && !isnan(p_sum)) ?
                   (p_row[rec] / p_sum) : (1.0 / length(p_row))
        prob = (isnan(rec_prob) || rec_prob < 1e-12) ? 1e-12 : min(rec_prob, 1.0)
        Turing.@addlogprob! log(prob)
    end
end

"""
    joint_survey_telemetry_turing_model(counts, depths, releases, recaptures, ks, W, hsi, land_mask)

Joint model integrating scientific survey count density (Negative Binomial likelihood)
with mark-recapture telemetry transition probabilities.
"""
@model function joint_survey_telemetry_turing_model(
    counts::Vector{Int},
    depths::Vector{Float64},
    releases::Vector{Int},
    recaptures::Vector{Int},
    ks::Vector{Int},
  W::SparseMatrixCSC{Float64, Int},
  hsi::Vector{Float64},
  land_mask::Union{Nothing, BitVector, Vector{Bool}};
  rest_coupling::Real = 0.0,
  rest_advantage_form::Symbol = :difference
  )
    beta0      ~ Normal(0.0, 5.0)
    beta_depth ~ Normal(0.0, 2.0)
    inv_r      ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    r_nb       = 1.0 / max(inv_r, 1e-6)

    velocity  ~ truncated(Normal(0.3, 0.2), 0.0, 0.95)
    diffusion ~ truncated(Normal(0.1, 0.2), 0.0, Inf)
    gamma     ~ Normal(1.0, 1.0)

    mu = exp.(clamp.(beta0 .+ beta_depth .* (depths ./ 100.0), -20.0, 20.0))
    prob_nb = r_nb ./ (r_nb .+ mu)
    for i in 1:length(counts)
        counts[i] ~ NegativeBinomial(r_nb, prob_nb[i])
    end

    tot = velocity + diffusion + 1e-6
    alpha = clamp(velocity / tot, 0.0, 1.0)
    rho   = clamp(1.0 / (1.0 + tot), 0.01, 0.95)

    P = build_sparse_transition_kernel(W, hsi, gamma, rho, alpha, land_mask;
        rest_coupling = rest_coupling, rest_advantage_form = rest_advantage_form)

    unique_rel_k = unique(zip(releases, ks))
    P_row_cache = Dict{Tuple{Int, Int}, Vector}()
    P_t = sparse(P')
    for (rel, k_n) in unique_rel_k
        v = zeros(eltype(P), size(W, 1))
        v[rel] = 1.0
        k_steps = max(0, k_n)
        for _ in 1:k_steps
            v = P_t * v
            s_v = sum(v)
            if s_v > 1e-12
                v ./= s_v
            end
        end
        P_row_cache[(rel, k_n)] = v
    end

    N = length(releases)
    for n in 1:N
        rel = releases[n]; rec = recaptures[n]; k_n = ks[n]
        p_row = P_row_cache[(rel, k_n)]
        p_sum = sum(p_row)
        rec_prob = (p_sum > 1e-12 && !isnan(p_sum)) ?
                   (p_row[rec] / p_sum) : (1.0 / length(p_row))
        prob = (isnan(rec_prob) || rec_prob < 1e-12) ? 1e-12 : min(rec_prob, 1.0)
        Turing.@addlogprob! log(prob)
    end
end

