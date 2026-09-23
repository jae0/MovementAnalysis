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
    gamma, residence, advection, land_mask
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
    rho   = clamp(T(residence), T(0), T(0.999))
    alpha = clamp(T(advection), T(0), T(1))
    w_move = one(T) - rho
    w_adv  = w_move * alpha
    w_diff = w_move * (one(T) - alpha)
    gam    = T(gamma)

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
    calculate_ssa_transition_row(Q::SparseMatrixCSC, dt::Real, rel::Int)

Computes the transition probability vector for a single release location `rel` over 
a continuous time interval `dt`, evaluating the specific row of the matrix exponential 
``T(\\Delta t) = \\exp(Q \\Delta t)``.

Uses the Uniformization (Poisson-Krylov) algorithm applied to a sparse unit vector 
to compute ``e_{\\text{rel}}^T \\exp(Q \\Delta t)``. This avoids forming the full dense 
matrix exponential, saving gigabytes of memory during continuous-time MCMC sampling.

# Mathematical Formulation
```math
\\text{Pr}(\\cdot \\mid s_{\\text{rel}}, \\Delta t) = \\sum_{k=0}^{\\infty} e^{-\\lambda} \\frac{\\lambda^k}{k!} v_0^T (T_{\\text{unif}})^k
```
where ``\\lambda = \\max |Q_{ii}| \\Delta t`` and ``T_{\\text{unif}} = I + Q / \\max |Q_{ii}|``.

# Arguments
- `Q::SparseMatrixCSC`: Infinitesimal generator matrix.
- `dt::Real`: Continuous elapsed time (``\\Delta t``).
- `rel::Int`: Starting release node index.

# Returns
- `Vector`: Dense probability distribution vector for the specified release node.
"""
function calculate_ssa_transition_row(Q::SparseMatrixCSC, dt::Real, rel::Int)
    S = size(Q, 1)
    q_diag = abs.(diag(Q))
    alpha = maximum(q_diag)
    
    T = promote_type(eltype(Q), typeof(dt))
    v = zeros(T, S)
    if alpha < 1e-12 || dt <= 0.0
        v[rel] = 1.0
        return v
    end

    lambda = alpha * dt
    P_unif_T = sparse(I, S, S) .+ sparse(Q') ./ alpha

    v_k = zeros(T, S)
    v_k[rel] = 1.0
    
    pois_k = exp(-lambda)
    v .= pois_k .* v_k
    
    k = 1
    while true
        pois_k = pois_k * lambda / k
        if pois_k < 1e-8 && k > lambda
            break
        end
        v_k = P_unif_T * v_k
        v .+= pois_k .* v_k
        k += 1
        if k > 1000
            break
        end
    end
    
    rs = sum(v)
    if rs > 1e-12
        v ./= rs
    else
        v .= 0.0
        v[rel] = 1.0
    end
    return v
end

"""
    pure_telemetry_turing_model(releases, recaptures, ks, groups, W, hsi, land_mask, G; max_k)

Pure mark-recapture telemetry transition model evaluated over spatial graph nodes.
Estimates group-stratified advection velocity, diffusion rate, and habitat gradient
responsiveness (gamma) parameters driving stochastic transition probability kernels:
    T_g = (1 - rho_g)[(1 - alpha_g) T_diff + alpha_g A_g(eta)] + rho_g I

# Arguments
- `releases::Vector{Int}`: Node index at release (1-indexed).
- `recaptures::Vector{Int}`: Node index at recapture (1-indexed).
- `ks::Vector{Int}`: Elapsed time steps between release and recapture.
- `groups::Vector{Int}`: Biological/demographic group index (1-indexed, 1:G).
- `W::SparseMatrixCSC{Float64, Int}`: Spatial mesh adjacency matrix.
- `hsi::Vector{Float64}`: Habitat suitability index per spatial unit.
- `land_mask`: Optional boolean vector denoting impermeable barriers.
- `G::Int`: Total number of demographic groups.
"""
@model function pure_telemetry_turing_model(
    releases::Vector{Int},
    recaptures::Vector{Int},
    ks::Vector{Int},
    groups::Vector{Int},
    W::SparseMatrixCSC{Float64, Int},
    hsi::Vector{Float64},
    land_mask::Union{Nothing, BitVector, Vector{Bool}},
    G::Int
)
    velocity  ~ filldist(truncated(Normal(0.3, 0.2), 0.0, 0.95), G)
    diffusion ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf), G)
    gamma     ~ filldist(Normal(1.0, 1.0), G)

    tot = velocity .+ diffusion .+ 1e-6
    alpha = clamp.(velocity ./ tot, 0.0, 1.0)
    rho   = clamp.(1.0 ./ (1.0 .+ tot), 0.01, 0.95)

    P_kernels = Vector{SparseMatrixCSC}(undef, G)
    for g in 1:G
        P_kernels[g] = build_sparse_transition_kernel(W, hsi, gamma[g], rho[g], alpha[g], land_mask)
    end

    unique_rel_k = unique(zip(releases, ks, groups))
    P_row_cache = Dict{Tuple{Int, Int, Int}, Vector}()
    for (rel, k_n, g) in unique_rel_k
        v = zeros(eltype(P_kernels[g]), size(W, 1))
        v[rel] = 1.0
        P_g_T = sparse(P_kernels[g]')
        k_steps = max(0, k_n)
        for _ in 1:k_steps
            v = P_g_T * v
            s_v = sum(v)
            if s_v > 1e-12
                v ./= s_v
            end
        end
        P_row_cache[(rel, k_n, g)] = v
    end

    N = length(releases)
    for n in 1:N
        rel = releases[n]; rec = recaptures[n]; k_n = ks[n]; g = groups[n]
        p_row = P_row_cache[(rel, k_n, g)]
        p_sum = sum(p_row)
        rec_prob = (p_sum > 1e-12 && !isnan(p_sum)) ?
                   (p_row[rec] / p_sum) : (1.0 / length(p_row))
        prob = (isnan(rec_prob) || rec_prob < 1e-12) ? 1e-12 : min(rec_prob, 1.0)
        Turing.@addlogprob! log(prob)
    end
end

"""
    joint_survey_telemetry_turing_model(counts, depths, releases, recaptures, ks, groups, W, hsi, land_mask, G)

Joint model integrating scientific survey count density (Negative Binomial likelihood)
with mark-recapture telemetry transition probabilities.
"""
@model function joint_survey_telemetry_turing_model(
    counts::Vector{Int},
    depths::Vector{Float64},
    releases::Vector{Int},
    recaptures::Vector{Int},
    ks::Vector{Int},
    groups::Vector{Int},
    W::SparseMatrixCSC{Float64, Int},
    hsi::Vector{Float64},
    land_mask::Union{Nothing, BitVector, Vector{Bool}},
    G::Int
)
    beta0      ~ Normal(0.0, 5.0)
    beta_depth ~ Normal(0.0, 2.0)
    inv_r      ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    r_nb       = 1.0 / max(inv_r, 1e-6)

    velocity  ~ filldist(truncated(Normal(0.3, 0.2), 0.0, 0.95), G)
    diffusion ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf), G)
    gamma     ~ filldist(Normal(1.0, 1.0), G)

    mu = exp.(clamp.(beta0 .+ beta_depth .* (depths ./ 100.0), -20.0, 20.0))
    prob_nb = r_nb ./ (r_nb .+ mu)
    for i in 1:length(counts)
        counts[i] ~ NegativeBinomial(r_nb, prob_nb[i])
    end

    tot = velocity .+ diffusion .+ 1e-6
    alpha = clamp.(velocity ./ tot, 0.0, 1.0)
    rho   = clamp.(1.0 ./ (1.0 .+ tot), 0.01, 0.95)

    P_kernels = Vector{SparseMatrixCSC}(undef, G)
    for g in 1:G
        P_kernels[g] = build_sparse_transition_kernel(
            W, hsi, gamma[g], rho[g], alpha[g], land_mask
        )
    end

    unique_rel_k = unique(zip(releases, ks, groups))
    P_row_cache = Dict{Tuple{Int, Int, Int}, Vector}()
    for (rel, k_n, g) in unique_rel_k
        v = zeros(eltype(P_kernels[g]), size(W, 1))
        v[rel] = 1.0
        P_g_T = sparse(P_kernels[g]')
        k_steps = max(0, k_n)
        for _ in 1:k_steps
            v = P_g_T * v
            s_v = sum(v)
            if s_v > 1e-12
                v ./= s_v
            end
        end
        P_row_cache[(rel, k_n, g)] = v
    end

    N = length(releases)
    for n in 1:N
        rel = releases[n]; rec = recaptures[n]; k_n = ks[n]; g = groups[n]
        p_row = P_row_cache[(rel, k_n, g)]
        p_sum = sum(p_row)
        rec_prob = (p_sum > 1e-12 && !isnan(p_sum)) ?
                   (p_row[rec] / p_sum) : (1.0 / length(p_row))
        prob = (isnan(rec_prob) || rec_prob < 1e-12) ? 1e-12 : min(rec_prob, 1.0)
        Turing.@addlogprob! log(prob)
    end
end

"""
    ssa_telemetry_turing_model(
        releases, recaptures, dts, groups, W, utility, land_mask, G
    )

Continuous-time Stochastic Simulation Algorithm (SSA) advection-diffusion mark-recapture
telemetry model. Infers physical advection velocity \$v\$, isotropic diffusion \$D\$,
and habitat utility sensitivity \$\\gamma\$ by evaluating the continuous-time Markov jump
infinitesimal generator \$Q\$ and matrix exponential transition probability kernel \$T(\\Delta t) = \\exp(Q \\Delta t)\$.

The time evolution of the state probability distribution follows the Master Equation 
(Nordsieck, Lamb & Uhlenbeck, 1940), also known as the Pauli Master Equation or M-equation, 
which is an equivalent linear form of the Chapman-Kolmogorov equation for Markov processes.

# Mathematical Formulation
For each demographic group \$g \\in \\{1, \\dots, G\\}\$:
```math
Q_g = \\text{construct\\_ssa\\_generator}(W, U; v_g, D_g, \\gamma_g)
```
The finite-time transition probability for observation \$n\$ across continuous interval \$\\Delta t_n\$ is:
```math
\\text{Pr}(s_{\\text{rec}} \\mid s_{\\text{rel}}, \\Delta t_n) = \\left[ \\exp(Q_g \\cdot \\Delta t_n) \\right]_{s_{\\text{rel}}, s_{\\text{rec}}}
```

# Arguments
- `releases::Vector{Int}`: Release spatial unit indices (1-indexed).
- `recaptures::Vector{Int}`: Recapture spatial unit indices (1-indexed).
- `dts::Vector{Float64}`: Continuous elapsed time between release and recapture.
- `groups::Vector{Int}`: Biological/demographic group assignment (1:G).
- `W::SparseMatrixCSC{Float64, Int}`: Spatial unit adjacency matrix.
- `utility::Vector{Float64}`: Spatial habitat utility or HSI vector.
- `land_mask`: Optional boolean mask denoting barrier units.
- `G::Int`: Number of demographic groups.
"""
@model function ssa_telemetry_turing_model(
    releases::Vector{Int},
    recaptures::Vector{Int},
    dts::Vector{Float64},
    groups::Vector{Int},
    W::SparseMatrixCSC{Float64, Int},
    utility::Vector{Float64},
    land_mask::Union{Nothing, BitVector, Vector{Bool}},
    G::Int
)
    velocity  ~ filldist(truncated(Normal(0.3, 0.2), 0.0, Inf), G)
    diffusion ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf), G)
    gamma     ~ filldist(Normal(1.0, 1.0), G)

    unique_rel_dt_g = unique(zip(releases, dts, groups))
    P_row_cache = Dict{Tuple{Int, Float64, Int}, Vector}()

    for g in 1:G
        Q_g = construct_ssa_generator(
            W, utility;
            velocity  = velocity[g],
            diffusion = diffusion[g],
            gamma     = gamma[g],
            land_mask = land_mask
        )
        for (rel, dt_val, g_id) in unique_rel_dt_g
            if g_id == g
                P_row_cache[(rel, dt_val, g)] = calculate_ssa_transition_row(Q_g, dt_val, rel)
            end
        end
    end

    N = length(releases)
    for n in 1:N
        rel = releases[n]; rec = recaptures[n]; g = groups[n]; dt = dts[n]
        p_row = P_row_cache[(rel, dt, g)]
        rec_prob = (p_sum > 1e-12 && !isnan(p_sum)) ?
                   (p_row[rec] / p_sum) : (1.0 / length(p_row))
        prob = (isnan(rec_prob) || rec_prob < 1e-12) ? 1e-12 : min(rec_prob, 1.0)
        Turing.@addlogprob! log(prob)
    end
end

"""
    joint_survey_ssa_telemetry_turing_model(
        counts, depths, releases, recaptures, dts, groups, W, utility, land_mask, G
    )

Joint model integrating scientific survey count density (Negative Binomial) with
continuous-time SSA advection-diffusion mark-recapture telemetry transition kernels.
"""
@model function joint_survey_ssa_telemetry_turing_model(
    counts::Vector{Int},
    depths::Vector{Float64},
    releases::Vector{Int},
    recaptures::Vector{Int},
    dts::Vector{Float64},
    groups::Vector{Int},
    W::SparseMatrixCSC{Float64, Int},
    utility::Vector{Float64},
    land_mask::Union{Nothing, BitVector, Vector{Bool}},
    G::Int
)
    beta0      ~ Normal(0.0, 5.0)
    beta_depth ~ Normal(0.0, 2.0)
    inv_r      ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    r_nb       = 1.0 / max(inv_r, 1e-6)

    velocity  ~ filldist(truncated(Normal(0.3, 0.2), 0.0, Inf), G)
    diffusion ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf), G)
    gamma     ~ filldist(Normal(1.0, 1.0), G)

    mu = exp.(clamp.(beta0 .+ beta_depth .* (depths ./ 100.0), -20.0, 20.0))
    prob_nb = r_nb ./ (r_nb .+ mu)
    for i in 1:length(counts)
        counts[i] ~ NegativeBinomial(r_nb, prob_nb[i])
    end

    unique_rel_dt_g = unique(zip(releases, dts, groups))
    P_row_cache = Dict{Tuple{Int, Float64, Int}, Vector}()

    for g in 1:G
        Q_g = construct_ssa_generator(
            W, utility;
            velocity  = velocity[g],
            diffusion = diffusion[g],
            gamma     = gamma[g],
            land_mask = land_mask
        )
        for (rel, dt_val, g_id) in unique_rel_dt_g
            if g_id == g
                P_row_cache[(rel, dt_val, g)] = calculate_ssa_transition_row(Q_g, dt_val, rel)
            end
        end
    end

    N = length(releases)
    for n in 1:N
        rel = releases[n]; rec = recaptures[n]; g = groups[n]; dt = dts[n]
        p_row = P_row_cache[(rel, dt, g)]
        p_sum = sum(p_row)
        rec_prob = (p_sum > 1e-12 && !isnan(p_sum)) ?
                   (p_row[rec] / p_sum) : (1.0 / length(p_row))
        prob = (isnan(rec_prob) || rec_prob < 1e-12) ? 1e-12 : min(rec_prob, 1.0)
        Turing.@addlogprob! log(prob)
    end
end
