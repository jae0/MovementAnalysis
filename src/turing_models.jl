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
    pure_telemetry_turing_model(releases, recaptures, ks, groups, W, hsi, land_mask, G; max_k)

Pure mark-recapture telemetry transition model evaluated over spatial graph nodes.
Estimates group-stratified advection velocity, diffusion rate, and habitat gradient
responsiveness (gamma) parameters driving stochastic transition probability kernels:
    P_g = (1 - rho_g)[(1 - alpha_g) T_diff + alpha_g A_g(eta)] + rho_g I

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
    # Biological group-stratified movement priors
    velocity  ~ filldist(truncated(Normal(0.3, 0.2), 0.0, 0.95), G)
    diffusion ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf), G)
    gamma     ~ filldist(Normal(1.0, 1.0), G)

    tot = velocity .+ diffusion .+ 1e-6
    alpha = clamp.(velocity ./ tot, 0.0, 1.0)
    rho   = clamp.(1.0 ./ (1.0 .+ tot), 0.01, 0.95)

    # Construct stochastic transition kernels per group
    P_kernels = construct_stochastic_transition_kernel(
        W, hsi;
        gamma     = gamma,
        residence = rho,
        advection = alpha,
        land_mask = land_mask
    )

    max_k = isempty(ks) ? 1 : maximum(ks)
    P_powers = Vector{Vector{Matrix{Float64}}}(undef, G)
    for g in 1:G
        P_powers[g] = Vector{Matrix{Float64}}(undef, max_k)
        P_powers[g][1] = P_kernels[g]
        for step in 2:max_k
            P_powers[g][step] = P_powers[g][step - 1] * P_kernels[g]
        end
    end

    # Multinomial transition log-likelihood across mark-recapture events
    N = length(releases)
    for n in 1:N
        rel = releases[n]
        rec = recaptures[n]
        g   = groups[n]
        k_n = ks[n]
        p_row = P_powers[g][k_n][rel, :]
        p_sum = sum(p_row)
        prob = p_sum > 1e-12 ? max(p_row[rec] / p_sum, 1e-12) : (1.0 / length(p_row))
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
    # Survey count priors
    beta0      ~ Normal(0.0, 5.0)
    beta_depth ~ Normal(0.0, 2.0)
    inv_r      ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    r_nb       = 1.0 / max(inv_r, 1e-6)

    # Telemetry priors
    velocity  ~ filldist(truncated(Normal(0.3, 0.2), 0.0, 0.95), G)
    diffusion ~ filldist(truncated(Normal(0.1, 0.2), 0.0, Inf), G)
    gamma     ~ filldist(Normal(1.0, 1.0), G)

    # Survey observation log-likelihood (Negative Binomial count model)
    mu = exp.(beta0 .+ beta_depth .* depths)
    prob_nb = r_nb ./ (r_nb .+ mu)
    for i in 1:length(counts)
        counts[i] ~ NegativeBinomial(r_nb, prob_nb[i])
    end

    # Telemetry transition log-likelihood
    tot = velocity .+ diffusion .+ 1e-6
    alpha = clamp.(velocity ./ tot, 0.0, 1.0)
    rho   = clamp.(1.0 ./ (1.0 .+ tot), 0.01, 0.95)

    P_kernels = construct_stochastic_transition_kernel(
        W, hsi;
        gamma     = gamma,
        residence = rho,
        advection = alpha,
        land_mask = land_mask
    )

    max_k = isempty(ks) ? 1 : maximum(ks)
    P_powers = Vector{Vector{Matrix{Float64}}}(undef, G)
    for g in 1:G
        P_powers[g] = Vector{Matrix{Float64}}(undef, max_k)
        P_powers[g][1] = P_kernels[g]
        for step in 2:max_k
            P_powers[g][step] = P_powers[g][step - 1] * P_kernels[g]
        end
    end

    N = length(releases)
    for n in 1:N
        rel = releases[n]
        rec = recaptures[n]
        g   = groups[n]
        k_n = ks[n]
        p_row = P_powers[g][k_n][rel, :]
        p_sum = sum(p_row)
        prob = p_sum > 1e-12 ? max(p_row[rec] / p_sum, 1e-12) : (1.0 / length(p_row))
        Turing.@addlogprob! log(prob)
    end
end
