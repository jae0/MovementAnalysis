"""
    ssa_movement.jl

Alternative continuous-time stochastic movement model based on the spatial Pauli
Master Equation and Stochastic Simulation Algorithm (SSA / Gillespie Direct Method),
derived from ModelSSA. Implements pure spatial kinematics with advective (directed
habitat taxis) and diffusive (isotropic random walk) transport across spatial
networks and lattices without demographic life-history mechanics.

# Mathematical Formulation
Let \$S\$ be the number of spatial units, \$W \\in \\mathbb{R}^{S \\times S}\$ the spatial
adjacency matrix, and \$U \\in \\mathbb{R}^S\$ the habitat utility or suitability field.
The spatial continuous-time jump process has infinitesimal generator
\$Q \\in \\mathbb{R}^{S \\times S}\$ with off-diagonal transition rates for \$j \\neq i\$:
```math
Q_{ij} = D \\frac{W_{ij}}{\\sum_{k} W_{ik}} +
         v \\frac{W_{ij} \\exp(\\gamma (U_j - U_i))}{\\sum_{k} W_{ik} \\exp(\\gamma (U_k - U_i))}
```
and diagonal conservation entries:
```math
Q_{ii} = -\\sum_{j \\neq i} Q_{ij}
```
where:
- \$D \\ge 0\$ is the isotropic diffusion coefficient (random exploration rate).
- \$v \\ge 0\$ is the advective velocity (directed habitat taxis jump rate).
- \$\\gamma \\ge 0\$ is the taxis sensitivity to habitat utility gradients.

The finite-time transition probability matrix over continuous elapsed time \$\\Delta t\$ is:
```math
P(\\Delta t) = \\exp(Q \\cdot \\Delta t)
```
"""

using LinearAlgebra
using SparseArrays
using Random
using Distributions
using DataFrames
using Dates

# ==============================================================================
# SECTION 1: PARAMETER SPECIFICATION & HABITAT UTILITY
# ==============================================================================

"""
    SSAMovementParams{T<:Real}

Configuration parameters for continuous-time SSA advective-diffusive movement.

# Fields
- `velocity::T`: Advective taxis jump rate (\$v \\ge 0\$).
- `diffusion::T`: Isotropic diffusion jump rate (\$D \\ge 0\$).
- `gamma::T`: Sensitivity to habitat utility gradient (\$\\gamma \\ge 0\$).
- `time_scale::Symbol`: Time unit interpretation (`:years` or `:days`). Default: `:years`.
- `taxis_temp_opt::T`: Optimal temperature for thermal habitat taxis (°C). Default: 2.5.
- `taxis_temp_sigma::T`: Thermal tolerance half-width (°C). Default: 1.5.
- `taxis_food_halfsat::T`: Food density half-saturation constant. Default: 2.0.
- `taxis_food_weight::T`: Relative weighting of food in composite utility. Default: 0.5.
"""
Base.@kwdef struct SSAMovementParams{T<:Real}
    velocity::T            = 0.20
    diffusion::T           = 0.15
    gamma::T               = 1.00
    time_scale::Symbol     = :years
    taxis_temp_opt::T      = 2.50
    taxis_temp_sigma::T    = 1.50
    taxis_food_halfsat::T  = 2.00
    taxis_food_weight::T   = 0.50
end

"""
    calculate_ssa_utility(
        hsi::AbstractVector{<:Real};
        temp::Union{Nothing, AbstractVector{<:Real}} = nothing,
        food::Union{Nothing, AbstractVector{<:Real}} = nothing,
        params::SSAMovementParams = SSAMovementParams()
    ) -> Vector{Float64}

Computes the spatial habitat utility vector \$U \\in \\mathbb{R}^S\$ driving directed
taxis. When temperature and food fields are provided, evaluates the bivariate
ModelSSA formulation; otherwise uses the normalized Habitat Suitability Index (HSI).

# Mathematical Formulation
When temperature \$T\$ and food \$F\$ are supplied:
```math
U(T, F) = \\exp\\!\\left(-\\frac{(T - T^*)^2}{2\\sigma_T^2}\\right) \\cdot
          \\left(1 + w_F \\frac{F}{F + F_{1/2}}\\right)
```
Otherwise:
```math
U(s) = \\text{clamp}(HSI(s), 10^{-4}, 1.0)
```

# Arguments
- `hsi::AbstractVector{<:Real}`: Habitat suitability index vector of length \$S\$.
- `temp`: Optional temperature vector of length \$S\$ (°C).
- `food`: Optional food density vector of length \$S\$.
- `params::SSAMovementParams`: Parameter configuration with utility weights.

# Returns
- `Vector{Float64}`: Non-negative spatial utility vector of length \$S\$.
"""
function calculate_ssa_utility(
    hsi::AbstractVector{<:Real};
    temp::Union{Nothing, AbstractVector{<:Real}} = nothing,
    food::Union{Nothing, AbstractVector{<:Real}} = nothing,
    params::SSAMovementParams = SSAMovementParams()
)::Vector{Float64}
    S = length(hsi)
    if !isnothing(temp) && length(temp) == S
        f_vec = (!isnothing(food) && length(food) == S) ? food : fill(1.0, S)
        u_out = Vector{Float64}(undef, S)
        t_opt = Float64(params.taxis_temp_opt)
        t_sig = Float64(params.taxis_temp_sigma)
        f_half = Float64(params.taxis_food_halfsat)
        w_f = Float64(params.taxis_food_weight)

        for i in 1:S
            ti = Float64(temp[i])
            fi = max(0.0, Float64(f_vec[i]))
            s_temp = exp(-((ti - t_opt)^2) / (2.0 * t_sig^2))
            s_food = fi / (fi + f_half)
            u_out[i] = max(1e-4, s_temp * (1.0 + w_f * s_food))
        end
        return u_out
    else
        return [max(1e-4, Float64(h)) for h in hsi]
    end
end


# ==============================================================================
# SECTION 2: INFINITESIMAL GENERATOR & TRANSITION KERNELS
# ==============================================================================

"""
    construct_ssa_generator(
        W::SparseMatrixCSC{Float64, Int},
        utility::AbstractVector{<:Real};
        velocity::Real = 0.20,
        diffusion::Real = 0.15,
        gamma::Real = 1.00,
        land_mask::Union{Nothing, BitVector, Vector{Bool}} = nothing
    ) -> SparseMatrixCSC{Float64, Int}

Constructs the continuous-time infinitesimal transition rate matrix
\$Q \\in \\mathbb{R}^{S \\times S}\$ (spatial Pauli Master Equation generator).

# Mathematical Formulation
For each node \$i\$ and neighbor \$j \\in \\mathcal{N}(i)\$:
```math
Q_{ij} = D \\frac{W_{ij}}{\\sum_{k} W_{ik}} +
         v \\frac{W_{ij} \\exp(\\gamma (U_j - U_i))}{\\sum_{k} W_{ik} \\exp(\\gamma (U_k - U_i))}
```
Diagonal elements satisfy \$Q_{ii} = -\\sum_{j \\neq i} Q_{ij}\$ ensuring row-stochastic
probability conservation (\$\\sum_j Q_{ij} = 0\$). Land units are treated as absorbing
or zero-rate boundaries.

# Arguments
- `W::SparseMatrixCSC{Float64, Int}`: Spatial unit adjacency matrix.
- `utility::AbstractVector{<:Real}`: Spatial habitat utility vector of length \$S\$.
- `velocity::Real`: Advective taxis jump rate (\$v \\ge 0\$).
- `diffusion::Real`: Isotropic diffusion jump rate (\$D \\ge 0\$).
- `gamma::Real`: Gradient sensitivity parameter (\$\\gamma \\ge 0\$).
- `land_mask`: Optional boolean mask (`true` for land/barrier units).

# Returns
- `SparseMatrixCSC{Float64, Int}`: Conservative infinitesimal generator \$Q\$.
"""
function construct_ssa_generator(
    W::SparseMatrixCSC{Float64, Int},
    utility::AbstractVector{<:Real};
    velocity::Real = 0.20,
    diffusion::Real = 0.15,
    gamma::Real = 1.00,
    land_mask::Union{Nothing, BitVector, Vector{Bool}} = nothing
)::SparseMatrixCSC{Float64, Int}
    S = size(W, 1)
    length(utility) == S ||
        throw(DimensionMismatch("utility length must match size(W, 1)"))

    rows = rowvals(W)
    vals = nonzeros(W)

    v = max(0.0, Float64(velocity))
    D = max(0.0, Float64(diffusion))
    gam = Float64(gamma)

    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    # Pre-allocate for sparse matrix assembly
    sizehint!(I_idx, nnz(W) + S)
    sizehint!(J_idx, nnz(W) + S)
    sizehint!(V_val, nnz(W) + S)

    for i in 1:S
        if !isnothing(land_mask) && land_mask[i]
            # Land node has zero transition rates
            push!(I_idx, i)
            push!(J_idx, i)
            push!(V_val, 0.0)
            continue
        end

        u_i = Float64(utility[i])
        nbrs = Int[]
        nbr_w = Float64[]

        # Collect eligible marine neighbors
        for idx in nzrange(W, i)
            j = rows[idx]
            if j != i
                if isnothing(land_mask) || !land_mask[j]
                    push!(nbrs, j)
                    push!(nbr_w, vals[idx])
                end
            end
        end

        if isempty(nbrs)
            push!(I_idx, i)
            push!(J_idx, i)
            push!(V_val, 0.0)
            continue
        end

        # 1. Diffusive weights
        sum_diff_w = sum(nbr_w)
        diff_weights = nbr_w ./ max(1e-12, sum_diff_w)

        # 2. Directed taxis weights
        taxis_raw = [w * exp(clamp(gam * (Float64(utility[j]) - u_i), -20.0, 20.0))
                     for (j, w) in zip(nbrs, nbr_w)]
        sum_taxis = sum(taxis_raw)
        taxis_weights = taxis_raw ./ max(1e-12, sum_taxis)

        # 3. Combined transition rate Q_ij
        total_exit_rate = 0.0
        for (k, j) in enumerate(nbrs)
            rate_ij = D * diff_weights[k] + v * taxis_weights[k]
            if rate_ij > 1e-12
                push!(I_idx, i)
                push!(J_idx, j)
                push!(V_val, rate_ij)
                total_exit_rate += rate_ij
            end
        end

        # 4. Diagonal rate Q_ii = -sum_{j != i} Q_ij
        push!(I_idx, i)
        push!(J_idx, i)
        push!(V_val, -total_exit_rate)
    end

    Q = sparse(I_idx, J_idx, V_val, S, S)
    return Q
end


"""
    calculate_ssa_transition_matrix(
        Q::AbstractMatrix{<:Real},
        dt::Real;
        method::Symbol = :taylor
    ) -> Matrix{Float64}

Computes the finite-time transition probability matrix \$P(\\Delta t) = \\exp(Q \\cdot \\Delta t)\$.

# Methods
- `:taylor` / `:uniformization`: Scaled truncated Taylor expansion. Highly efficient
  for sparse generators where jump rate \$\\max_i |Q_{ii}| \\cdot \\Delta t\$ is moderate.
- `:expm`: Exact dense matrix exponential via `LinearAlgebra.exp`.

# Returns
- `Matrix{Float64}`: Row-stochastic transition probability matrix where \$\\sum_j P_{ij} = 1\$.
"""
function calculate_ssa_transition_matrix(
    Q::AbstractMatrix{<:Real},
    dt::Real;
    method::Symbol = :taylor
)::Matrix{Float64}
    S = size(Q, 1)
    dt_f = max(0.0, Float64(dt))
    dt_f == 0.0 && return Matrix{Float64}(I, S, S)

    if method == :expm || S <= 150
        M = Matrix{Float64}(Q .* dt_f)
        P = exp(M)
        # Numerical cleanup: clamp and row-normalize
        P .= max.(P, 0.0)
        for i in 1:S
            rs = sum(P[i, :])
            if rs > 1e-12
                P[i, :] ./= rs
            else
                P[i, i] = 1.0
            end
        end
        return P
    end

    # Uniformization / Scaling & Squaring for Sparse Transition Generator
    # Q = alpha * (K - I), with alpha = max_i |Q_ii|
    q_diag = abs.(diag(Q))
    alpha = maximum(q_diag)
    if alpha < 1e-12
        return Matrix{Float64}(I, S, S)
    end

    # Determine scaling factor m such that (alpha * dt) / 2^m <= 1.0
    scaled_t = alpha * dt_f
    m_squaring = max(0, ceil(Int, log2(max(1.0, scaled_t))))
    step_dt = dt_f / (2.0 ^ m_squaring)

    # 4th-order Taylor polynomial for exp(Q * step_dt)
    A = Matrix{Float64}(Q .* step_dt)
    P_step = Matrix{Float64}(I, S, S) .+ A .+ 0.5 .* (A * A) .+
             (1.0 / 6.0) .* (A * A * A) .+ (1.0 / 24.0) .* (A * A * A * A)
    P_step .= max.(P_step, 0.0)

    # Squaring phases
    P = P_step
    for _ in 1:m_squaring
        P = P * P
    end

    # Row stochastic normalization
    for i in 1:S
        rs = sum(P[i, :])
        if rs > 1e-12
            P[i, :] ./= rs
        else
            P[i, i] = 1.0
        end
    end

    return P
end


# ==============================================================================
# SECTION 3: EXACT GILLESPIE SSA TRAJECTORY SIMULATION
# ==============================================================================

"""
    simulate_gillespie_trajectories(
        start_units::AbstractVector{<:Integer},
        tspan::Tuple{Real, Real},
        Q::SparseMatrixCSC{Float64, Int};
        saveat::Union{Nothing, Real} = nothing,
        rng::Random.AbstractRNG = Random.GLOBAL_RNG
    ) -> Vector{Vector{Tuple{Float64, Int}}}

Simulates exact continuous-time sample paths for individual agents executing
the Stochastic Simulation Algorithm (SSA / Gillespie 1977 Direct Method) on the
spatial network defined by generator \$Q\$.

# Algorithm (Gillespie Direct Method)
At current time \$t\$ and location \$s\$:
1. Total transition exit rate: \$\\lambda = -Q_{ss} = \\sum_{j \\neq s} Q_{sj}\$.
2. If \$\\lambda = 0\$, agent remains at \$s\$ indefinitely.
3. Time increment to next jump: \$\\tau \\sim \\text{Exponential}(\\lambda)\$.
4. Destination node \$j \\neq s\$ chosen with probability \$Q_{sj} / \\lambda\$.
5. Update: \$t \\leftarrow t + \\tau\$, \$s \\leftarrow j\$.

# Arguments
- `start_units`: Initial spatial unit index for each simulated agent.
- `tspan`: Simulation time interval `(t_start, t_end)`.
- `Q`: Infinitesimal generator matrix from `construct_ssa_generator`.
- `saveat`: Optional fixed time interval to sample and record discrete locations.
- `rng`: Random number generator instance.

# Returns
- `Vector{Vector{Tuple{Float64, Int}}}`: List of trajectories, where each trajectory
  contains pairs `(time, unit_index)`.
"""
function simulate_gillespie_trajectories(
    start_units::AbstractVector{<:Integer},
    tspan::Tuple{Real, Real},
    Q::SparseMatrixCSC{Float64, Int};
    saveat::Union{Nothing, Real} = nothing,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::Vector{Vector{Tuple{Float64, Int}}}
    t_start = Float64(tspan[1])
    t_end = Float64(tspan[2])
    t_end > t_start || throw(ArgumentError("tspan[2] must be greater than tspan[1]"))

    N_agents = length(start_units)
    rows = rowvals(Q)
    vals = nonzeros(Q)

    # Pre-extract transition lists for O(1) jump lookups
    S = size(Q, 1)
    nbr_nodes = [Int[] for _ in 1:S]
    nbr_rates = [Float64[] for _ in 1:S]
    total_rates = zeros(Float64, S)

    for i in 1:S
        for idx in nzrange(Q, i)
            j = rows[idx]
            w = vals[idx]
            if j != i && w > 0.0
                push!(nbr_nodes[i], j)
                push!(nbr_rates[i], w)
                total_rates[i] += w
            end
        end
    end

    trajectories = Vector{Vector{Tuple{Float64, Int}}}(undef, N_agents)

    for a in 1:N_agents
        traj = Tuple{Float64, Int}[]
        t_cur = t_start
        s_cur = Int(start_units[a])
        push!(traj, (t_cur, s_cur))

        while t_cur < t_end
            lambda = total_rates[s_cur]
            if lambda <= 1e-12
                # Absorbing node or zero exit rate
                push!(traj, (t_end, s_cur))
                break
            end

            # Sample time increment to next event
            tau = rand(rng, Exponential(1.0 / lambda))
            t_next = t_cur + tau
            if t_next > t_end
                push!(traj, (t_end, s_cur))
                break
            end

            # Select destination node proportionally to rate Q_sj
            rates = nbr_rates[s_cur]
            nodes = nbr_nodes[s_cur]
            r_pick = rand(rng) * lambda
            cum = 0.0
            next_s = nodes[end]
            for (k, rate_k) in enumerate(rates)
                cum += rate_k
                if r_pick <= cum
                    next_s = nodes[k]
                    break
                end
            end

            t_cur = t_next
            s_cur = next_s
            push!(traj, (t_cur, s_cur))
        end

        # Discretize onto regular saveat grid if requested
        if !isnothing(saveat) && saveat > 0.0
            dt_s = Float64(saveat)
            grid_t = collect(t_start:dt_s:t_end)
            disc_traj = Tuple{Float64, Int}[]
            step_idx = 1
            n_events = length(traj)

            for tg in grid_t
                while step_idx < n_events && traj[step_idx + 1][1] <= tg
                    step_idx += 1
                end
                push!(disc_traj, (tg, traj[step_idx][2]))
            end
            trajectories[a] = disc_traj
        else
            trajectories[a] = traj
        end
    end

    return trajectories
end


# ==============================================================================
# SECTION 4: DATASET GENERATOR MIRRORING MODEL_SSA
# ==============================================================================

"""
    generate_ssa_movement_data(;
        radius_km::Real = 10.0,
        domain_km::Real = 200.0,
        n_tags::Int = 50,
        n_steps::Int = 4,
        t_interval::Real = 0.25,
        velocity::Real = 0.35,
        diffusion::Real = 0.15,
        gamma::Real = 1.20,
        seed::Int = 42
    ) -> NamedTuple

Generates a synthetic mark-recapture telemetry and environmental mesh dataset
simulated via continuous-time Gillespie SSA advection-diffusion dynamics.

# Returns
- `NamedTuple`:
  - `telemetry`: DataFrame with columns `:tagid`, `:s_idx`, `:time`, `:tag`, `:lon`, `:lat`.
  - `mesh`: Hexagonal planar mesh partition NamedTuple.
  - `utility`: Spatial habitat utility vector.
  - `generator`: True infinitesimal generator \$Q\$.
  - `true_params`: `(velocity = velocity, diffusion = diffusion, gamma = gamma)`.
"""
function generate_ssa_movement_data(;
    radius_km::Real = 10.0,
    domain_km::Real = 200.0,
    n_tags::Int = 50,
    n_steps::Int = 4,
    t_interval::Real = 0.25,
    velocity::Real = 0.35,
    diffusion::Real = 0.15,
    gamma::Real = 1.20,
    seed::Int = 42
)::NamedTuple
    rng = MersenneTwister(seed)

    # 1. Build regular hexagonal mesh over planar domain
    n_pts_axis = max(4, round(Int, domain_km / (radius_km * sqrt(3.0))))
    xs = range(-domain_km / 2.0, domain_km / 2.0, length = n_pts_axis)
    ys = range(-domain_km / 2.0, domain_km / 2.0, length = n_pts_axis)
    grid_x = [x for x in xs for _ in ys]
    grid_y = [y for _ in xs for y in ys]

    mesh = build_hex_mesh_planar(grid_x, grid_y; radius_km = Float64(radius_km))
    S = mesh.n_units

    # 2. Synthetic bell-shaped Gaussian Habitat Suitability Field
    cx = [c[1] for c in mesh.centroids]
    cy = [c[2] for c in mesh.centroids]
    target_x = domain_km * 0.20
    target_y = domain_km * 0.20
    sigma_h = domain_km * 0.35
    hsi = [exp(-((cx[i] - target_x)^2 + (cy[i] - target_y)^2) / (2.0 * sigma_h^2))
           for i in 1:S]
    hsi_vec = [max(0.05, h / maximum(hsi)) for h in hsi]

    # 3. Construct Generator Q
    Q = construct_ssa_generator(
        mesh.W, hsi_vec;
        velocity = velocity,
        diffusion = diffusion,
        gamma = gamma
    )

    # 4. Simulate mark-recapture telemetry transitions via Gillespie Direct Algorithm
    start_units = rand(rng, 1:S, n_tags)
    t_total = Float64(n_steps) * Float64(t_interval)
    raw_trajs = simulate_gillespie_trajectories(
        start_units, (0.0, t_total), Q;
        saveat = t_interval, rng = rng
    )

    df_rows = NamedTuple[]
    for tag_id in 1:n_tags
        traj = raw_trajs[tag_id]
        for (step_idx, (t_pt, s_pt)) in enumerate(traj)
            c_pt = mesh.centroids_lonlat[s_pt]
            push!(df_rows, (
                tagid = tag_id,
                s_idx = s_pt,
                time = t_pt,
                tag = step_idx - 1,
                lon = c_pt[1],
                lat = c_pt[2],
                sex = 1,
                mat = 1,
                is_dead = false
            ))
        end
    end
    telemetry_df = DataFrame(df_rows)

    return (
        telemetry = telemetry_df,
        mesh = mesh,
        utility = hsi_vec,
        generator = Q,
        true_params = (
            velocity = velocity,
            diffusion = diffusion,
            gamma = gamma
        )
    )
end
