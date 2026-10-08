"""
    agent_movement.jl

Forward projection of synthetic animal trajectories over a fitted movement kernel.

This is a *projector*, not an independent model. It consumes the same pooled
transition kernel as the backward path reconstruction and introduces no
parameters and no likelihood of its own. What distinguishes it is the direction
of conditioning:

- `reconstruct_paths_bayesian_ensemble` bridges release -> recapture conditioned
  on **both** observed endpoints.
- This module projects forward from a release unit conditioned on **neither**
  endpoint.

A projected path may therefore visit units a bridge is forced to avoid, which
makes forward projection the appropriate tool for unconditioned space-use
questions: expected residence per unit, probability of visiting a unit, and where
a track would have gone had the animal not been recaptured.

# Scope limits

Agents are independent. There is no interaction, memory, mortality, reproduction,
or agent-level state beyond position, and every step for every agent obeys the
same transition law. The projection is stochastic, so a single run is a sample
from the kernel; conclusions that depend on the tail of that distribution need
several runs rather than one.
"""

using SparseArrays
using Random
using DataFrames
"""
    CrabAgent

One synthetic animal during a forward projection. `n_steps` is the horizon
assigned to this individual, so agents in the same run may have different
durations. `heading` is the current compass bearing in degrees, or `NaN` before
the first step; it is only consulted when `persistence` is non-zero.
"""
mutable struct CrabAgent
    id::Int
    pos::Int
    n_steps::Int
    heading::Float64
end

"""
    simulate_agent_trajectories(
        n_agents, start_nodes, n_steps, transition_kernel;
        seed = 42, centroids = nothing, persistence = 0.0, coord_space = :unknown
    )
    simulate_agent_trajectories(
        n_agents, start_nodes, n_steps::AbstractVector{<:Integer},
        transition_kernel; kwargs...
    )

Project `n_agents` independent trajectories forward under a single pooled
transition kernel.

Each agent advances by sampling one neighbour of its current unit from that
unit's row of `transition_kernel`. Every step for every agent shares the same
transition law; agents never interact.

# Directional persistence

With `persistence = 0` (the default) this is memoryless: an agent's step
distribution depends only on where it is. With `persistence > 0` and `centroids`
supplied, each agent carries a heading and its step distribution is reweighted by
`exp(persistence * cos(bearing - heading))` before sampling, so an animal tends to
continue the way it was already going. This is the agent-level counterpart of
`build_persistent_transition_kernel` and is the mechanism that makes the
simulation directionally autocorrelated.

The horizon may be a single integer, applied to all agents, or a vector whose
`i`th entry is the horizon for the agent released at `start_nodes[i]`. The
vector form is what lets a run inherit the empirical duration distribution
instead of an arbitrary fixed length.

# Arguments
- `n_agents::Int`: Number of independent synthetic agents.
- `start_nodes::Vector{Int}`: Release unit for each agent; `length` must equal `n_agents`.
- `n_steps`: Shared horizon, or a per-agent vector of horizons.
- `transition_kernel`: Row-stochastic pooled kernel (SparseMatrixCSC or Matrix), or
  a Vector of time-varying transition kernels for dynamic seasonal projections.
- `seed::Int`: Seed for the agent's own RNG, so a run is reproducible.
- `centroids`: Node coordinates, required when `persistence > 0`.
- `persistence`: Heading-persistence strength; `0.0` reproduces the memoryless chain.
- `coord_space`: `:geographic` or `:planar_km`; inferred from `centroids` when `:unknown`.

# Returns
- `DataFrame` with columns `tagid`, `step`, `mesh_unit`, where `tagid` identifies
  the agent and `step` runs from 0 to that agent's horizon inclusive. The frame
  therefore has one row per agent per step, plus its initial position.

# Throws
- `DimensionMismatch` if `length(start_nodes) != n_agents` or a horizon vector has
  the wrong length.
- `ArgumentError` if any horizon is negative, any start node is outside
  `1:size(kernel, 1)`, or `persistence > 0` without `centroids`.
"""
function simulate_agent_trajectories(
    n_agents::Int,
    start_nodes::AbstractVector{<:Integer},
    n_steps,
    transition_kernel::Union{
        AbstractMatrix{<:Real},
        AbstractVector{<:AbstractMatrix{<:Real}}
    };
    seed::Int = 42,
    centroids = nothing,
    persistence::Real = 0.0,
    coord_space::Symbol = :unknown
)
    rng = MersenneTwister(seed)

    length(start_nodes) == n_agents || throw(DimensionMismatch(
        "start_nodes length must equal n_agents."
    ))

    horizons = if n_steps isa Integer
        fill(Int(n_steps), n_agents)
    else
        Int.(collect(n_steps))
    end
    length(horizons) == n_agents || throw(DimensionMismatch(
        "n_steps length must equal n_agents."
    ))
    all(h -> h >= 0, horizons) ||
        throw(ArgumentError("Every horizon must be non-negative."))

    is_dyn = transition_kernel isa AbstractVector{<:AbstractMatrix}
    S = is_dyn ? size(first(transition_kernel), 1) : size(transition_kernel, 1)
    all(node -> 1 <= node <= S, start_nodes) || throw(ArgumentError(
        "Every start node must be within 1:$S."
    ))

    kappa = Float64(persistence)
    use_persistence = kappa != 0.0
    if use_persistence && centroids === nothing
        throw(ArgumentError(
            "persistence > 0 requires `centroids` so a step bearing can be computed."
        ))
    end
    if use_persistence && length(centroids) != S
        throw(DimensionMismatch(
            "centroids has $(length(centroids)) entries but the kernel has $S units."
        ))
    end
    space = use_persistence && coord_space === :unknown ?
            coordinate_space_of(centroids) : coord_space

    # Pre-transpose kernels to sample transitions from column i
    kernel_t_seq = if is_dyn
        [sparse(K') for K in transition_kernel]
    else
        [sparse(transition_kernel')]
    end

    total_records = n_agents + sum(horizons)
    agent_ids = Vector{Int}(undef, total_records)
    steps = Vector{Int}(undef, total_records)
    positions = Vector{Int}(undef, total_records)

    agents = [CrabAgent(i, Int(start_nodes[i]), horizons[i], NaN) for i in 1:n_agents]

    idx = 1
    for a in agents
        agent_ids[idx] = a.id
        steps[idx] = 0
        positions[idx] = a.pos
        idx += 1
    end

    max_h = maximum(horizons; init = 0)
    for step in 1:max_h
        kernel_t = is_dyn ?
            kernel_t_seq[mod1(step, length(kernel_t_seq))] :
            kernel_t_seq[1]

        for a in agents
            step > a.n_steps && continue
            ptr_range = nzrange(kernel_t, a.pos)
            neighbours = rowvals(kernel_t)[ptr_range]
            probs = nonzeros(kernel_t)[ptr_range]
            if length(probs) > 0
                weights = probs
                bearings = Vector{Float64}(undef, length(probs))
                if use_persistence
                    weights = Vector{Float64}(undef, length(probs))
                    for t in eachindex(probs)
                        b = neighbours[t] == a.pos ? NaN :
                            bearing_deg(centroids[a.pos], centroids[neighbours[t]];
                                        coord_space = space)
                        bearings[t] = b
                        weights[t] = Float64(probs[t]) *
                                     exp(_turn_logweight(kappa, b, a.heading))
                    end
                end

                total_w = sum(weights)
                if total_w > 0.0
                    r = rand(rng) * total_w
                    acc = 0.0
                    chosen = neighbours[end]
                    for t in eachindex(weights)
                        acc += weights[t]
                        if r <= acc
                            chosen = neighbours[t]
                            break
                        end
                    end
                    if use_persistence && chosen != a.pos
                        a.heading = bearings[findfirst(==(chosen), neighbours)]
                    end
                    a.pos = chosen
                end
            end

            agent_ids[idx] = a.id
            steps[idx] = step
            positions[idx] = a.pos
            idx += 1
        end
    end

    return DataFrame(
        tagid = agent_ids,
        step = steps,
        mesh_unit = positions
    )
end

"""
    forward_project_agents(
        release_nodes, durations;
        n_agents::Int,
        transition_kernel,
        seed::Int = 42
    )

Sample independent synthetic agents from an empirical release set and project each
one forward under the pooled kernel or a time-varying sequence of kernels.

This is the entry point that decouples the number of agents from the number of
observations. `n_agents` is a free choice: start units are drawn *with
replacement* from `release_nodes`, and each agent's horizon is drawn from
`durations`. A run with 500 agents is therefore not five times a run with 100 —
it is an independent sample of the same size, and increasing it reduces Monte
Carlo error without changing the underlying distribution.

Drawing horizons from an empirical vector keeps the projection on the same time
scale as the data, which a fixed step count does not.

# Arguments
- `release_nodes`: Pool of release units to sample starts from.
- `durations`: Pool of observed horizons to sample per-agent horizons from.
- `n_agents`: Number of independent agents to project.
- `transition_kernel`: Row-stochastic pooled kernel, or Vector of dynamic kernels.
- `seed`: Seed for sampling starts, horizons, and the trajectories.

# Returns
- `DataFrame` from `simulate_agent_trajectories`, plus a `start_node` and
  `n_steps` column recording each agent's sampled origin and horizon.
"""
function forward_project_agents(
    release_nodes::AbstractVector{<:Integer},
    durations::AbstractVector{<:Integer};
    n_agents::Int,
    transition_kernel,
    seed::Int = 42,
    centroids = nothing,
    persistence::Real = 0.0
)
    isempty(release_nodes) && throw(ArgumentError(
        "release_nodes is empty; there is nowhere to release an agent."
    ))
    isempty(durations) && throw(ArgumentError(
        "durations is empty; every agent would have a zero horizon."
    ))
    n_agents >= 1 || throw(ArgumentError("n_agents must be at least 1."))

    rng = MersenneTwister(seed)

    # With replacement: the agent count is an independent design choice, not a
    # by-product of how many animals happened to be observed.
    start_nodes = [
        Int(release_nodes[rand(rng, 1:length(release_nodes))]) for _ in 1:n_agents
    ]
    horizons = [
        max(1, Int(durations[rand(rng, 1:length(durations))])) for _ in 1:n_agents
    ]

    trajectories = simulate_agent_trajectories(
        n_agents, start_nodes, horizons, transition_kernel;
        seed = seed, centroids = centroids, persistence = persistence
    )

    trajectories.start_node = [Int(start_nodes[a]) for a in trajectories.tagid]
    trajectories.n_steps = [horizons[a] for a in trajectories.tagid]

    return trajectories
end

"""
    forward_space_use(trajectories, n_spatial::Int) -> NamedTuple

Aggregate a forward projection into per-unit space-use statistics.

Because the projection is conditioned on the release unit only, these are
unconditioned quantities: `visit_probability` is the fraction of agents that ever
occupied a unit, and `mean_dwell_steps` is the mean number of steps an agent
spent there given that it visited. They are the natural companion to a
bridge-derived residence distribution, which is conditioned on recapture and
therefore biased away from units the animal could not have visited between the two
observed endpoints.

# Returns
- `NamedTuple` with `unit_id`, `visit_probability`, `visits`, and
  `mean_dwell_steps`.
"""
function forward_space_use(
    trajectories::AbstractDataFrame,
    n_spatial::Int
)::NamedTuple
    n_agents = isempty(trajectories) ? 0 : length(unique(trajectories.tagid))
    visits = zeros(Int, n_spatial)
    dwell = zeros(Float64, n_spatial)

    # Single pass per agent. Counting each unique unit with `count(==(u), units)`
    # would be quadratic in the path length, which matters once `n_agent_projections`
    # and the horizons are large.
    if n_agents > 0
        for agent in groupby(trajectories, :tagid)
            seen = Set{Int}()
            for v in agent.mesh_unit
                u = Int(v)
                1 <= u <= n_spatial || continue
                dwell[u] += 1.0
                u in seen || (push!(seen, u); visits[u] += 1)
            end
        end
    end

    visit_probability = n_agents > 0 ? visits ./ n_agents : zeros(Float64, n_spatial)
    mean_dwell = zeros(Float64, n_spatial)
    for u in 1:n_spatial
        visits[u] > 0 && (mean_dwell[u] = dwell[u] / visits[u])
    end

    return (
        unit_id = collect(1:n_spatial),
        visit_probability = visit_probability,
        visits = visits,
        mean_dwell_steps = mean_dwell,
    )
end
