"""
    agent_movement.jl

Agent-based forward projection from a fitted telemetry kernel.

Given the posterior kernel ``P`` estimated from telemetry, individual animals are
advanced forward through it to produce trajectories that were never observed.
This is the projection counterpart to backward Viterbi reconstruction: the
reconstruction asks which single path best explains the tags, the projection
asks where untagged animals go.

The kernel is *pooled*. Telemetry is fitted with one scalar parameter set, so
there is exactly one ``P``; agents do not belong to strata and are not indexed by
one. Heading is the only auxiliary state, and it exists solely to let a step
change direction coherently.

Two kernels are accepted, and the distinction is detected from the size rather
than from a flag:

- ``n_units`` x ``n_units``: the first-order kernel. Heading is tracked
  geometrically from the step actually taken, so it needs ``centroids``.
- ``n_units * n_headings`` squared: the persistence kernel from
  `build_persistent_transition_kernel`, over states ``(unit, heading)``. Heading
  is sampled jointly with position, and no coordinates are needed.
"""

using SparseArrays
using Random
using DataFrames

mutable struct TrackedAgent
    id::Int
    pos::Int
    heading::Int
end

"""
    simulate_agent_trajectories(
        n_agents::Int,
        start_nodes::AbstractVector{<:Integer},
        start_headings::AbstractVector{<:Integer},
        P_kernel::SparseMatrixCSC{Float64, Int},
        n_steps::Int;
n_units::Int = 0,
        centroids = nothing,
        coord_space::Symbol = :unknown,
        heading_bins::Int = 8,
        seed::Int = 42
)

Project `n_agents` animals forward `n_steps` transitions through `P_kernel`,
one recorded row per agent per step, step 0 being the release location.

`start_nodes` gives each agent's release unit. `start_headings` gives its
heading bin; pass a vector of `1`s to ignore heading entirely. A persistent
kernel is used only when `P_kernel` has `n_units * n_headings` rows, which
distinguishes it from a first-order kernel by size alone. `centroids` is
required in that case to turn each step into a bearing; without it, and on a
first-order kernel, heading is carried forward unchanged, which is the correct
treatment of a zero-length step rather than a silent default. `heading_bins` sets how many
equally spaced bins the geometric heading uses; a persistence kernel derives its
own count from its size and ignores it.

# Returns
`DataFrame` with columns `tagid`, `step`, `mesh_unit`, `heading`.
"""
function simulate_agent_trajectories(
    n_agents::Int,
    start_nodes::AbstractVector{<:Integer},
    start_headings::AbstractVector{<:Integer},
    P_kernel::SparseMatrixCSC{Float64, Int},
    n_steps::Int;
    n_units::Int = 0,
    centroids = nothing,
    coord_space::Symbol = :unknown,
    heading_bins::Int = 8,
    seed::Int = 42
)
    n_agents >= 1 || throw(ArgumentError("n_agents must be positive, got $n_agents."))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative, got $n_steps."))
    length(start_nodes) >= n_agents || throw(DimensionMismatch(
        "start_nodes has $(length(start_nodes)) entries but n_agents is $n_agents."
    ))
    length(start_headings) >= n_agents || throw(DimensionMismatch(
        "start_headings has $(length(start_headings)) entries but n_agents is $n_agents."
    ))

    L = size(P_kernel, 1)
    size(P_kernel, 2) == L || throw(DimensionMismatch(
        "P_kernel must be square, got $(size(P_kernel))."
    ))

    # A persistence kernel has one row per (unit, heading) pair. Recover S and H by
    # requiring the start units to be in range once divided by the heading count.
    n_headings = 0
    S = n_units
    if n_units > 0 && L != n_units
        L % n_units == 0 || throw(DimensionMismatch(
            "P_kernel has $L rows, which is not a multiple of n_units=$n_units."
        ))
        n_headings = L ÷ n_units
    else
        S = L
    end

    for i in 1:n_agents
        1 <= start_nodes[i] <= S || throw(BoundsError(
            P_kernel, "start unit $(start_nodes[i]) for agent $i is outside 1:$S"
        ))
    end

    # CSC holds columns, so row `s` of the kernel is column `s` of the transpose.
    # One transpose serves every agent and every step.
    Pt = sparse(P_kernel')

    rng = MersenneTwister(seed)
    agents = [TrackedAgent(i, Int(start_nodes[i]), Int(start_headings[i])) for i in 1:n_agents]

    total_records = n_agents * (n_steps + 1)
    agent_ids = Vector{Int}(undef, total_records)
    steps     = Vector{Int}(undef, total_records)
    positions = Vector{Int}(undef, total_records)
    headings  = Vector{Int}(undef, total_records)

    idx = 1
    for a in agents
        agent_ids[idx] = a.id
        steps[idx] = 0
        positions[idx] = a.pos
        headings[idx] = a.heading
        idx += 1
    end

    for step in 1:n_steps
        for a in agents
            # persistence.jl indexes position-major: state (i, h) is (i-1)*H + h.
            src = n_headings > 0 ? (a.pos - 1) * n_headings + a.heading : a.pos
            if 1 <= src <= L
                rng_ = nzrange(Pt, src)
                probs = nonzeros(Pt)[rng_]
                if !isempty(probs)
                    dest = rowvals(Pt)[rng_]
                    r = rand(rng)
                    cum_p = 0.0
                    chosen = dest[end]          # guards against fp drift
                    for k in eachindex(probs)
                        cum_p += probs[k]
                        if r <= cum_p
                            chosen = dest[k]
                            break
                        end
                    end
                    if n_headings > 0
                        a.pos = fld(chosen - 1, n_headings) + 1
                        a.heading = (chosen - 1) % n_headings + 1
                    else
                        origin = a.pos
                        a.pos = chosen
                        if centroids !== nothing && chosen != origin
                            a.heading = heading_bin(
                                bearing_deg(centroids[origin], centroids[chosen];
                                            coord_space = coord_space),
                                heading_bins)
                        end
                    end
                end
            end

            agent_ids[idx] = a.id
            steps[idx] = step
            positions[idx] = a.pos
            headings[idx] = a.heading
            idx += 1
        end
    end

    return DataFrame(
        tagid = agent_ids,
        step = steps,
        mesh_unit = positions,
        heading = headings
    )
end