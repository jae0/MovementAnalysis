"""
    agent_movement.jl

Alternative Agent-Based Model (ABM) for animal movement.
Simulates individual animals undergoing advective (directed by habitat suitability) 
and diffusive (random) movement over a spatial mesh graph without growth/demographic processes.
"""

using SparseArrays
using Random
using Distributions
using DataFrames

mutable struct CrabAgent
    id::Int
    pos::Int
    group::Int
end

"""
    simulate_agent_trajectories(
        n_agents::Int,
        start_nodes::Vector{Int},
        groups::Vector{Int},
        transition_kernels::Vector{SparseMatrixCSC{Float64, Int}},
        n_steps::Int;
        seed::Int=42
    )

Simulates continuous forward trajectories of agents on a graph.
Each agent transitions based on the probabilities in its group's transition kernel.
Returns a DataFrame containing the agent trajectories.
"""
function simulate_agent_trajectories(
    n_agents::Int,
    start_nodes::Vector{Int},
    groups::Vector{Int},
    transition_kernels::Vector{SparseMatrixCSC{Float64, Int}},
    n_steps::Int;
    seed::Int=42
)
    rng = MersenneTwister(seed)
    
    agents = [CrabAgent(i, start_nodes[i], groups[i]) for i in 1:n_agents]
    
    # Transpose kernels to CSC format for fast row-slice access
    # (Since original is row-stochastic CSC, transposing gives fast access to outgoing edges from node i as a column)
    T_kernels_t = [sparse(T') for T in transition_kernels]
    
    total_records = n_agents * (n_steps + 1)
    agent_ids = Vector{Int}(undef, total_records)
    steps = Vector{Int}(undef, total_records)
    positions = Vector{Int}(undef, total_records)
    grp_out = Vector{Int}(undef, total_records)
    
    idx = 1
    for a in agents
        agent_ids[idx] = a.id
        steps[idx] = 0
        positions[idx] = a.pos
        grp_out[idx] = a.group
        idx += 1
    end
    
    for step in 1:n_steps
        for a in agents
            T_g_t = T_kernels_t[a.group]
            
            # The column a.pos in T_g_t corresponds to the row a.pos in T_g
            col_range = nzrange(T_g_t, a.pos)
            neighbors = rowvals(T_g_t)[col_range]
            probs = nonzeros(T_g_t)[col_range]
            
            if length(probs) > 0
                # Fast categorical sampling
                r = rand(rng)
                cum_p = 0.0
                chosen = neighbors[end] # fallback
                for i in 1:length(probs)
                    cum_p += probs[i]
                    if r <= cum_p
                        chosen = neighbors[i]
                        break
                    end
                end
                a.pos = chosen
            end
            
            agent_ids[idx] = a.id
            steps[idx] = step
            positions[idx] = a.pos
            grp_out[idx] = a.group
            idx += 1
        end
    end
    
    return DataFrame(
        tagid = agent_ids,
        step = steps,
        mesh_unit = positions,
        group = grp_out
    )
end
