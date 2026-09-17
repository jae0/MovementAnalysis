using Test
using MovementAnalysis
using SparseArrays
using Random
using DataFrames

@testset "Agent-Based Movement" begin
    # 1. Simple small graph
    # 1 - 2 - 3
    # |       |
    # 4 - 5 - 6
    W = sparse([
        0 1 0 1 0 0;
        1 0 1 0 0 0;
        0 1 0 0 0 1;
        1 0 0 0 1 0;
        0 0 0 1 0 1;
        0 0 1 0 1 0;
    ])
    
    # 2. Transition kernels
    # Group 1: diffuses
    # Group 2: advects to node 6
    hsi = [0.1, 0.2, 0.3, 0.1, 0.5, 1.0]
    
    T_1 = build_sparse_transition_kernel(W, hsi, 0.0, 0.1, 0.0, nothing) # pure diffusion
    T_2 = build_sparse_transition_kernel(W, hsi, 2.0, 0.1, 1.0, nothing) # strong advection
    
    T_kernels = [T_1, T_2]
    
    # Simulate
    start_nodes = [1, 1, 1, 1, 1]
    groups = [1, 1, 2, 2, 2]
    n_agents = 5
    n_steps = 10
    
    df = simulate_agent_trajectories(n_agents, start_nodes, groups, T_kernels, n_steps; seed=42)
    
    @test size(df, 1) == n_agents * (n_steps + 1)
    
    # Test group 2 reaches node 6 more often
    group2_df = filter(row -> row.group == 2 && row.step == 10, df)
    group1_df = filter(row -> row.group == 1 && row.step == 10, df)
    
    # Node 6 has highest HSI
    @test mean(group2_df.mesh_unit .== 6) >= mean(group1_df.mesh_unit .== 6)
end
