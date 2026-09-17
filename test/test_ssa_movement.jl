using Test
using MovementAnalysis
using LinearAlgebra
using SparseArrays
using Random
using DataFrames

@testset "Continuous-Time SSA Movement Model" begin
    # 1. Parameter construction & utility calculations
    p = SSAMovementParams(velocity=0.4, diffusion=0.2, gamma=1.5)
    @test p.velocity == 0.4
    @test p.diffusion == 0.2
    @test p.gamma == 1.5

    hsi = [0.1, 0.5, 0.9]
    u = calculate_ssa_utility(hsi; params=p)
    @test length(u) == 3
    @test all(u .> 0.0)

    # Bivariate thermal & food utility
    temp = [2.5, 5.0, 1.0]
    food = [2.0, 1.0, 0.5]
    u_biv = calculate_ssa_utility(hsi; temp=temp, food=food, params=p)
    @test length(u_biv) == 3
    @test u_biv[1] > u_biv[2] # optimal temperature (2.5) should have higher utility

    # 2. Generator construction & probability conservation
    # 3-node linear graph 1 -- 2 -- 3
    W = sparse([1, 2, 2, 3], [2, 1, 3, 2], [1.0, 1.0, 1.0, 1.0], 3, 3)
    Q = construct_ssa_generator(W, u; velocity=p.velocity, diffusion=p.diffusion, gamma=p.gamma)
    
    # Generator properties: Q_ii < 0, Q_ij >= 0 (i != j), sum_j Q_ij == 0
    @test size(Q) == (3, 3)
    @test all(diag(Q) .<= 0.0)
    for i in 1:3
        @test isapprox(sum(Q[i, :]), 0.0; atol=1e-10)
    end
    # Directed taxis bias: utility increases 1 -> 2 -> 3, so Q[2, 3] should exceed Q[2, 1]
    @test Q[2, 3] > Q[2, 1]

    # 3. Finite-time transition matrix P(dt) = exp(Q * dt)
    P1 = calculate_ssa_transition_matrix(Q, 0.5; method=:expm)
    P2 = calculate_ssa_transition_matrix(Q, 0.5; method=:taylor)
    @test all(P1 .>= 0.0)
    @test all(P2 .>= 0.0)
    for i in 1:3
        @test isapprox(sum(P1[i, :]), 1.0; atol=1e-6)
        @test isapprox(sum(P2[i, :]), 1.0; atol=1e-6)
    end
    @test isapprox(P1, P2; atol=1e-2)

    # 4. Exact Gillespie SSA trajectory simulation
    trajs = simulate_gillespie_trajectories(
        [1, 2], (0.0, 5.0), Q;
        saveat=1.0, rng=MersenneTwister(42)
    )
    @test length(trajs) == 2
    @test length(trajs[1]) == 6 # 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
    for pt in trajs[1]
        @test pt[2] in (1, 2, 3)
    end

    # 5. Synthetic SSA dataset generator
    sim_bundle = generate_ssa_movement_data(
        radius_km=15.0, domain_km=100.0, n_tags=10, n_steps=3, seed=123
    )
    @test haskey(sim_bundle, :telemetry)
    @test haskey(sim_bundle, :mesh)
    @test haskey(sim_bundle, :generator)
    @test nrow(sim_bundle.telemetry) == 40 # 10 tags * 4 steps
end
