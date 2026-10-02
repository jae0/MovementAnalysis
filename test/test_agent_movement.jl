using Test
using MovementAnalysis
using SparseArrays
using Random
using DataFrames
using Statistics: mean

@testset "Agent-Based Movement" begin
    #   1 - 2 - 3
    #   |       |
    #   4 - 5 - 6
    W = sparse([
        0 1 0 1 0 0;
        1 0 1 0 0 0;
        0 1 0 0 0 1;
        1 0 0 0 1 0;
        0 0 0 1 0 1;
        0 0 1 0 1 0;
    ])
    hsi = [0.1, 0.2, 0.3, 0.1, 0.5, 1.0]
    S = 6
    # Node 6 has the highest HSI, so strong advection should concentrate the walk.
    T = build_sparse_transition_kernel(W, hsi, 2.0, 0.1, 1.0, nothing)
    start_nodes = fill(1, 5)
    n_agents = 5

    # --- one row per agent per step, step 0 being the release ------------------
    df = simulate_agent_trajectories(n_agents, start_nodes, 10, sparse(T); seed = 42)
    @test names(df) == ["tagid", "step", "mesh_unit"]
    @test size(df, 1) == n_agents * 11
    @test all(1 .<= df.mesh_unit .<= S)
    @test sort(unique(df.tagid)) == collect(1:n_agents)
    for a in 1:n_agents
        block = df[df.tagid .== a, :]
        @test issorted(block.step)
        @test first(block.step) == 0 && last(block.step) == 10
        @test first(block.mesh_unit) == 1
    end

    # --- strong advection concentrates the population on the best cell --------
    long = simulate_agent_trajectories(200, fill(1, 200), 25, sparse(T); seed = 7)
    final = long[long.step .== 25, :mesh_unit]
    @test argmax([count(==(i), final) for i in 1:S]) == 6
    @test mean(final .== 6) > mean(final .== 1)

    # --- determinism ---------------------------------------------------------
    @test simulate_agent_trajectories(
        n_agents, start_nodes, 10, sparse(T); seed = 42) == df
    @test simulate_agent_trajectories(
        n_agents, start_nodes, 10, sparse(T); seed = 43) != df

    # --- per-agent horizons, the empirical-duration form ----------------------
    horizons = [1, 3, 5, 7, 9]
    dh = simulate_agent_trajectories(n_agents, start_nodes, horizons, sparse(T); seed = 42)
    @test size(dh, 1) == n_agents + sum(horizons)
    for (i, h) in enumerate(horizons)
        @test maximum(dh[dh.tagid .== i, :step]) == h
    end

    # --- directional persistence ---------------------------------------------
    # kappa > 0 needs coordinates, and straight-line walks should dominate
    # diffusive ones. Centroid 6 sits due east of 1, along the bottom row.
    cents = [(0.0, 1.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.0), (1.0, -1.0), (2.0, -1.0)]
    dp = simulate_agent_trajectories(
        300, fill(1, 300), 20, sparse(T); centroids = cents, persistence = 2.0, seed = 3)
    @test size(dp, 1) == 300 * 21
    @test any(dp.mesh_unit .!= 1)

    @test_throws ArgumentError simulate_agent_trajectories(
        2, [1, 1], 5, sparse(T); persistence = 1.0)          # no centroids
    @test_throws DimensionMismatch simulate_agent_trajectories(
        2, [1, 1], 5, sparse(T); centroids = cents[1:3], persistence = 1.0)

    # --- input validation ----------------------------------------------------
    @test_throws DimensionMismatch simulate_agent_trajectories(
        3, [1, 1], 5, sparse(T))                              # too few starts
    @test_throws DimensionMismatch simulate_agent_trajectories(
        2, [1, 1], [1, 2, 3], sparse(T))                      # horizon length
    @test_throws ArgumentError simulate_agent_trajectories(
        2, [1, 1], [-1, 5], sparse(T))                        # negative horizon
    @test_throws ArgumentError simulate_agent_trajectories(
        1, [S + 1], 5, sparse(T))                             # start off the mesh

    # --- forward_project_agents: decoupled count, empirical durations --------
    # n_agents is free and independent of the pool size. Starts are drawn with
    # replacement, so more agents means less Monte Carlo error, not a different
    # distribution.
    releases = [1, 2, 3]
    durations = [2, 4, 6, 8, 10]
    for na in (1, 7, 60)
        fp = forward_project_agents(
            releases, durations; n_agents = na, transition_kernel = sparse(T), seed = 5)
        @test length(unique(fp.tagid)) == na
        @test all(1 .<= fp.mesh_unit .<= S)
        # Every agent's horizon is one of the supplied durations, sampled with
        # replacement, so the row count is the agent count plus the drawn horizons.
        tops = [maximum(fp[fp.tagid .== a, :step]) for a in unique(fp.tagid)]
        @test all(in(durations), tops)
        @test size(fp, 1) == na + sum(tops)
    end

    # Deterministic under a fixed seed, and the count really is honoured.
    f1 = forward_project_agents(
        releases, durations; n_agents = 40, transition_kernel = sparse(T), seed = 11)
    f2 = forward_project_agents(
        releases, durations; n_agents = 40, transition_kernel = sparse(T), seed = 11)
    @test f1 == f2
    @test length(unique(f1.tagid)) == 40

    @test_throws ArgumentError forward_project_agents(
        Int[], durations; n_agents = 5, transition_kernel = sparse(T))
    @test_throws ArgumentError forward_project_agents(
        releases, Int[]; n_agents = 5, transition_kernel = sparse(T))

    # --- forward_space_use ---------------------------------------------------
    # Returns unit_id, visit_probability, visits, mean_dwell_steps. `visits`
    # counts a unit once per agent, so it is bounded by the agent count rather
    # than the row count; `mean_dwell_steps` is per agent, not a raw total.
    su = forward_space_use(f1, S)
    @test su.unit_id == collect(1:S)
    @test length(su.visits) == S
    @test length(su.visit_probability) == S
    @test length(su.mean_dwell_steps) == S
    n_agents = length(unique(f1.tagid))
    @test all(su.visits .>= 0) && all(su.visits .<= n_agents)
    @test all(isfinite, su.visit_probability)
    @test all(isfinite, su.mean_dwell_steps)
    # A cell no agent reached has zero dwell rather than a NaN.
    @test all(su.mean_dwell_steps[su.visits .== 0] .== 0.0)
    # visit_probability is a probability.
    @test all(0.0 .<= su.visit_probability .<= 1.0)
end