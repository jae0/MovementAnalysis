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
    n_agents, n_steps = 5, 10

    # --- the pooled kernel is the only kernel -------------------------------
    # Telemetry is fitted with one scalar parameter set, so projection takes one
    # matrix. The old signature took a per-group vector and indexed it by a group
    # column, which no longer exists anywhere upstream.
    T = build_sparse_transition_kernel(W, hsi, 2.0, 0.1, 1.0, nothing)
    start_nodes = fill(1, n_agents)
    start_headings = fill(1, n_agents)

    df = simulate_agent_trajectories(
        n_agents, start_nodes, start_headings, sparse(T), n_steps; seed = 42)

    @test size(df, 1) == n_agents * (n_steps + 1)
    @test names(df) == ["tagid", "step", "mesh_unit", "heading"]
    @test all(1 .<= df.mesh_unit .<= S)
    @test sort(unique(df.tagid)) == collect(1:n_agents)

    # Each agent occupies a contiguous block of n_steps+1 rows, and step 0 is the
    # release location, so every agent starts at node 1.
    for a in 1:n_agents
        block = df[df.tagid .== a, :]
        @test issorted(block.step)
        @test first(block.step) == 0 && last(block.step) == n_steps
        @test first(block.mesh_unit) == 1
    end

    # --- strong advection concentrates the population on the best cell ------
    # Node 6 has the highest HSI. Over many steps it should be the modal cell
    # and clearly beat the worst cell. Demanding an outright majority would be
    # asserting a concentration the parameters do not actually produce.
    df_long = simulate_agent_trajectories(
        200, fill(1, 200), fill(1, 200), sparse(T), 25; seed = 7)
    final = df_long[df_long.step .== 25, :mesh_unit]
    @test argmax([count(==(i), final) for i in 1:S]) == 6
    @test mean(final .== 6) > mean(final .== 1)

    # --- determinism --------------------------------------------------------
    @test simulate_agent_trajectories(
        n_agents, start_nodes, start_headings, sparse(T), n_steps;
        seed = 42) == df
    @test simulate_agent_trajectories(
        n_agents, start_nodes, start_headings, sparse(T), n_steps;
        seed = 43) != df

    # --- geometric heading requires coordinates, and then tracks the step ----
    centroids = [(0.0, 1.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.0), (1.0, -1.0), (2.0, -1.0)]
    dg = simulate_agent_trajectories(
        n_agents, start_nodes, start_headings, sparse(T), n_steps;
        centroids = centroids, coord_space = :km, seed = 42)
    @test all(1 .<= dg.heading .<= 8)
    # Heading survives the step: agents that moved must not all keep bin 1.
    moved = dg[(dg.step .> 0) .& (dg.mesh_unit .!= 1), :]
    @test !isempty(moved)
    @test any(moved.heading .!= 1)

    # --- persistence kernel: heading sampled jointly with position ----------
    # A (unit, heading) kernel has S*H rows; H is recovered from the size.
    H = 8
    Pk = build_persistent_transition_kernel(
        W, centroids, hsi; gamma = 2.0, residence = 0.1, advection = 1.0,
        persistence = 2.0, n_headings = H, coord_space = :km)
    @test size(Pk, 1) == S * H

    dp = simulate_agent_trajectories(
        n_agents, start_nodes, start_headings, Pk, n_steps;
        n_units = S, seed = 42)
    @test size(dp, 1) == n_agents * (n_steps + 1)
    @test all(1 .<= dp.mesh_unit .<= S)
    @test all(1 .<= dp.heading .<= H)
    # Strong kappa biases against turning, so headings must vary over a run
    # but the walk must still move off the release cell.
    @test length(unique(dp.heading)) > 1
    @test any(dp.mesh_unit .!= 1)

    # With kappa = 0 the (unit, heading) marginal reproduces the first-order
    # unit marginal. That is the property the whole formulation rests on.
    P0 = build_persistent_transition_kernel(
        W, centroids, hsi; gamma = 2.0, residence = 0.1, advection = 1.0,
        persistence = 0.0, n_headings = H, coord_space = :km)
    d0 = simulate_agent_trajectories(
        400, fill(1, 400), fill(1, 400), P0, 25; n_units = S, seed = 11)
    marg = [count(==(i), d0[d0.step .== 25, :mesh_unit]) for i in 1:S]
    @test argmax(marg) == 6

    # --- input validation ---------------------------------------------------
    @test_throws ArgumentError simulate_agent_trajectories(
        0, Int[], Int[], sparse(T), 1)
    @test_throws ArgumentError simulate_agent_trajectories(
        2, [1, 1], [1, 1], sparse(T), -1)
    @test_throws DimensionMismatch simulate_agent_trajectories(
        3, [1, 1], [1, 1, 1], sparse(T), 1)
    @test_throws DimensionMismatch simulate_agent_trajectories(
        1, [1], [1], sparse(T[1:2, 1:3]), 1)      # non-square kernel
    @test_throws BoundsError simulate_agent_trajectories(
        1, [S + 1], [1], sparse(T), 1)             # release cell off the mesh
    # A persistence kernel whose size is not a multiple of n_units is a bug, not
    # something to silently reinterpret as a first-order kernel.
    @test_throws DimensionMismatch simulate_agent_trajectories(
        1, [1], [1], Pk, 1; n_units = 5)

    # --- the pipeline call site, against a real mesh ------------------------
    # The agent branch of run_analysis resolves centroids from the loaded mesh
    # and passes n_units from the fitted kernel. Nothing else covers that wiring,
    # and a resolution that silently returned the wrong length would put the
    # bearing computation out of range at runtime.
    m = build_hex_mesh_planar([-64.0, -62.0], [44.0, 46.0]; radius_km = 30.0)
    Sm = m.n_units
    # _resolve_centroids returns (planar_km, lonlat, mesh_drawing); a caller that
    # wants one coordinate list must destructure it. Passing the tuple straight
    # through is what the agent branch used to do.
    planar, lonlat, draw = MovementAnalysis._resolve_centroids(m, Sm)
    @test length(planar) == Sm && length(lonlat) == Sm && length(draw) == Sm
    cents = planar === nothing ? lonlat : planar
    @test length(cents) == Sm

    Wm = m.W
    hm = MovementAnalysis.sanitise_hsi(collect(range(0.2, 0.9; length = Sm)))
    Pm = build_sparse_transition_kernel(Wm, hm, 1.5, 0.2, 0.5, nothing)
    dm = simulate_agent_trajectories(
        12, fill(1, 12), fill(1, 12), sparse(Pm), 15;
        n_units = Sm, centroids = cents, coord_space = :km, seed = 5)
    @test size(dm, 1) == 12 * 16
    @test all(1 .<= dm.mesh_unit .<= Sm)
    # Real meshes are long enough that an agent must be able to traverse it.
    @test any(dm.mesh_unit .!= 1)
end