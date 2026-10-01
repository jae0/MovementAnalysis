using Test
using MovementAnalysis
using SparseArrays
using DataFrames
using Statistics

@testset "Forward Agent Projection" begin
    # 1 - 2 - 3 - 4 - 5 chain with HSI rising to the right.
    W = sparse([1, 2, 2, 3, 3, 4, 4, 5], [2, 1, 3, 2, 4, 3, 5, 4],
               ones(8), 5, 5)
    hsi = [0.1, 0.3, 0.5, 0.7, 0.9]
    P = sparse(build_sparse_transition_kernel(W, hsi, 1.0, 0.15, 0.5, nothing))

    @testset "Per-agent horizons" begin
        starts = [1, 2, 3, 4, 5]
        durs = [3, 1, 5, 2, 4]
        df = simulate_agent_trajectories(5, starts, durs, P; seed = 7)
        @test Set(propertynames(df)) == Set([:tagid, :step, :mesh_unit])

        counts = DataFrames.combine(groupby(df, :tagid), :step => length => :n)
        @test sort(counts.n) == sort(durs .+ 1)
        for row in eachrow(counts)
            @test row.n == durs[row.tagid] + 1
        end
        @test all(df[df.step .== 0, :mesh_unit] .== starts[df[df.step .== 0, :tagid]])

        # A scalar horizon still applies to every agent.
        uni = simulate_agent_trajectories(3, [1, 2, 3], 4, P; seed = 1)
        cu = DataFrames.combine(groupby(uni, :tagid), :step => length => :n)
        @test all(cu.n .== 5)

        # Validation still covers the shared paths.
        @test_throws DimensionMismatch simulate_agent_trajectories(4, starts, durs, P; seed = 1)
        @test_throws DimensionMismatch simulate_agent_trajectories(5, starts, [1, 2, 3], P; seed = 1)
        @test_throws ArgumentError simulate_agent_trajectories(5, starts, [-1, 1, 2, 3, 4], P; seed = 1)
        @test_throws ArgumentError simulate_agent_trajectories(5, [0, 2, 3, 4, 5], durs, P; seed = 1)
    end

    @testset "Agent count is decoupled from the observation pool" begin
        pool = [1, 2, 3, 4, 5]
        durs = [3, 7, 2, 9, 4]
        proj = forward_project_agents(
            pool, durs; n_agents = 250, transition_kernel = P, seed = 11
        )
        @test length(unique(proj.tagid)) == 250
        @test hasproperty(proj, :start_node)
        @test hasproperty(proj, :n_steps)
        @test all(p -> p in pool, proj.start_node)
        # Horizons come from the empirical pool, so they stay on the data's scale.
        @test all(h -> h in durs, proj.n_steps)
        # A single observation does not cap the number of agents.
        @test length(unique(proj.tagid)) > length(pool)
        # `n_steps` is a per-row column, so the row total needs one horizon per
        # agent rather than the column sum.
        per_agent = DataFrames.combine(groupby(proj, :tagid), :n_steps => first => :horizon)
        @test nrow(proj) == 250 + sum(per_agent.horizon)

        @test_throws ArgumentError forward_project_agents(
            Int[], durs; n_agents = 5, transition_kernel = P
        )
        @test_throws ArgumentError forward_project_agents(
            pool, Int[]; n_agents = 5, transition_kernel = P
        )
        @test_throws ArgumentError forward_project_agents(
            pool, durs; n_agents = 0, transition_kernel = P
        )
    end

    @testset "Reproducibility" begin
        pool = [1, 2, 3, 4, 5]
        durs = [3, 7, 2, 9, 4]
        a = forward_project_agents(pool, durs; n_agents = 40, transition_kernel = P, seed = 5)
        b = forward_project_agents(pool, durs; n_agents = 40, transition_kernel = P, seed = 5)
        c = forward_project_agents(pool, durs; n_agents = 40, transition_kernel = P, seed = 6)
        @test a.mesh_unit == b.mesh_unit
        @test a.start_node == b.start_node
        @test a.mesh_unit != c.mesh_unit
    end

    @testset "Unconditioned space use" begin
        proj = forward_project_agents(
            [1, 2, 3, 4, 5], [4, 4, 4, 4, 4];
            n_agents = 300, transition_kernel = P, seed = 21
        )
        su = forward_space_use(proj, 5)
        @test su.unit_id == 1:5
        @test all(0.0 .<= su.visit_probability .<= 1.0)
        @test su.visits[5] > 0
        @test sum(su.visit_probability) .> 1.0   # agents visit several units
        # Dwell is only defined where a unit was visited.
        @test all(su.mean_dwell_steps[su.visits .== 0] .== 0.0)
        @test all(su.mean_dwell_steps[su.visits .> 0] .> 0.0)
        # An empty projection reports nothing visited rather than dividing by zero.
        empty_su = forward_space_use(
            DataFrame(tagid = Int[], step = Int[], mesh_unit = Int[]), 5
        )
        @test all(iszero, empty_su.visits)
        @test all(iszero, empty_su.visit_probability)
    end
end

@testset "Forward Projection Map" begin
    W = sparse([1, 2, 2, 3, 3, 4, 4, 5], [2, 1, 3, 2, 4, 3, 5, 4],
               ones(8), 5, 5)
    hsi = [0.1, 0.3, 0.5, 0.7, 0.9]
    P = Matrix(sparse(build_sparse_transition_kernel(W, hsi, 1.0, 0.15, 0.5, nothing)))
    polys = Vector{Vector{Tuple{Float64, Float64}}}()
    for i in 1:5
        push!(polys, [(i - 1.0, 0.0), (Float64(i), 0.0),
                      (Float64(i), 1.0), (i - 1.0, 1.0), (i - 1.0, 0.0)])
    end
    au = (
        centroids = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5), (4.5, 0.5)],
        polygons = polys,
    )

    m = leaflet_forward_projection_map(
        P, au; hsi = hsi, n_paths = 30, n_steps = 12, seed = 3
    )
    @test m isa LeafletMap
    @test m.metadata[:n_units] == 5
    @test m.metadata[:seed] == 3
    @test m.metadata[:n_paths] == 30
    @test m.metadata[:conditioning] == "forward_from_origin_only"

    html = m.html
    # The kernel and centroids must be embedded for client-side walking.
    @test occursin("var P = {", html)
    @test occursin("var CENTS = {", html)
    @test occursin("L.geoJSON", html)
    @test occursin("fitBounds", html)
    # Click-to-project wiring, and forward walking rather than a beam.
    @test occursin("layer.on('click', function() { project(uid); })", html)
    @test occursin("function stepOnce(unit)", html)
    @test occursin("nextRandom()", html)
    # Accumulated unconditioned visit frequency.
    @test occursin("visitCount", html)
    @test occursin("repaintUnits", html)
    # Projected tracks are dashed and are not labelled as recaptures.
    @test occursin("dashArray: '4,3'", html)
    @test occursin("unconditioned on recapture", html)
    # A caption containing quotes or a script tag must not add a closing tag.
    nasty = leaflet_forward_projection_map(
        P, au; hsi = hsi, title = "Proj \"quoted\" </script>"
    )
    benign = leaflet_forward_projection_map(P, au; hsi = hsi, title = "Proj")
    @test length(collect(eachmatch(r"</script>", nasty.html))) ==
          length(collect(eachmatch(r"</script>", benign.html)))
    @test occursin("\\u003c/script\\u003e", nasty.html)
    # Double quotes need no escaping in HTML text content, so the caption is
    # preserved verbatim while the angle brackets are neutralised.
    @test occursin("Proj \"quoted\" \\u003c/script\\u003e", nasty.html)

    @test_throws DimensionMismatch leaflet_forward_projection_map(P[1:4, 1:4], au)
    @test_throws DimensionMismatch leaflet_forward_projection_map(
        P, (centroids = [(0.5, 0.5)],)
    )
    @test_throws ArgumentError leaflet_forward_projection_map(
        P, au; n_paths = 0
    )
    @test_throws ArgumentError leaflet_forward_projection_map(
        P, au; n_steps = -1
    )
end

@testset "Agent Trajectories Render Distinctly" begin
    polys = Vector{Vector{Tuple{Float64, Float64}}}()
    for i in 1:3
        push!(polys, [(i - 1.0, 0.0), (Float64(i), 0.0),
                      (Float64(i), 1.0), (i - 1.0, 1.0), (i - 1.0, 0.0)])
    end
    au = (centroids = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)], polygons = polys)

    bridge = (
        tagid = "Tag 1", path = [1, 2],
        coords = [(0.5, 0.5), (1.5, 0.5)],
        n_steps = 1, duration_days = 4.0, total_dist_km = 1.0,
        displacement_km = 1.0, tortuosity = 1.0, mean_hsi = 0.4, color = "#38bdf8",
    )
    agent = (
        tagid = "Agent 7", path = [1, 2, 3],
        coords = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)],
        n_steps = 2, duration_days = 0.0, total_dist_km = 2.0,
        displacement_km = 2.0, tortuosity = 1.0, mean_hsi = 0.4, color = "#f97316",
        trajectory_kind = "Agent projection",
    )

    m = leaflet_tracks_map([bridge, agent], au; title = "t")
    html = m.html
    @test occursin("\"simulated\": true", html)
    @test occursin("\"simulated\": false", html)
    @test occursin("Agent projection", html)
    @test occursin("dashArray: feature.properties.simulated", html)
    # A forward projection has no recapture event, so neither the terminal marker
    # nor the popup may claim one.
    @test occursin("Projected endpoint", html)
    @test occursin("Projected start", html)
    @test occursin("Recapture (End)", html)   # still used for the bridge
    @test occursin("(p.simulated ? 'Projected to:' : 'Recapture:')", html)
    @test occursin(">Trajectory:<", html)
end
