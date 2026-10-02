using Test
using MovementAnalysis
using SparseArrays
using DataFrames

# The agent projection is only useful if its outputs are actually reachable.
# These tests assert the *wiring* -- that agent results reach maps, tables, and
# the interactive dashboard -- because none of that was covered before, and a
# broken thread fails silently at render time rather than at fit time.
@testset "Agent Outputs Reach Visualisation and Tables" begin
    m = build_hex_mesh_planar([-64.0, -62.0], [44.0, 46.0]; radius_km = 30.0)
    S = m.n_units
    hsi = MovementAnalysis.sanitise_hsi(collect(range(0.2, 0.9; length = S)))
    P = sparse(build_sparse_transition_kernel(m.W, hsi, 1.5, 0.2, 0.5, nothing))

    traj = forward_project_agents(
        collect(1:S), fill(10, S);
        n_agents = 12, transition_kernel = P, seed = 3)

    @test !isempty(traj)
    @test length(unique(traj.tagid)) == 12

    # --- the space-use summary carries every field a table needs -------------
    su = forward_space_use(traj, S)
    @test su.unit_id == collect(1:S)
    @test length(su.visits) == S == length(su.visit_probability) == length(su.mean_dwell_steps)
    @test all(0.0 .<= su.visit_probability .<= 1.0)
    @test all(isfinite, su.mean_dwell_steps)
    # Units nobody reached must be zero, not NaN: a table full of NaN is worse
    # than a table full of zeros.
    @test all(su.mean_dwell_steps[su.visits .== 0] .== 0.0)

    # --- the projected-use choropleth actually renders -----------------------
    su_map = leaflet_choropleth(
        m.polygons_lonlat, su.visit_probability;
        title = "Projected Space Use", vmin = 0.0, vmax = 1.0)
    @test su_map isa MovementAnalysis.LeafletMap
    html = su_map.html
    @test occursin("Projected Space Use", html)
    @test occursin("L.geoJSON", html) || occursin("FeatureCollection", html)

    # --- agent tracks map renders ------------------------------------------
    pts = [[(Float64(m.centroids_lonlat[u][1]), Float64(m.centroids_lonlat[u][2]))
            for u in traj.mesh_unit]]
    trk = leaflet_tracks_map(pts, m; max_paths = 1)
    @test trk isa MovementAnalysis.LeafletMap
    @test occursin("L.geoJSON", trk.html)

    # --- the agent layer on the interactive paths dashboard -----------------
    # Without agent_paths the dashboard still builds, so a dropped thread is
    # invisible; the layer and its GeoJSON must both be present when supplied.
    solo = leaflet_tracks_map(pts, m; max_paths = 1)
    threaded = leaflet_tracks_map(pts, m; agent_paths = pts, max_paths = 1)
    @test occursin("Projected Agents", threaded.html)
    @test occursin("agentTracksData", threaded.html)
    @test occursin("pooled kernel, no endpoints", threaded.html)
    # The layer label is always in the JS template, so it proves nothing on its
    # own. The injected GeoJSON features are the real evidence, and they must be
    # absent when no agent paths are supplied.
    @test occursin("\"id\": \"agent-1\"", threaded.html)
    @test !occursin("\"id\": \"agent-1\"", solo.html)

    # --- corridor explorer still builds without agent input -----------------
    corr = leaflet_interactive_corridor_dashboard(P, m; max_paths_render = 2)
    @test corr isa MovementAnalysis.LeafletMap
    @test occursin("FeatureCollection", corr.html)
end