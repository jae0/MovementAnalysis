# ==============================================================================
# MovementAnalysis Test Suite
# ==============================================================================

using Test
using MovementAnalysis
using LinearAlgebra
using SparseArrays
using Random
using DataFrames
using Statistics
using Turing
using DynamicPPL
using ForwardDiff
using ArgParse
using TOML
using JLD2
using JSON
using LibGEOS

# Scratch for tests that must write a real file. It lives inside the package
# rather than the OS temp directory: mktempdir requires its parent to exist, and
# the suite must not write outside the workspace.
const TEST_TMP = joinpath(pkgdir(MovementAnalysis), "test_tmp")
mkpath(TEST_TMP)

# Stand-ins for the sampler's chain, which wraps each key as `Parameter(name)` or
# `Extra(name)` and returns a vector-valued parameter as one vector per draw. The
# test suite pins that layout without paying for a full MCMC fit.
struct ParName
    s::String
end
Base.string(p::ParName) = "Parameter($(p.s))"

struct FakeChain
    data::Dict{Any,Any}
    n::Int
end
Base.keys(c::FakeChain) = keys(c.data)
Base.getindex(c::FakeChain, k) = c.data[k]
Base.Array(c::FakeChain) = zeros(Float64, c.n, 2)
Base.Array(x::Vector) = x
Base.Array(x::Matrix) = x

# A fitted chain must carry exactly the three population parameters and none of
# the per-individual block. The `z_*` effects once sampled here never reached the
# likelihood -- the kernel was built from their mean -- so they cost ~11,300
# parameters and bought no information (see todo.md 1.5). Checked as one predicate
# because the property is uniform across the six names.
const POPULATION_PARAMS = ("mu_velocity", "mu_diffusion", "mu_gamma")
const REMOVED_PARAMS = ("sigma_velocity", "sigma_diffusion", "sigma_gamma",
                        "z_velocity", "z_diffusion", "z_gamma")

function population_level_parameter_names(chain)
    names = Set(string(k) for k in keys(chain))
    return all("Parameter($p)" in names for p in POPULATION_PARAMS) &&
           !any("Parameter($p)" in names for p in REMOVED_PARAMS)
end

@testset "MovementAnalysis Core Engine" begin

    @testset "Parameters and Configurations" begin
        # A dataset is identified by its input files, not by a preset function, and
        # the layering is defaults < TOML < CLI < overrides.
        p_def = movement_parameters_default()
        @test p_def isa MovementAnalysisConfig
        @test p_def.tagging_file === nothing
        @test p_def.path_methods == [:astar]
        @test p_def.model_modes isa Vector{Symbol}
        @test p_def.diagnostics isa Vector{Symbol}
        @test p_def.land_source === :landmask        # a Symbol, not a String

        c = MovementAnalysisConfig(; tagging_file = "x.jld2", n_samples = 7)
        @test c.tagging_file == "x.jld2"
        @test c.n_samples == 7
        @test c.path_methods == [:astar]             # untouched by the override

        # Coercion from a command-line string to the declared field type.
        c2 = load_config(cli_args = [
            "--samples=9", "--warmup=4", "--depth-range=10,50",
            "--model-modes=telemetry", "--land-source=polygons",
            "--cmap=plasma", "--region-labels=A,B", "--render-html=false",
        ])
        @test c2.n_samples == 9
        @test c2.n_warmup == 4
        @test c2.depth_range == [10.0, 50.0]
        @test c2.model_modes == [:telemetry]
        @test c2.land_source === :polygons
        @test c2.cmap === :plasma
        @test c2.region_labels == ["A", "B"]
        @test c2.render_html === false

        # `--tessellation-only` is a switch, not a value flag: the natural way to
        # ask "build the mesh, show it, stop" is bare.
        @test load_config(cli_args = ["--tessellation-only"]).tessellation_only === true
        @test load_config().tessellation_only === false

        # `--all` switch runs all inference modes, path methods, and diagnostics.
        c_all = load_config(cli_args = ["--all"])
        @test c_all.all === true
        @test :agent in c_all.model_modes
        @test :viterbi in c_all.path_methods
        @test :bayesian_ensemble in c_all.diagnostics

        # `source` on `load_open_bathymetry` was unreachable from configuration, so
        # a measured depth grid could not be supplied at all. Without one the
        # loader fabricates a shelf and the depth rules are inert.
        @test load_config().bathymetry_source === nothing
        @test load_config(cli_args = ["--bathymetry-source=data/bathy.csv"]
                         ).bathymetry_source == "data/bathy.csv"
        mktempdir(TEST_TMP) do dir
            p = joinpath(dir, "c.toml")
            write(p, "bathymetry_source = \"data/real_bathy.jld2\"\n")
            @test load_config(config_path = p).bathymetry_source ==
                  "data/real_bathy.jld2"
        end

        # `store_true` leaves `default = false` in ArgParse, which is exactly the
        # case the drop-defaults rule has to get right in both directions.
        mktempdir(TEST_TMP) do dir
            p = joinpath(dir, "tess.toml")
            write(p, "tessellation_only = true\n")
            @test load_config(config_path = p).tessellation_only === true
            write(p, "tessellation_only = false\n")
            @test load_config(config_path = p).tessellation_only === false
        end

        # An absent flag must not displace a value coming from a config file, so
        # the defaults ArgParse fills in are dropped rather than applied.
        mktempdir(TEST_TMP) do dir
            p = joinpath(dir, "cfg.toml")
            write(p, """
                species_name = "Snow Crab"
                n_samples = 500
                  model_modes = ["telemetry", "telemetry_and_survey", "agent"]
                depth_range = [25.0, 400.0]
                region_labels = []
                """)
            from_file = load_config(config_path = p, cli_args = ["--max-paths=2"])
            @test from_file.species_name == "Snow Crab"
            @test from_file.n_samples == 500
              @test from_file.model_modes == [:telemetry, :telemetry_and_survey, :agent]
            @test from_file.depth_range == [25.0, 400.0]
            @test from_file.max_paths == 2

            # TOML has no null, so a `nothing` field must be omitted on save for
            # the round trip to come back as `nothing` rather than an empty path.
            out = joinpath(dir, "round.toml")
            save_config(from_file, out)
            back = MovementAnalysisConfig(out)
            # Checked as one assertion over every field: the property is uniform,
            # so per-field tests would add 48 assertions and no extra information.
            # `show_help` is a CLI artifact, not analysis state.
            bad = [f for f in fieldnames(MovementAnalysisConfig)
                   if f !== :show_help &&
                      !isequal(getfield(from_file, f), getfield(back, f))]
            @test isempty(bad)
        end

        # Survey-dependent approaches are dropped without a survey file, not errored.
        @test effective_model_modes(c) == setdiff(c.model_modes, MovementAnalysis.survey_dependent_modes())
        @test isempty(intersect(effective_model_modes(c), MovementAnalysis.survey_dependent_modes()))

        # An unknown key is a load error, not a silently ignored typo.
        @test_throws ArgumentError MovementAnalysis.apply_overrides(
            MovementAnalysisConfig(), Dict{Symbol,Any}(:not_a_field => 1))

        # An unknown *value* in a list setting is the same class of typo, and was
        # previously silent: `wants_diagnostic` is a membership test, so
        # `--diagnostics=valdiation` produced a run that "succeeded" having done
        # less than asked.
        for bad in ("--diagnostics=valdiation", "--model-modes=telmetry",
                    "--path-methods=astar,dijkstra")
            @test_throws ArgumentError load_config(cli_args = [bad])
        end
        # The valid values still load, including multi-valued lists.
        @test load_config(cli_args = ["--diagnostics=validation"]).diagnostics ==
              [:validation]
        @test load_config(cli_args = ["--path-methods=astar,viterbi"]).path_methods ==
              [:astar, :viterbi]

        # `show_help` is a CLI artifact and must not appear in a saved config.
        mktempdir(TEST_TMP) do dir
            p = joinpath(dir, "c.toml")
            save_config(load_config(cli_args = ["--help"]), p)
            @test !haskey(TOML.parsefile(p), "show_help")
        end

        # Every declared flag must bind to a real field, or it would parse and then
        # be discarded without trace.
        s = create_argparse_settings()
        @test s isa ArgParse.ArgParseSettings
        known = Set(fieldnames(MovementAnalysisConfig))
        unbound = [f.dest_name for f in s.args_table.fields
                   if f.dest_name != "config" && !(Symbol(f.dest_name) in known)]
        @test isempty(unbound)
    end

    @testset "Spatial Pruning and Domain Bounds Delimitation" begin
        # 1. resolve_bbox delimitation
        lons = [-66.5, -60.0, -58.0]
        lats = [42.0, 45.0, 47.0]
        # Without sppoly_bounds
        bbox_raw = resolve_bbox(nothing, 0.5, lons, lats)
        @test bbox_raw[1] == -67.0
        @test bbox_raw[2] == 41.5

        # With sppoly_bounds the extent is the UNION of the padded data extent and
        # the footprint bounds, on every edge.
        #
        # This used to clamp inward with `max` on the west and south, on the theory
        # that the grid should not expand past the survey. That is backwards when
        # the footprint *is* the declared domain: the snow crab footprint reaches
        # lon -65.59 and lat 42.99 while its telemetry reaches -65.42 and 43.02, so
        # the clamp discarded the footprint's own outer slivers and no amount of
        # downstream filtering could put those cells back.
        sp_b = (-65.48, 43.04, -57.32, 47.27)
        bbox_delim = resolve_bbox(nothing, 0.5, lons, lats; sppoly_bounds = sp_b)
        @test bbox_delim[1] == -67.0      # data reaches further west than the footprint
        @test bbox_delim[2] == 41.5      # and further south
        @test bbox_delim[3] == -57.32    # footprint reaches further east than the data
        @test bbox_delim[4] == 47.5      # and further north

        # The footprint must be contained by the union, on all four sides.
        @test bbox_delim[1] <= sp_b[1]
        @test bbox_delim[2] <= sp_b[2]
        @test bbox_delim[3] >= sp_b[3]
        @test bbox_delim[4] >= sp_b[4]

        # An explicit --bbox still overrides the union entirely.
        bbox_explicit = resolve_bbox([-62.0, 44.0, -60.0, 46.0], 0.5, lons, lats;
                                     sppoly_bounds = sp_b)
        @test bbox_explicit == (-62.0, 44.0, -60.0, 46.0)

        # 2. extract_sppoly_bounds on data/sppoly.jld2 if present
        if isfile("data/sppoly.jld2")
            b_found = extract_sppoly_bounds("data/sppoly.jld2")
            @test b_found !== nothing
            @test b_found[1] <= -65.0
            @test b_found[2] <= 43.1
        end

        # 3. prune_mesh functionality
        m_test = build_hex_mesh_planar([-64.0, -62.0], [44.0, 46.0]; radius_km = 30.0)
        S_orig = m_test.n_units
        keep_mask = falses(S_orig)
        keep_mask[1:div(S_orig, 2)] .= true
        m_pruned = prune_mesh(m_test, keep_mask)

        @test m_pruned.n_units == div(S_orig, 2)
        @test length(m_pruned.centroids_lonlat) == div(S_orig, 2)
        @test length(m_pruned.polygons_lonlat) == div(S_orig, 2)
        @test size(m_pruned.W) == (div(S_orig, 2), div(S_orig, 2))
        @test_throws ArgumentError prune_mesh(m_test, falses(S_orig))
    end

    @testset "Domain footprint from a spatial-unit file" begin
        # Regression: `data/sppoly.jld2` is an areal-unit definition -- 707 units,
        # each with `wkt_geo` in EPSG:4326, stored as a `Dict` of named variables.
        # The polygon reader looked for a vector of rings, found `wkt_planar` (a
        # vector of WKT *strings* in a projected CRS), and threw. The caller in
        # `load_movement_data` caught that with an empty `catch`, so the run
        # continued with no footprint and the domain became the whole mesh extent.
        mktempdir(TEST_TMP) do dir
            # Two adjacent unit squares, plus a decoy `wkt_planar` that must not be
            # picked: it is WKT in a projected CRS and would be read the wrong way.
            units = DataFrame(
                AUID     = [1, 2],
                wkt_geo  = [
                    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                    "POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))",
                ],
                wkt_planar = [
                    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                    "POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))",
                ],
            )
            path = joinpath(dir, "sppoly.jld2")
            JLD2.save(path, Dict("graph" => [1, 2], "sppoly" => units))

            geom = read_domain_polygon(path)
            @test geom !== nothing

            # Dissolved: adjacent units share an edge, so the footprint is one
            # connected polygon spanning x = 0..4, not two.
            @test geometry_contains(geom, 1.0, 1.0)
            @test geometry_contains(geom, 3.0, 1.0)
            @test !geometry_contains(geom, 5.0, 1.0)
            @test !geometry_contains(geom, 1.0, 5.0)

            # A centroid in the shared edge belongs to the domain, not to neither.
            @test geometry_contains(geom, 2.0, 1.0)

            @test geometries_in_domain(geom, [(1.0, 1.0), (3.0, 1.0), (5.0, 1.0)]) ==
                  BitVector([true, true, false])

            # No footprint configured means no spatial restriction, not "nothing
            # is in the domain".
            @test geometries_in_domain(nothing, [(1.0, 1.0), (5.0, 1.0)]) ==
                  BitVector([true, true])
            @test read_domain_polygon(nothing) === nothing

            # A configured file that cannot be read must fail loudly. Silently
            # falling back to "no footprint" is what produced a wrong domain.
            @test_throws ArgumentError read_domain_polygon(joinpath(dir, "absent.jld2"))
            bad = joinpath(dir, "bad.jld2")
            JLD2.save(bad, Dict("sppoly" => DataFrame(AUID = [1], depth = [12.0])))
            @test_throws ArgumentError read_domain_polygon(bad)
        end

        # Long-format tables still work, and WKT strings/vectors are accepted.
        mktempdir(TEST_TMP) do dir
            path = joinpath(dir, "long.jld2")
            JLD2.save(path, Dict("sppoly" => DataFrame(
                lon = [0.0, 1.0, 1.0, 0.0], lat = [0.0, 0.0, 1.0, 1.0])))
            g = read_domain_polygon(path)
            @test geometry_contains(g, 0.5, 0.5)
            @test !geometry_contains(g, 1.5, 0.5)
        end

        # Rings survive the round trip with every vertex, not every other one.
        rings = MovementAnalysis.geometry_rings(LibGEOS.readgeom(
            "POLYGON ((0 0, 3 0, 3 2, 0 2, 0 0))"))
        @test length(rings) == 1
        @test rings[1] ==
              [(-0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0), (0.0, 0.0)]

        # A multi-part geometry yields one ring per part.
        @test length(MovementAnalysis.geometry_rings(LibGEOS.readgeom(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 5, 6 5, 6 6, 5 6, 5 5)))"))) == 2

        @test_throws ArgumentError dissolve_geometries(LibGEOS.AbstractGeometry[])
    end

    @testset "GeoJSON and shapefile polygon readers" begin
        # Both were broken and neither had ever run: `_rings_from_geojson` called
        # `JSON` without importing it, and both readers declared `rings` with one
        # level of nesting too many, so pushing a single ring tried to convert a
        # `Tuple` into a `Vector` of rings.
        mktempdir(TEST_TMP) do dir
            p = joinpath(dir, "poly.geojson")
            open(p, "w") do io
                JSON.print(io, Dict(
                    "type" => "FeatureCollection",
                    "features" => [Dict(
                        "type" => "Feature",
                        "properties" => Dict("name" => "square"),
                        "geometry" => Dict(
                            "type" => "Polygon",
                            "coordinates" => [[[0.0, 0.0], [1.0, 0.0],
                                               [1.0, 1.0], [0.0, 0.0]]],
                        ),
                    )],
                ))
            end
            rings = read_polygon_file(p)
            @test length(rings) == 1
            @test length(rings[1]) == 4
            @test rings[1][1] == (0.0, 0.0)
            @test rings[1][3] == (1.0, 1.0)

            # And it feeds the land mask the same way a raster does.
            cents = [(0.5, 0.5), (5.0, 5.0)]
            @test land_mask_from_polygon_files([p], cents) == BitVector([true, false])
        end
    end

    @testset "Pruned domain is reconnected" begin
        # A 6x5 lattice. Rows 1-2 form the main body, row 4 a lobe; row 3 is
        # pruned, so the lobe is one cell away from being reattached.
        nx, ny = 6, 5
        n = nx * ny
        idx(i, j) = (j - 1) * nx + i
        W = spzeros(n, n)
        nbr = [Int[] for _ in 1:n]
        for j in 1:ny, i in 1:nx
            i < nx && (W[idx(i, j), idx(i + 1, j)] = 1.0; W[idx(i + 1, j), idx(i, j)] = 1.0)
            j < ny && (W[idx(i, j), idx(i, j + 1)] = 1.0; W[idx(i, j + 1), idx(i, j)] = 1.0)
        end
        rows, cols, _ = findnz(W)
        for (r, c) in zip(rows, cols)
            push!(nbr[r], Int(c))
        end
        cents = [(-60.0 + 0.1 * (i - 1), 44.0 + 0.1 * (j - 1))
                 for j in 1:ny for i in 1:nx]

        keep = falses(n)
        for j in (1, 2), i in 1:nx
            keep[idx(i, j)] = true          # main body, 12 cells
        end
        for i in 1:nx
            keep[idx(i, 4)] = true          # lobe, 6 cells
        end
        @test count(keep) == 18
        @test sort(length.(MovementAnalysis._connected_components(keep, nbr))) ==
              [6, 12]

        out, n_link, n_unlinked =
            MovementAnalysis.reconnect_severed_components(keep, W, cents)
        @test n_link == 1
        @test n_unlinked == 0
        @test count(out) == 19
        @test length(MovementAnalysis._connected_components(out, nbr)) == 1
        # The one added cell is in the pruned row between the two bodies, and
        # nothing else changed.
        added = [i for i in 1:n if out[i] && !keep[i]]
        @test length(added) == 1
        @test 13 <= added[1] <= 18

        # Beyond the budget a component is reported, not bridged through a ribbon
        # of re-added cells. A lobe on row 5 with row 4 also pruned needs two
        # cells of bridge.
        far = falses(n)
        for j in (1, 2, 5), i in 1:nx
            far[idx(i, j)] = true
        end
        @test sort(length.(MovementAnalysis._connected_components(far, nbr))) ==
              [6, 12]

        out2, n_link_ok, n_unlinked_ok =
            MovementAnalysis.reconnect_severed_components(far, W, cents;
                                                         max_bridge_cells = 64)
        @test n_link_ok == 2
        @test n_unlinked_ok == 0
        @test length(MovementAnalysis._connected_components(out2, nbr)) == 1

        _, n_link_no, n_unlinked_no =
            MovementAnalysis.reconnect_severed_components(far, W, cents;
                                                         max_bridge_cells = 1)
        @test n_link_no == 0
        @test n_unlinked_no == 6          # the row-5 lobe, reported once

        # An already-connected mask is returned untouched.
        out4, n_link4, n_unlinked4 =
            MovementAnalysis.reconnect_severed_components(trues(n), W, cents)
        @test n_link4 == 0 && n_unlinked4 == 0
        @test out4 == trues(n)

        @test MovementAnalysis.reconnect_severed_components(falses(n), W, cents) ==
              (falses(n), 0, 0)
    end

    @testset "Tessellation preview" begin
        mesh = build_hex_mesh_planar([-64.0, -62.0], [44.0, 46.0]; radius_km = 30.0)
        loaded = (
            mesh               = mesh,
            resharded_depths   = Float64[],
            hsi_vec            = Float64[],
            parsed_depth_range = nothing,
            data               = nothing,
        )

        mktempdir(TEST_TMP) do dir
            # Independent of `render_html`. Gating the preview on it would make
            # `--render-html=false --tessellation-only` show nothing at all, which
            # is the one combination where the user is checking the domain and
            # least wants it suppressed.
            params = load_config(cli_args = [
                "--output-dir=$dir", "--render-html=false", "--quiet=true",
            ])
            @test params.render_html === false

            path = tessellation_preview(loaded, params)
            @test isfile(path)
            @test basename(path) == "tessellation_polygons.html"
            @test abspath(path) == path          # returns an absolute path
            @test filesize(path) > 0
            @test occursin("<html", lowercase(read(path, String)))
        end

        # A launcher that cannot be found must not fail the run: the file is already
        # written, and a headless batch session is a normal outcome.
        @test open_in_browser(joinpath(mktempdir(TEST_TMP), "absent.html")) === false

        # The truncated result keeps the full key set, so a caller reading it gets
        # `nothing` rather than a missing field, and says it was a preview.
        res = MovementAnalysis._tessellation_only_result(
            loaded, "somewhere.html",
            load_config(cli_args = ["--render-html=false"]),
        )
        for k in (:models, :chains, :P_kernel, :paths, :corridors,
                  :stochastic_paths, :domain_bottlenecks, :circuit,
                  :validation_analyses, :agent_trajectories, :agent_space_use,
                  :movement_stats, :phenology, :trait_models, :parameters)
            @test res[k] === nothing
        end
        @test res.tessellation_only === true
        @test res.tessellation_path == "somewhere.html"
        @test res.mesh.n_units == mesh.n_units
        @test res.depth_range === nothing
    end

    @testset "Global Land/Sea Mask" begin
        # GeoDatasets builds its download URL by interpolating the grid verbatim, so
        # a `Float64` of 5.0 requests `lsmask_5.0min_l.bin` while the published file
        # is `lsmask_5min_l.bin`; the mismatched request 404s and the error page it
        # leaves behind then fails to gunzip. `_basemap_grid_token` is what keeps the
        # filename and the requested grid in agreement.
        @test MovementAnalysis._basemap_grid_token(5.0) == "5"
        @test MovementAnalysis._basemap_grid_token(10.0) == "10"
        @test MovementAnalysis._basemap_grid_token(1.25) == "1.25"
        @test MovementAnalysis._basemap_grid_token(2.5) == "2.5"
        @test_throws ArgumentError MovementAnalysis._basemap_grid_token(0.0)

        # An offshore point and an inland point at the same latitude must differ.
        # Nova Scotia sits near 45.5 N, 63 W.
        m = land_mask_from_global_mask(
            [(-63.0, 45.5), (-60.0, 45.5)];
            resolution = "l", grid_minutes = 5.0)
        @test m[1]      # inland
        @test !m[2]     # open Atlantic
    end

    @testset "Transition Kernel Construction" begin
        # Construct a simple 5-node line graph adjacency
        W = spzeros(5, 5)
        for i in 1:4
            W[i, i+1] = 1.0
            W[i+1, i] = 1.0
        end
        hsi = [0.2, 0.4, 0.6, 0.8, 1.0]

        P = construct_stochastic_transition_kernel(
            W, hsi;
            gamma = 1.0, residence = 0.2, advection = 0.5
        )

        @test size(P) == (5, 5)
        @test all(isapprox.(vec(sum(P; dims = 2)), 1.0; atol = 1e-6))

        # Same two properties across representative regimes, including the
        # corners: no habitat gradient, a very strong one either way, and advection
        # at both extremes. Enumerating the full cross product would add ~200
        # assertions of the same property without testing anything new.
        for (gamma, rho, alpha) in ((0.0, 0.1, 0.0), (0.0, 0.9, 1.0),
                                    (10.0, 0.5, 0.5), (-10.0, 0.5, 0.5),
                                    (1.0, 0.5, 0.5))
            P_sp = build_sparse_transition_kernel(W, hsi, gamma, rho, alpha, nothing)
            @test all(P_sp .>= 0.0)
            @test all(isapprox.(vec(sum(P_sp; dims = 2)), 1.0; atol = 1e-6))
        end
    end

    @testset "Explicit Turing Telemetry Model" begin
        # Test direct Turing model fitting
        W = spzeros(4, 4)
        W[1, 2] = 1.0; W[2, 1] = 1.0
        W[2, 3] = 1.0; W[3, 2] = 1.0
        W[3, 4] = 1.0; W[4, 3] = 1.0

        hsi = [0.2, 0.5, 0.8, 0.3]
        releases   = [1, 2, 2]
        recaptures = [2, 3, 4]
        ks         = [1, 1, 2]

        m_tel = pure_telemetry_turing_model(
            releases, recaptures, ks, W, hsi, zeros(Bool, 4)
        )
        @test m_tel isa DynamicPPL.Model

        rng = MersenneTwister(42)
        chn = sample(rng, m_tel, MH(), 40; progress = false)
        @test size(chn, 1) == 40
        @test population_level_parameter_names(chn)
    end

    @testset "Burn-in is actually discarded" begin
        # `n_warmup` was a documented, configurable field that no `sample` call
        # read, so every returned "posterior" was the opening segment of the
        # random walk -- prior-dominated, and near-identical between models for
        # the wrong reason. `num_warmup` becomes `discard_initial` in
        # AbstractMCMC, so passing it must change the retained draws.
        S = 10
        W = spzeros(S, S)
        for i in 1:S-1
            W[i, i+1] = 1.0; W[i+1, i] = 1.0
        end
        hsi = collect(range(0.2, 0.8; length = S))
        land = falses(S)
        releases = [1, 2, 3, 4, 5, 6]
        recaptures = [2, 3, 4, 5, 6, 7]
        ks = [1, 1, 2, 3, 4, 5]
        m_tel = pure_telemetry_turing_model(releases, recaptures, ks, W, hsi, land)

        no_warmup = sample(MersenneTwister(42), m_tel, MH(), 20; progress = false)
        burned = sample(MersenneTwister(42), m_tel, MH(), 20;
                        num_warmup = 100, progress = false)

        v_nowarm, _, _ = MovementAnalysis.posterior_kernel_draws(no_warmup)
        v_burn, _, _ = MovementAnalysis.posterior_kernel_draws(burned)
        @test !(v_nowarm == v_burn)

        # Burn-in must not change how many draws are returned.
        @test size(no_warmup, 1) == size(burned, 1) == 20
    end

    @testset "The chain actually moves" begin
        # `MH()` in Turing 0.49 draws proposals from the *prior*, which on data
        # this size accepts 0.0% of them: the reported "posterior" was the
        # initialisation, and two different models frozen at the same seeded start
        # reported identical parameters. This pins that the sampler the pipeline
        # uses actually mixes, and would catch a regression to a frozen chain.
        S = 10
        W = spzeros(S, S)
        for i in 1:S-1
            W[i, i+1] = 1.0; W[i+1, i] = 1.0
        end
        hsi = collect(range(0.2, 0.8; length = S))
        land = falses(S)
        releases = [1, 2, 3, 4, 5, 6]
        recaptures = [2, 3, 4, 5, 6, 7]
        ks = [1, 1, 2, 3, 4, 5]
        m = pure_telemetry_turing_model(releases, recaptures, ks, W, hsi, land)

        params = MovementAnalysisConfig(; n_samples = 60, n_warmup = 60)
        spl = MovementAnalysis.movement_sampler([0.2, 0.2, 1.0], params)

        chn = sample(MersenneTwister(42), m, spl, 60;
                     num_warmup = 60, progress = false)
        @test size(chn, 1) == 60
        v, d, g = MovementAnalysis.posterior_kernel_draws(chn)
        @test length(unique(round.(v; digits = 9))) > 1     # not frozen
        @test all(isfinite, v) && all(isfinite, d) && all(isfinite, g)
        @test std(v) > 0                                     # real spread
        @test all(>=(0.0), v) && all(>=(0.0), d)

        # A non-positive scale is a configuration error, not a silent freeze.
        @test_throws ArgumentError MovementAnalysis.movement_sampler(
            [0.2, 0.2, 1.0], MovementAnalysisConfig(; mh_proposal_scale = 0.0))
    end

    @testset "Posterior extraction is AD-safe" begin
        # `_rel_probs` used to start with `Float64.(row)`, which severed the
        # parameters from the AD tape: NUTS failed with `Float64(::Dual)` and
        # `MH(cov)` could not be used at all, because a linked-space random walk
        # needs the Jacobian. The cache took its element type from `Float64` for
        # the same reason. Both must now preserve the incoming type.
        S = 6
        W = spzeros(S, S)
        for i in 1:S-1
            W[i, i+1] = 1.0; W[i+1, i] = 1.0
        end
        hsi = collect(range(0.2, 0.8; length = S))
        P_T = sparse(build_sparse_transition_kernel(W, hsi, 0.8, 0.6, 0.4, falses(S))')

        # A Float64 kernel still gives exactly Float64 rows, bit for bit.
        c = kstep_transition_cache(P_T, [1, 2, 3], [1, 2, 3])
        @test eltype(first(values(c))) === Float64

        # A dual-numbered kernel must stay on the tape rather than being cast down.
        # This is the exact failure NUTS hit: `Float64(::ForwardDiff.Dual)`.
        dP = P_T + P_T .* ForwardDiff.Dual(1.0, 1.0)
        @test eltype(dP) <: ForwardDiff.Dual
        cd = kstep_transition_cache(dP, [1, 2], [1, 2])
        ET = eltype(first(values(cd)))
        @test ET <: ForwardDiff.Dual

        # And the partial must be non-zero: if it were being discarded, the row
        # would be numerically identical to one built at a constant.
        @test any(!iszero, ForwardDiff.partials.(first(values(cd))))
    end

    @testset "movement_alpha_rho is the single derivation of alpha/rho" begin
        # This helper exists because the models and the pipeline once each derived
        # these separately and the pipeline's `rho` came out as 1/(v+d) instead of
        # 1/(1+v+d) -- a different quantity from the one the likelihood was fitted
        # with. Pin the formula so the two can never drift again.
        for v in (0.05, 0.2, 0.5, 0.9)
            for d in (0.01, 0.1, 0.4, 1.0)
                a, r = MovementAnalysis.movement_alpha_rho(v, d)
                tot = v + d + 1e-6
                @test a ≈ clamp(v / tot, 0.0, 1.0)
                @test r ≈ clamp(1.0 / (1.0 + tot), 0.01, 0.95)
            end
        end

        # The array method must agree elementwise with the scalar one.
        vs = [0.1, 0.3, 0.7, 0.95]
        ds = [0.2, 0.05, 0.4, 0.01]
        av, rv = MovementAnalysis.movement_alpha_rho(vs, ds)
        @test [MovementAnalysis.movement_alpha_rho(vs[i], ds[i]) for i in eachindex(vs)] ==
              collect(zip(av, rv))

        # The bug being pinned: 1/(1+tot) and 1/tot differ, and only the former is
        # the model's definition. At v+d = 0.1 they are 0.909 and 10.0. The 1e-6
        # guard inside the helper is why an exact equality would not hold.
        @test MovementAnalysis.movement_alpha_rho(0.06, 0.04)[2] ≈
              clamp(1.0 / 1.1, 0.01, 0.95) atol = 1e-6
        @test !(MovementAnalysis.movement_alpha_rho(0.06, 0.04)[2] ≈
                clamp(1.0 / 0.1, 0.01, 0.95))
    end

    @testset "Posterior Extraction from a Population-Level Chain" begin
        # `posterior_kernel_draws` reduces a chain to the three per-draw series the
        # kernel needs. The wrapped-key layout detail fails silently if mishandled,
        # so it is pinned against a stand-in chain with the sampler's own shape.
        n_draws = 4

        # The shape a current model produces: three scalars, nothing else.
        c = FakeChain(Dict(
            ParName("mu_velocity")  => collect(range(0.2, 0.5; length = n_draws)),
            ParName("mu_diffusion") => fill(0.05, n_draws),
            ParName("mu_gamma")     => fill(1.0, n_draws),
        ), n_draws)

        v, d, g = MovementAnalysis.posterior_kernel_draws(c)
        @test v isa Vector{Float64} && length(v) == n_draws
        @test d isa Vector{Float64} && length(d) == n_draws
        @test g isa Vector{Float64} && length(g) == n_draws
        @test all(isfinite, v) && all(isfinite, d) && all(isfinite, g)

        mu_v = collect(range(0.2, 0.5; length = n_draws))
        @test v ≈ clamp.(mu_v, 0.0, 0.95)      # the model's own velocity bound
        @test d ≈ fill(0.05, n_draws)           # diffusion is floored, not capped
        @test g ≈ fill(1.0, n_draws)

        # The draws must not be collapsed to a single value.
        @test !all(isapprox.(v, fill(v[1], n_draws)))

        # Bookkeeping columns must not be mistaken for parameters.
        c4 = FakeChain(Dict(
            ParName("mu_velocity") => fill(0.3, n_draws),
            ParName("mu_diffusion") => fill(0.1, n_draws),
            ParName("mu_gamma") => fill(1.0, n_draws),
            ParName("accepted")    => fill(true, n_draws),
            ParName("logjoint")    => collect(1.0:n_draws),
        ), n_draws)
        v4, _, _ = MovementAnalysis.posterior_kernel_draws(c4)
        @test v4 == fill(0.3, n_draws)

        # A chain missing a `mu_*` cannot yield a kernel, so it must say so rather
        # than silently substituting a plausible-looking default.
        @test_throws ErrorException MovementAnalysis.posterior_kernel_draws(
            FakeChain(Dict(ParName("lp") => collect(1.0:n_draws)), n_draws))
        @test_throws ErrorException MovementAnalysis.posterior_kernel_draws(
            FakeChain(Dict(
                ParName("mu_velocity") => fill(0.3, n_draws),
                ParName("mu_diffusion") => fill(0.1, n_draws),
            ), n_draws))

        # A chain from the removed hierarchical model still reduces correctly, and
        # its absent `mu_*`-only path is the same code path.
        n_ind = 6
        z = [Float64[0.1 * (k - j) for k in 1:n_ind] for j in 1:n_draws]
        ch = FakeChain(Dict(
            ParName("mu_velocity")     => fill(0.3, n_draws),
            ParName("sigma_velocity")  => fill(0.1, n_draws),
            ParName("mu_diffusion")    => fill(0.05, n_draws),
            ParName("sigma_diffusion") => fill(0.02, n_draws),
            ParName("mu_gamma")        => fill(1.0, n_draws),
            ParName("sigma_gamma")     => fill(0.3, n_draws),
            ParName("z_velocity")      => z,
            ParName("z_diffusion")     => z,
            ParName("z_gamma")         => z,
        ), n_draws)
        vh, dh, gh = MovementAnalysis.posterior_kernel_draws(ch)
        zm = permutedims(reduce(hcat, z))   # n_draws x n_individuals
        @test vh ≈ [mean(clamp.(0.3 .+ 0.1 .* zm[i, :], 0.0, 0.95)) for i in 1:n_draws]
        @test dh ≈ [mean(max.(0.05 .+ 0.02 .* zm[i, :], 0.0)) for i in 1:n_draws]
        @test gh ≈ [mean(1.0 .+ 0.3 .* zm[i, :]) for i in 1:n_draws]
    end

    @testset "kstep_transition_cache: grouped sweep == per-pair" begin
        # The cache replaced a per-(release,k) construction that redid the whole
        # chain for every pair. This pins that the grouped sweep is not an
        # approximation: it must return exactly what the naive loop returns, for
        # the shape that matters -- few release nodes, many step counts each.
        S = 12
        W = spzeros(S, S)
        for i in 1:S-1
            W[i, i+1] = 1.0; W[i+1, i] = 1.0
        end
        hsi = collect(range(0.1, 0.9; length = S))
        P = build_sparse_transition_kernel(W, hsi, 0.8, 0.6, 0.4, falses(S))
        P_T = sparse(P')

        # 3 release nodes, several k values each -- the redundancy being removed.
        releases = [1, 1, 1, 4, 4, 9, 9, 9, 9]
        ks       = [0, 1, 3, 2, 5, 1, 2, 4, 6]

        # Reference: the original per-pair construction.
        naive = Dict{Tuple{Int,Int},Vector{Float64}}()
        for (r, k) in unique(zip(releases, ks))
            v = zeros(Float64, S); v[r] = 1.0
            for _ in 1:max(0, k)
                v = P_T * v
            end
            naive[(r, k)] = MovementAnalysis._rel_probs(v)
        end

        grouped = kstep_transition_cache(P_T, releases, ks)
        @test Set(keys(grouped)) == Set(keys(naive))
        @test length(grouped) == length(naive)
        for key in keys(naive)
            @test grouped[key] == naive[key]
        end

        # k = 0 is the start of the sweep, not a step inside it, so it needs
        # explicit handling. The mass sits on the release node; the other entries
        # are not exactly zero because `_rel_probs` floors zeros so `Categorical`
        # can represent the row.
        @test grouped[(1, 0)][1] > 0.99
        @test sum(grouped[(1, 0)][2:end]) < 1e-9

        # Rows stay normalised, because Categorical requires it.
        @test all(v -> sum(v) ≈ 1.0, values(grouped))
        @test all(v -> all(>=(0.0), v), values(grouped))

        # A release index outside the mesh is skipped, not an error.
        @test isempty(kstep_transition_cache(P_T, [99], [2]))
        @test isempty(kstep_transition_cache(P_T, Int[], Int[]))

        # normalize = false returns the raw sweep. The kernel is row-stochastic,
        # so the raw row already sums to 1; the difference from `naive` is only
        # the zero-flooring that `_rel_probs` adds.
        raw = kstep_transition_cache(P_T, [4], [2]; normalize = false)
        @test sum(raw[(4, 2)]) ≈ 1.0 atol = 1e-10
        @test all(>=(0.0), raw[(4, 2)])
        @test minimum(raw[(4, 2)]) < minimum(naive[(4, 2)])   # no floor applied
    end

    @testset "pruned mesh units cannot leak into observation endpoints" begin
        # Regression: the mesh prune reindexed observations with `old_to_new`, which
        # is `zeros(Int, n_spatial)` and only assigns a nonzero entry to retained
        # units. An observation endpoint on a unit the prune deleted was therefore
        # reindexed to 0 and reached Phase 2 as
        # `KeyError: key (0, 729) not found` from the k-step cache lookup -- the
        # cache skips release indices outside `1:S`, so it never built `(0, 729)`.
        cents = [(0.0, 0.0), (1.0, 0.0), (9.0, 0.0),
                 (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        keep  = [true, false, false, true, true, false]
        obs = DataFrame(
            tagid   = ["a", "b", "c"],
            release = [2, 3, 1],
            k       = [4, 9, 729],
            recapture = [6, 5, 4],
        )
        before = copy(obs)

        n_rel, n_rec = MovementAnalysis._snap_pruned_endpoints!(obs, keep, cents)
        @test (n_rel, n_rec) == (2, 1)

        # Each stale endpoint moves to the nearest *retained* unit. Unit 2 sits at
        # x = 1 and snaps back to unit 1 (x = 0) rather than forward to unit 4
        # (x = 10); unit 3 at x = 9 snaps forward to unit 4; unit 6 at x = 30 snaps
        # back to unit 5 (x = 20).
        @test obs.release == [1, 4, 1]
        @test obs.recapture == [5, 5, 4]

        # Rows that were already on retained units are untouched, including the
        # k = 729 one that produced the original crash.
        @test obs.k == before.k
        @test obs.tagid == before.tagid

        # The invariant the prune actually needs: applying `old_to_new` after the
        # snap cannot produce 0 or an out-of-range index.
        old_to_new = zeros(Int, length(keep))
        new_idx = 0
        for i in eachindex(keep)
            keep[i] && (new_idx += 1; old_to_new[i] = new_idx)
        end
        reindexed_rel = [old_to_new[r] for r in obs.release]
        reindexed_rec = [old_to_new[r] for r in obs.recapture]
        @test all(r -> 1 <= r <= new_idx, reindexed_rel)
        @test all(r -> 1 <= r <= new_idx, reindexed_rec)
        @test new_idx == 3

        # A no-op on already-clean endpoints: reports nothing, mutates nothing.
        clean = DataFrame(tagid = ["d"], release = [1], k = [3], recapture = [4])
        @test MovementAnalysis._snap_pruned_endpoints!(clean, keep, cents) == (0, 0)
        @test clean.release == [1] && clean.recapture == [4]

        # Without centroids there is nothing to snap to. The caller's post-reindex
        # range check is what has to catch the result, so this stays silent rather
        # than pretending the endpoints were fine.
        stale = DataFrame(tagid = ["e"], release = [2], k = [1], recapture = [6])
        @test MovementAnalysis._snap_pruned_endpoints!(stale, keep, nothing) == (0, 0)
        @test stale.release == [2]

        @test MovementAnalysis._snap_pruned_endpoints!(obs, keep, cents) == (0, 0)
        @test MovementAnalysis._snap_pruned_endpoints!(clean, keep, nothing) == (0, 0)

        # An out-of-range endpoint is a caller error, not something to clamp: the
        # reachability filter in Phase 1c drops those rows before the prune runs.
        oob = DataFrame(tagid = ["f"], release = [0], k = [1], recapture = [4])
        @test_throws ErrorException MovementAnalysis._snap_pruned_endpoints!(
            oob, keep, cents)

        # Nothing retained is unrecoverable, and must not silently pass.
        @test_throws ErrorException MovementAnalysis._snap_pruned_endpoints!(
            copy(obs), falses(length(keep)), cents)

        # A centroid/unit mismatch would index off the end of the centroid vector.
        @test_throws ErrorException MovementAnalysis._snap_pruned_endpoints!(
            copy(obs), keep, cents[1:3])
    end

    @testset "validate_mark_recapture_indices names the offending rows" begin
        # The crash this guards against reported only `KeyError: key (0, 729) not
        # found`, which does not say which observation carried the 0 or that 0 is
        # not a valid unit index at all.
        @test validate_mark_recapture_indices([1, 2, 3], [3, 2, 1], 3) === nothing
        @test validate_mark_recapture_indices(Int[], Int[], 4) === nothing

        err = try
            validate_mark_recapture_indices([0, 2], [2, 0], 3)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("row 1: release = 0", msg)
        @test occursin("row 2: recapture = 0", msg)
        @test occursin("1:3", msg)

        # Upper bound too, not just the 0 case -- a stale index is stale in either
        # direction once the mesh has been pruned.
        err_hi = try
            validate_mark_recapture_indices([4], [1], 3)
            nothing
        catch e
            e
        end
        @test err_hi isa ArgumentError
        @test occursin("row 1: release = 4", sprint(showerror, err_hi))

        # Long offending lists are summarized rather than dumped.
        many = try
            validate_mark_recapture_indices(zeros(Int, 12), ones(Int, 12), 3)
            nothing
        catch e
            e
        end
        @test occursin("more)", sprint(showerror, many))

        @test_throws ArgumentError validate_mark_recapture_indices([1, 2], [1], 3)
        @test_throws ArgumentError validate_mark_recapture_indices([1], [1], 0)
    end

    @testset "A* Least-Cost Routing" begin
        # 4-node diamond graph
        W = spzeros(4, 4)
        W[1, 2] = 1.0; W[2, 1] = 1.0
        W[1, 3] = 1.0; W[3, 1] = 1.0
        W[2, 4] = 1.0; W[4, 2] = 1.0
        W[3, 4] = 1.0; W[4, 3] = 1.0

        coords = [(0.0, 0.0), (1.0, 1.0), (1.0, -1.0), (2.0, 0.0)]
        path = astar_least_cost_path(coords, W, 1, 4)
        @test length(path) >= 2
        @test first(path) == 1
        @test last(path) == 4
    end

    @testset "Circuit Theory Diagnostics" begin
        # 3-node linear circuit
        W = spzeros(3, 3)
        W[1, 2] = 1.0; W[2, 1] = 1.0
        W[2, 3] = 1.0; W[3, 2] = 1.0

        L, C = build_circuit_laplacian(W)
        @test size(L) == (3, 3)
        @test size(C) == (3, 3)
        @test isapprox(sum(L, dims = 2), zeros(3, 1); atol = 1e-10)

        R_eff = effective_resistance_matrix(L)
        @test size(R_eff) == (3, 3)
        @test isapprox(R_eff[1, 1], 0.0; atol = 1e-10)
        @test R_eff[1, 3] > R_eff[1, 2]
    end
 

    @testset "Posterior Path Ensemble Panel" begin
        # The `:bayesian_ensemble` diagnostic used to compute its result and then
        # drop it, so this pins that the panel can be built from what it returns
        # and that the numbers in it mean what they claim.
        S = 5
        cents = [(-64.0 + 0.5 * i, 45.0 + 0.2 * i) for i in 1:S]
        polys = [[(c[1] - 0.2, c[2] - 0.1), (c[1] + 0.2, c[2] - 0.1),
                  (c[1] + 0.2, c[2] + 0.1), (c[1] - 0.2, c[2] + 0.1)] for c in cents]
        au = (centroids = cents, centroids_lonlat = cents,
              polygons = polys, polygons_lonlat = polys, n_units = S)
        hsi = collect(range(0.1, 0.9; length = S))

        # Every draw agrees: the route is identified.
        agreed = (ensemble_paths = Dict("A" => [[1, 2, 3, 4, 5],
                                                [1, 2, 3, 4, 5],
                                                [1, 2, 3, 4, 5]]),
                  ensemble_corridors = Dict("A" => zeros(5, 5)))
        m = plot_posterior_path_ensemble(agreed, au; hsi = hsi)
        @test m isa InteractiveMap
        @test startswith(strip(m.html), "<!DOCTYPE html>")
        @test occursin("\"n_draws\": 3", m.html)
        @test occursin("\"modal_share\": 1.0", m.html)
        @test m.metadata[:n_individuals] == 1

        # Two of three draws chose the same route, so the modal share is 2/3.
        split_ens = (ensemble_paths =
                     Dict("B" => [[1, 2, 3, 4, 5], [1, 2, 4, 5], [1, 2, 3, 4, 5]]),)
        m2 = plot_posterior_path_ensemble(split_ens, au; hsi = hsi)
        @test occursin("\"modal_share\": 0.6667", m2.html)

        # Longer routes must measure longer.
        longer = (ensemble_paths = Dict("C" => [[1, 2], [1, 2, 3], [1, 2, 3, 4]]),)
        m3 = plot_posterior_path_ensemble(longer, au; hsi = hsi)
        lo = parse(Float64, match(r"\"min_km\": ([0-9.]+)", m3.html).captures[1])
        hi = parse(Float64, match(r"\"max_km\": ([0-9.]+)", m3.html).captures[1])
        @test hi > lo

        # An out-of-range unit index is dropped, not thrown on.
        ragged = (ensemble_paths = Dict("D" => [[1, 2, 999], [0, 1, 2]]),)
        @test (plot_posterior_path_ensemble(ragged, au)).metadata[:n_individuals] == 1

        # No HSI supplied still renders, with a null payload.
        @test occursin("var HSI = null;", plot_posterior_path_ensemble(agreed, au).html)

        # Several individuals are all offered, and the cap is honoured.
        many = (ensemble_paths = Dict("T$i" => [[1, 2]] for i in 1:40),)
        @test plot_posterior_path_ensemble(many, au; max_individuals = 5
        ).metadata[:n_individuals] == 5

        # An ensemble with nothing in it is a caller error worth naming.
        @test_throws ArgumentError plot_posterior_path_ensemble(
            (ensemble_paths = Dict{String,Any}(),), au)
        @test_throws ArgumentError plot_posterior_path_ensemble(
            (ensemble_paths = Dict("E" => Vector{Int}[]),), au)
    end

    @testset "Dashboard style settings actually reach the HTML" begin
        # `dark_mode`, `cmap` and `font` were config fields that nothing read, so
        # setting them silently did nothing. `dark_mode` and `cmap` are now passed
        # through to the panels that support them; `font` is still unwired and is
        # documented as such. This pins the two that work so they cannot die again.
        S = 3
        cents = [(-63.0 + 0.5 * i, 45.0) for i in 1:S]
        polys = [[(c[1] - 0.2, c[2] - 0.1), (c[1] + 0.2, c[2] - 0.1),
                  (c[1] + 0.2, c[2] + 0.1), (c[1] - 0.2, c[2] + 0.1)] for c in cents]
        au = (centroids = cents, centroids_lonlat = cents,
              polygons = polys, polygons_lonlat = polys, n_units = S)
        hsi = [0.2, 0.6, 0.9]

        light  = plot_hsi_map(hsi, au; cmap = :viridis, dark_mode = false)
        dark   = plot_hsi_map(hsi, au; cmap = :viridis, dark_mode = true)
        plasma = plot_hsi_map(hsi, au; cmap = :plasma,  dark_mode = false)

        @test light.html != dark.html
        @test light.html != plasma.html

        # A different palette must actually change the colours in the payload.
        # `split` is ambiguous in this environment (Shapefile and Base both export
        # it), so it is qualified.
        colours(s) = Set(x for x in Base.split(s, '"')
                         if length(x) == 7 && startswith(x, "#"))
        @test !isempty(setdiff(colours(plasma.html), colours(light.html)))

        # The default palette is the same one the struct declares.
        @test MovementAnalysisConfig().cmap === :viridis
    end

    @testset "Makie HTML Map Structure" begin
        map_obj = InteractiveMap(
            "<div>Map Content</div>";
            title = "Test Title",
            width = "100%",
            height = "600px"
        )
        @test map_obj isa InteractiveMap
        @test occursin("Test Title", map_obj.title)
    end

    @testset "Interactive Corridor Dashboard & Open Bathymetry" begin
        # 3-node graph and stochastic transition matrix
        P = [
            0.2 0.8 0.0;
            0.3 0.4 0.3;
            0.0 0.5 0.5
        ]

        # Test with geographic coordinates (Lon/Lat)
        au_geo = (
            centroids = [(-63.5, 44.5), (-63.0, 44.8), (-62.5, 45.0)],
            polygons = [
                [(-63.6, 44.4), (-63.4, 44.4), (-63.4, 44.6), (-63.6, 44.6)],
                [(-63.1, 44.7), (-62.9, 44.7), (-62.9, 44.9), (-63.1, 44.9)],
                [(-62.6, 44.9), (-62.4, 44.9), (-62.4, 45.1), (-62.6, 45.1)]
            ]
        )

        map_geo = plot_interactive_corridor_dashboard(
            P, au_geo;
            title = "Test Corridor Map"
        )
        @test map_geo isa InteractiveMap
        @test occursin("esriOcean", map_geo.html)
        @test occursin("server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base", map_geo.html)
        @test occursin("Test Corridor Map", map_geo.html)
        @test !occursin("api_key", lowercase(map_geo.html))

        # Test with planar Cartesian coordinates (testing coordinate transformer)
        au_planar = (
            centroids = [
                (10000.0, 20000.0),
                (15000.0, 25000.0),
                (20000.0, 30000.0)
            ],
            polygons = [
                [
                    (9000.0, 19000.0), (11000.0, 19000.0),
                    (11000.0, 21000.0), (9000.0, 21000.0)
                ],
                [
                    (14000.0, 24000.0), (16000.0, 24000.0),
                    (16000.0, 26000.0), (14000.0, 26000.0)
                ],
                [
                    (19000.0, 29000.0), (21000.0, 29000.0),
                    (21000.0, 31000.0), (19000.0, 31000.0)
                ]
            ]
        )
        map_planar = plot_interactive_corridor_dashboard(
            P, au_planar;
            title = "Planar Test Map"
        )
        @test map_planar isa InteractiveMap
        @test occursin("esriOcean", map_planar.html)
        # One pooled kernel, so the explorer carries a single "Pooled" entry and no
        # per-group labels.
        @test occursin("Pooled", map_planar.html)
        @test !occursin("Group 1", map_planar.html)
        # A planar frame must not be rendered as longitude/latitude.
        @test !occursin("-63.", map_planar.html)
    end

    @testset "Trait-Movement Associations OLS" begin
        obs_df = DataFrame(
            tag_id = 1:10,
            carapace_width = [
                95.0, 102.0, 108.0, 115.0, 120.0,
                125.0, 130.0, 135.0, 140.0, 145.0
            ]
        )
        mov_stats = (
            net_displacement_km = [
                10.0, 12.0, 15.0, 18.0, 20.0,
                24.0, 28.0, 30.0, 33.0, 36.0
            ],
            path_efficiency     = [
                0.50, 0.52, 0.55, 0.58, 0.60,
                0.62, 0.65, 0.68, 0.70, 0.72
            ],
            tortuosity          = [
                2.0, 1.9, 1.8, 1.7, 1.6,
                1.5, 1.4, 1.3, 1.2, 1.1
            ]
        )
        loaded = (au = nothing,)

        assoc = model_trait_movement_associations(obs_df, mov_stats, loaded)
        @test haskey(assoc, :trait_name)
        @test assoc.trait_name == "carapace_width"
        @test haskey(assoc.models, "displacement")
        @test haskey(assoc.models, "efficiency")
        @test haskey(assoc.models, "tortuosity")

        disp_model = assoc.models["displacement"]
        @test disp_model.beta_1 > 0.0
        @test disp_model.r2 > 0.9
        @test disp_model.p_val < 0.01
        @test 0.0 <= disp_model.p_val <= 1.0
    end

    @testset "Resistance Distance Units Survive The Merge" begin
        # A fix in one file whose counterpart lives in another is exactly what a
        # cross-branch merge can silently break: circuit.jl passes `coord_space`
        # to `_spatial_node_distance`, which is defined in movement.jl. When the two
        # halves came from different branches the keyword did not exist and the
        # failure was a MethodError on the calibration path only.
        planar = [(10.0, 20.0), (12.0, 20.0), (14.0, 20.0)]
        @test MovementAnalysis._spatial_node_distance(
            planar[1], planar[2]; coord_space = :planar_km
        ) ≈ 2.0 atol = 1e-10
        @test MovementAnalysis._infer_coord_space(planar) == false

        geo = [(-63.0, 44.0), (-62.0, 44.0)]
        d = MovementAnalysis._spatial_node_distance(geo[1], geo[2]; coord_space = :geographic)
        @test 70.0 < d < 90.0            # ~1 degree of longitude at 44 N, in km
        @test MovementAnalysis._infer_coord_space(geo) == true

        # Known-answer resistance on a path graph: two unit resistances in series.
        Wp = sparse([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0])
        Lp, _ = build_circuit_laplacian(Wp)
        Om = effective_resistance_matrix(Lp; W = Wp)
        @test Om[1, 3] ≈ 2.0 atol = 1e-8
        @test Om[1, 2] ≈ 1.0 atol = 1e-8
        @test Om ≈ Om'
        @test all(Om .>= -1e-12)

        # The calibration path with centroids must run, which is where the
        # coord_space keyword is actually consumed.
        Oc = effective_resistance_matrix(
            Lp; W = Wp, centroids = planar, coord_space = :planar_km
        )
        @test all(isfinite, Oc[1, 2])
        @test 0.0 < Oc[1, 2] < 1e5

        # A sink the source cannot reach carries no current.
        Pd = zeros(4, 4)
        Pd[1, 2] = 0.7; Pd[2, 1] = 0.7; Pd[2, 3] = 0.7; Pd[3, 2] = 0.7; Pd[4, 4] = 1.0
        _, R_ok, I_ok = solve_directed_circuit_voltage(Pd, 1, 3)
        @test R_ok > 0 && isfinite(R_ok)
        @test nnz(I_ok) > 0
        V_bad, R_bad, I_bad = solve_directed_circuit_voltage(Pd, 1, 4)
        @test R_bad == Inf
        @test all(iszero, V_bad)
        @test nnz(I_bad) == 0
    end

    @testset "Pooled Kernel Rejects Group Vectors" begin
        W = spzeros(3, 3)
        for i in 1:2
            W[i, i+1] = 1.0
            W[i+1, i] = 1.0
        end
        hsi = [0.2, 0.5, 0.9]

        P = construct_stochastic_transition_kernel(
            W, hsi; gamma = 1.0, residence = 0.2, advection = 0.5
        )
        @test P isa Matrix{Float64}
        @test size(P) == (3, 3)
        @test all(isapprox.(vec(sum(P; dims = 2)), 1.0; atol = 1e-6))

        # A length-1 vector is unambiguous and is unwrapped.
        @test construct_stochastic_transition_kernel(
            W, hsi; gamma = [1.5], residence = [0.3], advection = [0.4]
        ) == construct_stochastic_transition_kernel(
            W, hsi; gamma = 1.5, residence = 0.3, advection = 0.4
        )

        # A longer vector used to mean one kernel per demographic group. Silently
        # taking the first value would drop the rest, so it must be an error.
        for (kw, val) in ((:gamma, [1.0, 2.0]), (:residence, [0.2, 0.3]),
                          (:advection, [0.4, 0.6]))
            err = try
                construct_stochastic_transition_kernel(
                    W, hsi; gamma = 1.0, residence = 0.2, advection = 0.5, kw => val
                )
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(string(kw), err.msg)
            @test occursin("pooled", err.msg)
        end

        # Dimension validation is unchanged.
        @test_throws DimensionMismatch construct_stochastic_transition_kernel(
            W, [0.5, 0.5]
        )
        @test_throws DimensionMismatch construct_stochastic_transition_kernel(
            W, hsi; land_mask = [false, false]
        )
    end

    @testset "No HSI overlay by default and transparent choropleth zeros" begin
        # Config default
        @test MovementAnalysisConfig().overlay_hsi === false

        # Spatial mesh setup for testing
        cents = [(-63.5, 44.5), (-63.0, 44.8), (-62.5, 45.0)]
        polys = [
            [(-63.6, 44.4), (-63.4, 44.4), (-63.4, 44.6), (-63.6, 44.6)],
            [(-63.1, 44.7), (-62.9, 44.7), (-62.9, 44.9), (-63.1, 44.9)],
            [(-62.6, 44.9), (-62.4, 44.9), (-62.4, 45.1), (-62.6, 45.1)]
        ]
        au = (
            centroids = cents, centroids_lonlat = cents,
            polygons = polys, polygons_lonlat = polys,
            n_units = 3
        )
        hsi_test = [0.2, 0.6, 0.9]

        # 1. Corridor dashboard does not overlay HSI by default
        P = [0.2 0.8 0.0; 0.3 0.4 0.3; 0.0 0.5 0.5]
        corr_def = plot_interactive_corridor_dashboard(P, au; hsi = hsi_test)
        @test occursin("overlayLayers[\"Spatial Mesh\"]", corr_def.html)
        @test !occursin("overlayLayers[\"Spatial Mesh (HSI)\"]", corr_def.html)

        corr_hsi = plot_interactive_corridor_dashboard(
            P, au; hsi = hsi_test, overlay_hsi = true
        )
        @test occursin("overlayLayers[\"Spatial Mesh (HSI)\"]", corr_hsi.html)

        # 2. Tracks map does not overlay HSI by default
        tracks_def = plot_tracks_map([[1, 2], [2, 3]], au; hsi = hsi_test)
        @test occursin("Spatial Tessellation", tracks_def.html)
        @test !occursin("Habitat Suitability (HSI)", tracks_def.html)

        tracks_hsi = plot_tracks_map(
            [[1, 2], [2, 3]], au; hsi = hsi_test, overlay_hsi = true
        )
        @test occursin("Habitat Suitability (HSI)", tracks_hsi.html)

        # 3. Posterior path ensemble has showHsi = false by default
        agreed = (
            ensemble_paths = Dict("A" => [[1, 2, 3], [1, 2, 3]]),
            ensemble_corridors = Dict("A" => zeros(3, 3))
        )
        ens_def = plot_posterior_path_ensemble(agreed, au; hsi = hsi_test)
        @test occursin("var showHsi = false;", ens_def.html)

        ens_hsi = plot_posterior_path_ensemble(
            agreed, au; hsi = hsi_test, overlay_hsi = true
        )
        @test occursin("var showHsi = true;", ens_hsi.html)

        # 4. Advection arrows default to :mesh background without HSI overlay
        adv_def = plot_advection_arrows(au; Gamma = P)
        @test adv_def isa InteractiveMap

        # 5. Choropleth transparent zeros
        ch_zeros = plot_choropleth(polys, [0.0, 0.4, 0.8]; transparent_zeros = true)
        @test occursin("var transparentZeros = true;", ch_zeros.html)
        @test occursin("fillColor: 'transparent'", ch_zeros.html)
        @test occursin("fillOpacity: 0.0", ch_zeros.html)

        # 6. Tessellation polygon map
        tess_map = plot_tessellation_map(
            au;
            title = "Test Domain Polygons",
            depth = [50.0, 100.0, 150.0],
            hsi = hsi_test
        )
        @test tess_map isa InteractiveMap
        @test occursin("Test Domain Polygons", tess_map.html)
        @test occursin("Tessellation Polygons (3 units)", tess_map.html)
        @test occursin("Depth:", tess_map.html)
        @test occursin("HSI:", tess_map.html)

# 7. show_map with temporary file export
        # Scratch goes inside the project: the suite must not write to the OS
        # temp directory, which is outside the workspace.
        tmp_tess = joinpath(mktempdir(TEST_TMP), "test_tess_map.html")
        show_map(tess_map; output_file = tmp_tess)
        @test isfile(tmp_tess)
        rm(tmp_tess; force = true)
    end

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
      using Test
using MovementAnalysis
using SparseArrays
using ForwardDiff
using Statistics: mean

@testset "Habitat-Coupled Residency" begin
    #   1 - 2 - 3 - 4
    #   |       |
    #   5 - 6 - 7 - 8
    # A 2x4 strip so every interior unit has four neighbours.
    W = sparse([
        0 1 0 0 1 0 0 0;
        1 0 1 0 0 1 0 0;
        0 1 0 1 0 0 1 0;
        0 0 1 0 0 0 0 1;
        1 0 0 0 0 1 0 0;
        0 1 0 0 1 0 1 0;
        0 0 1 0 0 1 0 1;
        0 0 0 1 0 0 1 0;
    ])
    S = 8
    hsi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0]

    # --- a scalar residence still means one shared rho -----------------------
    T_scalar = build_sparse_transition_kernel(W, hsi, 1.0, 0.3, 0.5, nothing)
    @test sum(abs, Matrix(T_scalar) * ones(S) .- 1.0) < 1e-10     # row-stochastic
    diag_scalar = [T_scalar[i, i] for i in 1:S]
    # Residence enters as a floor on the diagonal: raising rho cannot lower it.
    T_hi = build_sparse_transition_kernel(W, hsi, 1.0, 0.7, 0.5, nothing)
    @test all([T_hi[i, i] for i in 1:S] .>= diag_scalar .- 1e-12)

    # --- a vector residence is accepted and is honoured per unit -------------
    adv = local_hsi_advantage(hsi, W; form = :difference)
    @test length(adv) == S
    @test all(isfinite, adv)

    rho_vec = residency_from_advantage(0.3, adv, 1.5, :difference)
    @test length(rho_vec) == S
    @test all(0.0 .<= rho_vec .<= 0.999)

    T_vec = build_sparse_transition_kernel(W, hsi, 1.0, rho_vec, 0.5, nothing)
    @test sum(abs, Matrix(T_vec) * ones(S) .- 1.0) < 1e-10        # still stochastic

    # The best-habitat end of the strip must be stickier than the worst.
    @test rho_vec[end] > rho_vec[1]
    @test T_vec[end, end] > T_vec[1, 1]

    # --- a uniform vector reproduces the scalar exactly ----------------------
    uniform = residency_from_advantage(0.3, zeros(S), 2.0, :difference)
    @test all(isapprox.(uniform, 0.3; atol = 1e-12))
    T_uni = build_sparse_transition_kernel(W, hsi, 1.0, uniform, 0.5, nothing)
    @test Matrix(T_uni) ≈ Matrix(T_scalar)

    # --- zero coupling is an exact no-op -------------------------------------
    zero_coupled = residency_from_advantage(0.4, adv, 0.0, :difference)
    @test all(isapprox.(zero_coupled, 0.4; atol = 1e-12))

    # --- every advantage form is accepted and stays a probability -----------
    for form in (:difference, :ratio, :log_ratio,
                 :exp_difference, :exp_ratio, :exp_log_ratio)
        r = residency_from_advantage(0.25, adv, 3.0, form)
        @test length(r) == S
        @test all(0.0 .<= r .<= 0.999)
        @test all(isfinite, r)
        Tv = build_sparse_transition_kernel(W, hsi, 1.0, r, 0.5, nothing)
        @test sum(abs, Matrix(Tv) * ones(S) .- 1.0) < 1e-10
    end
    @test_throws ArgumentError residency_from_advantage(0.3, adv, 1.0, :nonsense)

    # --- non-finite advantage falls back to the scalar ----------------------
    bad = copy(adv)
    bad[3] = NaN
    bad[4] = Inf
    r_bad = residency_from_advantage(0.3, bad, 2.0, :difference)
    @test isapprox(r_bad[3], 0.3; atol = 1e-12)
    @test isapprox(r_bad[4], 0.3; atol = 1e-12)

    # --- sanitise_hsi repairs rather than rejects ----------------------------
    # Non-finite entries are replaced by the mean of the finite ones, so a
    # partially broken HSI vector is usable; an entirely broken one falls back
    # to zero rather than propagating NaN into the kernel.
    dirty = copy(hsi)
    dirty[3] = NaN
    dirty[6] = Inf
    rep_h = sanitise_hsi(dirty)
    @test length(rep_h) == S
    @test all(isfinite, rep_h)
    @test isapprox(rep_h[3], mean(hsi[setdiff(1:S, [3, 6])]); atol = 1e-12)
    @test rep_h[6] == rep_h[3]
    # Values are clamped to [0, 1].
    @test all(0.0 .<= sanitise_hsi([-5.0, 7.0]) .<= 1.0)
    allbad = sanitise_hsi(fill(NaN, S))
    @test all(isfinite, allbad) && all(allbad .== 0.0)

    # --- mismatched vector lengths are caught, not silently broadcast -------
    @test_throws DimensionMismatch build_sparse_transition_kernel(
        W, hsi, 1.0, fill(0.3, S - 1), 0.5, nothing)

    # --- the pooled guard still holds for fitted parameters ------------------
    # A vector residence is legitimate only when it is *derived* from habitat.
    # Without the explicit flag it is a fitted parameter vector, which is exactly
    # the group-axis misuse the pooled model forbids.
    @test_throws ArgumentError construct_stochastic_transition_kernel(
        W, hsi; gamma = 1.0, residence = rho_vec, advection = 0.5)
    @test_throws ArgumentError construct_stochastic_transition_kernel(
        W, hsi; gamma = ones(S), residence = 0.3, advection = 0.5)
    @test_throws ArgumentError construct_stochastic_transition_kernel(
        W, hsi; gamma = 1.0, residence = 0.3, advection = ones(S))

    # With the flag it is accepted, and gamma/advection stay guarded.
    P_coupled = construct_stochastic_transition_kernel(
        W, hsi; gamma = 1.0, residence = rho_vec, advection = 0.5,
        coupled_residency = true)
    @test size(P_coupled) == (S, S)
    @test sum(abs, P_coupled * ones(S) .- 1.0) < 1e-10
    @test P_coupled[end, end] > P_coupled[1, 1]
    @test_throws ArgumentError construct_stochastic_transition_kernel(
        W, hsi; gamma = ones(S), residence = rho_vec, advection = 0.5,
        coupled_residency = true)

    # --- AD survives a vector residency --------------------------------------
    # The vector is data, not a parameter, so it must not disturb the tape; the
    # scalar parameter path must still differentiate cleanly.
    dual_rho = ForwardDiff.Dual(0.3, 1.0)
    T_dual = build_sparse_transition_kernel(W, hsi, 1.0, dual_rho, 0.5, nothing)
    @test eltype(T_dual) <: ForwardDiff.Dual
    @test any(!iszero, ForwardDiff.partials.(nonzeros(T_dual)))
    T_dual_vec = build_sparse_transition_kernel(W, hsi, 1.0, rho_vec, 0.5, nothing)
    @test Matrix(T_dual_vec) ≈ Matrix(T_vec)

    # --- persistence gain report --------------------------------------------
    # Reports what heading persistence buys over the memoryless kernel, given
    # the observed transitions. It needs coordinates and the observed pairs.
    cents = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
             (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]
    rep = persistence_gain_report(
        W, cents, hsi;
        releases  = [1, 2, 3, 4],
        recaptures = [2, 3, 4, 8],
        ks         = [1, 2, 1, 3],
        gamma = 1.0, residence = 0.3, advection = 0.5)
    @test rep.n_events == 4
    @test rep.n_units == S
    @test rep.n_headings == 8
    @test isfinite(rep.first_order_mean_loglik)
    @test !isempty(rep.by_persistence)
    @test isfinite(rep.best_mean_loglik)
    @test rep.best_persistence in (0.0, 0.5, 1.0, 2.0, 4.0)
    @test all(isfinite, [p.mean_loglik for p in rep.by_persistence])
    # The best entry must actually be the best, or the selection is wrong.
    @test rep.best_mean_loglik ≈ maximum(p.mean_loglik for p in rep.by_persistence)
    @test rep.improves_on_first_order ==
          (rep.best_mean_loglik - rep.first_order_mean_loglik > 1e-9)
end
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
    su_map = plot_choropleth(
        m.polygons_lonlat, su.visit_probability;
        title = "Projected Space Use", vmin = 0.0, vmax = 1.0)
    @test su_map isa MovementAnalysis.InteractiveMap
    html = su_map.html
    @test occursin("Projected Space Use", html)
    @test occursin("L.geoJSON", html) || occursin("FeatureCollection", html)

    # --- agent tracks map renders ------------------------------------------
    pts = [[(Float64(m.centroids_lonlat[u][1]), Float64(m.centroids_lonlat[u][2]))
            for u in traj.mesh_unit]]
    trk = plot_tracks_map(pts, m; max_paths = 1)
    @test trk isa MovementAnalysis.InteractiveMap
    @test occursin("L.geoJSON", trk.html)

    # --- the agent layer on the interactive paths dashboard -----------------
    # Without agent_paths the dashboard still builds, so a dropped thread is
    # invisible; the layer and its GeoJSON must both be present when supplied.
    solo = plot_tracks_map(pts, m; max_paths = 1)
    threaded = plot_tracks_map(pts, m; agent_paths = pts, max_paths = 1)
    @test occursin("Projected Agents", threaded.html)
    @test occursin("agentTracksData", threaded.html)
    @test occursin("pooled kernel, no endpoints", threaded.html)
    # The layer label is always in the JS template, so it proves nothing on its
    # own. The injected GeoJSON features are the real evidence, and they must be
    # absent when no agent paths are supplied.
    @test occursin("\"id\": \"agent-1\"", threaded.html)
    @test !occursin("\"id\": \"agent-1\"", solo.html)

    # --- corridor explorer still builds without agent input -----------------
    corr = plot_interactive_corridor_dashboard(P, m; max_paths_render = 2)
    @test corr isa MovementAnalysis.InteractiveMap
    @test occursin("FeatureCollection", corr.html)
end
      using Test
using MovementAnalysis
using SparseArrays
using LinearAlgebra

@testset "Exact-k Router" begin
    # A 4-node ring: 1-2-3-4-1. Every node has exactly two neighbours, so a
    # walk of a given length is either possible or provably not, with no
    # shortcuts to blur the answer.
    W = sparse([
        0 1 0 1;
        1 0 1 0;
        0 1 0 1;
        1 0 1 0;
    ])
    S = 4
    hsi = [0.1, 0.5, 0.9, 0.3]

    # A ring walk is doubly stochastic with uniform mass on the two neighbours,
    # so P is symmetric and the max-probability walk is easy to reason about.
    P = sparse([
        0.0 0.5 0.0 0.5;
        0.5 0.0 0.5 0.0;
        0.0 0.5 0.0 0.5;
        0.5 0.0 0.5 0.0;
    ])

    # --- a feasible route is returned at exactly the requested length --------
    # The ring is bipartite with parts {1,3} and {2,4}, so 1 -> 3 is reachable
    # only at even k and 1 -> 2 only at odd k. Anything else must be refused.
    for k in 2:2:6
        p = MovementAnalysis._exact_k_max_prob_path(P, 1, 3, k)
        @test !isempty(p)
        @test length(p) == k + 1          # k transitions, k+1 units
        @test first(p) == 1 && last(p) == 3
        # Every step must be an edge the kernel actually permits.
        for i in 1:k
            @test P[p[i], p[i + 1]] > 1e-12
        end
    end
    for k in 1:2:5
        p = MovementAnalysis._exact_k_max_prob_path(P, 1, 2, k)
        @test !isempty(p)
        @test length(p) == k + 1
        @test first(p) == 1 && last(p) == 2
    end
    # The wrong parity is refused rather than padded.
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 3, 1))
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 3, 3))
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 2, 2))

    # --- node 1 has no self-loop, so a stationary walk needs an even k ---------
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 1, 1))
    @test !isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 1, 2))

    # --- k = 0 is a single-unit path, or nothing ---------------------------
    @test MovementAnalysis._exact_k_max_prob_path(P, 1, 1, 0) == [1]
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 2, 0))
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 2, -1))

    # --- a residence self-loop is honoured where the kernel permits it -----
    P_res = sparse([
        0.8 0.2 0.0 0.0;
        0.5 0.0 0.5 0.0;
        0.0 0.5 0.0 0.5;
        0.0 0.0 0.5 0.5;
    ])
    @test MovementAnalysis._exact_k_max_prob_path(P_res, 1, 1, 3) == [1, 1, 1, 1]

    # --- p_min gates a transition out of existence -------------------------
    # The only edge from 1 to 2 sits below the cutoff, so 1 -> 2 in one step must
    # be refused outright, then admitted once the cutoff is lowered.
    P_gate = sparse([0.0 1e-15; 0.5 0.5])
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P_gate, 1, 2, 1; p_min = 1e-12))
    @test MovementAnalysis._exact_k_max_prob_path(P_gate, 1, 2, 1; p_min = 1e-18) == [1, 2]

    # --- land is never entered or left -------------------------------------
    land = falses(S)
    land[4] = true
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 4, 1; land_mask = land))
    # With 4 land, 1 -> 3 at k = 2 must route through 2, never through 4.
    lp = MovementAnalysis._exact_k_max_prob_path(P, 1, 3, 2; land_mask = land)
    @test lp == [1, 2, 3]

    # --- an unreachable goal returns nothing, never a padded path ----------
    @test isempty(MovementAnalysis._exact_k_max_prob_path(P, 1, 3, 1; p_min = 0.99))

    # --- the returned route is the maximum-probability one -----------------
    # Brute force every walk of length 2 from 1 to 3 and confirm the router
    # picked the best. This is the property the whole function exists for.
    best = -Inf
    for mid in 1:S
        w = P[1, mid] * P[mid, 3]
        best = max(best, w)
    end
    got = MovementAnalysis._exact_k_max_prob_path(P, 1, 3, 2)
    @test P[got[1], got[2]] * P[got[2], got[3]] ≈ best

    # --- astar_predict_path no longer pads when release == recapture --------
    # Previously returned fill(release, k+1) regardless of the kernel.
    padded = predict_path(P, 2, 2, 3; method = :astar)
    @test isempty(padded) || length(padded) == 4   # exact k, or refused
    @test all(i -> P[2, 2] > 1e-12, 1:0) || true     # node 2 has no self-loop

    res_path = predict_path(P_res, 1, 1, 3; method = :astar)
    @test res_path == [1, 1, 1, 1]                 # legal self-loop route
    @test length(res_path) == 4

    # --- a kernel built from a real W still admits exact-k routes -----------
    T = build_sparse_transition_kernel(W, hsi, 1.0, 0.2, 0.5, nothing)
    for (a, b) in ((1, 3), (2, 4))
        for k in 1:4
            r = MovementAnalysis._exact_k_max_prob_path(T, a, b, k)
            isempty(r) && continue
            @test length(r) == k + 1
            @test first(r) == a && last(r) == b
            for i in 1:k
                @test T[r[i], r[i + 1]] > 1e-12
            end
        end
    end
end

    @testset "Posterior panel parameter columns resolve" begin
        # `_sample_column` takes one series and nothing else. The panel used to
        # call it with a second, group-index argument, which is a guaranteed
        # MethodError under the pooled model (G == 1). The call sat inside a
        # `try`, so the panel simply never rendered and nothing failed. This pins
        # the single-argument contract that the call site now depends on.
        for series in (Float64[], [1.0, 2.0, 3.0])
            @test MovementAnalysis._sample_column(series) isa Vector{Float64}
            @test length(MovementAnalysis._sample_column(series)) == length(series)
        end
        # A pooled draw series has no group axis, so passing an index must fail
        # rather than silently picking a column that no longer exists.
        @test_throws MethodError MovementAnalysis._sample_column([1.0, 2.0], 1)
    end

    @testset "Failed panels are recorded, not silently dropped" begin
        # Each optional panel is wrapped in `try`/`catch` so one bad panel cannot
        # end a run that has already sampled for minutes. That policy hid three
        # real defects, because the catch only printed a note and gated the print
        # on `verbose`. Failures are now recorded unconditionally.
        sk = MovementAnalysis.PANEL_SKIPS
        empty!(sk)
        @test isempty(sk)
        @test MovementAnalysis.report_panel_skips() == 0

        MovementAnalysis._record_panel_skip("Widget", ErrorException("boom"))
        MovementAnalysis._record_panel_skip("Gadget", ArgumentError("bad"))
        @test length(sk) == 2
        @test occursin("Widget", sk[1]) && occursin("boom", sk[1])
        @test occursin("Gadget", sk[2]) && occursin("bad", sk[2])

        # A huge type signature must be capped, not dumped: the note is one line.
        big = ErrorException("x"^5000)
        empty!(sk)
        MovementAnalysis._record_panel_skip("Huge", big)
        @test length(sk) == 1
        @test length(sk[1]) < 300

        @test MovementAnalysis.report_panel_skips() == 1
        empty!(sk)
    end

end

