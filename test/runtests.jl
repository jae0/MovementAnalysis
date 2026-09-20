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

@testset "MovementAnalysis Core Engine" begin

    @testset "Parameters and Configurations" begin
        p_def = movement_parameters_default()
        @test p_def isa NamedTuple
        @test haskey(p_def, :path_method)
        @test p_def.path_method == :astar
        @test haskey(p_def, :data_source)

        p_sc = movement_parameters_snowcrab()
        @test p_sc isa NamedTuple
        @test p_sc.data_source == :snowcrab
        @test length(p_sc.group_labels) == 3
        @test p_sc.species_name == "Snow Crab"
    end

    @testset "Synthetic ADR Movement Bundle" begin
        rng = MersenneTwister(123)
        bundle = generate_ADR_simulation_bundle(100.0, 15, 3, 5;
                                                area_method = :hexagonal, rng = rng)
        @test bundle isa NamedTuple
        @test haskey(bundle, :data)
        @test haskey(bundle, :telemetry_data)
        @test haskey(bundle, :au)
        @test bundle.n_spatial >= 10
        @test bundle.n_years == 3
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
        # Check row-stochastic property: rows sum to 1.0
        for i in 1:5
            @test isapprox(sum(P[i, :]), 1.0; atol = 1e-6)
        end

        # Verify sparse kernel row-stochasticity across parameter regimes
        for gamma in [0.0, 1.0, 10.0, -10.0], rho in [0.1, 0.5, 0.9], alpha in [0.0, 0.5, 1.0]
            P_sp = build_sparse_transition_kernel(W, hsi, gamma, rho, alpha, nothing)
            @test all(P_sp .>= 0.0)
            for i in 1:5
                @test isapprox(sum(P_sp[i, :]), 1.0; atol = 1e-6)
            end
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
        groups     = [1, 1, 1]

        m_tel = pure_telemetry_turing_model(
            releases, recaptures, ks, groups, W, hsi, nothing, 1
        )
        @test m_tel isa DynamicPPL.Model

        rng = MersenneTwister(42)
        chn = sample(rng, m_tel, MH(), 100; progress = false)
        @test size(chn, 1) == 100
        @test chn[:velocity] !== nothing
        @test chn[:diffusion] !== nothing
        @test chn[:gamma] !== nothing
        @test all(v -> all(!isnan, v), chn[:velocity])
        @test all(v -> all(!isnan, v), chn[:diffusion])
        @test all(v -> all(!isnan, v), chn[:gamma])
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

    @testset "Spectral Graph Wavelets (SGWT)" begin
        # 4-node ring graph
        W = spzeros(4, 4)
        W[1, 2] = 1.0; W[2, 1] = 1.0
        W[2, 3] = 1.0; W[3, 2] = 1.0
        W[3, 4] = 1.0; W[4, 3] = 1.0
        W[4, 1] = 1.0; W[1, 4] = 1.0

        L_norm = build_normalized_laplacian(W)
        @test size(L_norm) == (4, 4)
        lam_max = compute_laplacian_spectral_bounds(L_norm)
        @test lam_max >= 0.0
        @test lam_max <= 2.2

        f = [1.0, 2.0, 1.5, 0.5]
        res = spectral_graph_wavelet_transform(L_norm, f; num_scales = 3)
        @test res isa SpectralGraphWaveletResult
        @test length(res.scales) == 3
        @test size(res.details, 2) == 3
    end

    @testset "Leaflet HTML Map Structure" begin
        map_obj = LeafletMap(
            "<div>Map Content</div>";
            title = "Test Title",
            width = "100%",
            height = "600px"
        )
        @test map_obj isa LeafletMap
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

        map_geo = leaflet_interactive_corridor_dashboard(
            P, au_geo;
            title = "Test Corridor Map",
            group_labels = ["All"]
        )
        @test map_geo isa LeafletMap
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
        map_planar = leaflet_interactive_corridor_dashboard(
            [P, P], au_planar;
            title = "Planar Test Map",
            group_labels = ["Group 1", "Group 2"]
        )
        @test map_planar isa LeafletMap
        @test occursin("esriOcean", map_planar.html)
        @test occursin("Group 1", map_planar.html)
        @test occursin("Group 2", map_planar.html)
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

    include("test_ssa_movement.jl")
    include("test_agent_movement.jl")

end
