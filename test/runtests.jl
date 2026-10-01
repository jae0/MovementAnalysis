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

        # An absent flag must not displace a value coming from a config file, so
        # the defaults ArgParse fills in are dropped rather than applied.
        mktempdir() do dir
            p = joinpath(dir, "cfg.toml")
            write(p, """
                species_name = "Snow Crab"
                n_samples = 500
                model_modes = ["telemetry", "ssa", "agent"]
                depth_range = [25.0, 400.0]
                region_labels = []
                """)
            from_file = load_config(config_path = p, cli_args = ["--max-paths=2"])
            @test from_file.species_name == "Snow Crab"
            @test from_file.n_samples == 500
            @test from_file.model_modes == [:telemetry, :ssa, :agent]
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
        mktempdir() do dir
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

        # With sppoly_bounds (delimiting south and west)
        sp_b = (-65.48, 43.04, -57.32, 47.27)
        bbox_delim = resolve_bbox(nothing, 0.5, lons, lats; sppoly_bounds = sp_b)
        @test bbox_delim[1] == -65.48
        @test bbox_delim[2] == 43.04
        @test bbox_delim[3] == -57.5
        @test bbox_delim[4] == 47.5

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

    @testset "Continuous-Time SSA Uniformization and Model" begin
        # The generator itself, the `expm`/`taylor` transition matrices and the
        # Gillespie simulator are covered in `test_ssa_movement.jl`. What was
        # missing is the uniformization path -- `calculate_ssa_transition_row`, the
        # one the Turing model actually calls -- and the fact that the model had
        # never been fitted at all. Only those are checked here.
        S = 8
        W = spzeros(S, S)
        for i in 1:S-1
            W[i, i+1] = 1.0; W[i+1, i] = 1.0
        end
        hsi = collect(range(0.2, 0.8; length = S))
        land = falses(S)

        Q = construct_ssa_generator(W, hsi; velocity = 0.3, diffusion = 0.1,
                                    gamma = 0.9, land_mask = land)

        # Uniformization needs a positive rate; a generator with an all-zero
        # diagonal would send the row builder down its degenerate early exit.
        @test maximum(abs.(real(diag(Q)))) > 0

        # A row must be a probability distribution at every dt, including ones long
        # enough for the chain to have spread far.
        rows = [MovementAnalysis.calculate_ssa_transition_row(Q, dt, 1)
                for dt in (0.1, 1.0, 5.0, 50.0, 500.0)]
        @test all(r -> sum(r) ≈ 1.0, rows)
        @test all(r -> all(>=(0.0), r), rows)

        # Longer elapsed time must not concentrate probability back on the start.
        @test rows[2][1] > rows[4][1]

        # The model builds, samples, and yields the same three population
        # parameters as the discrete model.
        releases = [1, 2, 3, 4, 5]
        recaptures = [2, 3, 4, 5, 6]
        dts = [1.0, 1.0, 2.0, 3.0, 4.0]
        m_ssa = ssa_telemetry_turing_model(releases, recaptures, dts, W, hsi, land)
        @test m_ssa isa DynamicPPL.Model

        rng = MersenneTwister(42)
        chn_ssa = sample(rng, m_ssa, MH(), 40; num_warmup = 100, progress = false)
        @test size(chn_ssa, 1) == 40
        @test population_level_parameter_names(chn_ssa)

        vv, dd, gg = MovementAnalysis.posterior_kernel_draws(chn_ssa)
        @test all(x -> length(x) == 40, (vv, dd, gg))
        @test all(x -> all(isfinite, x), (vv, dd, gg))
        @test all(>=(0.0), vv) && all(>=(0.0), dd)
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

    @testset "Continuous-time path is differentiable" begin
        # The SSA path hard-cast to `Float64` in two places -- the generator's
        # parameters and the model's cache -- so any sampler running under
        # ForwardDiff failed with `Float64(::ForwardDiff.Dual)`. `movement_sampler`
        # is exactly such a sampler (a linked-space random walk needs the
        # Jacobian), so this took out the whole `:ssa` mode, which the default
        # snow crab config requests. Both must now preserve the element type.
        S = 6
        W = spzeros(S, S)
        for i in 1:S-1
            W[i, i+1] = 1.0; W[i+1, i] = 1.0
        end
        hsi = collect(range(0.2, 0.8; length = S))
        land = falses(S)

        # With concrete parameters the generator is unchanged.
        Qf = construct_ssa_generator(W, hsi; velocity = 0.3, diffusion = 0.1,
                                     gamma = 0.9, land_mask = land)
        @test eltype(Qf) === Float64
        @test all(abs.(vec(sum(Qf; dims = 2))) .< 1e-10)   # conservative

        # With duals it stays on the tape. The same parameters must give the same
        # numbers, so a Dual generator is checked against a Float64 one.
        Qd = construct_ssa_generator(W, hsi;
                                     velocity  = ForwardDiff.Dual(0.3, 1.0),
                                     diffusion = ForwardDiff.Dual(0.1, 1.0),
                                     gamma     = ForwardDiff.Dual(0.9, 1.0),
                                     land_mask = land)
        @test eltype(Qd) <: ForwardDiff.Dual
        @test ForwardDiff.value.(Qd.nzval) ≈ Qf.nzval
        @test any(!iszero, ForwardDiff.partials.(Qd.nzval))

        # A dual transition row is still a probability distribution.
        rowd = MovementAnalysis.calculate_ssa_transition_row(Qd, 2.0, 1)
        @test eltype(rowd) <: ForwardDiff.Dual
        @test sum(ForwardDiff.value.(rowd)) ≈ 1.0 atol = 1e-9

        # The regression itself: the SSA model must sample under the AD sampler the
        # pipeline uses. This is what failed before.
        m = ssa_telemetry_turing_model([1, 2, 3, 4, 5], [2, 3, 4, 5, 6],
                                      [1.0, 1.0, 2.0, 3.0, 4.0], W, hsi, land)
        spl = MovementAnalysis.movement_sampler(
            [0.2, 0.2, 1.0], MovementAnalysisConfig(; n_samples = 20, n_warmup = 20))
        chn = sample(MersenneTwister(42), m, spl, 20;
                     num_warmup = 20, progress = false)
        @test size(chn, 1) == 20
        @test population_level_parameter_names(chn)
        vv, dd, gg = MovementAnalysis.posterior_kernel_draws(chn)
        @test all(x -> length(x) == 20, (vv, dd, gg))
        @test all(x -> all(isfinite, x), (vv, dd, gg))
        @test length(unique(round.(vv; digits = 9))) > 1     # not frozen
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
        m = leaflet_posterior_path_ensemble(agreed, au; hsi = hsi)
        @test m isa LeafletMap
        @test startswith(strip(m.html), "<!DOCTYPE html>")
        @test occursin("\"n_draws\": 3", m.html)
        @test occursin("\"modal_share\": 1.0", m.html)
        @test m.metadata[:n_individuals] == 1

        # Two of three draws chose the same route, so the modal share is 2/3.
        split_ens = (ensemble_paths =
                     Dict("B" => [[1, 2, 3, 4, 5], [1, 2, 4, 5], [1, 2, 3, 4, 5]]),)
        m2 = leaflet_posterior_path_ensemble(split_ens, au; hsi = hsi)
        @test occursin("\"modal_share\": 0.6667", m2.html)

        # Longer routes must measure longer.
        longer = (ensemble_paths = Dict("C" => [[1, 2], [1, 2, 3], [1, 2, 3, 4]]),)
        m3 = leaflet_posterior_path_ensemble(longer, au; hsi = hsi)
        lo = parse(Float64, match(r"\"min_km\": ([0-9.]+)", m3.html).captures[1])
        hi = parse(Float64, match(r"\"max_km\": ([0-9.]+)", m3.html).captures[1])
        @test hi > lo

        # An out-of-range unit index is dropped, not thrown on.
        ragged = (ensemble_paths = Dict("D" => [[1, 2, 999], [0, 1, 2]]),)
        @test (leaflet_posterior_path_ensemble(ragged, au)).metadata[:n_individuals] == 1

        # No HSI supplied still renders, with a null payload.
        @test occursin("var HSI = null;", leaflet_posterior_path_ensemble(agreed, au).html)

        # Several individuals are all offered, and the cap is honoured.
        many = (ensemble_paths = Dict("T$i" => [[1, 2]] for i in 1:40),)
        @test leaflet_posterior_path_ensemble(many, au; max_individuals = 5
        ).metadata[:n_individuals] == 5

        # An ensemble with nothing in it is a caller error worth naming.
        @test_throws ArgumentError leaflet_posterior_path_ensemble(
            (ensemble_paths = Dict{String,Any}(),), au)
        @test_throws ArgumentError leaflet_posterior_path_ensemble(
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

        light  = leaflet_hsi_map(hsi, au; cmap = :viridis, dark_mode = false)
        dark   = leaflet_hsi_map(hsi, au; cmap = :viridis, dark_mode = true)
        plasma = leaflet_hsi_map(hsi, au; cmap = :plasma,  dark_mode = false)

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
        corr_def = leaflet_interactive_corridor_dashboard(P, au; hsi = hsi_test)
        @test occursin("overlayLayers[\"Spatial Mesh\"]", corr_def.html)
        @test !occursin("overlayLayers[\"Spatial Mesh (HSI)\"]", corr_def.html)

        corr_hsi = leaflet_interactive_corridor_dashboard(
            P, au; hsi = hsi_test, overlay_hsi = true
        )
        @test occursin("overlayLayers[\"Spatial Mesh (HSI)\"]", corr_hsi.html)

        # 2. Tracks map does not overlay HSI by default
        tracks_def = leaflet_tracks_map([[1, 2], [2, 3]], au; hsi = hsi_test)
        @test occursin("Spatial Tessellation", tracks_def.html)
        @test !occursin("Habitat Suitability (HSI)", tracks_def.html)

        tracks_hsi = leaflet_tracks_map(
            [[1, 2], [2, 3]], au; hsi = hsi_test, overlay_hsi = true
        )
        @test occursin("Habitat Suitability (HSI)", tracks_hsi.html)

        # 3. Posterior path ensemble has showHsi = false by default
        agreed = (
            ensemble_paths = Dict("A" => [[1, 2, 3], [1, 2, 3]]),
            ensemble_corridors = Dict("A" => zeros(3, 3))
        )
        ens_def = leaflet_posterior_path_ensemble(agreed, au; hsi = hsi_test)
        @test occursin("var showHsi = false;", ens_def.html)

        ens_hsi = leaflet_posterior_path_ensemble(
            agreed, au; hsi = hsi_test, overlay_hsi = true
        )
        @test occursin("var showHsi = true;", ens_hsi.html)

        # 4. Advection arrows default to :mesh background without HSI overlay
        adv_def = leaflet_advection_arrows(au; Gamma = P)
        @test adv_def isa LeafletMap

        # 5. Choropleth transparent zeros
        ch_zeros = leaflet_choropleth(polys, [0.0, 0.4, 0.8]; transparent_zeros = true)
        @test occursin("var transparentZeros = true;", ch_zeros.html)
        @test occursin("fillColor: 'transparent'", ch_zeros.html)
        @test occursin("fillOpacity: 0.0", ch_zeros.html)

        # 6. Tessellation polygon map
        tess_map = leaflet_tessellation_map(
            au;
            title = "Test Domain Polygons",
            depth = [50.0, 100.0, 150.0],
            hsi = hsi_test
        )
        @test tess_map isa LeafletMap
        @test occursin("Test Domain Polygons", tess_map.html)
        @test occursin("Tessellation Polygons (3 units)", tess_map.html)
        @test occursin("Depth:", tess_map.html)
        @test occursin("HSI:", tess_map.html)

        # 7. show_map with temporary file export
        tmp_tess = joinpath(tempdir(), "test_tess_map.html")
        show_map(tess_map; output_file = tmp_tess)
        @test isfile(tmp_tess)
        rm(tmp_tess; force = true)
    end

    include("test_ssa_movement.jl")
    include("test_agent_movement.jl")

end
