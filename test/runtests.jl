using Test
using MovementAnalysis
using LinearAlgebra
using SparseArrays
using Random
using DataFrames
using Statistics
using Turing
using DynamicPPL

# ==============================================================================
# Pruned regression suite.
#
# Kept only where a test guards a defect that actually occurred, or fixes a known
# answer. Removed: duplicate coverage of the same property in adjacent testsets,
# trivial smoke assertions (`haskey`, `isa` on things the test itself just built),
# and the snow-crab domain-mask tests, which assert against checked-in empirical
# files and belong with the end-to-end run (todo 7.5) rather than here.
#
# Agent projection and directional persistence live in their own files.
# ==============================================================================

@testset "MovementAnalysis" begin

    @testset "Configuration" begin
        p_def = movement_parameters_default()
        p_sc = movement_parameters_snowcrab()
        @test p_def.path_method == :astar
        @test p_def.time_interval == :daily
        @test p_def.depth_range === nothing
        @test p_sc.data_source == :snowcrab
        @test p_sc.depth_range == (25.0, 400.0)
        @test p_sc.species_name == "Snow Crab"
        # Both presets must expose the same keys, or config loading rejects one.
        @test Set(propertynames(p_def)) == Set(propertynames(p_sc))
        @test p_def.persistence == 0.0
        @test p_def.n_agent_projections == 200

        cli = MovementAnalysis.parse_movement_cli_args([
            "--config", "configs/snowcrab.toml", "--samples", "250"
        ])
        @test cli.config_path == "configs/snowcrab.toml"
        @test cli.n_samples == 250
    end

    @testset "Pooled Kernel And Turing Model" begin
        W = spzeros(5, 5)
        for i in 1:4
            W[i, i+1] = 1.0
            W[i+1, i] = 1.0
        end
        hsi = [0.2, 0.4, 0.6, 0.8, 1.0]

        P = construct_stochastic_transition_kernel(
            W, hsi; gamma = 1.0, residence = 0.2, advection = 0.5
        )
        @test size(P) == (5, 5)
        @test all(isapprox.(vec(sum(P; dims = 2)), 1.0; atol = 1e-6))

        # Row-stochasticity must hold across the whole parameter regime,
        # including degenerate gamma and saturated residence.
        for g in (0.0, 1.0, 10.0, -10.0), rho in (0.1, 0.5, 0.9), a in (0.0, 0.5, 1.0)
            K = build_sparse_transition_kernel(W, hsi, g, rho, a, nothing)
            @test all(K .>= 0.0)
            @test all(isapprox.(vec(sum(K; dims = 2)), 1.0; atol = 1e-6))
        end

        # The pooled model has exactly three parameters and no group index.
        m = pure_telemetry_turing_model([1, 2, 2], [2, 3, 4], [1, 1, 2], W, hsi, nothing)
        @test m isa DynamicPPL.Model
        chn = sample(MersenneTwister(42), m, MH(), 100; progress = false)
        @test size(chn, 1) == 100
        for p in (:velocity, :diffusion, :gamma)
            @test all(v -> all(isfinite, v), chn[p])
        end
    end

    @testset "Directional Persistence" begin
        n = 5
        W = spzeros(n, n)
        for i in 1:(n - 1)
            W[i, i+1] = 1.0
            W[i+1, i] = 1.0
        end
        cents = [(-64.0 + 0.5 * i, 44.0) for i in 1:n]
        hsi = fill(0.5, n)
        H = 8
        P0 = sparse(build_sparse_transition_kernel(W, hsi, 1.0, 0.2, 0.5, nothing))

        @testset "Bearings" begin
            @test isapprox(mod(bearing_deg(cents[1], cents[2]; coord_space = :geographic), 360), 90.0; atol = 1e-6)
            @test isapprox(mod(bearing_deg(cents[2], cents[1]; coord_space = :geographic), 360), 270.0; atol = 1e-6)
            @test isnan(bearing_deg(cents[1], cents[1]))
            @test MovementAnalysis.heading_bin(90.0, 8) == 3
            @test MovementAnalysis.heading_bin(NaN, 8) == 1
        end

        @testset "Augmented Kernel" begin
            for kappa in (0.0, 2.0, 6.0)
                T = build_persistent_transition_kernel(
                    W, cents, hsi; gamma = 1.0, residence = 0.2, advection = 0.5,
                    persistence = kappa, n_headings = H, coord_space = :geographic
                )
                @test size(T, 1) == n * H
                @test all(isapprox.(vec(sum(T; dims = 2)), 1.0; atol = 1e-10))
            end

            # kappa = 0 must reduce exactly to the first-order unit marginal, so
            # the extra parameter is a strict generalisation.
            T0 = build_persistent_transition_kernel(
                W, cents, hsi; gamma = 1.0, residence = 0.2, advection = 0.5,
                persistence = 0.0, n_headings = H, coord_space = :geographic
            )
            M0 = Matrix(persistent_unit_marginal(T0, n))
            @test M0 ≈ Matrix(P0) atol = 1e-12
            @test all(isapprox.(vec(sum(M0; dims = 2)), 1.0; atol = 1e-10))

            @test_throws DimensionMismatch build_persistent_transition_kernel(
                W, [cents[1]], hsi
            )
            @test_throws ArgumentError build_persistent_transition_kernel(
                W, cents, hsi; n_headings = 1
            )
        end

        @testset "Persistence Lengthens Directed Transits" begin
            # Released at the west end with no residence, so a memoryless chain
            # wanders while a persistent one commits to a direction.
            K = 12
            far = function (kappa)
                T = build_persistent_transition_kernel(
                    W, cents, hsi; gamma = 1.0, residence = 0.0, advection = 0.5,
                    persistence = kappa, n_headings = H, coord_space = :geographic
                )
                Tk = T^K
                last_unit = n * H
                acc = 0.0
                for h in 1:H
                    acc += sum(Tk[h, (last_unit - H + 1):last_unit])
                end
                acc / H
            end
            @test far(4.0) > far(0.0) + 0.3
            @test far(4.0) > 0.9
        end

        @testset "Agents Turn With A Heading" begin
            function reversal_rate(kappa)
                df = simulate_agent_trajectories(
                    300, fill(1, 300), 14, sparse(P0);
                    seed = 3, centroids = cents, persistence = kappa,
                    coord_space = :geographic
                )
                rev = 0
                tot = 0
                for g in groupby(df, :tagid)
                    u = Int.(g.mesh_unit)
                    for i in 3:length(u)
                        d1 = u[i-1] - u[i-2]
                        d2 = u[i] - u[i-1]
                        (d1 == 0 || d2 == 0) && continue
                        tot += 1
                        sign(d1) != sign(d2) && (rev += 1)
                    end
                end
                tot == 0 ? NaN : rev / tot
            end
            # A memoryless 1-D walk reverses constantly; persistence suppresses it.
            @test reversal_rate(0.0) > 0.4
            @test reversal_rate(6.0) < 0.05
            @test reversal_rate(6.0) < reversal_rate(0.0)

            @test_throws ArgumentError simulate_agent_trajectories(
                2, [1, 2], 3, sparse(P0); persistence = 1.0
            )
        end

        @testset "Symmetric Grid Stays Isotropic" begin
            # A 3x3 grid with uniform habitat is symmetric under reflection in
            # both axes, so the augmented chain must be too. A stay that reset the
            # heading to a fixed bin would break that symmetry and inject a
            # systematic drift, which a one-dimensional chain cannot reveal.
            G = spzeros(9, 9)
            for r in 1:3, c in 1:3
                i = (r - 1) * 3 + c
                for (dr, dc) in ((0, 1), (0, -1), (1, 0), (-1, 0))
                    rr, cc = r + dr, c + dc
                    1 <= rr <= 3 && 1 <= cc <= 3 && (G[i, (rr - 1) * 3 + cc] = 1.0)
                end
            end
            gcent = [
                (-64.0 + 0.1 * ((i - 1) % 3), 44.0 + 0.1 * ((i - 1) ÷ 3))
                for i in 1:9
            ]
            blk(u) = ((u - 1) * H + 1):((u - 1) * H + H)
            for kappa in (0.0, 4.0)
                Tg = build_persistent_transition_kernel(
                    G, gcent, fill(0.5, 9);
                    gamma = 0.0, residence = 0.1, advection = 0.0,
                    persistence = kappa, n_headings = H, coord_space = :geographic
                )
                T80 = Tg^80
                m(x) = sum(T80[blk(5), x]) / H
                north = m(vcat(blk(1), blk(2), blk(3)))
                south = m(vcat(blk(7), blk(8), blk(9)))
                east = m(vcat(blk(2), blk(5), blk(8)))
                west = m(vcat(blk(4), blk(5), blk(6)))
                @test north + south + east + west ≈ 1.0 atol = 1e-6
                @test north ≈ south atol = 1e-9
                @test east ≈ west atol = 1e-9
            end
        end

        @testset "Gain Report" begin
            # A straight chain makes the truth easy: released at one end, most
            # animals are recaptured some way along it.
            releases = [1, 1, 1, 2, 2, 3]
            recaptures = [3, 4, 5, 4, 5, 5]
            ks = [6, 6, 6, 6, 6, 6]
            rep = persistence_gain_report(
                W, cents, hsi;
                releases = releases, recaptures = recaptures, ks = ks,
                gamma = 1.0, residence = 0.0, advection = 0.5,
                kappas = [0.0, 1.0, 3.0], n_headings = H
            )
            @test rep.n_events == 6
            @test rep.n_units == n
            @test length(rep.by_persistence) == 3
            @test all(isfinite, [r.mean_loglik for r in rep.by_persistence])
            @test rep.by_persistence[1].persistence == 0.0
            @test rep.by_persistence[1].gain_vs_first_order ≈ 0.0 atol = 1e-9
            @test rep.best_persistence in (0.0, 1.0, 3.0)

            @test_throws ArgumentError persistence_gain_report(
                W, cents, hsi;
                releases = [1], recaptures = [2], ks = [1], max_units = 2
            )
            @test_throws DimensionMismatch persistence_gain_report(
                W, cents, hsi; releases = [1], recaptures = [2, 3], ks = [1]
            )
        end
    end

    @testset "Path Reconstruction" begin
        # Every consecutive transition on a returned path must be one the model
        # permits. The previous implementation violated this by padding.
        W = spzeros(3, 3)
        W[1, 2] = 1.0; W[2, 1] = 1.0; W[2, 3] = 1.0; W[3, 2] = 1.0
        hsi = [0.3, 0.5, 0.8]
        P = construct_stochastic_transition_kernel(
            W, hsi; gamma = 0.0, residence = 0.3, advection = 0.0
        )
        @test all(P[i, i] > 0.0 for i in 1:3)

        p4 = predict_path(P, 1, 3, 4; method = :astar)
        @test length(p4) == 5
        @test first(p4) == 1
        @test last(p4) == 3
        @test all(P[p4[i], p4[i+1]] > 0.0 for i in 1:(length(p4) - 1))

        # A shorter request must shorten, not return the longer route.
        Pd = [0.0 0.9 0.1; 0.0 0.0 0.9; 0.0 0.0 1.0]
        p1 = predict_path(Pd, 1, 3, 1; method = :astar)
        @test p1 == [1, 3]

        # Unsatisfiable horizons are reported, never fabricated.
        Pn = [0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 0.0 1.0]
        @test predict_path(Pn, 1, 3, 1; method = :astar) == Int[]
        Pstay = [0.0 1.0; 1.0 0.0]
        @test predict_path(Pstay, 1, 1, 3; method = :astar) == Int[]
        @test predict_path(Pstay, 1, 1, 2; method = :astar) == [1, 2, 1]
        @test predict_path(P, 1, 3; method = :astar) == [1, 2, 3]

        @test predict_path(Pn, 1, 3, 2; method = :viterbi) == [1, 2, 3]
        @test predict_path(Pn, 1, 3, 1; method = :viterbi) == Int[]
    end

    @testset "Markov Bridge And HMM Validity" begin
        # Disconnected endpoints: the bridge is undefined and must be reported.
        Pd = zeros(4, 4)
        Pd[1, 2] = 1.0; Pd[2, 1] = 1.0; Pd[3, 4] = 1.0; Pd[4, 3] = 1.0
        unreachable = predict_corridor(Pd, 1, 3, 5)
        @test size(unreachable) == (4, 6)
        @test all(isnan, unreachable)
        @test all(isnan, predict_dynamic_corridor([Pd], 1, 3))

        # A connected bridge is a genuine conditional distribution.
        Pc = zeros(3, 3)
        Pc[1, 2] = 1.0; Pc[2, 3] = 1.0; Pc[3, 3] = 1.0
        c = predict_corridor(Pc, 1, 3, 2)
        @test all(j -> isapprox(sum(c[:, j]), 1.0; atol = 1e-12), 1:3)
        @test c[1, 1] == 1.0
        @test c[2, 2] == 1.0
        @test c[3, 3] == 1.0

        # HMM decode reports failure rather than returning the argmax of a
        # forbidden score field.
        res = viterbi_hmm_path_smoothing([1, 3], [1, 3], Pc)
        @test res.valid == true
        @test res.path == [1, 2, 3]
        @test isfinite(res.log_likelihood)
        @test viterbi_hmm_path_smoothing([1, 3], [1, 3], Pc; land_mask = trues(3)).valid == false
    end

    @testset "Residence And Trait Alignment" begin
        # Per-unit residence must account for exactly the elapsed time. A path of
        # m+1 nodes spans m intervals, and a terminal node starts none.
        paths_rich = [
            (tagid = "A", path = [1, 2, 3, 4],
             coords = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)],
             total_dist_km = 3.0, displacement_km = 3.0, tortuosity = 1.0,
             mean_hsi = 0.5, duration_days = 6.0, color = "#38bdf8"),
            (tagid = "B", path = [2, 3],
             coords = [(0.0, 1.0), (0.0, 2.0)],
             total_dist_km = 1.0, displacement_km = 1.0, tortuosity = 1.0,
             mean_hsi = 0.5, duration_days = 2.0, color = "#38bdf8"),
        ]
        stats = compute_movement_statistics(
            paths_rich, (paths = Dict{String, Vector{Int}}(),), (n_spatial = 4,)
        )
        rt = stats.residence_time_per_patch
        @test sum(rt) ≈ 8.0
        @test rt[1] ≈ 2.0
        @test rt[4] == 0.0
        @test stats.tagid == ["A", "B"]

        # Traits are joined to metrics by tag, never by row position.
        widths = [95.0, 102.0, 108.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0]
        obs = DataFrame(tagid = string.(1:10), carapace_width = widths)
        mov = (
            tagid = string.(1:10),
            net_displacement_km = [10.0, 12.0, 15.0, 18.0, 20.0, 24.0, 28.0, 30.0, 33.0, 36.0],
            path_efficiency = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72],
            tortuosity = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1],
        )
        loaded = (au = nothing,)
        assoc = model_trait_movement_associations(obs, mov, loaded)
        @test assoc.trait_name == "carapace_width"
        d = assoc.models["displacement"]
        @test d.beta_1 > 0.0
        @test d.r2 > 0.9
        @test d.p_val < 0.01

        # Reversing the observation table must not move the fit.
        rev = model_trait_movement_associations(obs[10:-1:1, :], mov, loaded)
        @test rev.models["displacement"].beta_1 ≈ d.beta_1
        @test rev.trait_vals == assoc.trait_vals

        # Missing traits are skipped, never imputed with a placeholder.
        miss = DataFrame(
            tagid = string.(1:10),
            carapace_width = Vector{Union{Missing, Float64}}(
                vcat(fill(missing, 7), widths[8:end])
            ),
        )
        @test length(model_trait_movement_associations(miss, mov, loaded).trait_vals) == 3

        # No trait column: explicit unavailable, never synthesised.
        none = model_trait_movement_associations(DataFrame(tagid = string.(1:10)), mov, loaded)
        @test none.trait_name == "none"
        @test isempty(none.models)
        @test haskey(none, :message)
    end

    @testset "Posterior Predictive Uses Full Horizon" begin
        # 3-node chain. From unit 1 the far unit is unreachable in one step but
        # easily reached in twelve, so scoring the same recapture at both
        # horizons separates a full-horizon propagation from a truncated one.
        n = 3
        W = spzeros(n, n)
        W[1, 2] = 1.0; W[2, 1] = 1.0; W[2, 3] = 1.0; W[3, 2] = 1.0
        loaded = (
            W = W, hsi_vec = fill(0.5, n), land_mask = falses(n), n_spatial = n,
            mesh = (n_units = n,),
        )
        kernels = (alpha_samples = [0.5], rho_samples = [0.05], gamma_samples = [1.0])

        obs_df(k) = DataFrame(release = [1], recapture = [3], k = [k], tagid = ["A"])
        loaded_long = merge(loaded, (obs_df = obs_df(12),))
        loaded_short = merge(loaded, (obs_df = obs_df(1),))
        ppc_short = posterior_predictive_check(
            loaded_short, NamedTuple(), kernels, (seed = 1,)
        )
        ppc_long = posterior_predictive_check(
            loaded_long, NamedTuple(), kernels, (seed = 1,)
        )

        s_long = ppc_long.summary
        s_short = ppc_short.summary
        @test s_long.n_draws == 1
        @test s_long.n_observations == 1
        @test s_long.n_observations_total == 1
        @test isfinite(s_long.brier_mean)
        @test s_long.brier_mean >= 0.0
        # A one-step horizon cannot reach the far unit, so it must score worse
        # than the twelve-step horizon on the identical transition.
        @test s_long.brier_mean < s_short.brier_mean
        # Event-level calibration is bounded: a one-hot target against any
        # distribution gives a squared error in [0, 2].
        @test 0.0 <= s_long.brier_mean <= 2.0
    end

    @testset "Habitat-Coupled Residency" begin
        # A 5-node chain whose suitability peaks in the middle, so the centre unit
        # has a positive local advantage and the ends a negative one.
        W = spzeros(5, 5)
        for i in 1:4
            W[i, i+1] = 1.0
            W[i+1, i] = 1.0
        end
        hsi = [0.1, 0.4, 0.9, 0.4, 0.1]

        @testset "Sanitising Guards Non-Finite Habitat" begin
            dirty = [0.2, NaN, Inf, -Inf, 1.4, -0.3]
            clean = sanitise_hsi(dirty)
            @test length(clean) == length(dirty)
            @test all(isfinite, clean)
            @test all(0.0 .<= clean .<= 1.0)
            # Non-finite entries fall back to the mean of the finite ones.
            @test clean[2] ≈ clean[3] ≈ clean[4]
            # An entirely non-finite field degrades to zeros, not NaN.
            @test sanitise_hsi([NaN, Inf]) == [0.0, 0.0]
            @test sanitise_hsi(Float64[]) == Float64[]
        end

        @testset "Advantage Forms" begin
            for form in (:difference, :ratio, :log_ratio,
                         :exp_difference, :exp_ratio, :exp_log_ratio)
                a = local_hsi_advantage(hsi, W; form = form)
                @test length(a) == 5
                @test all(isfinite, a)
                # A local peak has positive advantage; a local trough negative.
                @test a[3] > 0
                @test a[1] < 0
                # The neutral point is exactly zero for every form.
                @test local_hsi_advantage(fill(0.5, 5), W; form = form) ≈ zeros(5) atol = 1e-9
            end

            # Ratio forms cannot be formed against a zero neighbourhood, and must
            # degrade to neutral rather than to Inf or NaN.
            zero_hsi = zeros(5)
            for form in (:ratio, :log_ratio, :exp_ratio, :exp_log_ratio)
                @test local_hsi_advantage(zero_hsi, W; form = form) ≈ zeros(5) atol = 1e-12
            end
            @test all(isfinite, local_hsi_advantage(zero_hsi, W; form = :ratio))

            # An isolated unit has no neighbourhood, so the ratio is undefined and
            # must degrade to neutral. Units 3-5 here are isolated; units 1 and 2
            # do have each other and so get a real (non-zero) contrast.
            iso = spzeros(5, 5)
            iso[1, 2] = 1.0; iso[2, 1] = 1.0
            a_iso = local_hsi_advantage(hsi, iso; form = :ratio)
            @test all(isfinite, a_iso)
            @test a_iso[3:5] ≈ zeros(3) atol = 1e-12

            @test_throws ArgumentError local_hsi_advantage(hsi, W; form = :nonsense)
            @test_throws DimensionMismatch local_hsi_advantage([0.5, 0.5], W)
        end

        @testset "Residency Is Exact At Zero Coupling" begin
            adv = local_hsi_advantage(hsi, W; form = :difference)
            for form in (:difference, :ratio, :log_ratio,
                         :exp_difference, :exp_ratio, :exp_log_ratio)
                @test residency_from_advantage(0.3, adv, 0.0, form) ≈ fill(0.3, 5) atol = 0
            end
            # A NaN advantage must not poison the stay probability.
            bad = [0.1, NaN, 0.3, Inf, -Inf]
            r = residency_from_advantage(0.3, bad, 1.0, :exp_ratio)
            @test all(isfinite, r)
            @test r[2] == 0.3 && r[4] == 0.3 && r[5] == 0.3
            # Strong coupling is clamped at the documented 0.999 ceiling rather
            # than saturating to exactly 1, and stays finite.
            strong = residency_from_advantage(
                0.5, fill(50.0, 3), 10.0, :exp_ratio
            )
            @test all(isfinite, strong)
            @test all(0.0 .< strong .<= 0.999)
            @test strong ≈ fill(0.999, 3) atol = 1e-12
            @test_throws ArgumentError residency_from_advantage(0.3, adv, 1.0, :bogus)
        end

        @testset "Kernel Reduces And Responds" begin
            base = construct_stochastic_transition_kernel(
                W, hsi; gamma = 1.0, residence = 0.3, advection = 0.5
            )
            # Zero coupling must reproduce the fixed-residence kernel exactly.
            for form in (:difference, :ratio, :log_ratio,
                         :exp_difference, :exp_ratio, :exp_log_ratio)
                K = construct_stochastic_transition_kernel(
                    W, hsi; gamma = 1.0, residence = 0.3, advection = 0.5,
                    rest_coupling = 0.0, rest_advantage_form = form
                )
                @test K == base
            end

            # Positive coupling makes a habitat peak stickier than a trough.
            Kp = construct_stochastic_transition_kernel(
                W, hsi; gamma = 1.0, residence = 0.3, advection = 0.5,
                rest_coupling = 1.0, rest_advantage_form = :difference
            )
            @test all(isapprox.(vec(sum(Kp; dims = 2)), 1.0; atol = 1e-6))
            @test Kp[3, 3] > base[3, 3]      # local peak
            @test Kp[1, 1] < base[1, 1]      # local trough

            # Every form keeps the kernel stochastic, including a hostile field.
            for form in (:difference, :ratio, :log_ratio,
                         :exp_difference, :exp_ratio, :exp_log_ratio)
                K = construct_stochastic_transition_kernel(
                    W, hsi; gamma = 1.0, residence = 0.3, advection = 0.5,
                    rest_coupling = 2.0, rest_advantage_form = form
                )
                @test all(K .>= 0.0)
                @test all(isapprox.(vec(sum(K; dims = 2)), 1.0; atol = 1e-6))
            end

            # A field containing NaN and Inf must not produce a NaN kernel.
            hostile = [0.2, NaN, Inf, 0.5, -Inf]
            for form in (:ratio, :log_ratio, :exp_ratio, :exp_log_ratio)
                K = construct_stochastic_transition_kernel(
                    W, hostile; gamma = 1.0, residence = 0.3, advection = 0.5,
                    rest_coupling = 1.0, rest_advantage_form = form
                )
                @test all(isfinite, K)
                @test all(isapprox.(vec(sum(K; dims = 2)), 1.0; atol = 1e-6))
            end

            # All-zero habitat: ratios are undefined, and the kernel must fall
            # back to the plain global residence rather than to Inf.
            K0 = construct_stochastic_transition_kernel(
                W, zeros(5); gamma = 1.0, residence = 0.3, advection = 0.5,
                rest_coupling = 1.0, rest_advantage_form = :ratio
            )
            @test all(isfinite, K0)
            @test all(isapprox.(vec(sum(K0; dims = 2)), 1.0; atol = 1e-6))
        end
    end

    @testset "Deterministic Seeding" begin
        # Julia salts `hash` per process, so a fixed seed must not depend on it.
        h = MovementAnalysis._stable_string_hash("tag_0042")
        @test h isa UInt64
        @test MovementAnalysis._stable_string_hash("tag_0042") == h
        @test MovementAnalysis._stable_string_hash("tag_0043") != h
        @test MovementAnalysis._stable_string_hash("") == 0xcbf29ce484222325
    end

    @testset "Circuit Known Answers" begin
        Wp = sparse([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0])
        Lp, _ = build_circuit_laplacian(Wp)
        @test all(isapprox.(vec(sum(Lp; dims = 2)), 0.0; atol = 1e-10))

        # Two unit resistances in series: R(1,3) = 2, R(1,2) = 1.
        Om = effective_resistance_matrix(Lp; W = Wp)
        @test Om[1, 3] ≈ 2.0 atol = 1e-8
        @test Om[1, 2] ≈ 1.0 atol = 1e-8
        @test diag(Om) ≈ zeros(3)
        @test Om ≈ Om'
        @test all(Om .>= -1e-12)

        # Triangle: a direct 1 ohm edge in parallel with the two-edge path.
        Ws = sparse([0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0])
        Os = effective_resistance_matrix(build_circuit_laplacian(Ws)[1]; W = Ws)
        @test Os[1, 2] ≈ 2 / 3 atol = 1e-8

        # Disconnected components get the sentinel, not a measured distance.
        Wd = sparse([0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0;
                     0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0])
        Od = effective_resistance_matrix(
            build_circuit_laplacian(Wd)[1]; W = Wd, disconnected_dist = 1e6
        )
        @test Od[1, 2] ≈ 1.0 atol = 1e-8
        @test Od[1, 3] == 1e6
        @test Od[3, 4] ≈ 1.0 atol = 1e-8

        # Land-masked edges leave the conductance entirely.
        Wl = sparse([0.0 1.0 1.0; 1.0 0.0 0.0; 1.0 0.0 0.0])
        _, Cl = build_circuit_laplacian(Wl; land_mask = [false, true, true])
        @test Cl[1, 2] == 0.0
        @test Cl[1, 3] == 0.0

        # Conductance from habitat is exponential in the summed suitability.
        for g in (0.5, 1.0, 2.0, 3.0)
            h3 = [0.0, 0.8, 0.8]
            _, Cg = build_circuit_laplacian(Wl; hsi = h3, hsi_exponent = g)
            @test Cg[1, 2] ≈ exp(g * (h3[1] + h3[2]) / 2) atol = 1e-10
        end
        @test build_circuit_laplacian(Wl; hsi = [0.0, 0.8, 0.8])[2] ≈
              build_circuit_laplacian(Wl; hsi = [0.0, 0.8, 0.8], hsi_exponent = 1.0)[2]

        # A sink the source cannot reach carries no current.
        P = zeros(4, 4)
        P[1, 2] = 0.7; P[2, 1] = 0.7; P[2, 3] = 0.7; P[3, 2] = 0.7; P[4, 4] = 1.0
        _, R_ok, I_ok = solve_directed_circuit_voltage(P, 1, 3)
        @test R_ok > 0 && isfinite(R_ok)
        @test nnz(I_ok) > 0
        V_bad, R_bad, I_bad = solve_directed_circuit_voltage(P, 1, 4)
        @test R_bad == Inf
        @test all(iszero, V_bad)
        @test nnz(I_bad) == 0
        @test solve_directed_circuit_voltage(P, 1, 1)[2] == 0.0
    end

    @testset "Effective Resistance Distance Units" begin
        # haversine returns metres; a planar-km frame inside the degree box must
        # not be read as longitude/latitude.
        planar = [(10.0, 20.0), (12.0, 20.0), (14.0, 20.0)]
        @test MovementAnalysis._spatial_node_distance(
            planar[1], planar[2]; coord_space = :planar_km
        ) ≈ 2.0 atol = 1e-10
        @test MovementAnalysis._infer_coord_space(planar) == false

        geo = [(-63.0, 44.0), (-62.0, 44.0)]
        d = MovementAnalysis._spatial_node_distance(geo[1], geo[2]; coord_space = :geographic)
        @test 70.0 < d < 90.0
        @test MovementAnalysis._infer_coord_space(geo) == true

        Wp = sparse([0.0 1.0 1.0; 1.0 0.0 0.0; 1.0 0.0 0.0])
        Om = effective_resistance_matrix(
            build_circuit_laplacian(Wp)[1];
            W = Wp, centroids = planar, coord_space = :planar_km
        )
        @test 0.0 < Om[1, 2] < 1e5
    end

    @testset "Spatial Input Integrity" begin
        @test movement_parameters_default().bathymetry_source isa Symbol

        # Survey geometry is surfaced as polygons, never as a Boolean mask.
        @test MovementAnalysis._snowcrab_land_polygons((mesh = (n_units = 3,),)) === nothing
        @test MovementAnalysis._snowcrab_land_polygons(
            (mesh = (sppoly_geometries = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]],),)
        ) == [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]

        # Empirical HSI survives a mesh change rather than being flattened.
        src = (centroids = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],)
        dst = (centroids = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0),
                            (2.0, 0.0), (2.5, 0.0), (3.0, 0.0)],)
        hsi_src = [0.1, 0.4, 0.7, 1.0]
        P = compute_network_transfer_matrix(src, dst; method = :idw)
        tr = reshard_spatial_field(hsi_src, src, dst)
        @test all(abs.(sum(P; dims = 2) .- 1.0) .< 1e-8)
        @test all(minimum(hsi_src) .- 1e-9 .<= tr .<= maximum(hsi_src) .+ 1e-9)
        @test issorted(tr)
        # A mesh with an explicit `nothing` polygon field must not crash transfer.
        @test compute_network_transfer_matrix(
            (centroids = src.centroids, polygons = nothing,), dst; method = :idw
        ) isa AbstractMatrix

        # Resharding keeps HSI aligned, retains the composed mapping, and leaves
        # every endpoint on a navigable marine unit.
        pr = merge(movement_parameters_default(), (reshard_hex = true, verbose = false,))
        loaded = load_movement_data(pr)
        @test length(loaded.hsi_vec) == loaded.n_spatial
        @test length(unique(round.(loaded.hsi_vec; digits = 8))) > 1
        @test [s.name for s in loaded.unit_mapping.stages] == [:source, :fine]
        @test all(1 .<= loaded.unit_mapping.to_final .<= loaded.n_spatial)
        @test loaded.bathymetry_provenance.file_backed == false
        @test loaded.bathymetry_provenance.hsi_origin != :empirical_transferred
        @test nrow(loaded.obs_df) > 0
        @test all(1 .<= loaded.obs_df.release .<= loaded.n_spatial)
        @test all(1 .<= loaded.obs_df.recapture .<= loaded.n_spatial)
        @test all(.!loaded.land_mask[loaded.obs_df.release])
        @test all(.!loaded.land_mask[loaded.obs_df.recapture])
        @test length(loaded.hsi_vec) == length(loaded.land_mask)

        # The adaptive stage must not wipe the habitat field to a constant.
        pr2 = merge(
            movement_parameters_default(),
            (reshard_hex = true, adaptive_mesh = true, verbose = false,),
        )
        ad = load_movement_data(pr2)
        @test length(ad.hsi_vec) == ad.n_spatial
        @test length(unique(round.(ad.hsi_vec; digits = 8))) > 1
        @test [s.name for s in ad.unit_mapping.stages] == [:source, :fine, :adaptive]
        @test all(1 .<= ad.unit_mapping.to_final .<= ad.n_spatial)
        @test all(.!ad.land_mask[ad.obs_df.release])
        @test all(.!ad.land_mask[ad.obs_df.recapture])
    end

    @testset "Output Encoding And Palette Bounds" begin
        # A quote, backslash, control character, or script tag must not produce
        # invalid JSON or terminate the enclosing script.
        @test MovementAnalysis._json_string("plain") == "\"plain\""
        @test MovementAnalysis._json_string("a\"b") == "\"a\\\"b\""
        @test MovementAnalysis._json_string("a\\b") == "\"a\\\\b\""
        @test MovementAnalysis._json_string("a\nb") == "\"a\\nb\""
        @test MovementAnalysis._json_string("</script>") == "\"\\u003c/script\\u003e\""
        @test MovementAnalysis._js_escape_content("it's") == "it\\'s"

        pal = ["#000000", "#ffffff"]
        for v in (NaN, Inf, -Inf, 0.0, 0.5, 1.0, -1.0, 2.0)
            hex = MovementAnalysis._map_val_to_hex(v, 0.0, 1.0, pal)
            @test length(hex) == 7
            @test all(c -> c in "0123456789abcdef", hex[2:end])
        end
        @test MovementAnalysis._map_val_to_hex(NaN, 0.0, 1.0, pal) == "#888888"
        @test MovementAnalysis._map_val_to_hex(Inf, 0.0, 1.0, pal) == "#888888"
        mid = MovementAnalysis._map_val_to_hex(0.5, 0.0, 1.0, pal)
        @test MovementAnalysis._map_val_to_hex(0.7, 2.0, 2.0, pal) == mid
        @test MovementAnalysis._map_val_to_hex(99.0, 2.0, 2.0, pal) == mid
        @test MovementAnalysis._map_val_to_hex(0.5, 0.0, 1.0, ["#123456"]) == "#123456"
        @test MovementAnalysis._map_val_to_hex(0.5, 0.0, 1.0, String[]) == "#888888"
    end

    @testset "Map Structure And Coordinates" begin
        @test LeafletMap("<div>x</div>"; title = "T").title == "T"

        # A degree of longitude is not a degree of latitude away from the equator.
        ci = (-63.0, 44.0)
        east_km, north_km = MovementAnalysis._metric_offset_km(:geographic, ci, (-62.0, 44.0))
        @test north_km ≈ 0.0 atol = 1e-9
        @test 70.0 < east_km < 90.0
        _, n2 = MovementAnalysis._metric_offset_km(:geographic, ci, (-63.0, 45.0))
        @test n2 ≈ 111.32 atol = 1e-6
        @test east_km < n2
        @test collect(MovementAnalysis._km_to_map_offset(
            :geographic, ci, east_km, north_km
        )) ≈ [1.0, 0.0] atol = 1e-9
        @test MovementAnalysis._metric_offset_km(:planar_km, (0.0, 0.0), (3.0, 4.0)) == (3.0, 4.0)

        # Declared feature count must match the input, and a mismatch is rejected.
        polys = [
            [(-63.6, 44.4), (-63.4, 44.4), (-63.4, 44.6), (-63.6, 44.6), (-63.6, 44.4)],
            [(-63.1, 44.7), (-62.9, 44.7), (-62.9, 44.9), (-63.1, 44.9), (-63.1, 44.7)],
        ]
        au = (centroids_lonlat = [(-63.5, 44.5), (-63.0, 44.8)],)
        m = leaflet_choropleth(polys, [0.25, 0.75]; au = au, title = "G")
        @test occursin("\"type\": \"FeatureCollection\"", m.html)
        @test count("\"type\": \"Feature\"", m.html) == 2
        @test_throws AssertionError leaflet_choropleth(polys, [0.1, 0.2, 0.3])

        m2 = leaflet_choropleth(
            polys, [0.25, 0.75]; au = au,
            extra_props = Dict(1 => ["label" => "quote\" and </script>"], 2 => ["l" => "b\\s"]),
        )
        @test occursin("\\u003c/script\\u003e", m2.html)
        @test !occursin("\"quote\" and", m2.html)

        # The corridor explorer must not present a pass-through route as a k-step
        # arrival, and the map must declare a coordinate space.
        P = [0.2 0.8 0.0; 0.3 0.4 0.3; 0.0 0.5 0.5]
        corr = leaflet_interactive_corridor_dashboard(P, au; title = "C")
        @test corr isa LeafletMap
        @test !occursin("seq.indexOf(end) !== -1", corr.html)
    end

    @testset "Hydrodynamic Axis Validation" begin
        S = 3
        nz = 2
        polys = [
            [(-63.6, 44.4), (-63.4, 44.4), (-63.4, 44.6), (-63.6, 44.6), (-63.6, 44.4)],
            [(-63.1, 44.7), (-62.9, 44.7), (-62.9, 44.9), (-63.1, 44.9), (-63.1, 44.7)],
            [(-62.6, 44.9), (-62.4, 44.9), (-62.4, 45.1), (-62.6, 45.1), (-62.6, 44.9)],
        ]
        au = (centroids_lonlat = [(-63.5, 44.5), (-63.0, 44.8), (-62.5, 45.0)], polygons = polys)
        depths = [-10.0, -50.0]

        ok = leaflet_hydrodynamic_dashboard(
            au, (depths = depths, temperature = [1.0 2.0; 3.0 4.0; 5.0 6.0],)
        )
        @test ok.metadata[:n_units] == S
        @test ok.metadata[:n_depths] == nz
        @test ok.metadata[:depths_assumed] == false
        @test ok.metadata[:provenance] == :unknown
        @test occursin("1.00", ok.html)

        # Stored (depth x cell) is transposed, not discarded.
        tr = leaflet_hydrodynamic_dashboard(
            au, (depths = depths, temperature = [1.0 3.0 5.0; 2.0 4.0 6.0],)
        )
        @test occursin("1.00", tr.html)
        @test tr.metadata[:n_depths] == nz

        # A field matching neither axis renders the default with a warning.
        @test leaflet_hydrodynamic_dashboard(
            au, (depths = depths, temperature = [1.0 2.0; 3.0 4.0],)
        ).metadata[:n_depths] == nz

        # Assumed depths and synthetic provenance are both flagged.
        nodesp = leaflet_hydrodynamic_dashboard(au, (temperature = [1.0 2.0 3.0 4.0],))
        @test nodesp.metadata[:depths_assumed] == true
        @test nodesp.metadata[:n_depths] == 4
        syn = leaflet_hydrodynamic_dashboard(
            au, (depths = depths, temperature = [1.0 2.0; 3.0 4.0; 5.0 6.0],);
            provenance = :synthetic,
        )
        @test occursin("SYNTHETIC", syn.title)
        @test syn.metadata[:provenance] == :synthetic
    end

    @testset "Optional Plotting Backend" begin
        # The base package must not call `plot` directly: it either dispatches to
        # the extension or names the missing dependency.
        @test hasmethod(MovementAnalysis.plot_posterior_predictive_check, Tuple{Any, Any})
        err = try
            MovementAnalysis._ad_ratio_histogram_via_plots()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Plots", err.msg)
        @test MovementAnalysis.plot_ad_ratio_distribution(
            [0.1, 0.2], [0.3, 0.4]; mode = :leaflet
        ) isa Any
    end

    @testset "Graph Wavelets" begin
        W = sparse([0.0 1.0 1.0 0.0; 1.0 0.0 1.0 0.0;
                    1.0 1.0 0.0 1.0; 0.0 0.0 1.0 0.0])
        L = build_normalized_laplacian(W)
        @test size(L) == (4, 4)
        lam = compute_laplacian_spectral_bounds(L)
        @test 0.0 <= lam <= 2.2
        res = spectral_graph_wavelet_transform(L, [1.0, 2.0, 1.5, 0.5]; num_scales = 3)
        @test length(res.scales) == 3
        @test size(res.details, 2) == 3
    end

    include("test_agent_movement.jl")

end
