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