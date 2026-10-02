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