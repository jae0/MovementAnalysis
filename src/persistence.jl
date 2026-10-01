"""
    persistence.jl

Directional persistence for movement: a second-order chain over
(position, incoming heading).

# Why this is a different model

The pooled kernel used throughout the package is *memoryless*: `P[i, j]` depends
only on the current unit `i`, the candidate `j`, and the habitat field. An animal
that has been moving north-east is therefore given exactly the same step
distribution as one that has just arrived from the south-west. Real trajectories
are directionally autocorrelated, and no first-order kernel can represent that
regardless of how its parameters are tuned.

Persistence adds a heading state. An agent at unit `i` with heading ``h`` samples
a move to ``j`` with probability

```math
T[(i,h),(j,h')] \\propto P_{ij} \\exp\\left(\\kappa \\cos(\\beta_{ij} - \\theta_h)\\right)
```

where ``\\beta_{ij}`` is the bearing of the step ``i \\to j``, ``\\theta_h`` the
centre of heading bin ``h``, and ``\\kappa`` the persistence parameter. Heading
turning and movement are coupled through geometry, so a heading is only
reachable if a neighbour lies in that direction.

# Backward compatibility

``\\kappa = 0`` makes the exponential factor constant, so the reweighting is a
no-op and the heading-uniform unit marginal is *exactly* the first-order kernel.
The default is therefore 0.0 and nothing changes unless persistence is enabled.

# Cost, and why the likelihood is not simply extended

The state space grows from ``S`` to ``S H``. With ``H = 8`` and a few hundred mesh
units that is a few thousand states, and evaluating ``T^k`` for the mark-recapture
likelihood inside an MCMC loop is far more expensive than the ``P^k`` the
discrete-time model already pays. Persistence is therefore applied where it is
affordable and meaningful:

- `build_persistent_transition_kernel` gives the exact augmented kernel for
  simulation and analysis, with the first-order case as a documented special
  point.
- `simulate_agent_trajectories` carries a per-agent heading, so individual
  trajectories actually turn, which is what makes the agent formulation worth
  having.
- `persistence_gain_report` scores held-out events over a grid of ``\\kappa`` on
  small graphs, which is how one decides whether the extra parameter is earning
  its keep before paying for it in the full pipeline.

Integrating ``\\kappa`` into the fitted telemetry likelihood for a full-resolution
mesh is not currently affordable and is not attempted.
"""

using SparseArrays
using LinearAlgebra
using Random
using Statistics

const _KM_PER_DEG_P = 111.32

"""
    AdvantageForm

How a unit's local habitat advantage over its neighbourhood is measured, and how
that advantage enters the stay probability.

# Additive forms

`rho_i = clamp(residence + beta * A_i, 0, 0.999)` with

- `:difference` — ``A_i = h_i - \\bar{h}_{N(i)}``, bounded in ``[-1, 1]`` for
  habitat in ``[0, 1]``.
- `:ratio` — ``A_i = h_i / \\bar{h}_{N(i)} - 1``, the relative excess. Centred on
  zero, so "no better than your neighbours" leaves the stay probability alone.
- `:log_ratio` — ``A_i = \\log h_i - \\log \\bar{h}_{N(i)}``, the log ratio. Avoids
  the division entirely and is the natural scale-free contrast.

# Exponential forms

`rho_i = logistic(logit(residence) + beta * A_i)`, which keeps
``\\rho \\in (0, 1)`` without ever clamping to the boundary. Useful when the
coupling is strong, because the additive form saturates at ``0.999`` and then
stops responding.

- `:exp_difference`, `:exp_ratio`, `:exp_log_ratio` — as above, applied as a
  logit shift.

# Guards

`beta == 0` short-circuits before any arithmetic, because `0 * Inf` is `NaN`.
Habitat is sanitised to finite values in ``[0, 1]`` first, a neighbourhood with no
usable habitat yields zero advantage (so the stay probability reverts to the
global `residence`), the logit shift is clamped, and the resulting probability is
clamped as a final backstop. A ratio is therefore never formed against a zero or
non-finite denominator.
"""
const AdvantageForm = Symbol

const _ADVANTAGE_FORMS = (
    :difference, :ratio, :log_ratio,
    :exp_difference, :exp_ratio, :exp_log_ratio,
)

"""
    sanitise_hsi(hsi) -> Vector{Float64}

Habitat suitability forced to finite values in `[0, 1]`.

Non-finite entries (NaN, ±Inf) are replaced by the mean of the finite entries, or
by `0.0` when nothing in the field is finite. Every downstream habitat statistic
assumes a bounded finite field, and an unbounded HSI would otherwise propagate
through the softmax into a NaN kernel.
"""
function sanitise_hsi(hsi::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(hsi))
    finite_sum = 0.0
    n_finite = 0
    for i in eachindex(hsi)
        v = Float64(hsi[i])
        if isfinite(v)
            v = clamp(v, 0.0, 1.0)
            finite_sum += v
            n_finite += 1
        end
        out[i] = v
    end
    fallback = n_finite > 0 ? finite_sum / n_finite : 0.0
    for i in eachindex(out)
        isfinite(out[i]) || (out[i] = fallback)
    end
    return out
end

"""
    local_hsi_advantage(hsi, W; form = :difference) -> Vector{Float64}

Per-unit local habitat advantage over its own neighbourhood, for the requested
`form` (see [`AdvantageForm`](@ref)).

Zero advantage means "this unit is no better or worse than the options around it",
which is the neutral point for every supported form. Where a form is undefined —
an all-zero neighbourhood makes any ratio meaningless — zero is returned, so the
caller's stay probability degrades to the plain global `residence` rather than to
an arbitrary number.

# Throws
- `ArgumentError` if `form` is not one of the supported symbols.
- `DimensionMismatch` if `hsi` and `W` disagree on unit count.
"""
function local_hsi_advantage(
    hsi::AbstractVector{<:Real},
    W::AbstractMatrix;
    form::AdvantageForm = :difference
)::Vector{Float64}
    form in _ADVANTAGE_FORMS || throw(ArgumentError(
        "Unsupported rest_advantage_form :$form; expected one of " *
        "$(_ADVANTAGE_FORMS)."
    ))
    S = size(W, 1)
    length(hsi) == S || throw(DimensionMismatch(
        "hsi has $(length(hsi)) entries but W has $S rows."
    ))

    h = sanitise_hsi(hsi)

    # Row i of W is the set of neighbours, which is column i of the transpose.
    Wt = SparseMatrixCSC(transpose(sparse(W)))
    bar = zeros(Float64, S)
    for i in 1:S
        s = 0.0
        wsum = 0.0
        for ptr in nzrange(Wt, i)
            j = Wt.rowval[ptr]
            j == i && continue
            s += Wt.nzval[ptr] * h[j]
            wsum += Wt.nzval[ptr]
        end
        bar[i] = wsum > 0.0 ? s / wsum : NaN
    end

    eps_h = 1e-9
    out = zeros(Float64, S)
    for i in 1:S
        hi = h[i]
        b = bar[i]
        if !isfinite(b) || b <= eps_h
            # No usable neighbourhood: the ratio is undefined, so stay neutral.
            out[i] = 0.0
            continue
        end
        if form === :difference
            out[i] = hi - b
        elseif form === :ratio
            out[i] = hi / b - 1.0
        elseif form === :log_ratio
            out[i] = log(hi + eps_h) - log(b + eps_h)
        else
            a = if form === :exp_difference
                hi - b
            elseif form === :exp_ratio
                hi / b - 1.0
            else
                log(hi + eps_h) - log(b + eps_h)
            end
            out[i] = isfinite(a) ? a : 0.0
        end
    end
    return out
end

"""
    residency_from_advantage(residence, advantage, coupling, form) -> Vector{Float64}

Per-unit stay probability from a local habitat advantage.

`coupling == 0` returns `residence` unchanged for every unit, so the coupling is a
strict generalisation of the fixed-residence model. Exponential forms apply the
coupling as a bounded logit shift; additive forms apply it directly and clamp.
"""
function residency_from_advantage(
    residence::Real,
    advantage::AbstractVector{<:Real},
    coupling::Real,
    form::AdvantageForm
)::Vector{Float64}
    rho0 = clamp(Float64(residence), 0.0, 0.999)
    beta = Float64(coupling)
    # Short-circuit: 0 * Inf is NaN, and a zero coupling must be an exact no-op.
    (beta == 0.0 || isempty(advantage)) && return fill(rho0, length(advantage))

    form in _ADVANTAGE_FORMS || throw(ArgumentError(
        "Unsupported rest_advantage_form :$form."
    ))

    logit0 = log(rho0 / (1.0 - rho0))
    out = Vector{Float64}(undef, length(advantage))
    for i in eachindex(advantage)
        a = Float64(advantage[i])
        if !isfinite(a)
            out[i] = rho0
            continue
        end
        if form === :exp_difference || form === :exp_ratio || form === :exp_log_ratio
            # Clamp the shift so logistic() cannot saturate to exactly 0 or 1.
            shift = clamp(beta * a, -20.0, 20.0)
            out[i] = clamp(1.0 / (1.0 + exp(-(logit0 + shift))), 0.0, 0.999)
        else
            out[i] = clamp(rho0 + beta * a, 0.0, 0.999)
        end
    end
    return out
end

"""
    coordinate_space_of(centroids) -> Symbol

`:geographic` or `:planar_km`, resolved conservatively. Degrees and kilometres
are both small numbers, so the distinction must be declared or inferred from a
known marine region rather than read off a magnitude.
"""
function coordinate_space_of(centroids)::Symbol
    isempty(centroids) && return :planar_km
    xs = Float64[]
    ys = Float64[]
    for pt in centroids
        length(pt) >= 2 || continue
        push!(xs, Float64(pt[1]))
        push!(ys, Float64(pt[2]))
    end
    (isempty(xs) || isempty(ys)) && return :planar_km
    min_x, max_x = minimum(xs), maximum(xs)
    min_y, max_y = minimum(ys), maximum(ys)
    (min_y < -90.0 || max_y > 90.0 || min_x < -180.0 || max_x > 180.0) && return :planar_km
    known = (min_x <= -20.0 && max_x <= -10.0 && min_y >= 30.0 && max_y <= 85.0) ||
            (min_x >= 100.0 && max_x <= 180.0 && min_y >= -50.0 && max_y <= 70.0) ||
            (min_x >= -180.0 && max_x <= -50.0 && min_y >= -60.0 && max_y <= 75.0)
    return known ? :geographic : :planar_km
end

"""
    bearing_deg(from, to; coord_space = :unknown) -> Float64

Compass bearing of the step `from -> to`, in degrees clockwise from north.

Returns `NaN` for a zero-length step, which is the self-transition case; callers
treat that as "heading unchanged" because staying put preserves direction.
"""
function bearing_deg(from, to; coord_space::Symbol = :unknown)::Float64
    space = coord_space === :unknown ? coordinate_space_of((from, to)) : coord_space
    if space === :geographic
        ex = (Float64(to[1]) - Float64(from[1])) * cosd(Float64(from[2]))
        ny = Float64(to[2]) - Float64(from[2])
    else
        ex = Float64(to[1]) - Float64(from[1])
        ny = Float64(to[2]) - Float64(from[2])
    end
    (ex == 0.0 && ny == 0.0) && return NaN
    return mod(rad2deg(atan(ex, ny)), 360.0)
end

"""
    heading_bin(bearing::Float64, n_headings::Int) -> Int

Bin index in `1:n_headings` for a bearing, with bin 1 centred on north.

A `NaN` bearing means a zero-length step, which is the *self* transition. Callers
must resolve that case themselves by preserving the current heading: falling
through to bin 1 here would make every residence step reset the heading to north
and inject a systematic drift on any mesh, because bin 1 is not symmetric under
reflection.
"""
function heading_bin(bearing::Float64, n_headings::Int)::Int
    isnan(bearing) && return 1
    w = 360.0 / n_headings
    b = mod(bearing, 360.0)
    return clamp(floor(Int, b / w) + 1, 1, n_headings)
end

@inline function _turn_logweight(kappa::Float64, bearing::Float64, theta::Float64)::Float64
    # No heading yet (the agent's first step): no preference in any direction.
    # Without this the cosine would be NaN and the agent would freeze in place.
    isnan(theta) && return 0.0
    # A zero-length step preserves heading exactly, so it aligns perfectly.
    isnan(bearing) && return kappa
    return kappa * cosd(bearing - theta)
end

"""
    build_persistent_transition_kernel(
        W, centroids, hsi; gamma, residence, advection, persistence = 0.0,
        n_headings = 8, land_mask = nothing, coord_space = :unknown
    ) -> SparseMatrixCSC{Float64, Int}

Exact second-order transition kernel over `(unit, heading)` states.

State ``(i, h)`` is row `(i - 1) * n_headings + h` of the returned matrix, i.e. the
agent is at unit ``i`` travelling on heading bin ``h``. The move distribution
within a heading row is the pooled first-order kernel reweighted by heading
alignment and renormalised, so:

- ``persistence = 0`` reproduces the first-order kernel as the heading-uniform
  unit marginal, exactly.
- ``persistence > 0`` concentrates successive steps in a similar direction, which
  lengthens directed transits and suppresses turning.

# Arguments
- `W`: Spatial adjacency, `n_units x n_units`.
- `centroids`: Node coordinates, in degrees or kilometres per `coord_space`.
- `hsi`: Habitat suitability, length `n_units`.
- `gamma`, `residence`, `advection`: Passed to the first-order builder.
- `persistence`: ``\\kappa`` in the formulation above (default `0.0`).
- `n_headings`: Number of equally spaced heading bins (default `8`).
- `land_mask`, `coord_space`: As for the first-order builder.

# Returns
- `SparseMatrixCSC{Float64, Int}` of size `n_units * n_headings` squared, with
  every row summing to 1.
"""
function build_persistent_transition_kernel(
    W::AbstractMatrix{<:Real},
    centroids::AbstractVector,
    hsi::AbstractVector{<:Real};
    gamma = 1.0,
    residence = 0.25,
    advection = 0.4,
    persistence::Real = 0.0,
    n_headings::Integer = 8,
    land_mask = nothing,
    coord_space::Symbol = :unknown
)
    S = size(W, 1)
    size(W, 2) == S || throw(DimensionMismatch("W must be square, got $(size(W))."))
    length(centroids) == S || throw(DimensionMismatch(
        "centroids has $(length(centroids)) entries but W has $S rows."
    ))
    length(hsi) == S || throw(DimensionMismatch(
        "hsi has $(length(hsi)) entries but W has $S rows."
    ))
    H = Int(n_headings)
    H >= 2 || throw(ArgumentError("n_headings must be at least 2."))
    kappa = Float64(persistence)
    space = coord_space === :unknown ? coordinate_space_of(centroids) : coord_space

    P0 = sparse(build_sparse_transition_kernel(
        W, hsi, gamma, residence, advection, land_mask
    ))
    P0 = P0 isa SparseMatrixCSC ? P0 : SparseMatrixCSC(P0)

    # CSC stores columns, so row i of the kernel is column i of the transpose.
    # Reading `nzrange(P0, i)` directly would return the wrong entries.
    P0_t = SparseMatrixCSC(transpose(P0))

    bin_width = 360.0 / H
    theta = [(h - 1) * bin_width for h in 1:H]

    N = S * H
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    for i in 1:S
        nb_rows = collect(nzrange(P0_t, i))
        isempty(nb_rows) && continue
        for h in 1:H
            th = theta[h]
            acc = 0.0
            m = length(nb_rows)
            wts = Vector{Float64}(undef, m)
            targets = Vector{Int}(undef, m)
            for (k, ptr) in enumerate(nb_rows)
                j = P0_t.rowval[ptr]
                p = Float64(P0_t.nzval[ptr])
                if j == i
                    # A stay involves no turn, so there is nothing to reweight and
                    # nothing to re-bin: the agent keeps the heading it arrived
                    # with. Assigning the stay to a fixed bin instead (via
                    # heading_bin(NaN)) would reset the heading on every
                    # residence step and inject a systematic drift on any mesh,
                    # because the reset bin is not symmetric under reflection.
                    w = p
                    targets[k] = (i - 1) * H + h
                else
                    b = bearing_deg(centroids[i], centroids[j]; coord_space = space)
                    w = p * exp(_turn_logweight(kappa, b, th))
                    targets[k] = (j - 1) * H + heading_bin(b, H)
                end
                wts[k] = w
                acc += w
            end
            acc > 0.0 || continue
            row_id = (i - 1) * H + h
            for k in 1:m
                w = wts[k] / acc
                w > 0.0 || continue
                push!(I_idx, row_id)
                push!(J_idx, targets[k])
                push!(V_val, w)
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, N, N)
end

"""
    persistent_unit_marginal(T, n_units) -> SparseMatrixCSC

Heading-averaged unit transition matrix implied by an augmented kernel. This is
the first-order kernel a marginal observer would infer: for `persistence = 0` it
equals the original kernel, and for larger `persistence` it is the shadow
distribution an agent presents when its heading is not observed.
"""
function persistent_unit_marginal(
    T::SparseMatrixCSC{<:Real, <:Integer},
    n_units::Int
)::SparseMatrixCSC{Float64, Int}
    N = size(T, 1)
    N % n_units == 0 || throw(DimensionMismatch(
        "Augmented state space $N is not a multiple of $n_units units."
    ))
    H = N ÷ n_units
    H >= 1 || throw(ArgumentError("n_units must be positive."))
    # As above, CSC columns of the transpose are the kernel's rows. Row r of T is
    # column r of T_t, and each stored entry names its destination in `rowval`.
    T_t = SparseMatrixCSC(transpose(T))
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]
    for r in 1:N
        i = (r - 1) ÷ H + 1
        for ptr in nzrange(T_t, r)
            j = (T_t.rowval[ptr] - 1) ÷ H + 1
            push!(I_idx, i)
            push!(J_idx, j)
            push!(V_val, Float64(T_t.nzval[ptr]) / H)
        end
    end
    return sparse(I_idx, J_idx, V_val, n_units, n_units)
end

"""
    persistence_gain_report(
        W, centroids, hsi; releases, recaptures, ks, gamma, residence, advection,
        kappas = [0.0, 0.5, 1.0, 2.0, 4.0], n_headings = 8, land_mask = nothing
    ) -> NamedTuple

Score held-out mark-recapture events under the first-order model and under the
persistent model at each ``\\kappa`` in `kappas`, so the extra parameter can be
judged rather than assumed.

For each event the release heading is unknown and taken as uniform, so the
event probability is the heading-averaged ``T^k`` entry. The report gives mean
log-likelihood per model and the improvement over ``\\kappa = 0``.

This is intended for small graphs: it forms ``T^k`` explicitly and is guarded by
`max_units`. On a full-resolution mesh the same comparison belongs in the
pipeline, not in a diagnostic.
"""
function persistence_gain_report(
    W::AbstractMatrix{<:Real},
    centroids::AbstractVector,
    hsi::AbstractVector{<:Real};
    releases::AbstractVector{<:Integer},
    recaptures::AbstractVector{<:Integer},
    ks::AbstractVector{<:Integer},
    gamma = 1.0,
    residence = 0.25,
    advection = 0.4,
    kappas::AbstractVector{<:Real} = [0.0, 0.5, 1.0, 2.0, 4.0],
    n_headings::Integer = 8,
    land_mask = nothing,
    max_units::Int = 400
)
    S = size(W, 1)
    S <= max_units || throw(ArgumentError(
        "persistence_gain_report forms T^k explicitly and is limited to " *
        "$max_units units; got $S."
    ))
    n_ev = length(releases)
    (length(recaptures) == n_ev && length(ks) == n_ev) || throw(DimensionMismatch(
        "releases, recaptures, and ks must have the same length."
    ))
    n_ev > 0 || throw(ArgumentError("No events to score."))

    P0 = sparse(build_sparse_transition_kernel(
        W, hsi, gamma, residence, advection, land_mask
    ))
    P0 = P0 isa SparseMatrixCSC ? P0 : SparseMatrixCSC(P0)

    distinct_k = sort(unique(Int.(ks)))
    first_order = zeros(n_ev)
    for k in distinct_k
        idx = findall(==(k), Int.(ks))
        k == 0 && continue
        Pk = P0^k
        for i in idx
            first_order[i] = log(max(Float64(Pk[releases[i], recaptures[i]]), 1e-300))
        end
    end

    baseline = mean(first_order)
    per_kappa = NamedTuple[]
    for kappa in kappas
        T = build_persistent_transition_kernel(
            W, centroids, hsi;
            gamma = gamma, residence = residence, advection = advection,
            persistence = kappa, n_headings = n_headings, land_mask = land_mask
        )
        H = Int(n_headings)
        scores = zeros(n_ev)
        for k in distinct_k
            idx = findall(==(k), Int.(ks))
            k == 0 && continue
            Tk = T^k
            for i in idx
                r0 = (releases[i] - 1) * H
                c0 = (recaptures[i] - 1) * H
                p = 0.0
                for h in 1:H
                    for g in 1:H
                        p += Tk[r0 + h, c0 + g]
                    end
                end
                scores[i] = log(max(p / H, 1e-300))
            end
        end
        m = mean(scores)
        push!(per_kappa, (
            persistence = Float64(kappa),
            mean_loglik = m,
            gain_vs_first_order = m - baseline,
        ))
    end

    best = per_kappa[argmax([r.mean_loglik for r in per_kappa])]
    return (
        n_events = n_ev,
        n_units = S,
        n_headings = Int(n_headings),
        first_order_mean_loglik = baseline,
        by_persistence = per_kappa,
        best_persistence = best.persistence,
        best_mean_loglik = best.mean_loglik,
        improves_on_first_order = best.gain_vs_first_order > 1e-9,
    )
end
