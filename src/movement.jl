"""
    movement.jl

Advection-Diffusion-Reaction (ADR) spatiotemporal movement, spatial telemetry,
and particle trajectory simulation engine for the MovementAnalysis package.

Version: v1.0.0
"""

"""
    TelemetryData <: AbstractMatrix{Float64}

Structured container for mark-recapture telemetry observation events. Supports categorical
group-stratified transition modeling while maintaining matrix indexing compatibility.

# Fields
- `releases`: Vector of unit indices at initial detection / release (1-indexed).
- `recaps`: Vector of unit indices at subsequent detection / recapture (1-indexed).
- `ks`: Elapsed discrete time intervals between consecutive detections.
- `groups`: Biological stratum / group identifiers (1-indexed integers).
- `covariates`: Individual-level continuous covariates.
- `G`: Total count of distinct biological strata / groups.
- `max_k`: Maximum observed elapsed transition step count.
- `matrix`: `Matrix{Float64}` representation of size `(N, 4)`.
"""
struct TelemetryData <: AbstractMatrix{Float64}
    releases::Vector{Int}
    recaps::Vector{Int}
    ks::Vector{Int}
    groups::Vector{Int}
    covariates::Vector{Float64}
    G::Int
    max_k::Int
    matrix::Matrix{Float64}
end

Base.size(td::TelemetryData) = size(td.matrix)
Base.size(td::TelemetryData, d::Int) = size(td.matrix, d)
Base.getindex(td::TelemetryData, i::Int, j::Int) = td.matrix[i, j]
Base.getindex(td::TelemetryData, i::Int) = td.matrix[i]
Base.IndexStyle(::Type{TelemetryData}) = IndexLinear()

TelemetryData(input::Union{DataFrame, AbstractMatrix}; kwargs...) =
    _process_telemetry_data(input; kwargs...)

"""
    _process_telemetry_data(telemetry_input; mark_recapture_G=nothing)

Transforms input telemetry data (DataFrame or Matrix) into a validated `TelemetryData`
structure for movement modeling. Supports both longitudinal event sequences and
pre-aggregated transition event pairs.
"""
function _process_telemetry_data(telemetry_input; mark_recapture_G=nothing)
    if telemetry_input isa TelemetryData
        return telemetry_input
    elseif telemetry_input isa AbstractMatrix
        mat = Matrix{Float64}(telemetry_input)
        n_rows = size(mat, 1)
        rel = n_rows > 0 ? Int.(mat[:, 1]) : Int[]
        rec = n_rows > 0 && size(mat, 2) >= 2 ? Int.(mat[:, 2]) : Int[]
        ks  = n_rows > 0 && size(mat, 2) >= 3 ? Int.(mat[:, 3]) : Int[]
        cov = n_rows > 0 && size(mat, 2) >= 4 ? mat[:, 4] : zeros(Float64, n_rows)
        grp = if size(mat, 2) >= 5
            Int.(mat[:, 5])
        elseif !isnothing(mark_recapture_G) && mark_recapture_G > 1 &&
               all(c -> isinteger(c) && c >= 1, cov)
            Int.(cov)
        else
            ones(Int, n_rows)
        end
        G_val = isnothing(mark_recapture_G) ? (isempty(grp) ? 1 : maximum(grp)) :
            Int(mark_recapture_G)
        max_k_val = isempty(ks) ? 1 : maximum(ks)
        return TelemetryData(rel, rec, ks, grp, cov, G_val, max_k_val, mat)

    elseif telemetry_input isa DataFrame
        df = telemetry_input
        has_rel = hasproperty(df, :release) || hasproperty(df, :releases)
        has_rec = hasproperty(df, :recapture) || hasproperty(df, :recaps)

        if has_rel && has_rec
            rel_col = hasproperty(df, :release) ? :release : :releases
            rec_col = hasproperty(df, :recapture) ? :recapture : :recaps
            k_col = hasproperty(df, :k) ? :k : (hasproperty(df, :ks) ? :ks : nothing)
            grp_col = hasproperty(df, :group) ? :group :
                (hasproperty(df, :groups) ? :groups : nothing)
            cov_col = hasproperty(df, :covariate) ? :covariate :
                (hasproperty(df, :individual_covariate) ? :individual_covariate : nothing)

            rel = Int.(df[!, rel_col])
            rec = Int.(df[!, rec_col])
            ks  = !isnothing(k_col) ? Int.(df[!, k_col]) : ones(Int, nrow(df))
            grp = !isnothing(grp_col) ? Int.(df[!, grp_col]) : ones(Int, nrow(df))
            cov = !isnothing(cov_col) ? Float64.(df[!, cov_col]) : zeros(Float64, nrow(df))
            G_val = isnothing(mark_recapture_G) ? (isempty(grp) ? 1 : maximum(grp)) :
                Int(mark_recapture_G)
            max_k_val = isempty(ks) ? 1 : maximum(ks)
            mat = hcat(Float64.(rel), Float64.(rec), Float64.(ks), cov)
            return TelemetryData(rel, rec, ks, grp, cov, G_val, max_k_val, mat)
        end

        time_col = if hasproperty(df, :time)
            :time
        elseif hasproperty(df, :timestamp)
            :timestamp
        else
            _detect_time_column(df; allow_nothing=true)
        end

        tag_col = nothing
        for cand in [:tagid, :tag_id, :tag, :id, :individual_id, :animal_id]
            if hasproperty(df, cand)
                tag_col = cand
                break
            end
        end

        s_col = if hasproperty(df, :s_idx)
            :s_idx
        else
            _detect_spatial_unit_column(df; allow_nothing=true)
        end

        if isnothing(time_col) || isnothing(tag_col) || isnothing(s_col)
            error("Telemetry DataFrame must contain either event pairs " *
                  "(:release, :recapture) or longitudinal observations " *
                  "(tag, spatial unit, and time).")
        end

        releases = Int[]
        recaps = Int[]
        ks = Int[]
        groups = Int[]
        covariates = Float64[]

        grp_col = hasproperty(df, :group) ? :group :
            (hasproperty(df, :groups) ? :groups : nothing)
        cov_col = hasproperty(df, :individual_covariate) ? :individual_covariate :
            (hasproperty(df, :covariate) ? :covariate : nothing)

        _t_val(v) = v isa Dates.TimeType ?
            Float64(Dates.datetime2epochms(DateTime(v))) / 86400000.0 : Float64(v)

        gdf = groupby(df, tag_col)
        for sub_df in gdf
            if nrow(sub_df) < 2
                continue
            end
            sub_sorted = sort(sub_df, [order(time_col)])

            for i in 1:(nrow(sub_sorted) - 1)
                row_rel = sub_sorted[i, :]
                row_rec = sub_sorted[i+1, :]
                push!(releases, Int(getproperty(row_rel, s_col)))
                push!(recaps, Int(getproperty(row_rec, s_col)))
                t_rel = _t_val(getproperty(row_rel, time_col))
                t_rec = _t_val(getproperty(row_rec, time_col))
                push!(ks, max(1, round(Int, t_rec - t_rel)))
                grp_val = !isnothing(grp_col) ? Int(getproperty(row_rel, grp_col)) : 1
                push!(groups, grp_val)
                cov_val = !isnothing(cov_col) ? Float64(getproperty(row_rel, cov_col)) : 0.0
                push!(covariates, cov_val)
            end
        end

        n_events = length(releases)
        mat = if n_events > 0
            hcat(Float64.(releases), Float64.(recaps), Float64.(ks), covariates)
        else
            Matrix{Float64}(undef, 0, 4)
        end
        G_val = isnothing(mark_recapture_G) ? (isempty(groups) ? 1 : maximum(groups)) :
            Int(mark_recapture_G)
        max_k_val = isempty(ks) ? 1 : maximum(ks)
        return TelemetryData(releases, recaps, ks, groups, covariates, G_val, max_k_val, mat)
    else
        error("Unsupported format for mark_recapture_data: $(typeof(telemetry_input)). " *
              "Expected DataFrame or Matrix.")
    end
end

"""
    simulate_correlated_density_vector(habitat_prob, rho_target, log_mu, sigma_resid;
                                        rng=Random.GLOBAL_RNG)

Simulates a spatial population density vector correlated with habitat suitability index (HSI).

# Arguments
- `habitat_prob::AbstractVector{<:Real}`: Habitat suitability probabilities for spatial units.
- `rho_target::Real`: Desired correlation between habitat suitability and log-density.
- `log_mu::Real`: Mean of the log-density field.
- `sigma_resid::Real`: Marginal residual standard deviation of log-density.
- `rng::AbstractRNG`: Random number generator.

# Returns
- `Vector{Float64}`: Simulated positive population density values.
"""
function simulate_correlated_density_vector(
    habitat_prob::AbstractVector{<:Real},
    rho_target::Real,
    log_mu::Real,
    sigma_resid::Real;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)
    n = length(habitat_prob)
    p_std = (habitat_prob .- mean(habitat_prob)) ./ (std(habitat_prob) + 1e-9)
    epsilon = randn(rng, n)
    log_n_signal = (rho_target .* p_std) .+ (sqrt(max(0.0, 1.0 - rho_target^2)) .* epsilon)
    log_n = log_mu .+ (log_n_signal .* sigma_resid)
    return exp.(log_n)
end

"""
    generate_ADR_simulation_bundle(domain_size, n_units, n_years, n_marks;
                                   area_method=:hexagonal, rng=Random.GLOBAL_RNG)

Generates a complete synthetic simulation bundle for joint Advection-Diffusion-Reaction (ADR)
population density surveys and individual mark-recapture telemetry.

# Arguments
- `domain_size::Real`: Spatial bounding box size (e.g. 1000.0 km).
- `n_units::Int`: Target number of spatial partitioning units.
- `n_years::Int`: Number of discrete temporal observation years.
- `n_marks::Int`: Number of tagged individuals released in mark-recapture telemetry.
- `area_method::Symbol`: Spatial tessellation method (`:hexagonal`, `:cvt`, `:voronoi`, `:grid`).
- `rng::AbstractRNG`: Random number generator instance.

# Returns
- `NamedTuple`:
  - `data::DataFrame`: Long-format observation table with columns `(:density, :unit_id, ...)`.
  - `telemetry_data::DataFrame`: Long-format telemetry observations with columns `(:tagid, ...)`.
  - `au::NamedTuple`: Spatial areal units object containing boundaries, centroids, and `W`.
  - `n_spatial::Int`: Number of spatial units.
  - `n_years::Int`: Number of temporal survey periods.
"""
function generate_ADR_simulation_bundle(
    domain_size::Real,
    n_units::Int,
    n_years::Int,
    n_marks::Int;
    area_method::Symbol = :cvt,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)
    # 1. Generate spatial partitioning using MovementAnalysis's partitioning engine
    s_x_init = rand(rng, 1000) .* domain_size
    s_y_init = rand(rng, 1000) .* domain_size
    
    au = assign_spatial_units(
        s_x_init, s_y_init;
        area_method = area_method,
        target_units = n_units,
        exact_units = false
    )
    
    centroids = au.centroids
    n_spatial = length(centroids)
    cent_x = [c[1] for c in centroids]
    cent_y = [c[2] for c in centroids]

    # 2. Dynamic Spatiotemporal Habitat Suitability Index (HSI)
    habitat_p = zeros(Float64, n_spatial, n_years)
    center_x = domain_size / 2.0
    center_y = domain_size / 2.0
    
    p_init = [exp(-sqrt((cent_x[i] - center_x)^2 + (cent_y[i] - center_y)^2) / (domain_size
      / 3.0)) for i in 1:n_spatial]
    habitat_p[:, 1] = p_init ./ (maximum(p_init) + 1e-9)
    
    for t in 2:n_years
        noise = randn(rng, n_spatial) .* 0.05
        habitat_p[:, t] = min.(1.0, max.(0.01, habitat_p[:, t-1] .+ noise))
    end

    # 3. Simulate Spatiotemporal Density Observations
    density_n = zeros(Float64, n_spatial, n_years)
    for t in 1:n_years
        density_n[:, t] = simulate_correlated_density_vector(habitat_p[:, t], 0.75, 3.5,
          0.6; rng=rng)
    end

    # 4. Simulate Individual Mark-Recapture Telemetry Transitions
    telemetry_df = DataFrame(
        tagid = Int[],
        s_idx = Int[],
        time = Float64[],
        tag = Int[],
        individual_covariate = Float64[]
    )
    
    for i in 1:n_marks
        release_unit = rand(rng, 1:n_spatial)
        year_start = rand(rng, 1:max(1, n_years - 2))
        time_steps = rand(rng, 1:2)
        
        # Simulate physical dispersal distance
        d_travel = rand(rng, Uniform(domain_size / 20.0, domain_size / 2.5)) * time_steps
        dists = [abs(sqrt((cent_x[release_unit] - cent_x[j])^2 + (cent_y[release_unit] -
          cent_y[j])^2) - d_travel) for j in 1:n_spatial]
        recapture_unit = argmin(dists)
        
        ind_cov = randn(rng)

        # Release event (tag = 0)
        push!(telemetry_df, (tagid=i, s_idx=release_unit, time=Float64(year_start), tag=0,
          individual_covariate=ind_cov))
        # Recapture event (tag = 1)
        push!(telemetry_df, (tagid=i, s_idx=recapture_unit, time=Float64(year_start +
          time_steps), tag=1, individual_covariate=ind_cov))
    end

    # 5. Build Long-Format Survey DataFrame
    df = DataFrame()
    for t in 1:n_years
        temp_df = DataFrame(
            density = density_n[:, t],
            unit_id = 1:n_spatial,
            s_idx = 1:n_spatial,
            year = t,
            time_idx = t,
            s_x = cent_x,
            s_y = cent_y,
            habitat_p = habitat_p[:, t]
        )
        append!(df, temp_df)
    end

    return (
        data = df,
        telemetry_data = telemetry_df,
        au = au,
        n_spatial = n_spatial,
        n_years = n_years
    )
end

"""
    compute_velocity_field(prob_vec, grid_dim, strength; mode=:exponential)

Computes an advection velocity vector field from a spatial gradient of habitat suitability.

# Arguments
- `prob_vec::AbstractVector{<:Real}`: Spatial habitat suitability values.
- `grid_dim::Int`: Dimension of regular grid (for lattice geometries).
- `strength::Real`: Scaling factor for advection velocity.
- `mode::Symbol`: `:exponential` (relative gradient) or `:linear` (absolute gradient).

# Returns
- `NamedTuple`: `(vx = vec(vx), vy = vec(vy))` velocity components.
"""
function compute_velocity_field(
    prob_vec::AbstractVector{<:Real},
    grid_dim::Int,
    strength::Real;
    mode::Symbol = :exponential
)
    grid = reshape(prob_vec, grid_dim, grid_dim)
    rows, cols = size(grid)
    vx = zeros(Float64, rows, cols)
    vy = zeros(Float64, rows, cols)
    eps_val = 1e-6

    for r in 1:rows, c in 1:cols
        gx = (c == 1) ? (grid[r, 2] - grid[r, 1]) :
             (c == cols ? (grid[r, cols] - grid[r, cols-1]) : ((grid[r, c+1] - grid[r, c-1]) / 2.0))
        gy = (r == 1) ? (grid[2, c] - grid[1, c]) :
             (r == rows ? (grid[rows, c] - grid[rows-1, c]) : ((grid[r+1, c] - grid[r-1, c]) / 2.0))

        if mode == :exponential
            denom_x = sqrt(grid[r, c] * (grid[r, c] + eps_val))
            denom_y = sqrt(grid[r, c] * (grid[r, c] + eps_val))
            vx[r, c] = (gx / denom_x) * strength
            vy[r, c] = (gy / denom_y) * strength
        else
            vx[r, c] = gx * strength
            vy[r, c] = gy * strength
        end
    end
    return (vx = vec(vx), vy = vec(vy))
end

"""
    calculate_multistep_transition(Gamma_base::AbstractMatrix{T}, steps::Int) where T <: Real

Calculates the multi-step dispersal transition matrix via Markov matrix exponentiation:
\$\\Gamma^{(k)} = \\Gamma^k\$.
"""
function calculate_multistep_transition(Gamma_base::AbstractMatrix{T}, steps::Int) where T <: Real
    n_spatial = size(Gamma_base, 1)
    if steps < 1
        return Matrix{T}(I, n_spatial, n_spatial)
    end
    
    G_step = copy(Gamma_base)
    for i in 1:n_spatial
        row_sum = sum(G_step[i, :])
        if row_sum > 0
            G_step[i, :] ./= row_sum
        end
    end
    
    Gamma_k = G_step ^ steps
    
    for i in 1:n_spatial
        final_sum = sum(Gamma_k[i, :])
        if final_sum > 0
            Gamma_k[i, :] ./= final_sum
        end
    end
    
    return Gamma_k
end

"""
    simulate_posterior_trajectories(Gamma_base, start_units, n_steps, au_context;
                                   rho_persistence=0.0, rng=Random.GLOBAL_RNG)

Simulates individual movement trajectories from a stationary transition matrix \$\\Gamma\$,
with optional directional persistence (Correlated Random Walk / CRW).

# Arguments
- `Gamma_base::AbstractMatrix`: Transition probability matrix (\$S \\times S\$).
- `start_units::Vector{Int}`: Starting spatial unit indices for each tracked individual.
- `n_steps::Int`: Number of discrete forward movement steps.
- `au_context::NamedTuple`: Spatial units object containing `centroids`.
- `rho_persistence::Real`: Directional persistence parameter (\$\\rho \\ge 0\$).
- `rng::AbstractRNG`: Random number generator.

# Returns
- `Matrix{Int}`: Matrix of shape `(n_indiv, n_steps + 1)` with unit indices visited over time.
"""
function simulate_posterior_trajectories(
    Gamma_base::AbstractMatrix{T},
    start_units::Vector{Int},
    n_steps::Int,
    au_context::NamedTuple;
    rho_persistence::Real = 0.0,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
) where T <: Real
    
    n_indiv = length(start_units)
    n_spatial = size(Gamma_base, 1)
    centroids = if hasproperty(au_context, :centroids)
        au_context.centroids
    elseif hasproperty(au_context, :centroids_km)
        au_context.centroids_km
    elseif hasproperty(au_context, :centroids_lonlat)
        au_context.centroids_lonlat
    else
        throw(ArgumentError(
            "au_context must contain :centroids, :centroids_km, or :centroids_lonlat."
        ))
    end
    paths = zeros(Int, n_indiv, n_steps + 1)
    paths[:, 1] = start_units
    
    G_sampling = copy(Gamma_base)
    for i in 1:n_spatial
        G_sampling[i, :] .= max.(0.0, G_sampling[i, :])
        rs = sum(G_sampling[i, :])
        if rs > 1e-12
            G_sampling[i, :] ./= rs
        else
            G_sampling[i, :] .= 0.0
            G_sampling[i, i] = 1.0
        end
    end
    
    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for t in 2:(n_steps + 1)
            p_row = max.(0.0, vec(G_sampling[curr_node, :]))
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                norm_prev = norm(v_prev)
                if norm_prev > 1e-9
                    for j in 1:n_spatial
                        if j != curr_node && p_row[j] > 0.0
                            v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                            norm_cand = norm(v_cand)
                            if norm_cand > 1e-9
                                cos_theta = dot(v_prev, v_cand) / (norm_prev * norm_cand)
                                p_row[j] *= exp(rho_persistence * cos_theta)
                            end
                        end
                    end
                end
            end
            row_sum = sum(p_row)
            if row_sum > 1e-12
                p_row ./= row_sum
            else
                p_row .= 0.0
                p_row[curr_node] = 1.0
            end
            next_node = rand(rng, Categorical(p_row))
            paths[i, t] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

"""
    simulate_mechanistic_trajectories(Gamma_sequence, start_units, t_start, au_context;
                                      rho_persistence=0.0, n_years_sim=1,
                                      rng=Random.GLOBAL_RNG)

Simulates individual movement paths through a dynamic, non-stationary environment where transition
kernels \$\\Gamma_t\$ vary over time.
"""
function simulate_mechanistic_trajectories(
    Gamma_sequence::Vector{<:AbstractMatrix{T}},
    start_units::Vector{Int},
    t_start::Int,
    au_context::NamedTuple;
    rho_persistence::Real = 0.0,
    n_years_sim::Int = 1,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
) where T <: Real

    n_indiv = length(start_units)
    n_spatial = size(Gamma_sequence[1], 1)
    centroids = if hasproperty(au_context, :centroids)
        au_context.centroids
    elseif hasproperty(au_context, :centroids_km)
        au_context.centroids_km
    elseif hasproperty(au_context, :centroids_lonlat)
        au_context.centroids_lonlat
    else
        throw(ArgumentError(
            "au_context must contain :centroids, :centroids_km, or :centroids_lonlat."
        ))
    end
    max_available_time = length(Gamma_sequence)
    
    actual_steps = n_years_sim
    if t_start + n_years_sim > max_available_time
        actual_steps = max_available_time - t_start
    end

    paths = zeros(Int, n_indiv, actual_steps + 1)
    paths[:, 1] = start_units
    
    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for step in 1:actual_steps
            t_current = t_start + step - 1
            if t_current > max_available_time
                break
            end
            Gamma_t = Gamma_sequence[t_current]
            p_row = max.(0.0, vec(Gamma_t[curr_node, :]))
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                norm_prev = norm(v_prev)
                if norm_prev > 1e-9
                    for j in 1:n_spatial
                        if j != curr_node && p_row[j] > 0.0
                            v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                            norm_cand = norm(v_cand)
                            if norm_cand > 1e-9
                                cos_theta = dot(v_prev, v_cand) / (norm_prev * norm_cand)
                                p_row[j] *= exp(rho_persistence * cos_theta)
                            end
                        end
                    end
                end
            end
            row_sum = sum(p_row)
            if row_sum > 1e-12
                p_row ./= row_sum
            else
                p_row .= 0.0
                p_row[curr_node] = 1.0
            end
            next_node = rand(rng, Categorical(p_row))
            paths[i, step + 1] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

"""
    compute_suitability_transition_kernel(
        suitability_vec, W;
        sensitivity=1.0, diffusion_weight=0.1, relationship=:exponential
    )

Generates a spatial Markov transition probability kernel based on local habitat
suitability index (HSI) differences and network topology:
- `:exponential` (default): \$\\Gamma_{ij} \\propto \\exp(\\beta \\cdot \\text{HSI}_j) + D_{\\text{weight}}\$
- `:linear`: \$\\Gamma_{ij} \\propto (1 + \\beta \\cdot \\text{HSI}_j) + D_{\\text{weight}}\$
- `:logistic`: \$\\Gamma_{ij} \\propto \\frac{1}{1 + \\exp(-\\beta \\cdot \\text{HSI}_j)} + D_{\\text{weight}}\$

# Arguments
- `suitability_vec::AbstractVector{<:Real}`: Habitat Suitability Index (HSI) values (\$S\$).
- `W::AbstractMatrix`: Spatial adjacency weights matrix (\$S \\times S\$).
- `sensitivity::Real`: Advective response sensitivity (\$\\beta \\ge 0\$).
- `diffusion_weight::Real`: Baseline isotropic dispersal weight (\$D_{\\text{weight}} \\ge 0\$).
- `relationship::Symbol`: Functional relationship (`:exponential`, `:linear`, `:logistic`).

# Returns
- `SparseMatrixCSC{Float64, Int}`: Row-stochastic Markov transition probability matrix.
"""
function compute_suitability_transition_kernel(
    suitability_vec::AbstractVector{T},
    W::AbstractMatrix;
    sensitivity::Real = 1.0,
    diffusion_weight::Real = 0.1,
    relationship::Symbol = :exponential
) where T <: Real

    n_spatial = length(suitability_vec)
    Gamma = spzeros(T, n_spatial, n_spatial)
    rows = rowvals(W)
    vals = nonzeros(W)
    
    for i in 1:n_spatial
        h_i = suitability_vec[i]
        bias_i = if relationship == :exponential
            exp(sensitivity * h_i)
        elseif relationship == :logistic
            1.0 / (1.0 + exp(-sensitivity * h_i))
        else # :linear
            max(0.01, 1.0 + sensitivity * h_i)
        end
        Gamma[i, i] = bias_i

        for j_idx in nzrange(W, i)
            j = rows[j_idx]
            if i == j
                continue
            end
            h_j = suitability_vec[j]
            suitability_bias = if relationship == :exponential
                exp(sensitivity * h_j)
            elseif relationship == :logistic
                1.0 / (1.0 + exp(-sensitivity * h_j))
            else # :linear
                max(0.01, 1.0 + sensitivity * h_j)
            end
            edge_weight = vals[j_idx]
            Gamma[i, j] = (suitability_bias + diffusion_weight) * edge_weight
        end
    end
    
    for i in 1:n_spatial
        row_sum = sum(Gamma[i, :])
        if row_sum > 1e-12
            Gamma[i, :] ./= row_sum
        else
            Gamma[i, i] = 1.0
        end
    end
    return Gamma
end

"""
    calculate_regional_connectivity(Gamma, strata_definition)

Aggregates fine-scale spatial unit transition probabilities into a macro-regional
connectivity matrix.
"""
function calculate_regional_connectivity(Gamma::AbstractMatrix, strata_definition::AbstractVector)
    n_units = size(Gamma, 1)
    unique_strata = unique(strata_definition)
    n_strata = length(unique_strata)
    strata_map = Dict(s => i for (i, s) in enumerate(unique_strata))
    
    C = zeros(Float64, n_strata, n_strata)

    for i in 1:n_units
        from_stratum_idx = strata_map[strata_definition[i]]
        for j in 1:n_units
            to_stratum_idx = strata_map[strata_definition[j]]
            C[from_stratum_idx, to_stratum_idx] += Gamma[i, j]
        end
    end

    for r in 1:n_strata
        row_sum = sum(C[r, :])
        if row_sum > 0
            C[r, :] ./= row_sum
        end
    end

    return C
end

"""
    plot_ad_ratio_distribution(advection_field, diffusion_field; mode=:plots)

Generates a diagnostic histogram of the Advection-to-Diffusion ratio across spatial units.
Supports `mode=:plots` (default Plots.Plot) or `mode=:leaflet` / `mode=:html` (interactive HTML).
"""
function plot_ad_ratio_distribution(advection_field::AbstractVector{<:Real},
  diffusion_field::AbstractVector{<:Real}; mode::Symbol=:plots)
    if mode == :leaflet || mode == :html
        return leaflet_ad_ratio_distribution(advection_field, diffusion_field)
    end
    ratios = advection_field ./ (mean(diffusion_field) .+ 1e-6)
    plt = Plots.histogram(
        ratios, bins=25, title="Advection-to-Diffusion Ratio (Péclet-like)",
        xlabel="Ratio (Advection / Diffusion)", ylabel="Frequency",
        label="Spatial Units", color=:plum, linecolor=:white
    )
    Plots.vline!(plt, [1.0], color=:red, linestyle=:dash, linewidth=2.0, label="Equilibrium
      Threshold")
    return plt
end

"""
    synthesize_adr_results(chain_or_res, sim_data, vel_vectors; au=sim_data.au)

Performs comprehensive post-processing, parameter extraction, and diagnostic visualization
for a fitted Advection-Diffusion-Reaction movement model.

# Returns
- `NamedTuple`: Containing:
  - `parameters::NamedTuple`: Posterior mean estimates for velocity, diffusion, sigma.
  - `propagator_matrix::Matrix{Float64}`: Reconstructed forward propagator \$M_{\\text{prop}}\$.
  - `transition_matrix::Matrix{Float64}`: Reconstructed one-step Markov transition matrix.
  - `plots::NamedTuple`: Rendered diagnostic plots (`regional_connectivity`, etc.).
"""
function synthesize_adr_results(
    chain_or_res,
    sim_data::NamedTuple,
    vel_vectors::NamedTuple;
    au = sim_data.au
)
    # 1. Extract Posterior Parameter Estimates
    S_strength_mean = if hasproperty(chain_or_res, :effects) &&
      hasproperty(chain_or_res.effects, :velocity)
        chain_or_res.effects.velocity.mean
    elseif hasproperty(chain_or_res, :value) && :velocity_movement in names(chain_or_res,
      :parameters)
        mean(chain_or_res[:velocity_movement])
    else
        1.0
    end

    D_coeff_mean = if hasproperty(chain_or_res, :effects) &&
      hasproperty(chain_or_res.effects, :diffusion)
        chain_or_res.effects.diffusion.mean
    elseif hasproperty(chain_or_res, :value) && :diffusion_movement in names(chain_or_res,
      :parameters)
        mean(chain_or_res[:diffusion_movement])
    else
        0.5
    end

    # 2. Reconstruct Spatial Graph Operators
    W = au.W
    n_spatial = size(W, 1)
    
    # Graph Laplacian L = D_deg - W
    deg = vec(sum(W, dims=2))
    L = spdiagm(0 => deg) - W
    
    # Directed Advection Operator A
    W_dir = tril(W, -1)
    out_deg = vec(sum(W_dir, dims=2))
    D_inv = spdiagm(0 => 1.0 ./ (out_deg .+ 1e-9))
    A = D_inv * W_dir
    
    # Propagator M_prop = I - v*A - D*L
    # NOTE: Inverting M_prop to get a transition matrix assumes that the propagator 
    # defines a solvable linear system. However, for advection-diffusion on bounded domains, 
    # this may not be the intended transition kernel. The transition matrix should emerge 
    # directly from discretization, not from inverting a propagator.
    # Fix: Verify this is the intended mathematical model, or construct 
    # Gamma_mean directly from the advection/diffusion operators without inversion.

    M_prop_mean = Matrix(I(n_spatial) - (S_strength_mean .* A) - (D_coeff_mean .* L))
    Gamma_mean = inv(M_prop_mean)
    
    # Normalize rows of Gamma
    for i in 1:n_spatial
        rs = sum(Gamma_mean[i, :])
        if rs > 0
            Gamma_mean[i, :] ./= rs
        else
            Gamma_mean[i, i] = 1.0
        end
    end

    # 3. Visualization 1: Regional Connectivity Matrix
    cents_x = [c[1] for c in au.centroids]
    mid_x = (minimum(cents_x) + maximum(cents_x)) / 2.0
    strata = [x > mid_x ? "East" : "West" for x in cents_x]
    conn_mat = calculate_regional_connectivity(Gamma_mean, strata)
    
    plt_conn = Plots.heatmap(
        ["West", "East"], ["West", "East"], conn_mat,
        title = "Regional Transfer Probability Matrix",
        xlabel = "To Region", ylabel = "From Region", color = :viridis, clims = (0, 1)
    )

    # 4. Visualization 2: Advection-to-Diffusion Ratio Distribution
    advection_magnitude = sqrt.(vel_vectors.vx.^2 .+ vel_vectors.vy.^2) .* S_strength_mean
    diffusion_magnitude = fill(D_coeff_mean, n_spatial)
    plt_ad = plot_ad_ratio_distribution(advection_magnitude, diffusion_magnitude)

    # 5. Visualization 3: Path Simulation with & without Persistence
    random_starts = rand(1:n_spatial, min(4, n_spatial))
    n_sim_steps = 15
    paths_standard = simulate_posterior_trajectories(Gamma_mean, random_starts, n_sim_steps,
      au; rho_persistence=0.0)
    paths_persistent = simulate_posterior_trajectories(Gamma_mean, random_starts,
      n_sim_steps, au; rho_persistence=2.0)
    
    plt_comp = Plots.plot(layout=(1, 2), size=(900, 450), aspect_ratio=:equal)
    render_paths!(plt_comp[1], paths_standard; au=au, color=:crimson, lw=1.5, labels=["Mark
      $i" for i in 1:length(random_starts)])
    Plots.plot!(plt_comp[1], title="Standard Markovian Random Walk")
    
    render_paths!(plt_comp[2], paths_persistent; au=au, color=:navy, lw=1.5, labels=["Mark
      $i" for i in 1:length(random_starts)])
    Plots.plot!(plt_comp[2], title="Persistent Movement (\\rho=2.0)")

    # 6. Visualization 4: Dynamic Multi-Year Projections
    gamma_seq = [Gamma_mean for _ in 1:sim_data.n_years]
    dynamic_paths = simulate_mechanistic_trajectories(gamma_seq, random_starts, 1, au;
      rho_persistence=1.5, n_years_sim=min(4, sim_data.n_years - 1))
    
    plt_dyn = Plots.plot(aspect_ratio=:equal, title="Multi-Year Mechanistic Path
      Projections", legend=:outerright)
    render_paths!(plt_dyn, dynamic_paths; au=au, color=:darkgreen, lw=1.5)

    # 7. Visualization 5: Comprehensive Interactive Leaflet Dashboard
    dash_html = leaflet_movement_dashboard(
        (au = au, transition_matrix = Gamma_mean, opts = (hsi = nothing,),
         d_val = D_coeff_mean, v_val = S_strength_mean),
        paths_persistent;
        strata = strata,
        title = "ADR Movement Estimation Dashboard"
    )

    plots_bundle = (
        regional_connectivity = plt_conn,
        ad_ratio = plt_ad,
        paths_comparison = plt_comp,
        dynamic_paths = plt_dyn,
        leaflet_dashboard = dash_html
    )

    params_bundle = (
        velocity_mean = S_strength_mean,
        diffusion_mean = D_coeff_mean
    )

    return (
        parameters = params_bundle,
        propagator_matrix = M_prop_mean,
        transition_matrix = Gamma_mean,
        plots = plots_bundle
    )
end

# ==============================================================================
# SECTION: GENERAL TELEMETRY & POSTERIOR TRANSITION KERNEL UTILITIES
# ==============================================================================

"""
    validate_telemetry(df::DataFrame) -> nothing

Validates that a telemetry / mark-recapture DataFrame conforms to MovementAnalysis movement
schema requirements and logical consistency constraints.

# Validation Checks
1. Presence of required identifier, coordinate, and detection type columns:
   `:tagid`, `:lon`, `:lat`, `:tag`, and either `:timestamp` or `:time`.
2. Valid detection codes: `:tag ≥ 0` (where `0` indicates initial release/marking,
   and `n ≥ 1` indicates the \$n\$-th recapture event).
3. Individual trajectory consistency: exactly one release event (`tag == 0`) per
   `:tagid`, and all recapture events chronologically follow the initial release.

Throws an informative `ArgumentError` if any check fails.
"""
function validate_telemetry(df::DataFrame)
    time_col = if hasproperty(df, :time)
        :time
    elseif hasproperty(df, :timestamp)
        :timestamp
    else
        nothing
    end

    if isnothing(time_col)
        throw(ArgumentError("Telemetry DataFrame must contain either :time or :timestamp."))
    end

    required = [:tagid, :lon, :lat, :tag]
    missing_cols = filter(c -> !hasproperty(df, c), required)
    if !isempty(missing_cols)
        throw(ArgumentError("Telemetry DataFrame missing required columns: $(missing_cols)"))
    end

    if any(df.tag .< 0)
        throw(ArgumentError("Column :tag must be ≥ 0 (0 = initial mark, n ≥ 1 = recaptures)."))
    end

    for sub in groupby(df, :tagid)
        release_rows = filter(r -> r.tag == 0, sub)
        if nrow(release_rows) != 1
            tid = first(sub.tagid)
            throw(ArgumentError("tagid=$(tid): expected exactly one release event (tag=0), " *
                                "found $(nrow(release_rows))."))
        end
        release_time = Float64(release_rows[1, time_col])
        recapture_times = Float64.(filter(r -> r.tag > 0, sub)[!, time_col])
        if !all(recapture_times .> release_time)
            tid = first(sub.tagid)
            throw(ArgumentError("tagid=$(tid): one or more recaptures precede or coincide with release."))
        end
    end
    return nothing
end

"""
    map_point_to_units(df::DataFrame, au::NamedTuple)::DataFrame

Projects observation coordinates `(:lon, :lat)` in a telemetry DataFrame to
their nearest active, navigable spatial unit centroids, returning a copy of `df`
with an appended `:s_idx` column.

When `au` contains `:land_mask` or `:W`, mapping is restricted strictly to active
water units (`!land_mask && degree(W) > 0`), ensuring observations near coastlines
or outside the survey domain snap to valid water units rather than land barriers.

# Mathematical Formulation
For each observation \$\\mathbf{x}_i = (\\text{lon}_i, \\text{lat}_i)\$:
\$s_i = \\arg\\min_{j \\in \\mathcal{M}} \\|\\mathbf{x}_i - \\mathbf{c}_j\\|_2\$
where \$\\mathcal{M} = \\{ j \\mid \\neg \\text{land\\_mask}[j] \\land \\text{deg}(j) > 0 \\}\$.
"""
function map_point_to_units(
    df::DataFrame, au::NamedTuple;
    target_col::Union{Symbol, AbstractString, Nothing} = nothing
)::DataFrame
    land_mask = hasproperty(au, :land_mask) ? au.land_mask : nothing
    W = hasproperty(au, :W) ? au.W : nothing
    if hasproperty(au, :centroids_km) &&
       hasproperty(au, :center_lon) &&
       hasproperty(au, :center_lat)
        return map_point_to_units(
            df, au.centroids_km, au.center_lon, au.center_lat;
            target_col = target_col, land_mask = land_mask, W = W
        )
    end
    cents = hasproperty(au, :centroids) ? au.centroids :
            (hasproperty(au, :centroids_lonlat) ? au.centroids_lonlat : nothing)
    if cents === nothing
        throw(ArgumentError("au must contain :centroids or :centroids_km."))
    end
    S = length(cents)
    active_mask = trues(S)
    if land_mask !== nothing
        active_mask .&= .!land_mask
    end
    if W !== nothing
        deg = vec(sum(W, dims=2))
        active_mask .&= (deg .> 0.0)
    end
    active_indices = findall(active_mask)
    if isempty(active_indices)
        active_indices = collect(1:S)
    end
    active_cents = cents[active_indices]
    col_x, col_y = _detect_xy_columns(df)
    mapped_local = map_to_units(df[!, col_x], df[!, col_y], active_cents)
    out = copy(df)
    mapped_s = [active_indices[k] for k in mapped_local]
    out[!, :s_idx] = mapped_s
    if !isnothing(target_col)
        out[!, Symbol(target_col)] = mapped_s
    end
    return out
end

"""
    time_steps_between(t_release::Real, t_recapture::Real)::Int

Converts continuous decimal-year temporal difference to discrete integer time steps.
Enforces a minimum interval of 1 step: \$\\Delta t = \\max(1, \\lfloor t_{\\text{rec}} - t_{\\text{rel}} \\rceil)\$.
"""
function time_steps_between(t_release::Real, t_recapture::Real)::Int
    return max(1, round(Int, t_recapture - t_release))
end

"""
    extract_scalar_param(chain, param_prefix::String)::Vector{Float64}

Extracts posterior samples for a scalar parameter identified by `param_prefix`
(e.g., `"velocity"` or `"diffusion"`), resolving variable name suffixes dynamically
across MCMC chains, FlexiChains, or DataFrames.

# Suffix Resolution Precedence
1. Suffix matching formula indices: `"\$(param_prefix)_s_idx_t_idx"`
2. Suffix matching module name: `"\$(param_prefix)_movement"`
3. Exact parameter name: `param_prefix`
4. Any sampled parameter starting with `param_prefix`
"""
function extract_scalar_param(chain, param_prefix::String)::Vector{Float64}
    if chain isa NamedTuple || chain isa AbstractDict
        for k in keys(chain)
            sk = string(k)
            if sk == param_prefix || sk == "$(param_prefix)_s_idx_t_idx" ||
               sk == "$(param_prefix)_movement" || startswith(sk, param_prefix)
                val = chain[k]
                return isa(val, AbstractVector) ? Float64.(val) : [Float64(val)]
            end
        end
    end

    p_names = if occursin("FlexiChain", string(typeof(chain)))
        string.(keys(chain))
    elseif hasmethod(names, Tuple{typeof(chain), Symbol})
        string.(names(chain, :parameters))
    elseif hasmethod(names, Tuple{typeof(chain)})
        string.(names(chain))
    elseif hasmethod(keys, Tuple{typeof(chain)})
        string.(keys(chain))
    else
        String[]
    end

    target = ""
    for candidate in [
        "$(param_prefix)_s_idx_t_idx",
        "$(param_prefix)_movement",
        param_prefix
    ]
        matched = _find_parameter(p_names, candidate, 1, false)
        if !isempty(matched)
            target = matched
            break
        end
    end

    if isempty(target)
        idx = findfirst(n -> startswith(n, param_prefix), p_names)
        if !isnothing(idx)
            target = p_names[idx]
        end
    end

    if isempty(target)
        error("Could not find parameter matching prefix '$(param_prefix)' in MCMC chain.")
    end

    return get_params_vector(chain, target, 1)[:, 1]
end

const _extract_scalar_param = extract_scalar_param

"""
    reconstruct_posterior_kernel(
        chain,
        W::AbstractMatrix;
        hsi::Union{Nothing, AbstractVector}=nothing,
        relationship::Symbol=:exponential
    )::Matrix{Float64}

Reconstructs the posterior-mean row-stochastic Markov transition matrix \$\\bar{\\mathbf{\\Gamma}}\$
across all MCMC samples.

# Mathematical Formulation
For each posterior sample \$s \\in \\{1, \\dots, N_s\\}\$:
\$\\mathbf{M}_s = \\mathbf{I} - v_s \\mathbf{A} - D_s \\mathbf{L}\$
\$\\mathbf{\\Gamma}_s = \\text{row\\_normalize}\\left(\\mathbf{M}_s^{-1}\\right)\$
\$\\bar{\\mathbf{\\Gamma}} = \\frac{1}{N_s} \\sum_{s=1}^{N_s} \\mathbf{\\Gamma}_s\$

where \$\\mathbf{L} = \\text{diag}(\\mathbf{W} \\mathbf{1}) - \\mathbf{W}\$ is the graph Laplacian,
and \$\\mathbf{A}\$ is the directed advection operator derived from \$\\nabla \\text{HSI}\$
(or spatial topology).

# Arguments
- `chain`: MCMC posterior chain (Turing `Chains` or `FlexiChain`).
- `W::AbstractMatrix`: Adjacency / spatial weights matrix (\$S \\times S\$).
- `hsi::Union{Nothing, AbstractVector}`: Optional Habitat Suitability Index vector of length \$S\$.
- `relationship::Symbol`: Functional form for HSI gradient: `:exponential`, `:logistic`, `:linear`.

# Returns
- `Matrix{Float64}`: Row-stochastic \$S \\times S\$ Markov transition probability matrix.
"""
function reconstruct_posterior_kernel(
    chain,
    W::AbstractMatrix;
    hsi::Union{Nothing, AbstractVector}=nothing,
    relationship::Symbol=:exponential
)::Matrix{Float64}
    S = size(W, 1)

    deg = vec(sum(W, dims=2))
    L = spdiagm(0 => deg) - W

    if !isnothing(hsi)
        hsi_vec = Float64.(hsi)
        W_dir = spzeros(Float64, S, S)
        rows = rowvals(W)
        vals = nonzeros(W)
        for i in 1:S, j_idx in nzrange(W, i)
            j = rows[j_idx]
            i == j && continue
            dh = hsi_vec[j] - hsi_vec[i]
            dh > 0.0 || continue
            W_dir[i, j] = if relationship == :exponential
                vals[j_idx] * exp(dh)
            elseif relationship == :logistic
                vals[j_idx] / (1.0 + exp(-4.0 * dh))
            else
                vals[j_idx] * dh
            end
        end
        out_deg = vec(sum(W_dir, dims=2))
        D_inv = spdiagm(0 => [od > 1e-12 ? 1.0 / od : 0.0 for od in out_deg])
        A_base = Matrix(D_inv * W_dir)
    else
        W_dir = tril(W, -1)
        out_deg = vec(sum(W_dir, dims=2))
        D_inv = spdiagm(0 => 1.0 ./ (out_deg .+ 1e-9))
        A_base = Matrix(D_inv * W_dir)
    end

    L_dense = Matrix(L)
    I_S = Matrix{Float64}(I, S, S)

    v_samps = extract_scalar_param(chain, "velocity")
    d_samps = extract_scalar_param(chain, "diffusion")
    n_s = length(v_samps)

    Gamma_acc = zeros(Float64, S, S)
    for s in 1:n_s
        M_prop = I_S .- (v_samps[s] .* A_base) .- (d_samps[s] .* L_dense)
        Gamma_s = inv(M_prop)
        for i in 1:S
            Gamma_s[i, :] .= max.(0.0, Gamma_s[i, :])
            rs = sum(Gamma_s[i, :])
            if rs > 1e-12
                Gamma_s[i, :] ./= rs
            else
                Gamma_s[i, :] .= 0.0
                Gamma_s[i, i] = 1.0
            end
        end
        Gamma_acc .+= Gamma_s
    end
    Gamma_out = Gamma_acc ./ max(1, n_s)
    for i in 1:S
        Gamma_out[i, :] .= max.(0.0, Gamma_out[i, :])
        rs = sum(Gamma_out[i, :])
        if rs > 1e-12
            Gamma_out[i, :] ./= rs
        else
            Gamma_out[i, :] .= 0.0
            Gamma_out[i, i] = 1.0
        end
    end
    return Gamma_out
end

const _reconstruct_posterior_kernel = reconstruct_posterior_kernel

"""
    reshard_hsi_field(
        hsi_raw::AbstractVector{<:Real},
        au_dest::NamedTuple;
        au_src::Union{Nothing, NamedTuple} = nothing,
        hsi_coords::Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
        hsi_area_method::Symbol = :grid,
        domain_bbox::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
    )::Vector{Float64}

Reshards an input Habitat Suitability Index (HSI) vector or surface from an
arbitrary source spatial geometry onto the destination spatial tessellation
`au_dest` (defaulting to `:hexagonal`).

# Mathematical Formulation
Let ``\\mathbf{h}_{\\text{src}} \\in [0, 1]^{S_{\\text{src}}}`` be the source HSI values
and ``\\mathbf{P} \\in \\mathbb{R}^{S_{\\text{dest}} \\times S_{\\text{src}}}`` be the
spatial interpolation / area-overlap transfer matrix between `au_src` and `au_dest`:
```math
\\mathbf{h}_{\\text{dest}} = \\text{clamp}\\left(\\mathbf{P} \\, \\mathbf{h}_{\\text{src}}, \\, 0.0, \\, 1.0\\right)
```

# Arguments
- `hsi_raw::AbstractVector{<:Real}`: Raw source HSI values of length ``S_{\\text{src}}``.
- `au_dest::NamedTuple`: Destination spatial tessellation (e.g., hexagonal or CVT units).
- `au_src::Union{Nothing, NamedTuple}`: Explicit source spatial tessellation with `:centroids` and `:polygons`.
- `hsi_coords::Union{Nothing, Vector{Tuple{Float64, Float64}}}`: Explicit coordinate points for each source HSI unit.
- `hsi_area_method::Symbol`: Source tessellation geometry when constructing from coordinates/bbox (`:grid`, `:cvt`, `:voronoi`, `:hexagonal`). Default: `:grid`.
- `domain_bbox::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}}`: Optional domain bounding box `(min_lon, max_lon, min_lat, max_lat)`.

# Returns
- `Vector{Float64}`: Resharded HSI vector aligned with `au_dest.centroids` (length ``S_{\\text{dest}}``).
"""
function reshard_hsi_field(
    hsi_raw         :: AbstractVector{<:Real},
    au_dest         :: NamedTuple;
    au_src          :: Union{Nothing, NamedTuple} = nothing,
    hsi_coords      :: Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
    hsi_area_method :: Symbol = :grid,
    domain_bbox     :: Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
)::Vector{Float64}
    S_dest = length(au_dest.centroids)
    S_src = length(hsi_raw)

    # 1. If explicit source tessellation is provided
    if !isnothing(au_src) && length(au_src.centroids) == S_src
        hsi_dest = reshard_spatial_field(hsi_raw, au_src, au_dest)
        return clamp.(Vector{Float64}(hsi_dest), 0.0, 1.0)
    end

    # 2. If source and destination sizes match and no distinct geometry/coords specified
    if S_src == S_dest && isnothing(hsi_coords)
        return clamp.(Float64.(hsi_raw), 0.0, 1.0)
    end

    # 3. Determine source points for interpolation
    src_points = if !isnothing(hsi_coords) && length(hsi_coords) == S_src
        hsi_coords
    else
        # Construct synthetic regular grid over destination domain bounding box
        dest_xs = [c[1] for c in au_dest.centroids]
        dest_ys = [c[2] for c in au_dest.centroids]
        min_x = !isnothing(domain_bbox) ? domain_bbox[1] : minimum(dest_xs)
        max_x = !isnothing(domain_bbox) ? domain_bbox[2] : maximum(dest_xs)
        min_y = !isnothing(domain_bbox) ? domain_bbox[3] : minimum(dest_ys)
        max_y = !isnothing(domain_bbox) ? domain_bbox[4] : maximum(dest_ys)

        pad_x = (max_x - min_x) * 0.05
        pad_y = (max_y - min_y) * 0.05
        grid_side = ceil(Int, sqrt(S_src))
        gx = range(min_x - pad_x, max_x + pad_x, length=grid_side)
        gy = range(min_y - pad_y, max_y + pad_y, length=grid_side)
        all_pts = Tuple{Float64, Float64}[(x, y) for x in gx for y in gy]
        all_pts[1:S_src]
    end

    # 4. Inverse Distance Weighting (k-d Tree) from source points to destination centroids
    src_coords_mat = hcat([[c[1], c[2]] for c in src_points]...)
    kdtree = KDTree(src_coords_mat)
    k_nn = min(4, S_src)

    hsi_dest = zeros(Float64, S_dest)
    for j in 1:S_dest
        dest_pt = [au_dest.centroids[j][1], au_dest.centroids[j][2]]
        idxs, dists = knn(kdtree, dest_pt, k_nn, true)
        if any(dists .< 1e-9)
            exact_idx = idxs[findfirst(dists .< 1e-9)]
            hsi_dest[j] = Float64(hsi_raw[exact_idx])
        else
            inv_d = 1.0 ./ dists
            w = inv_d ./ sum(inv_d)
            hsi_dest[j] = sum(w .* hsi_raw[idxs])
        end
    end

    return clamp.(hsi_dest, 0.0, 1.0)
end

# ==============================================================================
# SECTION: SNOW CRAB TELEMETRY & MARK-RECAPTURE DATA INGESTION & PROCESSING
# ==============================================================================

"""
    haversine_distance(lon1::Real, lat1::Real, lon2::Real, lat2::Real; radius::Real=6378137.0)::Float64

Computes the great-circle geodesic distance in metres between two points `(lon1, lat1)`
and `(lon2, lat2)` in decimal degrees on a spherical Earth of radius `radius` (WGS84 mean radius).

# Mathematical Formula
\$\\Delta\\phi = \\text{deg2rad}(\\text{lat}_2 - \\text{lat}_1), \\quad \\Delta\\lambda = \\text{deg2rad}(\\text{lon}_2 - \\text{lon}_1)\$
\$a = \\sin^2\\left(\\frac{\\Delta\\phi}{2}\\right) + \\cos(\\text{lat}_1) \\cos(\\text{lat}_2) \\sin^2\\left(\\frac{\\Delta\\lambda}{2}\\right)\$
\$d = 2 R \\operatorname{atan2}\\left(\\sqrt{a}, \\sqrt{1-a}\\right)\$
"""
function haversine_distance(
    lon1::Real, lat1::Real, lon2::Real, lat2::Real;
    radius::Real=6378137.0
)::Float64
    if !isfinite(lon1) || !isfinite(lat1) || !isfinite(lon2) || !isfinite(lat2)
        return NaN
    end
    phi1 = deg2rad(Float64(lat1))
    phi2 = deg2rad(Float64(lat2))
    dphi = deg2rad(Float64(lat2 - lat1))
    dlam = deg2rad(Float64(lon2 - lon1))

    a = sin(dphi / 2.0)^2 + cos(phi1) * cos(phi2) * sin(dlam / 2.0)^2
    a = clamp(a, 0.0, 1.0)
    return 2.0 * radius * atan(sqrt(a), sqrt(1.0 - a))
end

"""
    tag_to_study_id(tag_id)::Union{Int, Nothing}

Maps a numeric or alphanumeric snow crab tag identifier to its corresponding historical
study index (1 to 57) based on the Scotian Shelf and Gulf historical tagging metadata.

# Arguments
- `tag_id`: Integer, Float, or String representation of the tag number.

# Returns
- `Int`: Study identifier between 1 and 57, or `nothing` if unmapped.
"""
function tag_to_study_id(tag_id)::Union{Int, Nothing}
    if ismissing(tag_id) || isnothing(tag_id)
        return nothing
    end
    raw_str = string(tag_id)
    # Strip leading 'G' or 'g' (e.g. "G1234" -> "1234")
    clean_str = replace(raw_str, r"^[Gg]" => "")

    # Handle prefixed t-tags (e.g. "t1605")
    if startswith(clean_str, "t") || startswith(clean_str, "T")
        t_num = tryparse(Int, clean_str[2:end])
        if !isnothing(t_num)
            if 1605 <= t_num <= 1659
                return 49
            elseif (1217 <= t_num <= 1350) || (1601 <= t_num <= 1603)
                return 52
            elseif 1676 <= t_num <= 1694
                return 56
            elseif 3026 <= t_num <= 3175
                return 57
            end
        end
    end

    # Handle prefixed s-tags (e.g. "s99102")
    if startswith(clean_str, "s") || startswith(clean_str, "S")
        s_num = tryparse(Int, clean_str[2:end])
        if !isnothing(s_num) && 99102 <= s_num <= 99196
            return 57
        end
    end

    # Parse numeric integer tag ID
    id = tryparse(Int, clean_str)
    if isnothing(id)
        f_num = tryparse(Float64, clean_str)
        if !isnothing(f_num)
            id = round(Int, f_num)
        else
            return nothing
        end
    end

    if 0 <= id <= 600
        return 1
    elseif 2350 <= id <= 2399
        return 5
    elseif 1000 <= id <= 1600
        return 6
    elseif (2401 <= id <= 2403) || (2411 <= id <= 2450)
        return 7
    elseif 1601 <= id <= 2349
        return 8
    elseif 6000 <= id <= 6349
        return 9
    elseif 2716 <= id <= 2850
        return 10
    elseif 2456 <= id <= 2715
        return 11
    elseif 5050 <= id <= 5099
        return 12
    elseif 5100 <= id <= 5149
        return 13
    elseif (2851 <= id <= 2900) || (3051 <= id <= 3262) || (3446 <= id <= 3545)
        return 14
    elseif (3263 <= id <= 3444) || (3546 <= id <= 3600)
        return 15
    elseif 4000 <= id <= 4246
        return 16
    elseif 7450 <= id <= 7542
        return 17
    elseif 7543 <= id <= 7699
        return 18
    elseif (4798 <= id <= 4999) || (5250 <= id <= 5520) || (6350 <= id <= 6999)
        return 19
    elseif 5521 <= id <= 5808
        return 20
    elseif 4250 <= id <= 4480
        return 21
    elseif 4482 <= id <= 4797
        return 22
    elseif (7000 <= id <= 7449) || (7700 <= id <= 7999)
        return 23
    elseif (5809 <= id <= 5999) || (8501 <= id <= 8570) || (9229 <= id <= 9649)
        return 24
    elseif 8571 <= id <= 8828
        return 25
    elseif 8829 <= id <= 9228
        return 26
    elseif (10482 <= id <= 10499) || (11082 <= id <= 11334) || (11350 <= id <= 11387)
        return 27
    elseif id == 4301 || (8040 <= id <= 8050) || (8137 <= id <= 8150) ||
           (8201 <= id <= 8298) || (9684 <= id <= 9741) || (9748 <= id <= 9784) ||
           (10254 <= id <= 10406) || (10468 <= id <= 10481) || (11039 <= id <= 11081)
        return 28
    elseif (4248 <= id <= 4249) || (8000 <= id <= 8039) || (8051 <= id <= 8068) ||
           (8070 <= id <= 8136) || (8151 <= id <= 8200) || (10407 <= id <= 10467) ||
           (11000 <= id <= 11036)
        return 29
    elseif (8299 <= id <= 8500) || (9650 <= id <= 9683)
        return 30
    elseif 10032 <= id <= 10253
        return 31
    elseif (9742 <= id <= 9747) || (9785 <= id <= 10031)
        return 32
    elseif (10501 <= id <= 10530) || (10609 <= id <= 10716)
        return 33
    elseif (10531 <= id <= 10608) || (10718 <= id <= 10798)
        return 34
    elseif (10800 <= id <= 10949) || (11400 <= id <= 11449)
        return 35
    elseif (10950 <= id <= 10999) || (11335 <= id <= 11349) || (11388 <= id <= 11399) ||
           (11450 <= id <= 11499) || (13000 <= id <= 13149)
        return 36
    elseif 13150 <= id <= 13490
        return 37
    elseif (13491 <= id <= 13649) || (15650 <= id <= 15673)
        return 38
    elseif 13650 <= id <= 13834
        return 39
    elseif 13835 <= id <= 14109
        return 40
    elseif 14110 <= id <= 14226
        return 41
    elseif (14227 <= id <= 14306) || (14308 <= id <= 14311) || (14313 <= id <= 14314) ||
           id == 14316 || (14318 <= id <= 14345) || (14347 <= id <= 14430)
        return 42
    elseif 14431 <= id <= 14880
        return 43
    elseif (15750 <= id <= 15779) || (15800 <= id <= 15899) || (15950 <= id <= 15999)
        return 44
    elseif (15780 <= id <= 15799) || (15900 <= id <= 15949)
        return 45
    elseif 14881 <= id <= 14951
        return 46
    elseif 14952 <= id <= 15001
        return 47
    elseif 15002 <= id <= 15199
        return 48
    elseif 15400 <= id <= 15499
        return 49
    elseif (15500 <= id <= 15649) || (15674 <= id <= 15749) || (16000 <= id <= 16711)
        return 50
    elseif (16712 <= id <= 17199) || (17300 <= id <= 17399) || (17500 <= id <= 17607)
        return 51
    elseif 15200 <= id <= 15399
        return 52
    elseif (17608 <= id <= 17849) || id == 17953
        return 53
    elseif (17200 <= id <= 17299) || (17850 <= id <= 17999)
        return 54
    elseif 18043 <= id <= 18099
        return 55
    elseif (17401 <= id <= 17499) || (18000 <= id <= 18042) || (18100 <= id <= 18158)
        return 56
    elseif 18226 <= id <= 18326
        return 57
    end

    return nothing
end

"""
    filter_dead_tags(df::DataFrame; time_threshold_days::Real=30.0,
                     dist_threshold_meters::Real=50.0)::DataFrame

Identifies and flags acoustic telemetry records belonging to a terminal stationary period,
which indicates a deceased animal or shed tag (e.g. tag stationary within `dist_threshold_meters`
for \$\\ge\$ `time_threshold_days` continuously until the end of detections).

# Arguments
- `df::DataFrame`: Telemetry table with columns `:tagid`, `:lon`, `:lat`, and `:timestamp`
  (or `:time`).
- `time_threshold_days::Real`: Minimum duration in days of terminal non-movement (default 30.0).
- `dist_threshold_meters::Real`: Maximum radius in metres from terminal location (default 50.0).

# Returns
- `DataFrame`: A copy of `df` with an added boolean column `:is_dead`.
"""
function filter_dead_tags(
    df::DataFrame;
    time_threshold_days::Real=30.0,
    dist_threshold_meters::Real=50.0
)::DataFrame
    out = copy(df)
    time_col = hasproperty(out, :timestamp) ? :timestamp : :time
    if !hasproperty(out, time_col) || !hasproperty(out, :tagid) ||
       !hasproperty(out, :lon) || !hasproperty(out, :lat)
        error("DataFrame must contain :tagid, :lon, :lat, and :timestamp or :time.")
    end

    t_days = if eltype(out[!, time_col]) <: Dates.TimeType
        [Dates.value(Dates.Millisecond(Dates.DateTime(t) - Dates.DateTime(1970, 1, 1))) / 8.64e7 for t in out[!, time_col]]
    else
        Float64.(out[!, time_col]) .* 365.25
    end
    out._t_days = t_days

    sort!(out, [:tagid, :_t_days])

    is_dead_flags = zeros(Bool, nrow(out))
    gdf = groupby(out, :tagid)

    for sub in gdf
        n_obs = nrow(sub)
        if n_obs == 0
            continue
        end

        row_indices = parentindices(sub)[1]
        last_lon = Float64(sub.lon[end])
        last_lat = Float64(sub.lat[end])
        last_t = sub._t_days[end]

        dists_to_last = [haversine_distance(Float64(sub.lon[i]), Float64(sub.lat[i]), last_lon, last_lat) for i in 1:n_obs]
        days_to_last = [last_t - sub._t_days[i] for i in 1:n_obs]

        cummax_dist = zeros(Float64, n_obs)
        curr_max = 0.0
        for i in n_obs:-1:1
            d = isfinite(dists_to_last[i]) ? dists_to_last[i] : Inf
            curr_max = max(curr_max, d)
            cummax_dist[i] = curr_max
        end

        in_terminal_cluster = [cummax_dist[i] <= dist_threshold_meters for i in 1:n_obs]
        cluster_durations = [in_terminal_cluster[i] ? days_to_last[i] : 0.0 for i in 1:n_obs]
        max_cluster_dur = maximum(cluster_durations)

        for i in 1:n_obs
            if in_terminal_cluster[i] && max_cluster_dur >= time_threshold_days
                is_dead_flags[row_indices[i]] = true
            end
        end
    end

    out.is_dead = is_dead_flags
    select!(out, Not(:_t_days))
    return out
end

"""
    _parse_flexible_date(date_val)::Union{Date, Nothing}

Parses date strings, DateTime, Date, or numeric timestamp representations safely.
"""
function _parse_flexible_date(date_val)::Union{Date, Nothing}
    if ismissing(date_val) || isnothing(date_val)
        return nothing
    end
    if date_val isa Date
        return date_val
    elseif date_val isa Dates.DateTime
        return Date(date_val)
    elseif date_val isa Real
        yr = round(Int, date_val)
        if 1950 <= yr <= 2050
            return Date(yr, 6, 1)
        end
    elseif date_val isa AbstractString
        s = strip(date_val)
        isempty(s) && return nothing
        m_iso = match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
        if !isnothing(m_iso)
            y, m, d = parse(Int, m_iso.captures[1]), parse(Int, m_iso.captures[2]), parse(Int, m_iso.captures[3])
            return Date(y, clamp(m, 1, 12), clamp(d, 1, 31))
        end
        m_us = match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})", s)
        if !isnothing(m_us)
            m, d, y = parse(Int, m_us.captures[1]), parse(Int, m_us.captures[2]), parse(Int, m_us.captures[3])
            return Date(y, clamp(m, 1, 12), clamp(d, 1, 31))
        end
        dt = tryparse(DateTime, s)
        if !isnothing(dt)
            return Date(dt)
        end
    end
    return nothing
end

"""
    _date_to_decimal_year(d::Date)::Float64

Converts a `Date` to a continuous decimal year representation.
"""
function _date_to_decimal_year(d::Date)::Float64
    yr = Dates.year(d)
    doy = Dates.dayofyear(d)
    days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
    return Float64(yr) + (Float64(doy) - 1.0) / days_in_yr
end

_date_to_decimal_year(dt::Dates.TimeType)::Float64 = _date_to_decimal_year(Date(dt))



"""
    summarize_tag_activity(df::DataFrame)::DataFrame

Calculates summary statistics per individual tag from a telemetry / mark-recapture DataFrame.

# Computed Metrics per Tag
- `duration_days::Float64`: Elapsed days from first to last observation.
- `cw_change::Float64`: Net carapace width growth in mm (`last_cw - first_cw`).
- `cc_change::Float64`: Net carapace condition progression.
- `total_dist_m::Float64`: Cumulative physical track distance in metres across consecutive pings.
- `n_points::Int`: Total number of detection records.
"""
function summarize_tag_activity(df::DataFrame)::DataFrame
    time_col = hasproperty(df, :timestamp) ? :timestamp : (hasproperty(df, :time) ? :time : nothing)
    if isnothing(time_col) || !hasproperty(df, :tagid)
        error("Input DataFrame must have :tagid and either :timestamp or :time.")
    end

    gdf = groupby(df, :tagid)
    summary_rows = []

    for sub in gdf
        tid = first(sub.tagid)
        n_pts = nrow(sub)

        times_parsed = [_parse_flexible_date(t) for t in sub[!, time_col]]
        valid_times = filter(!isnothing, times_parsed)
        dur_days = if length(valid_times) > 1
            Float64(Dates.value(maximum(valid_times) - minimum(valid_times)))
        else
            0.0
        end

        cw_raw = hasproperty(sub, :cw) ? [tryparse(Float64, string(x)) for x in sub.cw] : nothing
        cw_vals = !isnothing(cw_raw) ? Float64[x for x in cw_raw if !isnothing(x) && isfinite(x)] : Float64[]
        cw_chg = length(cw_vals) > 1 ? cw_vals[end] - cw_vals[1] : NaN

        cc_raw = hasproperty(sub, :cc) ? [tryparse(Float64, string(c)) for c in sub.cc] : nothing
        cc_vals = !isnothing(cc_raw) ? Float64[x for x in cc_raw if !isnothing(x) && isfinite(x)] : Float64[]
        cc_chg = length(cc_vals) > 1 ? cc_vals[end] - cc_vals[1] : NaN

        tot_dist = 0.0
        if hasproperty(sub, :lon) && hasproperty(sub, :lat) && n_pts > 1
            for i in 1:(n_pts - 1)
                lon1, lat1 = Float64(sub.lon[i]), Float64(sub.lat[i])
                lon2, lat2 = Float64(sub.lon[i+1]), Float64(sub.lat[i+1])
                d = haversine_distance(lon1, lat1, lon2, lat2)
                if isfinite(d)
                    tot_dist += d
                end
            end
        end

        push!(summary_rows, (
            tagid = tid,
            duration_days = dur_days,
            cw_change = cw_chg,
            cc_change = cc_chg,
            total_dist_m = tot_dist,
            n_points = n_pts
        ))
    end

    return DataFrame(summary_rows)
end

"""
    sample_markov_bridge(Gamma::AbstractMatrix{<:Real},
                         u_start::Integer, u_end::Integer,
                         n_steps::Integer;
                         graph::Union{Nothing, SimpleGraph} = nothing,
                         W::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                         hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
                         powers::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
                         rng::Random.AbstractRNG = Random.GLOBAL_RNG)::Vector{Int}

Samples a discrete Markov state-space trajectory ``z_0 = u_{\\text{start}} \\to z_1 \\to
\\dots \\to z_T = u_{\\text{end}}`` conditioned on fixed start and terminal endpoints under
the transition kernel ``\\boldsymbol{\\Gamma}`` while strictly respecting the marine domain
and coastline movement boundaries of adjacency graph ``G(V, E)``.

# Mathematical Formulation:
Let ``G(V, E)`` be the topological water graph where edges ``(i, j) \\in E`` connect
only physically adjacent water units across marine boundaries.
1. The shortest geodesic graph distance ``d_G(u_{\\text{start}}, u_{\\text{end}})`` is computed via
   ``A^*`` search. To prevent artificial jumps across land barriers or peninsulas (e.g. Cape Breton),
   the effective trajectory step count satisfies:
   ```math
   T_{\\text{eff}} = \\max(n_{\\text{steps}}, d_G(u_{\\text{start}}, u_{\\text{end}}))
   ```
2. For each intermediate step ``t \\in \\{1, \\dots, T_{\\text{eff}}-1\\}``, given previous state
   ``z_{t-1} = i`` and destination ``u_{\\text{end}}``, transitions are sampled strictly from
   valid marine graph neighbors ``j \\in \\mathcal{N}(i) \\cup \\{i\\}`` via Chapman-Kolmogorov:
   ```math
   P(z_t = j \\mid z_{t-1} = i, z_T = u_{\\text{end}}) = \\frac{T_{ij} \\, (\\mathbf{T}^{T_{\\text{eff}}-t})_{j, u_{\\text{end}}}}{(\\mathbf{T}^{T_{\\text{eff}}-t+1})_{i, u_{\\text{end}}}}
   ```
   where ``T_{ij} > 0`` only if ``(i, j) \\in E`` or ``i = j``.

# Inputs:
- `Gamma`: ``S \\times S`` transition matrix.
- `u_start`: Initial spatial unit index ``1 \\le u_{\\text{start}} \\le S``.
- `u_end`: Terminal spatial unit index ``1 \\le u_{\\text{end}} \\le S``.
- `n_steps`: Requested discrete time steps ``T \\ge 1``.
- `graph`: Optional `SimpleGraph` topology for marine coastline path routing.
- `W`: Optional ``S \\times S`` sparse adjacency matrix.
- `hsi`: Optional spatial HSI suitability vector for directional advective weighting.
- `powers`: Precomputed matrix powers of the 1-step kernel.
- `rng`: Random number generator.

# Outputs:
- `path::Vector{Int}`: Sequence of visited spatial unit indices strictly on the marine graph.
"""
function sample_markov_bridge(
    Gamma::AbstractMatrix{<:Real},
    u_start::Integer,
    u_end::Integer,
    n_steps::Integer;
    graph::Union{Nothing, SimpleGraph} = nothing,
    W::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    powers::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::Vector{Int}
    u_s = Int(u_start)
    u_e = Int(u_end)
    S = size(Gamma, 1)

    if u_s == u_e
        return fill(u_s, max(2, n_steps + 1))
    end

    # If graph is provided, ensure path stays strictly on valid water edges
    if !isnothing(graph)
        sp = a_star(graph, u_s, u_e)
        sp_nodes = isempty(sp) ? [u_s, u_e] : vcat([src(e) for e in sp], [dst(last(sp))])
        d_min = length(sp_nodes) - 1

        effective_steps = max(n_steps, d_min)
        if effective_steps <= d_min
            return sp_nodes
        end

        # Build 1-step water transition matrix T if W is provided
        T_step = if !isnothing(W)
            T_mat = zeros(Float64, S, S)
            for i in 1:S
                nbrs = findall(>(0.0), W[i, :])
                if isempty(nbrs)
                    T_mat[i, i] = 1.0
                    continue
                end
                w_vec = if !isnothing(hsi)
                    [exp(0.5 * (Float64(hsi[j]) - Float64(hsi[i]))) for j in nbrs]
                else
                    ones(Float64, length(nbrs))
                end
                w_sum = sum(w_vec)
                w_norm = w_sum > 0 ? (w_vec ./ w_sum) : fill(1.0 / length(nbrs), length(nbrs))
                T_mat[i, i] = 0.35 # probability of remaining in cell during unit step
                for (idx, j) in enumerate(nbrs)
                    T_mat[i, j] = 0.65 * w_norm[idx]
                end
            end
            T_mat
        else
            Matrix{Float64}(Gamma)
        end

        # Precompute powers up to effective_steps
        P_mats = Vector{Matrix{Float64}}(undef, effective_steps)
        P_mats[1] = copy(T_step)
        for k in 2:effective_steps
            P_mats[k] = P_mats[k-1] * T_step
        end

        path = zeros(Int, effective_steps + 1)
        path[1] = u_s
        path[end] = u_e
        curr = u_s

        for t in 1:(effective_steps - 1)
            rem = effective_steps - t
            cand_nodes = findall(>(0.0), T_step[curr, :])
            weights = Float64[]
            for j in cand_nodes
                t_ij = T_step[curr, j]
                p_j_end = (rem == 1) ? T_step[j, u_e] : P_mats[rem][j, u_e]
                push!(weights, t_ij * p_j_end)
            end
            w_sum = sum(weights)
            if w_sum <= 1e-15
                sp_curr = a_star(graph, curr, u_e)
                next_node = !isempty(sp_curr) ? dst(first(sp_curr)) : curr
                path[t + 1] = next_node
                curr = next_node
            else
                probs = weights ./ w_sum
                r = rand(rng)
                cum = 0.0
                chosen = cand_nodes[end]
                for (idx, j) in enumerate(cand_nodes)
                    cum += probs[idx]
                    if r <= cum
                        chosen = j
                        break
                    end
                end
                path[t + 1] = chosen
                curr = chosen
            end
        end

        return path
    end

    # Fallback when no graph topology is supplied
    if n_steps <= 1
        return [u_s, u_e]
    end

    local P_mats_dense::Vector{Matrix{Float64}}
    if !isnothing(powers) && length(powers) >= n_steps
        P_mats_dense = [Matrix{Float64}(p) for p in powers[1:n_steps]]
    else
        P_mats_dense = Vector{Matrix{Float64}}(undef, n_steps)
        P_mats_dense[1] = Matrix{Float64}(Gamma)
        for k in 2:n_steps
            P_mats_dense[k] = P_mats_dense[k-1] * P_mats_dense[1]
        end
    end

    path = zeros(Int, n_steps + 1)
    path[1] = u_s
    path[end] = u_e
    curr = u_s

    for t in 1:(n_steps - 1)
        rem_steps = n_steps - t
        weights = zeros(Float64, S)
        for j in 1:S
            g_ij = Float64(Gamma[curr, j])
            p_j_end = (rem_steps == 1) ? Float64(Gamma[j, u_e]) : Float64(P_mats_dense[rem_steps][j, u_e])
            weights[j] = g_ij * p_j_end
        end
        w_sum = sum(weights)
        if w_sum <= 1e-15
            weights = Float64[Float64(Gamma[curr, j]) for j in 1:S]
            w_sum = sum(weights)
        end
        probs = weights ./ (w_sum > 0.0 ? w_sum : 1.0)

        r = rand(rng)
        cum = 0.0
        next_s = S
        for j in 1:S
            cum += probs[j]
            if r <= cum
                next_s = j
                break
            end
        end
        path[t + 1] = next_s
        curr = next_s
    end

    return path
end

"""
    reconstruct_mark_recapture_paths(tagging::DataFrame, result::NamedTuple;
                                     time_interval::Symbol = :monthly,
                                     max_paths::Union{Nothing, Int} = nothing,
                                     smooth_jitter::Bool = true,
                                     rng::Random.AbstractRNG = Random.GLOBAL_RNG)::Vector{NamedTuple}

Reconstructs the latent continuous state-space trajectory for every individual
mark-recapture event in `tagging` using an exact discrete Markov Bridge filter conditioned
on the true observed release and recapture coordinates under the fitted transition matrix
and HSI field. Trajectories strictly follow the marine water graph and avoid land crossing.

# Mathematical Formulation:
For an individual crab ``k`` with release observation at ``(\\text{lon}_0, \\text{lat}_0)``
at time ``t_0`` and recapture at ``(\\text{lon}_T, \\text{lat}_T)`` at time ``t_T``:
1. Coordinates are mapped to nearest discrete marine areal units ``u_0, u_T \\in \\mathcal{U}``.
   When observations lie beyond the domain, the last encountered boundary unit HSI probability is extended.
2. The elapsed time interval ``\\Delta t = \\max(1, \\text{round}(\\text{Int}, |t_T - t_0| / \\Delta t_{\\text{interval}}))``.
3. Conditional latent intermediate units ``(z_0 = u_0, z_1, \\dots, z_T = u_T)`` are sampled
   along connected marine graph edges.
4. Each discrete state ``z_t`` is projected to continuous geographic coordinates ``(\\text{lon}_t, \\text{lat}_t)``
   with boundary pinning at true release and recapture points.

# Inputs:
- `tagging`: Observation DataFrame containing `:tagid`, `:lon`, `:lat`, and `:time` or `:timestamp`.
- `result`: Model bundle containing `au`, `transition_matrix`, and `opts`.
- `time_interval`: Temporal step resolution (`:monthly`, `:weekly`, `:daily`).
- `max_paths`: Optional integer limit on number of individuals to reconstruct (default `nothing` for all).
- `smooth_jitter`: Whether to apply subtle spatial smoothing inside marine polygons.

# Outputs:
- `trajectories::Vector{NamedTuple}`: Reconstructed trajectories for all individuals.
"""
function reconstruct_mark_recapture_paths(
    tagging::DataFrame,
    result::NamedTuple;
    time_interval::Symbol = :monthly,
    max_paths::Union{Nothing, Int} = nothing,
    smooth_jitter::Bool = true,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::Vector{NamedTuple}
    au = result.au
    Gamma = Matrix{Float64}(result.transition_matrix)
    S = au.n_units
    g = hasproperty(au, :graph) ? au.graph : nothing
    W = hasproperty(au, :W) ? au.W : nothing
    hsi_vec = hasproperty(result, :hsi) && !isnothing(result.hsi) ? result.hsi : nothing

    # Pre-extract geographic centroids of au in WGS84 degrees
    cents_deg = Tuple{Float64, Float64}[]
    wkt_str = hasproperty(au, :wkt) ? string(au.wkt) : ""
    is_utm = contains(lowercase(wkt_str), "utm") || (!isempty(au.centroids) && au.centroids[1][2] > 1000.0)

    for c in au.centroids
        if is_utm
            push!(cents_deg, utm_to_lonlat(c[1], c[2]; zone=20, is_km=true))
        else
            push!(cents_deg, (Float64(c[1]), Float64(c[2])))
        end
    end

    # Group observations by tag
    time_col = hasproperty(tagging, :time) ? :time : (hasproperty(tagging, :timestamp) ? :timestamp : nothing)
    isnothing(time_col) && error("tagging DataFrame must have :time or :timestamp column.")

    gdf = groupby(tagging, :tagid)
    tag_keys = collect(keys(gdf))
    n_tags_total = length(tag_keys)
    n_to_process = isnothing(max_paths) ? n_tags_total : min(n_tags_total, max_paths)

    # Unit step duration in decimal years
    dt_step = (time_interval == :weekly) ? (1.0 / 52.0) : ((time_interval == :daily) ? (1.0 / 365.0) : (1.0 / 12.0))

    trajectories = NamedTuple[]

    for i in 1:n_to_process
        sub = sort(gdf[tag_keys[i]], time_col)
        nrow(sub) < 2 && continue

        tid = string(first(sub.tagid))
        all_path_units = Int[]
        all_path_coords = Tuple{Float64, Float64}[]

        # Reconstruct piecewise between consecutive observation records
        for seg in 1:(nrow(sub) - 1)
            r1 = sub[seg, :]
            r2 = sub[seg + 1, :]

            lon1, lat1 = Float64(r1.lon), Float64(r1.lat)
            lon2, lat2 = Float64(r2.lon), Float64(r2.lat)
            t1 = Float64(r1[time_col])
            t2 = Float64(r2[time_col])

            # Find nearest spatial units in au strictly on navigable water
            au_lmask = hasproperty(au, :land_mask) ? au.land_mask : nothing
            u_start = _nearest_unit_index(
                lon1, lat1, cents_deg;
                land_mask = au_lmask, W = W
            )
            u_end = _nearest_unit_index(
                lon2, lat2, cents_deg;
                land_mask = au_lmask, W = W
            )

            elapsed_t = max(0.0, t2 - t1)
            n_steps = clamp(round(Int, elapsed_t / dt_step), 1, 60)

            # Sample water-constrained Markov bridge
            bridge = sample_markov_bridge(
                Gamma, u_start, u_end, n_steps;
                graph = g, W = W, hsi = hsi_vec, rng = rng
            )

            # Convert bridge units to continuous geographic coordinates
            for (step_idx, u_id) in enumerate(bridge)
                if seg > 1 && step_idx == 1
                    continue # Avoid duplicate point at segment boundary
                end

                push!(all_path_units, u_id)

                if step_idx == 1
                    push!(all_path_coords, (lon1, lat1))
                elseif step_idx == length(bridge)
                    push!(all_path_coords, (lon2, lat2))
                else
                    c_pt = cents_deg[u_id]
                    pt_lon, pt_lat = c_pt[1], c_pt[2]
                    if smooth_jitter
                        # Subtle perturbation inside marine polygon (approx 0.003 deg ~ 300 m)
                        jit_r = 0.003 * sqrt(rand(rng))
                        jit_theta = 2.0 * pi * rand(rng)
                        pt_lon += jit_r * cos(jit_theta)
                        pt_lat += jit_r * sin(jit_theta)
                    end
                    push!(all_path_coords, (pt_lon, pt_lat))
                end
            end
        end

        length(all_path_coords) < 2 && continue

        # Compute trajectory distance and net displacement (km)
        total_dist_km = 0.0
        for k in 2:length(all_path_coords)
            total_dist_km += haversine_distance(
                all_path_coords[k-1][1], all_path_coords[k-1][2],
                all_path_coords[k][1], all_path_coords[k][2]
            ) / 1000.0
        end

        net_disp_km = haversine_distance(
            all_path_coords[1][1], all_path_coords[1][2],
            all_path_coords[end][1], all_path_coords[end][2]
        ) / 1000.0

        dur_days = Float64((sub[end, time_col] - sub[1, time_col]) * 365.25)

        push!(trajectories, (
            tagid           = tid,
            coords          = all_path_coords,
            units           = all_path_units,
            n_steps         = length(all_path_coords) - 1,
            start_date      = string(sub[1, time_col]),
            end_date        = string(sub[end, time_col]),
            duration_days   = dur_days,
            total_dist_km   = total_dist_km,
            displacement_km = net_disp_km
        ))
    end

    return trajectories
end

function _nearest_unit_index(
    lon::Float64,
    lat::Float64,
    cents_deg::Vector{Tuple{Float64, Float64}};
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    W::Union{Nothing, AbstractMatrix} = nothing
)::Int
    S = length(cents_deg)
    active_mask = trues(S)
    if land_mask !== nothing
        active_mask .&= .!land_mask
    end
    if W !== nothing
        deg = vec(sum(W, dims=2))
        active_mask .&= (deg .> 0.0)
    end
    active_indices = findall(active_mask)
    if isempty(active_indices)
        active_indices = collect(1:S)
    end

    best_idx = first(active_indices)
    best_dist = Inf
    for i in active_indices
        c = cents_deg[i]
        d = (c[1] - lon)^2 + (c[2] - lat)^2
        if d < best_dist
            best_dist = d
            best_idx = i
        end
    end
    return best_idx
end



"""
    _build_A_ad(adj_rows, hsi, gamma_g, S)

AD-compatible directed adjacency build. Optimized for ForwardDiff.
"""
function _build_A_ad(adj_rows, hsi, gamma_g, S)
    # 1. Mathematical simplification & Vectorization:
    # exp(γ(HSI_j - HSI_i)) / Σ exp(γ(HSI_k - HSI_i)) simplifies to
    # exp(γ HSI_j) / Σ exp(γ HSI_k). The HSI_i term cancels out 
    # We compute this once for all units 
    exp_hsi = exp.(gamma_g .* hsi)
    
    # Extract the promoted AD type (e.g., ForwardDiff.Dual) directly from the math
    T = eltype(exp_hsi)
    A = zeros(T, S, S)
    
    @inbounds for i in 1:S
        nbrs = adj_rows[i]
        isempty(nbrs) && continue
        
        # 2. Allocation-free denominator accumulation
        sw = zero(T)
        for j in nbrs
            sw += exp_hsi[j]
        end
        
        sw <= 0 && continue
        
        # 3. Allocation-free assignment
        for j in nbrs
            A[i, j] = exp_hsi[j] / sw
        end
    end
    
    return A
end


function _row_normalise(M, S)
    T = eltype(M)
    M_rect = max.(zero(T), M)
    s = sum(M_rect, dims=2)
    
    # Non-mutating division-by-zero protection
    s_safe = s .+ (s .== zero(T))
    
    # Broadcast normalization and uniform distribution injection for zero-rows
    return (M_rect ./ s_safe) .+ ((s .== zero(T)) ./ T(S))
end

# ── Directed adjacency and resolvent operator ──────────────────────────────────
 

"""
    compute_directed_adjacency(hsi, W; gamma=1.0) -> Matrix{Float64}

Construct the directed adjacency matrix A from the symmetric adjacency W and
habitat suitability values HSI. Row i has non-zero entries only at W-neighbours
of i, weighted by exp(γ HSI_j) and row-normalised:

```math
A[i,j] = \\frac{\\exp(\\gamma_i \\cdot \\text{HSI}_j)}{\\sum_{k \\in \\mathcal{N}(i)} \\exp(\\gamma_i \\cdot \\text{HSI}_k)}
```

γ > 0 biases movement towards higher HSI; γ = 0 gives the uniform random walk.

# Arguments
- `hsi::AbstractVector{<:Real}`: Habitat suitability per spatial unit (length S).
- `W::SparseMatrixCSC`: Symmetric binary adjacency matrix (S × S).
- `gamma::Union{Real, AbstractVector{<:Real}}`: Advection sensitivity to HSI gradient
  (default 1.0). May be a scalar or a spatially varying vector of length S.

# Returns
- `Matrix{Float64}`: Dense S × S matrix A with row-stochastic structure over W-neighbours.
"""
function compute_directed_adjacency(
    hsi::AbstractVector{<:Real},
    W::SparseMatrixCSC;
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0
)::SparseMatrixCSC{Float64, Int}
    S = size(W, 1)

    is_spatial = gamma isa AbstractVector
    if is_spatial
        n_g = length(gamma)
        if n_g != S && n_g != 1
            throw(DimensionMismatch(
                "gamma vector has length $n_g, but must be scalar or match spatial units S=$S."
            ))
        end
        if n_g == 1
            gamma_scalar = Float64(gamma[1])
            is_spatial = false
        end
    else
        gamma_scalar = Float64(gamma)
    end

    # Build sparse triplets -- same sparsity pattern as W
    nnz_W = nnz(W)
    I_out = Vector{Int}(undef, nnz_W)
    J_out = Vector{Int}(undef, nnz_W)
    V_out = Vector{Float64}(undef, nnz_W)
    ptr_out = 0

    if !is_spatial
        # Fast path: precompute exponents once for uniform scalar gamma
        exp_hsi = exp.(gamma_scalar .* Float64.(hsi))
        for i in 1:S
            col_start = W.colptr[i]
            col_end   = W.colptr[i+1] - 1
            col_start > col_end && continue

            sw = 0.0
            @inbounds for ptr in col_start:col_end
                sw += exp_hsi[W.rowval[ptr]]
            end
            sw <= 0.0 && continue

            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                ptr_out += 1
                I_out[ptr_out] = i
                J_out[ptr_out] = j
                V_out[ptr_out] = exp_hsi[j] / sw
            end
        end
    else
        # Spatially-varying gamma[i] per unit i
        for i in 1:S
            col_start = W.colptr[i]
            col_end   = W.colptr[i+1] - 1
            col_start > col_end && continue

            g_i = Float64(gamma[i])
            sw = 0.0
            @inbounds for ptr in col_start:col_end
                sw += exp(g_i * Float64(hsi[W.rowval[ptr]]))
            end
            sw <= 0.0 && continue

            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                ptr_out += 1
                I_out[ptr_out] = i
                J_out[ptr_out] = j
                V_out[ptr_out] = exp(g_i * Float64(hsi[j])) / sw
            end
        end
    end

    return sparse(
        view(I_out, 1:ptr_out),
        view(J_out, 1:ptr_out),
        view(V_out, 1:ptr_out),
        S, S
    )
end


 
"""
    resolvent_transition(beta, D_diff, A, L, S) -> Matrix{Float64}

Compute the resolvent transition operator:

    Γ̄ = (I − β A − D L)^{-1}

where L is the symmetric graph Laplacian. Rows of Γ̄ are rectified (negative
entries set to zero) and row-normalised to form valid probability distributions.

Note: the inverse exists when ‖β A + D L‖ < 1 in an appropriate operator norm.
The NUTS priors (β < 0.95, D ≥ 0) are chosen to help ensure this, but
near-boundary samples may produce poorly conditioned M; the try/catch below
falls back to `M \\ I` in those cases.

# Arguments
- `beta`: Advective weight (scalar, 0 ≤ β < 1).
- `D_diff`: Diffusion coefficient (scalar, ≥ 0).
- `A::Matrix{Float64}`: Directed adjacency matrix (S × S, row-stochastic over nbrs).
- `L::Matrix{Float64}`: Graph Laplacian (S × S, positive semidefinite).
- `S::Int`: Number of spatial units.

# Returns
- Dense S × S matrix Γ̄ with row-stochastic rows.
"""
function resolvent_transition(
    beta::Real,
    D_diff::Real,
    A::AbstractMatrix{<:Real},
    L::AbstractMatrix{<:Real},
    S::Int
)::Matrix{Float64}

    # Lock types upfront for type-stability
    b = Float64(beta)
    d = Float64(D_diff)

    # 1. Construct M in a single allocation-free, column-major pass
    M = Matrix{Float64}(undef, S, S)
    @inbounds for j in 1:S
        for i in 1:S
            diag = (i == j) ? 1.0 : 0.0
            M[i, j] = diag - b * A[i, j] - d * L[i, j]
        end
    end

    # 2. Invert M
    Gamma = try
        inv(M)
    catch e
        @warn "resolvent_transition: inv failed, falling back to M\\I: $(e)"
        M \ Matrix{Float64}(I, S, S)
    end

    # 3. Rectify and accumulate row sums (Column-Major for peak cache efficiency)
    row_sums = zeros(Float64, S)
    @inbounds for j in 1:S
        for i in 1:S
            v = max(0.0, Gamma[i, j])
            Gamma[i, j] = v
            row_sums[i] += v
        end
    end

    # 4. Row-normalise using the accumulated sums (Column-Major)
    @inbounds for j in 1:S
        for i in 1:S
            if row_sums[i] > 0.0
                Gamma[i, j] /= row_sums[i]
            end
        end
    end

    return Gamma
end

 

"""
    _powerm(M, k) -> Matrix

Compute M^k using iterative binary (repeated squaring) exponentiation.
Optimized to perform all multiplications in-place, reducing memory allocations 
to \$O(1)\$ regardless of \$k\$.
"""
function _powerm(M::AbstractMatrix{T}, k::Integer) where {T <: Real}
    n = size(M, 1)
    
    # Ensure type stability by matching the precision of M (e.g., Float64)
    OutType = float(T) 
    
    k <= 0 && return Matrix{OutType}(I, n, n)
    k == 1 && return Matrix{OutType}(M)

    # Preallocate active matrices
    R = Matrix{OutType}(I, n, n)
    B = Matrix{OutType}(M)
    
    # Preallocate temporary buffers for in-place multiplication
    R_tmp = similar(R)
    B_tmp = similar(B)

    # Iterative repeated squaring
    while k > 0
        if isodd(k)
            # R = R * B (in-place)
            mul!(R_tmp, R, B)
            R, R_tmp = R_tmp, R  # Swap references (zero cost)
        end
        
        k >>= 1 # Fast bitwise division by 2  ..  isodd(k) and k >>= 1 is faster than k % 2 != 0 and k ÷ 2.
        
        if k > 0
            # B = B * B (in-place)
            mul!(B_tmp, B, B)
            B, B_tmp = B_tmp, B  # Swap references (zero cost)
        end
    end
    
    return R
end

"""
    power_transition(Gamma::AbstractMatrix{Float64}, k::Integer) -> Matrix{Float64}

Computes the ``k``-step Markov transition probability matrix ``\\boldsymbol{\\Gamma}^k``
via repeated matrix squaring (binary exponentiation) and applies non-negativity
rectification and row-stochastic normalization to eliminate floating-point drift.

# Mathematical Formulation
For a discrete spatial Markov transition matrix
``\\boldsymbol{\\Gamma} \\in \\mathbb{R}^{S \\times S}``:
```math
\\boldsymbol{\\Gamma}^k = \\underbrace{\\boldsymbol{\\Gamma} \\times \\cdots \\times \\boldsymbol{\\Gamma}}_{k \\text{ times}}
```
Each entry ``[\\boldsymbol{\\Gamma}^k]_{ij} = \\mathbb{P}(X_{t+k} = j \\mid X_t = i)``
denotes the conditional probability that an individual transitions from node ``i`` to
node ``j`` after ``k`` time steps. Small negative entries arising from numerical
rounding are clamped to zero and rows are re-normalized:
```math
[\\boldsymbol{\\Gamma}^k]_{ij} \\leftarrow \\frac{\\max(0, [\\boldsymbol{\\Gamma}^k]_{ij})}{\\sum_{l=1}^S \\max(0, [\\boldsymbol{\\Gamma}^k]_{il})}
```

# Arguments
- `Gamma`: Row-stochastic one-step transition matrix (size ``S \\times S``).
- `k`: Discrete time step exponent (``k \\ge 1``).

# Returns
- `Matrix{Float64}`: Numerically stable row-stochastic ``k``-step transition matrix.
"""
function power_transition(Gamma::AbstractMatrix{Float64}, k::Integer)::Matrix{Float64}
    # Uses the optimized O(1) allocation _powerm we built previously
    Gk = _powerm(Gamma, max(1, k))
    S = size(Gk, 1)
    
    # 1. Rectify negative entries and accumulate row sums (Column-Major)
    row_sums = zeros(Float64, S)
    @inbounds for j in 1:S
        for i in 1:S
            v = max(0.0, Gk[i, j])
            Gk[i, j] = v
            row_sums[i] += v
        end
    end
    
    # 2. Row-normalise using the accumulated sums (Column-Major)
    @inbounds for j in 1:S
        for i in 1:S
            if row_sums[i] > 0.0
                Gk[i, j] /= row_sums[i]
            end
        end
    end
    
    return Gk
end


"""
    construct_stochastic_transition_kernel(W, hsi; gamma=1.0, residence=0.2, advection=0.5, spatial=false)

Constructs strictly row-stochastic, unconditionally non-negative discrete movement
transition matrices ``P`` over spatial graph ``W``:

```math
P = (1 - \\rho) \\left[ (1 - \\alpha) T_{\\text{diff}} + \\alpha A \\right] + \\rho I
```

# Arguments
- `W::SparseMatrixCSC`: Spatial adjacency matrix (size ``S \\times S``).
- `hsi::AbstractVector{<:Real}`: Habitat suitability values per unit (length ``S``).
- `gamma::Union{Real, AbstractVector{<:Real}}`: Sensitivity of directional advection to
  the habitat gradient (default 1.0). May be scalar or vector across groups or units.
- `residence::Union{Real, AbstractVector{<:Real}}`: Probability ``\\rho \\in [0, 1)`` of
  remaining in current unit (default 0.2). May be scalar or vector across groups or units.
- `advection::Union{Real, AbstractVector{<:Real}}`: Fraction ``\\alpha \\in [0, 1]`` of
  directed movement vs random diffusion (default 0.5). May be scalar or vector.
- `spatial::Bool`: When `true`, vector parameters of length ``S`` are interpreted as
  spatially varying per unit ``s \\in 1:S``, returning a single ``S \\times S`` matrix.
  When `false` (default), vector parameters of length ``G`` represent group-level
  parameters across ``G`` biological groups, returning a `Vector{Matrix{Float64}}`.

# Returns
- `Matrix{Float64}`: If all parameters are scalars (or `spatial=true`), dense ``S \\times S``
  row-stochastic transition probability matrix.
- `Vector{Matrix{Float64}}`: If any parameter is an `AbstractVector` (and `spatial=false`),
  vector of ``G`` dense ``S \\times S`` row-stochastic transition matrices.
"""
function construct_stochastic_transition_kernel(
    W::SparseMatrixCSC,
    hsi::AbstractVector{<:Real};
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0,
    residence::Union{Real, AbstractVector{<:Real}} = 0.2,
    advection::Union{Real, AbstractVector{<:Real}} = 0.5,
    spatial::Bool = false,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Union{Matrix{Float64}, Vector{Matrix{Float64}}}
    S = size(W, 1)

    # 1. Unbiased topological random walk matrix T_diff (row-stochastic, independent of params)
    T_diff = zeros(Float64, S, S)
    for i in 1:S
        col_start = W.colptr[i]
        col_end   = W.colptr[i+1] - 1
        deg_i = col_end - col_start + 1
        if deg_i > 0 && col_start <= col_end
            inv_deg = 1.0 / deg_i
            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                T_diff[i, j] = inv_deg
            end
        else
            T_diff[i, i] = 1.0
        end
    end

    # Helper to enforce land zero-transition barrier and row stochasticity
    function _postprocess_kernel!(P_mat::Matrix{Float64})
        if land_mask !== nothing
            for i in 1:S
                if land_mask[i]
                    P_mat[i, :] .= 0.0
                    P_mat[i, i] = 1.0
                else
                    for j in 1:S
                        if land_mask[j]
                            P_mat[i, j] = 0.0
                        end
                    end
                    rs = sum(view(P_mat, i, :))
                    if rs > 0.0
                        P_mat[i, :] ./= rs
                    else
                        P_mat[i, i] = 1.0
                    end
                end
            end
        else
            for i in 1:S
                rs = sum(view(P_mat, i, :))
                if rs > 0.0
                    P_mat[i, :] ./= rs
                else
                    P_mat[i, i] = 1.0
                end
            end
        end
        return P_mat
    end

    # 2. Check for vector parameters
    any_vector = (gamma isa AbstractVector) ||
                 (residence isa AbstractVector) ||
                 (advection isa AbstractVector)

    if !any_vector
        # Standard scalar parameter execution — build P as sparse combination
        rho   = clamp(Float64(residence), 0.0, 0.9999)
        alpha = clamp(Float64(advection), 0.0, 1.0)
        A = compute_directed_adjacency(hsi, W; gamma=Float64(gamma))

        w_move = 1.0 - rho
        w_adv  = w_move * alpha
        w_diff = w_move * (1.0 - alpha)

        # Sparse combination: rho * I + w_adv * A + w_diff * T_diff
        P_sparse = w_adv * A + w_diff * T_diff + rho * sparse(I, S, S)
        P = Matrix{Float64}(P_sparse)

        return _postprocess_kernel!(P)

    elseif spatial
        # Spatially-varying parameters across S spatial units
        for (name, p) in (("gamma", gamma), ("residence", residence), ("advection", advection))
            if p isa AbstractVector && length(p) != S && length(p) != 1
                throw(DimensionMismatch(
                    "In spatial mode, parameter `$name` has length $(length(p)), " *
                    "but must match spatial units S=$S or be a scalar."
                ))
            end
        end

        rho_vec = residence isa AbstractVector ?
            (length(residence) == 1 ? fill(Float64(residence[1]), S) : Float64.(residence)) :
            fill(Float64(residence), S)
        adv_vec = advection isa AbstractVector ?
            (length(advection) == 1 ? fill(Float64(advection[1]), S) : Float64.(advection)) :
            fill(Float64(advection), S)
        g_vec = gamma isa AbstractVector ?
            (length(gamma) == 1 ? fill(Float64(gamma[1]), S) : Float64.(gamma)) :
            fill(Float64(gamma), S)

        clamp!(rho_vec, 0.0, 0.9999)
        clamp!(adv_vec, 0.0, 1.0)

        A = compute_directed_adjacency(hsi, W; gamma=g_vec)

        # Spatially-varying: build sparse P row by row over W-neighbours only
        P_rows = Int[]; P_cols = Int[]; P_vals = Float64[]
        A_csc  = SparseMatrixCSC(A)
        Td_csc = SparseMatrixCSC(T_diff)
        for i in 1:S
            rho_i    = rho_vec[i]
            alpha_i  = adv_vec[i]
            w_adv_i  = (1.0 - rho_i) * alpha_i
            w_diff_i = (1.0 - rho_i) * (1.0 - alpha_i)
            # Collect neighbours from A and T_diff (union of sparsity patterns)
            nbr_vals = Dict{Int, Float64}()
            for ptr in nzrange(A_csc, i)
                j = A_csc.rowval[ptr]
                nbr_vals[j] = get(nbr_vals, j, 0.0) + w_adv_i * A_csc.nzval[ptr]
            end
            for ptr in nzrange(Td_csc, i)
                j = Td_csc.rowval[ptr]
                nbr_vals[j] = get(nbr_vals, j, 0.0) + w_diff_i * Td_csc.nzval[ptr]
            end
            nbr_vals[i] = get(nbr_vals, i, 0.0) + rho_i
            for (j, v) in nbr_vals
                push!(P_rows, i); push!(P_cols, j); push!(P_vals, v)
            end
        end
        P_sparse = sparse(P_rows, P_cols, P_vals, S, S)
        P = Matrix{Float64}(P_sparse)

        return _postprocess_kernel!(P)

    else
        # Group vector mode: construct distinct transition matrix per group g in 1:G
        v_lengths = [length(p) for p in (gamma, residence, advection) if p isa AbstractVector]
        G = maximum(v_lengths)

        for (name, p) in (("gamma", gamma), ("residence", residence), ("advection", advection))
            if p isa AbstractVector && length(p) != G && length(p) != 1
                throw(DimensionMismatch(
                    "Parameter `$name` has length $(length(p)), but expected length G=$G or 1."
                ))
            end
        end

        g_vec = gamma isa AbstractVector ?
            (length(gamma) == 1 ? fill(Float64(gamma[1]), G) : Float64.(gamma)) :
            fill(Float64(gamma), G)
        rho_vec = residence isa AbstractVector ?
            (length(residence) == 1 ? fill(Float64(residence[1]), G) : Float64.(residence)) :
            fill(Float64(residence), G)
        adv_vec = advection isa AbstractVector ?
            (length(advection) == 1 ? fill(Float64(advection[1]), G) : Float64.(advection)) :
            fill(Float64(advection), G)

        kernels = Vector{Matrix{Float64}}(undef, G)
        I_diag = sparse(I, S, S)
        for g in 1:G
            rho_g   = clamp(rho_vec[g], 0.0, 0.9999)
            alpha_g = clamp(adv_vec[g], 0.0, 1.0)
            A_g     = compute_directed_adjacency(hsi, W; gamma=g_vec[g])

            w_move = 1.0 - rho_g
            w_adv  = w_move * alpha_g
            w_diff = w_move * (1.0 - alpha_g)

            P_g_sparse = w_adv * A_g + w_diff * T_diff + rho_g * I_diag
            kernels[g] = _postprocess_kernel!(Matrix{Float64}(P_g_sparse))
        end
        return kernels
    end
end


function _spatial_node_distance(c1, c2)::Float64
    x1, y1 = Float64(c1[1]), Float64(c1[2])
    x2, y2 = Float64(c2[1]), Float64(c2[2])
    if abs(x1) <= 180.0 && abs(x2) <= 180.0 && abs(y1) <= 90.0 && abs(y2) <= 90.0
        return haversine_distance(x1, y1, x2, y2)
    else
        return sqrt((x1 - x2)^2 + (y1 - y2)^2)
    end
end

"""
    _find_navigable_node(u::Int, centroids, land_mask, g::AbstractGraph) -> Int

Resolves an endpoint unit to the nearest topologically active and navigable
marine node in graph `g`. If `u` is already active (not in `land_mask` and has
degree > 0 in `g`), returns `u` directly. Otherwise finds the closest marine
unit in Euclidean space.
"""
function _find_navigable_node(
    u::Int,
    centroids,
    land_mask::Union{Nothing, AbstractVector{Bool}},
    g::AbstractGraph
)::Int
    S = nv(g)
    is_valid_u = (1 <= u <= S) &&
                 (land_mask === nothing || !land_mask[u]) &&
                 (degree(g, u) > 0)
    if is_valid_u
        return u
    end

    best_v = u
    min_dist = Inf
    for v in 1:S
        if (land_mask === nothing || !land_mask[v]) && degree(g, v) > 0
            d = centroids !== nothing ?
                _spatial_node_distance(centroids[u], centroids[v]) : abs(u - v)
            if d < min_dist
                min_dist = d
                best_v = v
            end
        end
    end
    if best_v != u
        @warn "Unit $u is not navigable (land or severed); mapped to " *
              "nearest marine unit $best_v."
    end
    return best_v
end

"""
    astar_predict_path(
        P::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        centroids = nothing,
        k::Union{Nothing, Int} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        p_min::Real = 1e-12
    ) -> Vector{Int}

Reconstructs the most likely individual movement trajectory between `release` and
`recapture` locations using the goal-directed \$A^*\$ heuristic search algorithm on the
negative log-likelihood transition graph.

# Mathematical Formulation
Given row-stochastic transition matrix \$\\mathbf{P} \\in [0, 1]^{S \\times S}\$, the
probability of a discrete trajectory \$\\pi = (u_0=u_{\\text{rel}}, \\dots, u_m=u_{\\text{rec}})\$ is:
```math
\\mathbb{P}(\\pi \\mid u_{\\text{rel}}, u_{\\text{rec}}) = \\prod_{\\tau=0}^{m-1} P_{u_\\tau, u_{\\tau+1}}
```
Maximizing path probability is equivalent to finding the shortest path with additive non-negative costs:
```math
c(u, v) = -\\ln P_{u, v} \\ge 0
```
When node spatial centroids \$\\mathbf{c}_u\$ are provided, the distance \$D(u, u_{\\text{rec}})\$
gives an admissible and consistent heuristic:
```math
h(u) = \\left\\lceil \\frac{D(u, u_{\\text{rec}})}{\\Delta x_{\\max}} \\right\\rceil \\cdot \\min_{i \\neq j} (-\\ln P_{i, j})
```
guaranteeing that \$A^*\$ identifies the exact global maximum-likelihood trajectory while
expanding an order of magnitude fewer nodes than a full trellis search.

# Arguments
- `P`: Row-stochastic transition probability matrix (dense or sparse \$S \\times S\$).
- `release`: Source spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index (1-indexed).
- `centroids`: Optional collection of centroid coordinate tuples `(lon, lat)` or `(x, y)`.
- `k`: Optional target step duration (integer).
- `land_mask`: Optional boolean vector of length \$S\$ (`true` for impermeable land).
- `p_min`: Numerical cutoff below which transitions are treated as zero probability (default `1e-12`).

# Returns
- `Vector{Int}`: Ordered sequence of spatial unit indices connecting `release` to `recapture`.
"""
function astar_predict_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    centroids = nothing,
    k::Union{Nothing, Int} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    p_min::Real = 1e-12
)::Vector{Int}
    S = size(P, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        throw(ArgumentError("Release unit ($release) and recapture unit ($recapture) must be within 1:$S."))
    end
    if release == recapture
        return k !== nothing ? fill(release, max(1, k + 1)) : [release]
    end

    # Build directed graph and sparse cost matrix
    g = SimpleDiGraph(S)
    rows_c = Int[]
    cols_c = Int[]
    vals_c = Float64[]

    max_p_trans = 0.0

    if P isa SparseMatrixCSC
        rows = rowvals(P)
        vals = nonzeros(P)
        for j in 1:S
            if land_mask !== nothing && land_mask[j]
                continue
            end
            for ptr in nzrange(P, j)
                i = rows[ptr]
                if i == j || (land_mask !== nothing && land_mask[i])
                    continue
                end
                p_val = Float64(vals[ptr])
                if p_val > p_min
                    add_edge!(g, i, j)
                    push!(rows_c, i)
                    push!(cols_c, j)
                    cost = -log(p_val)
                    push!(vals_c, cost)
                    if p_val > max_p_trans
                        max_p_trans = p_val
                    end
                end
            end
        end
    else
        for i in 1:S
            if land_mask !== nothing && land_mask[i]
                continue
            end
            for j in 1:S
                if i == j || (land_mask !== nothing && land_mask[j])
                    continue
                end
                p_val = Float64(P[i, j])
                if p_val > p_min
                    add_edge!(g, i, j)
                    push!(rows_c, i)
                    push!(cols_c, j)
                    cost = -log(p_val)
                    push!(vals_c, cost)
                    if p_val > max_p_trans
                        max_p_trans = p_val
                    end
                end
            end
        end
    end

    cents_vec = if centroids !== nothing
        hasproperty(centroids, :centroids_lonlat) ? centroids.centroids_lonlat :
        (hasproperty(centroids, :centroids) ? centroids.centroids : centroids)
    else
        nothing
    end

    # Ensure endpoints are valid, navigable marine nodes in g
    u_start = _find_navigable_node(release, cents_vec, land_mask, g)
    u_end   = _find_navigable_node(recapture, cents_vec, land_mask, g)
    if u_start == u_end
        return k !== nothing ? fill(u_start, max(1, k + 1)) : [u_start]
    end

    has_cents = cents_vec !== nothing && length(cents_vec) == S
    heuristic = if has_cents && !isempty(rows_c)
        max_d = 1e-6
        for k_idx in 1:length(rows_c)
            u = rows_c[k_idx]
            v = cols_c[k_idx]
            d = _spatial_node_distance(cents_vec[u], cents_vec[v])
            if d > max_d
                max_d = d
            end
        end
        min_cost_hop = max_p_trans > 0.0 ? -log(max_p_trans) : 0.01

        v -> begin
            if v == u_end
                return 0.0
            end
            d_v = _spatial_node_distance(cents_vec[v], cents_vec[u_end])
            hops = ceil(d_v / max_d)
            return hops * min_cost_hop
        end
    else
        v -> 0.0
    end

    distmx = sparse(rows_c, cols_c, vals_c, S, S)
    sp = a_star(g, u_start, u_end, distmx, heuristic)

    if isempty(sp)
        @warn "astar_predict_path: no marine path found between $release and " *
              "$recapture in graph topology; endpoints may reside in " *
              "disconnected marine basins."
        return [u_start]
    end

    raw_path = vcat([src(e) for e in sp], [dst(last(sp))])

    if k !== nothing && k >= 1
        m = length(raw_path) - 1
        if m < k
            p_self = [Float64(P[u, u]) for u in raw_path]
            expanded_path = copy(raw_path)
            while length(expanded_path) < k + 1
                best_idx = argmax(p_self)
                insert!(expanded_path, best_idx, expanded_path[best_idx])
                p_self[best_idx] *= 0.90
            end
            return expanded_path
        end
    end

    return raw_path
end

"""
    astar_least_cost_path(
        centroids::AbstractVector,
        W::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        land_polygons = nothing
    ) -> Vector{Int}

Computes the ecological least-cost migration corridor between `release` and `recapture`
spatial units across an environmental resistance/friction surface using \$A^*\$ graph search.

# Mathematical Formulation
Given node centroids \$\\mathbf{c}_u\$ and network adjacency \$W\$, the physical distance
between adjacent nodes is \$d(u, v) = \\text{dist}(\\mathbf{c}_u, \\mathbf{c}_v)\$.
Traversing node \$v\$ incurs an environmental friction/resistance \$\\Phi(v) \\ge 1.0\$.
The directed edge traversal cost is:
```math
c(u, v) = d(u, v) \\times \\frac{\\Phi(u) + \\Phi(v)}{2}
```
where \$\\Phi(v)\$ can be parameterized from habitat suitability:
```math
\\Phi(v) = 1.0 + 3.0 \\times (1.0 - \\text{HSI}_v)^2
```
With admissible Euclidean / Haversine heuristic \$h(u) = d(u, u_{\\text{rec}}) \\times \\min_w \\Phi(w)\$,
\$A^*\$ identifies the optimal least-resistance corridor avoiding environmental barriers.

# Arguments
- `centroids`: Spatial centroids coordinate vector (length \$S\$).
- `W`: Spatial adjacency matrix (\$S \\times S\$).
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index (1-indexed).
- `resistance`: Optional explicit resistance vector of length \$S\$ (\$\\Phi \\ge 1.0\$).
- `hsi`: Optional habitat suitability index vector (used if `resistance` is not provided).
- `land_mask`: Optional boolean vector denoting impermeable land units.
- `land_polygons`: Optional land barrier polygons for topological edge severing.

# Returns
- `Vector{Int}`: Sequence of spatial unit indices tracing the least-cost path.
"""
function astar_least_cost_path(
    centroids::AbstractVector,
    W::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    resistance::Union{Nothing, AbstractVector{<:Real}} = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    land_polygons = nothing,
    centroids_lonlat = nothing
)::Vector{Int}
    S = size(W, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        throw(ArgumentError(
            "Release ($release) and recapture ($recapture) must be within 1:$S."
        ))
    end
    if release == recapture
        return [release]
    end

    phi = if resistance !== nothing
        Float64.(resistance)
    elseif hsi !== nothing
        [1.0 + 3.0 * (1.0 - clamp(Float64(h), 0.0, 1.0))^2 for h in hsi]
    else
        ones(Float64, S)
    end
    min_phi = minimum(phi)

    W_active = copy(sparse(Float64.(W)))
    if land_mask !== nothing
        for l in findall(land_mask)
            W_active[l, :] .= 0.0
            W_active[:, l] .= 0.0
        end
        dropzeros!(W_active)
    end
    if land_polygons !== nothing
        sever_coords = centroids_lonlat !== nothing ? centroids_lonlat : centroids
        sever_land_crossing_edges!(
            W_active, sever_coords; land_polygons=land_polygons
        )
    end

    g = SimpleGraph(S)
    rows_c = Int[]
    cols_c = Int[]
    vals_c = Float64[]

    rows = rowvals(W_active)
    for col in 1:S
        for ptr in nzrange(W_active, col)
            row = rows[ptr]
            if row > col
                add_edge!(g, col, row)
                d_ij = _spatial_node_distance(centroids[col], centroids[row])
                cost = d_ij * (phi[col] + phi[row]) / 2.0
                push!(rows_c, col)
                push!(cols_c, row)
                push!(vals_c, cost)
                push!(rows_c, row)
                push!(cols_c, col)
                push!(vals_c, cost)
            end
        end
    end

    # Ensure endpoints are valid, navigable marine nodes in g
    u_start = _find_navigable_node(release, centroids, land_mask, g)
    u_end   = _find_navigable_node(recapture, centroids, land_mask, g)
    if u_start == u_end
        return [u_start]
    end

    distmx = sparse(rows_c, cols_c, vals_c, S, S)
    heuristic = v -> begin
        if v == u_end
            return 0.0
        end
        return _spatial_node_distance(centroids[v], centroids[u_end]) * min_phi
    end

    sp = a_star(g, u_start, u_end, distmx, heuristic)
    if isempty(sp)
        @warn "astar_least_cost_path: no marine path found between $release and " *
              "$recapture in active water graph; endpoints may reside in " *
              "disconnected marine basins."
        return [u_start]
    end
    return vcat([src(e) for e in sp], [dst(last(sp))])
end

"""
    smooth_marine_path(
        path::Vector{Int},
        centroids::AbstractVector;
        land_polygons = nothing,
        centroids_lonlat = nothing
    ) -> Vector{Int}

Applies line-of-sight shortcutting ("string-pulling") to an animal trajectory
on a discrete mesh, removing artificial cell-to-cell hexagonal zig-zagging
while strictly preserving clearance around terrestrial land barriers (islands,
peninsulas, headlands).

# Mathematical Formulation
For path waypoints \$\\mathbf{w} = [u_1, u_2, \\dots, u_m]\$, the algorithm
casts a line of sight between non-adjacent waypoints \$u_i\$ and \$u_j\$
(\$j > i + 1\$). If the line segment:
```math
L(u_i, u_j) = \\{ (1 - t) \\mathbf{c}_{u_i} + t \\mathbf{c}_{u_j} \\mid t \\in [0, 1] \\}
```
does not intersect any terrestrial barrier polygon (evaluated via
`_line_crosses_polygon_or_in`), intermediate waypoints \$u_{i+1}, \\dots, u_{j-1}\$
are removed. If an intersection occurs, waypoints around the headland are kept.

# Arguments
- `path`: Ordered sequence of spatial unit indices.
- `centroids`: Spatial centroids coordinate vector matching unit indices.
- `land_polygons`: Terrestrial boundary polygons (defaults to `:maritimes`).
- `centroids_lonlat`: Optional coordinates in degrees for polygon intersection.

# Returns
- `Vector{Int}`: Smoothed subset of waypoints with direct lines of sight.
"""
function smooth_marine_path(
    path::Vector{Int},
    centroids::AbstractVector;
    land_polygons = nothing,
    centroids_lonlat = nothing
)::Vector{Int}
    if length(path) <= 2
        return path
    end

    polys = if land_polygons === nothing
        _DEFAULT_MARITIMES_LAND_POLYGONS
    elseif land_polygons in (:none, :false, false)
        nothing
    elseif land_polygons in (:maritimes, :default)
        _DEFAULT_MARITIMES_LAND_POLYGONS
    else
        land_polygons
    end

    coords_check = centroids_lonlat !== nothing ? centroids_lonlat : centroids
    is_planar = abs(coords_check[1][1]) > 180.0 || abs(coords_check[1][2]) > 90.0

    if is_planar && polys !== nothing
        @warn "smooth_marine_path: centroids appear to be planar coordinates " *
              "while land_polygons are in geographic degrees; provide " *
              "centroids_lonlat to enable barrier-respecting smoothing." maxlog = 1
        return path
    end

    smoothed = Int[path[1]]
    curr_idx = 1
    n_pts = length(path)

    while curr_idx < n_pts
        furthest_idx = curr_idx + 1
        for look_idx in n_pts:-1:(curr_idx + 2)
            p_curr = (
                Float64(coords_check[path[curr_idx]][1]),
                Float64(coords_check[path[curr_idx]][2])
            )
            p_look = (
                Float64(coords_check[path[look_idx]][1]),
                Float64(coords_check[path[look_idx]][2])
            )
            crosses = polys !== nothing ?
                _line_crosses_polygon_or_in(p_curr, p_look, polys) : false
            if !crosses
                furthest_idx = look_idx
                break
            end
        end
        push!(smoothed, path[furthest_idx])
        curr_idx = furthest_idx
    end

    return smoothed
end

"""
    StochasticAStarResult

Container holding posterior inference results for stochastic A* pathfinding
propagating uncertainty in habitat suitability, friction parameters, or observation
error terms.

# Fields
- `corridor_prob::Vector{Float64}`: Posterior probability of node inclusion in corridor.
- `edge_prob::SparseMatrixCSC{Float64, Int}`: Posterior traversal probability of edge.
- `medoid_path::Vector{Int}`: Most representative trajectory across posterior draws.
- `all_paths::Vector{Vector{Int}}`: Vector of individual trajectory realizations.
- `path_costs::Vector{Float64}`: Realized path costs across all draws.
- `path_distances::Vector{Float64}`: Realized physical path distances (km).
- `mean_distance::Float64`: Posterior expected physical distance.
- `ci_distance::Tuple{Float64, Float64}`: 95% credible interval for travel distance.
- `release::Int`: Origin node.
- `recapture::Int`: Destination node.
- `n_draws::Int`: Number of stochastic draws evaluated.
"""
struct StochasticAStarResult
    corridor_prob::Vector{Float64}
    edge_prob::SparseMatrixCSC{Float64, Int}
    medoid_path::Vector{Int}
    all_paths::Vector{Vector{Int}}
    path_costs::Vector{Float64}
    path_distances::Vector{Float64}
    mean_distance::Float64
    ci_distance::Tuple{Float64, Float64}
    release::Int
    recapture::Int
    n_draws::Int
end

function Base.show(io::IO, res::StochasticAStarResult)
    S = length(res.corridor_prob)
    n_corridor = count(p -> p > 0.0, res.corridor_prob)
    n_bottleneck = count(p -> p >= 0.80, res.corridor_prob)
    println(io, "StochasticAStarResult:")
    println(io, "  Release -> Recapture:        $(res.release) -> $(res.recapture)")
    println(io, "  Stochastic Draws (M):        $(res.n_draws)")
    println(io, "  Corridor Envelope Units:     $n_corridor / $S units")
    println(io, "  Consensus Bottlenecks (P>=0.8): $n_bottleneck units")
    println(io, "  Medoid Path Waypoints:       $(length(res.medoid_path))")
    print(io,   "  Expected Path Distance:      $(round(res.mean_distance, digits=2)) km " *
                "(95% CI: $(round(res.ci_distance[1], digits=2)) - " *
                "$(round(res.ci_distance[2], digits=2)) km)")
end

"""
    astar_stochastic_least_cost_path(
        centroids::AbstractVector,
        W::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        hsi_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        hsi_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hsi_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
        resistance_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        resistance_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
        resistance_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
        n_draws::Int = 50,
        friction_power::Real = 2.0,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        land_polygons = nothing,
        smooth::Bool = false,
        ci_alpha::Real = 0.05,
        seed::Union{Nothing, Int} = nothing
    ) -> StochasticAStarResult

Performs Bayesian stochastic A* least-cost pathfinding by propagating posterior
uncertainty in habitat suitability (HSI), environmental friction, or observation error
terms across the marine graph.

# Mathematical Formulation
Deterministic A* yields a single trajectory ``\\pi^* = \\arg\\min_{\\pi} \\sum c(u, v)``
conditioned on a fixed point-estimate resistance surface ``\\hat{\\boldsymbol{\\Phi}}``.
In stochastic A*, environmental friction varies across posterior draws:
```math
H_i^{(m)} = \\text{clamp}(H_i^{\\text{obs}} + \\sigma_{H, i} Z_i^{(m)}, 0, 1)
```
or ``\\mathbf{H}^{(m)} \\sim \\pi(\\mathbf{H} \\mid \\mathbf{y})``.
Nodal friction is parameterized as:
```math
\\Phi_i^{(m)} = 1.0 + 3.0 \\left(1.0 - H_i^{(m)}\\right)^\\gamma
```
For each draw ``m = 1, \\dots, M``, A* identifies the optimal trajectory ``\\pi^{(m)}``.
The posterior corridor utilization probability across the graph is computed:
```math
P(i \\in \\text{Corridor} \\mid \\text{data}) =
  \\frac{1}{M} \\sum_{m=1}^M \\mathbb{I}(i \\in \\pi^{(m)})
```
Nodes with ``P(i) \\to 1.0`` indicate mandatory migratory bottleneck pinch-points that
must be traversed regardless of habitat uncertainty, while nodes with intermediate
``0 < P(i) < 1.0`` reveal viable alternative corridors.

# Arguments
- `centroids`: Vector of spatial unit coordinates (lon/lat tuples or planar points).
- `W`: Adjacency matrix of the spatial graph (size ``S \\times S``).
- `release`: Starting spatial unit index.
- `recapture`: Destination spatial unit index.
- `hsi_samples`: Optional ``S \\times M`` matrix of posterior HSI MCMC draws.
- `hsi_mean`: Optional posterior mean / estimated HSI vector (length ``S``).
- `hsi_se`: Optional HSI standard error / observation error vector (length ``S``).
- `resistance_samples`: Optional ``S \\times M`` matrix of friction/resistance draws.
- `resistance_mean`: Optional mean resistance vector.
- `resistance_se`: Optional resistance standard error vector.
- `n_draws`: Number of stochastic realizations (default: 50).
- `friction_power`: Exponent ``\\gamma`` for converting HSI into friction (default: 2.0).
- `land_mask`: Optional boolean vector (`true` for land units).
- `land_polygons`: Optional land boundary geometries for raycasting line-of-sight checks.
- `smooth`: If `true`, applies line-of-sight raycasting (`smooth_marine_path`) to each draw.
- `ci_alpha`: Credible interval significance level (default: 0.05 for 95% CI).
- `seed`: Optional random seed for reproducible sampling.

# Returns
- `StochasticAStarResult`: Posterior corridor probabilities, edge traversal frequencies,
  medoid path, path distance and cost credible intervals.

# References
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic
  determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics.
- Adriaensen, F., et al. (2003). The application of least-cost modelling as a functional
  landscape model. Landscape and Urban Planning, 64(4), 233-247.
"""
function astar_stochastic_least_cost_path(
    centroids::AbstractVector,
    W::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    hsi_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    hsi_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
    hsi_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
    resistance_samples::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    resistance_mean::Union{Nothing, AbstractVector{<:Real}} = nothing,
    resistance_se::Union{Nothing, AbstractVector{<:Real}} = nothing,
    n_draws::Int = 50,
    friction_power = 2.0,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    land_polygons = nothing,
    centroids_lonlat = nothing,
    smooth::Bool = false,
    ci_alpha::Real = 0.05,
    seed::Union{Nothing, Int} = nothing
)::StochasticAStarResult
    S = size(W, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        throw(ArgumentError(
            "Release ($release) and recapture ($recapture) must be within 1:$S."
        ))
    end

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    _get_power_vec(f_arg, n_sim) = if f_arg isa AbstractVector{<:Real}
        [max(0.05, Float64(f_arg[min(i, length(f_arg))])) for i in 1:n_sim]
    elseif f_arg isa Distribution
        [max(0.05, Float64(rand(rng, f_arg))) for _ in 1:n_sim]
    elseif f_arg isa Real
        fill(max(0.05, Float64(f_arg)), n_sim)
    else
        fill(2.0, n_sim)
    end

    # 1. Assemble resistance realizations
    r_draws::Matrix{Float64} = if !isnothing(resistance_samples)
        if size(resistance_samples, 1) != S
            throw(DimensionMismatch(
                "resistance_samples rows ($(size(resistance_samples, 1))) != units S ($S)"
            ))
        end
        Float64.(resistance_samples)
    elseif !isnothing(resistance_mean)
        if length(resistance_mean) != S
            throw(DimensionMismatch(
                "resistance_mean length ($(length(resistance_mean))) != units S ($S)"
            ))
        end
        M_sim = max(1, n_draws)
        draws = zeros(Float64, S, M_sim)
        mean_vec = Float64.(resistance_mean)
        if !isnothing(resistance_se)
            se_vec = Float64.(resistance_se)
            for m in 1:M_sim
                z = randn(rng, S)
                draws[:, m] = max.(1.0, mean_vec .+ se_vec .* z)
            end
        else
            for m in 1:M_sim
                draws[:, m] = max.(1.0, mean_vec)
            end
        end
        draws
    elseif !isnothing(hsi_samples)
        if size(hsi_samples, 1) != S
            throw(DimensionMismatch(
                "hsi_samples rows ($(size(hsi_samples, 1))) != units S ($S)"
            ))
        end
        M_sim = size(hsi_samples, 2)
        draws = zeros(Float64, S, M_sim)
        p_vec = _get_power_vec(friction_power, M_sim)
        for m in 1:M_sim
            p_m = p_vec[m]
            h_col = clamp.(Float64.(hsi_samples[:, m]), 0.0, 1.0)
            draws[:, m] = [1.0 + 3.0 * (1.0 - h)^p_m for h in h_col]
        end
        draws
    elseif !isnothing(hsi_mean)
        if length(hsi_mean) != S
            throw(DimensionMismatch(
                "hsi_mean length ($(length(hsi_mean))) != units S ($S)"
            ))
        end
        M_sim = max(1, n_draws)
        draws = zeros(Float64, S, M_sim)
        mean_h = Float64.(hsi_mean)
        p_vec = _get_power_vec(friction_power, M_sim)
        if !isnothing(hsi_se)
            se_h = Float64.(hsi_se)
            for m in 1:M_sim
                p_m = p_vec[m]
                z = randn(rng, S)
                h_m = clamp.(mean_h .+ se_h .* z, 0.0, 1.0)
                draws[:, m] = [1.0 + 3.0 * (1.0 - h)^p_m for h in h_m]
            end
        else
            for m in 1:M_sim
                p_m = p_vec[m]
                h_m = clamp.(mean_h, 0.0, 1.0)
                draws[:, m] = [1.0 + 3.0 * (1.0 - h)^p_m for h in h_m]
            end
        end
        draws
    else
        throw(ArgumentError(
            "Either hsi_mean/se, hsi_samples, or resistance_mean/se must be provided."
        ))
    end

    M_total = size(r_draws, 2)
    all_paths = Vector{Vector{Int}}(undef, M_total)
    path_costs = zeros(Float64, M_total)
    path_distances = zeros(Float64, M_total)
    node_counts = zeros(Int, S)
    edge_counts = spzeros(Float64, S, S)

    # 2. Iterate across stochastic draws
    for m in 1:M_total
        r_m = r_draws[:, m]

        # Solve A* least-cost trajectory
        p_m = astar_least_cost_path(
            centroids,
            W,
            release,
            recapture;
            resistance = r_m,
            land_mask = land_mask,
            land_polygons = land_polygons,
            centroids_lonlat = centroids_lonlat
        )

        if smooth && length(p_m) > 2
            p_m = smooth_marine_path(
                p_m,
                centroids;
                land_polygons = land_polygons,
                centroids_lonlat = centroids_lonlat
            )
        end

        all_paths[m] = p_m

        # Accumulate node visits
        for u in p_m
            node_counts[u] += 1
        end

        # Accumulate edge visits and metrics
        dist_m = 0.0
        cost_m = 0.0
        for idx in 1:(length(p_m) - 1)
            u = p_m[idx]
            v = p_m[idx + 1]
            edge_counts[u, v] += 1.0
            d_uv = _spatial_node_distance(centroids[u], centroids[v])
            dist_m += d_uv
            cost_m += d_uv * (r_m[u] + r_m[v]) / 2.0
        end

        path_distances[m] = dist_m
        path_costs[m] = cost_m
    end

    # 3. Compute posterior corridor probabilities
    corridor_prob = node_counts ./ Float64(M_total)
    edge_prob = edge_counts ./ Float64(M_total)

    # 4. Identify medoid trajectory (highest mean corridor confidence)
    best_score = -Inf
    best_idx = 1
    for m in 1:M_total
        p_len = length(all_paths[m])
        score_m = p_len > 0 ? sum(corridor_prob[u] for u in all_paths[m]) / p_len : 0.0
        if score_m > best_score
            best_score = score_m
            best_idx = m
        end
    end
    medoid = all_paths[best_idx]

    # 5. Compute distance summary statistics
    mean_dist = mean(path_distances)
    alpha_lo = clamp(Float64(ci_alpha) / 2.0, 0.0, 0.5)
    alpha_hi = 1.0 - alpha_lo
    ci_dist = (quantile(path_distances, alpha_lo), quantile(path_distances, alpha_hi))

    return StochasticAStarResult(
        corridor_prob,
        edge_prob,
        medoid,
        all_paths,
        path_costs,
        path_distances,
        mean_dist,
        ci_dist,
        release,
        recapture,
        M_total
    )
end

"""
    astar_stochastic_predict_path(
        P::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int;
        P_samples::Union{Nothing, Vector{<:AbstractMatrix{<:Real}}} = nothing,
        temperature::Real = 0.05,
        n_draws::Int = 50,
        centroids = nothing,
        k::Union{Nothing, Int} = nothing,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        ci_alpha::Real = 0.05,
        seed::Union{Nothing, Int} = nothing
    ) -> StochasticAStarResult

Performs stochastic A* path prediction across uncertain transition probability matrices.
"""
function astar_stochastic_predict_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    P_samples::Union{Nothing, Vector{<:AbstractMatrix{<:Real}}} = nothing,
    temperature::Real = 0.05,
    n_draws::Int = 50,
    centroids = nothing,
    k::Union{Nothing, Int} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    ci_alpha::Real = 0.05,
    seed::Union{Nothing, Int} = nothing
)::StochasticAStarResult
    S = size(P, 1)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    M_total = !isnothing(P_samples) ? length(P_samples) : max(1, n_draws)

    all_paths = Vector{Vector{Int}}(undef, M_total)
    path_costs = zeros(Float64, M_total)
    path_distances = zeros(Float64, M_total)
    node_counts = zeros(Int, S)
    edge_counts = spzeros(Float64, S, S)
    temp = max(1e-6, Float64(temperature))

    for m in 1:M_total
        P_m = if !isnothing(P_samples)
            P_samples[m]
        else
            P_pert = copy(P)
            I_nz, J_nz, V_nz = findnz(sparse(P))
            V_new = zeros(Float64, length(V_nz))
            for idx in eachindex(V_nz)
                v = max(1e-12, Float64(V_nz[idx]))
                log_v = log(v) + temp * randn(rng)
                V_new[idx] = exp(log_v)
            end
            P_sparse = sparse(I_nz, J_nz, V_new, S, S)
            row_sums = vec(sum(P_sparse, dims=2))
            D_inv = Diagonal([rs > 0.0 ? 1.0 / rs : 1.0 for rs in row_sums])
            D_inv * P_sparse
        end

        p_m = astar_predict_path(
            P_m,
            release,
            recapture;
            centroids = centroids,
            k = k,
            land_mask = land_mask
        )

        all_paths[m] = p_m
        for u in p_m
            node_counts[u] += 1
        end

        dist_m = 0.0
        cost_m = 0.0
        for idx in 1:(length(p_m) - 1)
            u = p_m[idx]
            v = p_m[idx + 1]
            edge_counts[u, v] += 1.0
            if !isnothing(centroids)
                dist_m += _spatial_node_distance(centroids[u], centroids[v])
            else
                dist_m += 1.0
            end
            p_uv = P_m[u, v]
            cost_m += p_uv > 0.0 ? -log(p_uv) : 25.0
        end
        path_distances[m] = dist_m
        path_costs[m] = cost_m
    end

    corridor_prob = node_counts ./ Float64(M_total)
    edge_prob = edge_counts ./ Float64(M_total)

    best_score = -Inf
    best_idx = 1
    for m in 1:M_total
        p_len = length(all_paths[m])
        score_m = p_len > 0 ? sum(corridor_prob[u] for u in all_paths[m]) / p_len : 0.0
        if score_m > best_score
            best_score = score_m
            best_idx = m
        end
    end
    medoid = all_paths[best_idx]

    mean_dist = mean(path_distances)
    alpha_lo = clamp(Float64(ci_alpha) / 2.0, 0.0, 0.5)
    alpha_hi = 1.0 - alpha_lo
    ci_dist = (quantile(path_distances, alpha_lo), quantile(path_distances, alpha_hi))

    return StochasticAStarResult(
        corridor_prob,
        edge_prob,
        medoid,
        all_paths,
        path_costs,
        path_distances,
        mean_dist,
        ci_dist,
        release,
        recapture,
        M_total
    )
end

"""
    predict_path(P::AbstractMatrix{<:Real}, release::Int, recapture::Int,
                 k::Union{Nothing, Int}=nothing;
                 centroids=nothing, method=:astar, land_mask=nothing) -> Vector{Int}

Computes the single most likely sequence of spatial units visited by an individual
between `release` and `recapture` using either goal-directed \$A^*\$ heuristic search
(`method=:astar`, default) or the classic fixed-horizon Viterbi trellis (`method=:viterbi`).

# Arguments
- `P`: Row-stochastic transition matrix (size ``S \\times S``).
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index (1-indexed).
- `k`: Optional discrete time steps elapsed between release and recapture (``k \\ge 1``).
- `centroids`: Optional spatial centroids coordinate vector for \$A^*\$ distance heuristic.
- `method`: Algorithm selector (`:astar` for high-performance \$A^*\$,
  `:viterbi` for classic trellis).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units).

# Returns
- `Vector{Int}`: Sequence of spatial unit indices from `release` to `recapture`.
"""
function predict_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    centroids = nothing,
    method::Symbol = :astar,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    if k === nothing || method == :astar
        return astar_predict_path(
            P, release, recapture;
            centroids = centroids,
            k = k,
            land_mask = land_mask
        )
    elseif method == :viterbi
        S = size(P, 1)
        if !(1 <= release <= S) || !(1 <= recapture <= S)
            throw(ArgumentError(
                "Release unit ($release) and recapture unit ($recapture) must be within 1:$S."
            ))
        end
        if k < 1
            return [release]
        end
        if release == recapture
            return fill(release, k + 1)
        end
        if k == 1
            has_p = Float64(P[release, recapture]) > 1e-12
            no_land = land_mask === nothing ||
                      (!land_mask[release] && !land_mask[recapture])
            if has_p && no_land
                return [release, recapture]
            else
                return astar_predict_path(
                    P, release, recapture;
                    centroids = centroids, k = k, land_mask = land_mask
                )
            end
        end

        logP = Matrix{Float64}(undef, S, S)
        @inbounds for j in 1:S
            for i in 1:S
                p_ij = Float64(P[i, j])
                logP[i, j] = p_ij > 1e-15 ? log(p_ij) : -1e12
            end
        end

        if land_mask !== nothing
            for l in 1:S
                if land_mask[l]
                    logP[:, l] .= -1e12
                    logP[l, :] .= -1e12
                end
            end
        end

        delta = fill(-1e12, S, k + 1)
        psi   = zeros(Int, S, k + 1)
        delta[release, 1] = 0.0

        for tau in 2:(k + 1)
            prev_tau = tau - 1
            for j in 1:S
                best_val = -Inf
                best_prev = 1
                for i in 1:S
                    score = delta[i, prev_tau] + logP[i, j]
                    if score > best_val
                        best_val = score
                        best_prev = i
                    end
                end
                delta[j, tau] = best_val
                psi[j, tau]   = best_prev
            end
        end

        if delta[recapture, k + 1] <= -1e11
            @warn "Viterbi: no valid marine path of length $k found between " *
                  "$release and $recapture; falling back to marine A* path."
            return astar_predict_path(
                P, release, recapture;
                centroids = centroids, k = k, land_mask = land_mask
            )
        end

        path = zeros(Int, k + 1)
        path[k + 1] = recapture
        for tau in (k + 1):-1:2
            path[tau - 1] = psi[path[tau], tau]
        end
        return path
    else
        throw(ArgumentError("Unknown path method '$method'. Expected :astar or :viterbi."))
    end
end


"""
    predict_corridor(P::AbstractMatrix{<:Real}, release::Int, recapture::Int, k::Int;
                     land_mask=nothing) -> Matrix{Float64}

Computes the Markov bridge probability distribution across all spatial units at each
intermediate time step ``\\tau \\in \\{0, 1, \\dots, k\\}``:

```math
\\mathbb{P}(X_\\tau = j \\mid X_0 = u_{\\text{rel}}, X_k = u_{\\text{rec}}) = 
\\frac{[P^\\tau]_{u_{\\text{rel}}, j} \\cdot [P^{k - \\tau}]_{j, u_{\\text{rec}}}}{[P^k]_{u_{\\text{rel}}, u_{\\text{rec}}}}
```

# Arguments
- `P`: Row-stochastic transition matrix (size ``S \\times S``).
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Total discrete time steps elapsed between release and recapture (``k \\ge 1``).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units). When provided,
  all probability mass on land units is strictly zeroed out and columns are renormalized
  over navigable marine units.

# Returns
- `Matrix{Float64}`: Array of size ``(S, k + 1)`` where column ``\\tau + 1`` gives the spatial
  probability distribution over all ``S`` units at time step ``\\tau``. Each column sums to 1.0.
"""
function predict_corridor(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    S = size(P, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        error("Release unit ($release) and recapture unit ($recapture) must be within 1:$S.")
    end

    corridor = zeros(Float64, S, k + 1)

    if k <= 0
        corridor[release, 1] = 1.0
        return corridor
    end

    # Iterative sparse-vector multiplication instead of dense matrix powers
    # Forward probabilities: v_fwd[j, tau+1] = P(X_tau = j | X_0 = release)
    v_fwd = zeros(Float64, S, k + 1)
    v_fwd[release, 1] = 1.0
    P_t = P' # Transpose once for efficiency
    for tau in 1:k
        v_fwd[:, tau + 1] = P_t * v_fwd[:, tau]
    end

    # Backward probabilities: v_bwd[j, m+1] = P(X_k = recapture | X_{k-m} = j)
    v_bwd = zeros(Float64, S, k + 1)
    v_bwd[recapture, 1] = 1.0
    for m in 1:k
        v_bwd[:, m + 1] = P * v_bwd[:, m]
    end

    # Total likelihood of transitioning from release to recapture in k steps
    P_total = v_fwd[recapture, k + 1]

    if P_total <= 1e-15
        @warn "Recapture unit $recapture has near-zero reachability from release $release in $k steps."
        if land_mask !== nothing
            n_water = count(!, land_mask)
            w_val = n_water > 0 ? 1.0 / n_water : 1.0 / S
            for j in 1:S
                corridor[j, :] .= land_mask[j] ? 0.0 : w_val
            end
        else
            corridor[:, :] .= 1.0 / S
        end
        corridor[release, 1] = 1.0
        corridor[recapture, k + 1] = 1.0
        return corridor
    end

    # Markov bridge formula for each intermediate step tau = 0 .. k
    for tau in 0:k
        tau_idx = tau + 1
        rem_idx = (k - tau) + 1
        for j in 1:S
            prob_fwd = v_fwd[j, tau_idx]
            prob_bwd = v_bwd[j, rem_idx]
            corridor[j, tau_idx] = (prob_fwd * prob_bwd) / P_total
        end
        # Enforce land barrier: strictly zero probability on land
        if land_mask !== nothing
            for l in 1:S
                if land_mask[l]
                    corridor[l, tau_idx] = 0.0
                end
            end
        end
        # Normalize column
        col_sum = sum(view(corridor, :, tau_idx))
        if col_sum > 0.0
            corridor[:, tau_idx] ./= col_sum
        end
    end

    return corridor
end


"""
    predict_path(res::NamedTuple, release::Int, recapture::Int, k=nothing; kwargs...) -> Vector{Int}

Extracts the group transition matrix and domain mesh from a fitted MovementAnalysis result
NamedTuple and predicts the most probable movement trajectory between release and
recapture nodes using goal-directed search (`:astar`) or dynamic programming (`:viterbi`).

# Arguments
- `res`: Result NamedTuple containing `:transition_matrices` and `:mesh`.
- `release`: Starting spatial unit index.
- `recapture`: Destination spatial unit index.
- `k`: Total discrete time steps (optional for `:astar`).
- `group`: Demographic group identifier (string, symbol, or integer ID).
- `centroids`: Optional centroid coordinates (defaults to `res.mesh`).
- `method`: Algorithm (`:astar` or `:viterbi`).
- `land_mask`: Optional boolean land mask vector.

# Returns
- `Vector{Int}`: Sequence of traversed spatial unit indices.
"""
function predict_path(
    res::NamedTuple,
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    group::Union{String, Symbol, Int} = 1,
    centroids = nothing,
    method::Symbol = :astar,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    mask = land_mask !== nothing ? land_mask :
        (hasproperty(res, :land_mask) ? res.land_mask : nothing)
    cents = centroids !== nothing ? centroids :
        (hasproperty(res, :mesh) ? res.mesh : nothing)
    if hasproperty(res, :transition_matrices)
        tm = res.transition_matrices
        key = group isa Int ?
            (hasproperty(res, :group_lookup) ? res.group_lookup[group] : first(keys(tm))) :
            string(group)
        P = tm[key]
        return predict_path(
            P, release, recapture, k;
            centroids = cents,
            method = method,
            land_mask = mask
        )
    else
        throw(ArgumentError("Expected a result NamedTuple with field `:transition_matrices`."))
    end
end

"""
    predict_corridor(res::NamedTuple, release::Int, recapture::Int, k::Int; kwargs...) -> Matrix{Float64}

Extracts the group transition matrix from a fitted MovementAnalysis result NamedTuple and
computes the space-time Markov bridge corridor matrix over ``k`` steps.

# Arguments
- `res`: Result NamedTuple containing `:transition_matrices`.
- `release`: Starting spatial unit index.
- `recapture`: Destination spatial unit index.
- `k`: Total discrete time steps (``k \\ge 1``).
- `group`: Demographic group identifier (string, symbol, or integer ID).
- `land_mask`: Optional boolean land mask vector.

# Returns
- `Matrix{Float64}`: State probability matrix of size ``S \\times (k + 1)``.
"""
function predict_corridor(
    res::NamedTuple, release::Int, recapture::Int, k::Int;
    group::Union{String, Symbol, Int} = 1,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)
    mask = land_mask !== nothing ? land_mask :
        (hasproperty(res, :land_mask) ? res.land_mask : nothing)
    if hasproperty(res, :transition_matrices)
        tm = res.transition_matrices
        key = group isa Int ?
            (hasproperty(res, :group_lookup) ? res.group_lookup[group] : first(keys(tm))) :
            string(group)
        P = tm[key]
        return predict_corridor(P, release, recapture, k; land_mask=mask)
    else
        error("Expected a result NamedTuple with field `:transition_matrices`.")
    end
end

"""
    predict_path(P_vec::AbstractVector{<:AbstractMatrix{<:Real}}, release::Int, recapture::Int, k::Int;
                 group::Union{Integer, Symbol, AbstractString} = 1,
                 land_mask::Union{Nothing, AbstractVector{Bool}} = nothing) -> Vector{Int}

Group-aware overload for `predict_path` when given a vector of group-specific transition matrices.
Reconstructs the most likely sequence of spatial units (Viterbi path) for the specified group.

# Mathematical Formulation
Given group index ``g``, the dynamic programming Viterbi trellis identifies:
```math
\\mathbf{s}^* = \\arg\\max_{\\mathbf{s}} \\prod_{\\tau=1}^k [P_g]_{s_{\\tau-1}, s_\\tau}
```
subject to ``s_0 = u_{\\text{rel}}`` and ``s_k = u_{\\text{rec}}``.

# Arguments
- `P_vec`: Collection of row-stochastic transition matrices for each biological group.
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Number of discrete time intervals elapsed.
- `group`: 1-based integer index or group label.
- `land_mask`: Optional boolean mask of impermeable barrier units.

# Returns
- `Vector{Int}`: Sequence of ``k + 1`` spatial unit indices from `release` to `recapture`.
"""
function predict_path(
    P_vec::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    group::Union{Integer, Symbol, AbstractString} = 1,
    centroids = nothing,
    method::Symbol = :astar,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    if isempty(P_vec)
        throw(ArgumentError("P_vec transition kernel vector cannot be empty."))
    end
    g_idx = if group isa Integer
        Int(group)
    else
        parsed = tryparse(Int, string(group))
        parsed !== nothing ? parsed : 1
    end
    if !(1 <= g_idx <= length(P_vec))
        throw(ArgumentError("Group index $g_idx is out of bounds (1:$(length(P_vec)))."))
    end
    return predict_path(
        P_vec[g_idx], release, recapture, k;
        centroids = centroids,
        method = method,
        land_mask = land_mask
    )
end

"""
    predict_corridor(P_vec::AbstractVector{<:AbstractMatrix{<:Real}}, release::Int, recapture::Int, k::Int;
                     group::Union{Integer, Symbol, AbstractString} = 1,
                     land_mask::Union{Nothing, AbstractVector{Bool}} = nothing) -> Matrix{Float64}

Group-aware overload for `predict_corridor` when given a vector of group-specific transition matrices.
Computes the Markov bridge probability distribution across spatial units for the specified group.

# Mathematical Formulation
Given group index ``g``, the Markov bridge probability at intermediate step ``\\tau`` is:
```math
\\mathbb{P}(X_\\tau = j \\mid X_0 = u_{\\text{rel}}, X_k = u_{\\text{rec}}) = 
\\frac{[P_g^\\tau]_{u_{\\text{rel}}, j} \\cdot [P_g^{k - \\tau}]_{j, u_{\\text{rec}}}}{[P_g^k]_{u_{\\text{rel}}, u_{\\text{rec}}}}
```

# Arguments
- `P_vec`: Collection of row-stochastic transition matrices for each biological group.
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Number of discrete time intervals elapsed.
- `group`: 1-based integer index or group label.
- `land_mask`: Optional boolean mask of impermeable barrier units.

# Returns
- `Matrix{Float64}`: Array of size ``(S, k + 1)`` where column ``\\tau + 1`` gives the spatial
  probability distribution over all units at time step ``\\tau``.
"""
function predict_corridor(
    P_vec::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int,
    k::Int;
    group::Union{Integer, Symbol, AbstractString} = 1,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    if isempty(P_vec)
        throw(ArgumentError("P_vec transition kernel vector cannot be empty."))
    end
    g_idx = if group isa Integer
        Int(group)
    else
        parsed = tryparse(Int, string(group))
        parsed !== nothing ? parsed : 1
    end
    if !(1 <= g_idx <= length(P_vec))
        throw(ArgumentError("Group index $g_idx is out of bounds (1:$(length(P_vec)))."))
    end
    return predict_corridor(P_vec[g_idx], release, recapture, k; land_mask=land_mask)
end


# ==============================================================================
# Domain Tessellation, Autocorrelation Infilling & Land Barrier Engine
# ==============================================================================

"""
    point_in_polygon(x::Real, y::Real, poly) -> Bool

Tests whether planar or geographic coordinate `(x, y)` lies inside a closed 2D polygon
or MultiPolygon using the classical ray-casting (even-odd crossing) algorithm.

# Mathematical Formulation
A horizontal ray is cast from point ``(x, y)`` along the positive ``x``-axis to ``+\\infty``.
For each polygon edge connecting vertices ``(x_i, y_i)`` and ``(x_j, y_j)``:
```math
\\text{crosses}(e) = ((y_i > y) \\ne (y_j > y)) \\land 
\\left( x < \\frac{(x_j - x_i)(y - y_i)}{y_j - y_i} + x_i \\right)
```
The query point is inside if and only if the total edge crossing count is odd:
```math
\\text{inside} = \\left( \\sum_{e \\in E} \\mathbb{I}[\\text{crosses}(e)] \\right) \\pmod 2 = 1
```

# Arguments
- `x::Real`: Longitude or planar x-coordinate of query point.
- `y::Real`: Latitude or planar y-coordinate of query point.
- `poly`: Polygon geometry representation. Supports:
  - `Vector{Tuple{<:Real, <:Real}}` or `Vector{<:AbstractVector{<:Real}}`: Single polygon ring.
  - `Vector{Vector{...}}`: Collection of polygon rings (MultiPolygon / archipelago).
  - `LibGEOS.AbstractGeometry`: Geometric polygon object via LibGEOS.

# Returns
- `Bool`: `true` if `(x, y)` is inside the polygon interior, `false` otherwise.
"""
function point_in_polygon(x::Real, y::Real, poly)::Bool
    if poly isa AbstractVector && !isempty(poly) && first(poly) isa AbstractVector
        for sub_poly in poly
            if point_in_polygon(x, y, sub_poly)
                return true
            end
        end
        return false
    end

    n = length(poly)
    n < 3 && return false
    inside = false
    j = n
    px = Float64(x)
    py = Float64(y)
    @inbounds for i in 1:n
        pt_i = poly[i]
        pt_j = poly[j]
        xi = Float64(pt_i[1])
        yi = Float64(pt_i[2])
        xj = Float64(pt_j[1])
        yj = Float64(pt_j[2])

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

"""
    point_in_polygon(x::Real, y::Real, geom::LibGEOS.AbstractGeometry) -> Bool

Evaluates geometric inclusion of 2D coordinates ``(x, y)`` within a LibGEOS
spatial polygon or multi-polygon geometry object.

# Arguments
- `x, y`: Coordinate values in the geometry's coordinate system.
- `geom`: `LibGEOS.AbstractGeometry` polygon representation.

# Returns
- `Bool`: `true` if point lies strictly inside or on the boundary of the geometry.
"""
function point_in_polygon(x::Real, y::Real, geom::LibGEOS.AbstractGeometry)::Bool
    pt = LibGEOS.Point(Float64(x), Float64(y))
    return LibGEOS.contains(geom, pt)
end


"""
Curated geographic boundary coordinates for the Canadian Maritime region, enclosing
major terrestrial landmasses while preserving marine straits (Cabot Strait, Northumberland
Strait, Bay of Fundy, Chedabucto Bay, and Scotian Shelf coastal waters).
"""
const _DEFAULT_MARITIMES_LAND_POLYGONS = [
    # 1. Cape Breton Island
    [
        (-60.45, 47.05), (-60.38, 46.90), (-60.38, 46.65), (-60.45, 46.30),
        (-60.15, 46.25), (-59.75, 46.15), (-59.85, 45.85), (-60.50, 45.60),
        (-61.00, 45.48), (-61.40, 45.65), (-61.50, 45.90), (-61.25, 46.25),
        (-61.05, 46.65), (-60.75, 46.90), (-60.55, 47.05), (-60.45, 47.05)
    ],
    # 2. Nova Scotia Mainland
    [
        (-61.40, 45.60), (-61.50, 45.45), (-61.20, 45.35), (-60.99, 45.33),
        (-61.50, 45.10), (-62.50, 44.80), (-63.50, 44.55), (-64.20, 44.40),
        (-64.70, 44.00), (-65.30, 43.50), (-65.65, 43.40), (-66.15, 43.80),
        (-66.25, 44.20), (-65.75, 44.65), (-65.00, 45.15), (-64.35, 45.35),
        (-64.40, 45.85), (-63.70, 45.85), (-62.60, 45.75), (-61.90, 45.75),
        (-61.40, 45.60)
    ],
    # 3. Prince Edward Island
    [
        (-64.40, 46.95), (-64.00, 46.60), (-63.50, 46.50), (-62.50, 46.45),
        (-61.95, 46.40), (-62.40, 46.00), (-63.00, 45.95), (-63.80, 46.20),
        (-64.40, 46.60), (-64.40, 46.95)
    ],
    # 4. New Brunswick & Quebec / Gaspe coast
    [
        (-64.40, 45.85), (-64.50, 45.80), (-64.70, 45.50), (-65.50, 45.30),
        (-67.00, 45.00), (-68.50, 45.50), (-69.50, 48.00), (-68.00, 49.00),
        (-65.00, 49.20), (-64.20, 48.80), (-64.80, 48.00), (-64.80, 47.00),
        (-64.50, 46.20), (-64.40, 45.85)
    ]
]


"""
    identify_land_units(
        centroids::AbstractVector;
        land_polygons::Union{Nothing, Symbol, AbstractVector} = nothing,
        depth::Union{Nothing, AbstractVector{<:Real}} = nothing,
        depth_threshold::Real = 0.0
    ) -> Vector{Bool}

Classifies spatial partitioning units as terrestrial land (`true`) or navigable marine water
(`false`) using boundary polygons, bathymetric depth thresholds, or regional defaults.

# Mathematical Formulation
A spatial unit ``s \\in \\{1, \\dots, S\\}`` with centroid ``\\mathbf{c}_s = (x_s, y_s)`` is classified
as land if it satisfies either geometric boundary polygon inclusion or the bathymetric threshold:
```math
\\text{is\\_land}(s) = (\\exists p \\in \\mathcal{P} : \\mathbf{c}_s \\in p) \\lor (d_s \\le d_{\\text{thresh}})
```
where ``\\mathcal{P}`` is the set of land boundary polygons and ``d_s`` is the bathymetric depth.

# Arguments
- `centroids`: Vector of coordinate tuples `(x, y)` or `(lon, lat)` for all ``S`` units.
- `land_polygons`: Optional polygon or collection of polygons.
  - `nothing`: If `depth` is also `nothing`, defaults to `:maritimes` (curated NW Atlantic coastline).
  - `:maritimes` or `:default`: Uses built-in Nova Scotia, Cape Breton, PEI, and NB/QC polygons.
  - `Vector{...}`: User-provided polygon rings or MultiPolygon coordinate vectors.
- `depth`: Optional bathymetric depth vector of length ``S`` (positive values indicate water depth).
- `depth_threshold`: Scalar threshold below which a unit is classified as land (default `0.0`).

# Returns
- `Vector{Bool}`: Boolean mask of length ``S`` where `true` indicates a terrestrial land unit.
"""
function identify_land_units(
    centroids::AbstractVector;
    polygons::Union{Nothing, AbstractVector} = nothing,
    land_polygons::Union{Nothing, Symbol, AbstractVector} = nothing,
    depth::Union{Nothing, AbstractVector{<:Real}} = nothing,
    depth_threshold::Real = 0.0,
    land_fraction_threshold::Real = 0.5
)::Vector{Bool}
    S = length(centroids)
    land_mask = falses(S)

    # 1. Depth thresholding
    if depth !== nothing
        if length(depth) != S
            throw(DimensionMismatch(
                "Depth vector length ($(length(depth))) must match centroids length ($S)."
            ))
        end
        for i in 1:S
            if isfinite(depth[i]) && depth[i] <= depth_threshold
                land_mask[i] = true
            end
        end
    end

    # 2. Polygon boundaries
    polys = if land_polygons === nothing
        _DEFAULT_MARITIMES_LAND_POLYGONS
    elseif land_polygons in (:none, :false, false)
        nothing
    elseif land_polygons in (:maritimes, :default)
        _DEFAULT_MARITIMES_LAND_POLYGONS
    else
        land_polygons
    end

    if polys !== nothing
        for i in 1:S
            land_mask[i] && continue
            c = centroids[i]
            x, y = Float64(c[1]), Float64(c[2])
            # Check centroid inclusion
            if point_in_polygon(x, y, polys)
                land_mask[i] = true
                continue
            end
            # Check unit polygon vertices if provided
            if polygons !== nothing && i <= length(polygons)
                u_poly = polygons[i]
                if length(u_poly) >= 3
                    n_verts = length(u_poly) - (u_poly[1] == u_poly[end] ? 1 : 0)
                    n_in = count(k -> point_in_polygon(u_poly[k][1], u_poly[k][2], polys), 1:n_verts)
                    if n_in >= ceil(Int, n_verts * land_fraction_threshold)
                        land_mask[i] = true
                    end
                end
            end
        end
    end

    return land_mask
end


"""
    apply_land_barrier(
        W::AbstractMatrix{<:Real},
        hsi::AbstractVector{<:Real},
        land_mask::AbstractVector{Bool}
    ) -> Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}

Enforces impassable terrestrial movement barriers on the spatial adjacency graph ``W`` and
habitat suitability vector ``\\mathbf{h}``, severing all topological connections across land.

# Mathematical Formulation
Let ``\\mathcal{L} = \\{l \\mid \\text{land\\_mask}[l] = \\text{true}\\}`` be the set of land units,
and ``\\mathcal{W} = \\{w \\mid \\text{land\\_mask}[w] = \\text{false}\\}`` be marine water units.
All directed and undirected edges incident to any land unit are strictly eliminated:
```math
W^{\\text{water}}_{ij} = \\begin{cases} 
W_{ij} & \\text{if } i \\in \\mathcal{W} \\land j \\in \\mathcal{W} \\\\
0 & \\text{otherwise}
\\end{cases}
```
Habitat suitability values on land are set to zero:
```math
h^{\\text{water}}_i = \\begin{cases}
h_i & \\text{if } i \\in \\mathcal{W} \\\\
0.0 & \\text{if } i \\in \\mathcal{L}
\\end{cases}
```
This guarantees that topological random walks (``T_{\\text{diff}}``) and directed advection (``A``)
have zero probability of transitioning onto or through land masses:
```math
P(X_{t+1} \\in \\mathcal{L} \\mid X_t \\in \\mathcal{W}) = 0
```

# Arguments
- `W`: Spatial graph adjacency matrix (size ``S \\times S``).
- `hsi`: Habitat suitability vector (length ``S``).
- `land_mask`: Boolean vector of length ``S`` (`true` for land units).

# Returns
- `Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}`:
  - `W_water`: Severed adjacency matrix with all land connections removed and zeros dropped.
  - `hsi_water`: Habitat suitability vector with land units zeroed out.
"""
function apply_land_barrier(
    W::AbstractMatrix{<:Real},
    hsi::AbstractVector{<:Real},
    land_mask::AbstractVector{Bool}
)::Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}
    S = size(W, 1)
    if length(hsi) != S || length(land_mask) != S
        throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), hsi has length $(length(hsi)), " *
            "and land_mask has length $(length(land_mask))."
        ))
    end

    W_water = copy(sparse(Float64.(W)))
    for l in findall(land_mask)
        W_water[l, :] .= 0.0
        W_water[:, l] .= 0.0
    end
    dropzeros!(W_water)

    hsi_water = copy(Float64.(hsi))
    hsi_water[land_mask] .= 0.0

    return (W_water, hsi_water)
end


function _extract_polygon_rings(polys)
    if polys isa AbstractVector && !isempty(polys)
        first_elem = first(polys)
        if first_elem isa Tuple || (first_elem isa AbstractVector && length(first_elem) == 2 && first_elem[1] isa Real)
            return [polys]
        elseif first_elem isa AbstractVector
            return polys
        end
    end
    return [polys]
end

function _line_crosses_polygon_or_in(p1, p2, polys)::Bool
    # Check midpoint inclusion in land polygon
    mid_x = (p1[1] + p2[1]) / 2.0
    mid_y = (p1[2] + p2[2]) / 2.0
    if point_in_polygon(mid_x, mid_y, polys)
        return true
    end

    # Orientation test for 2D line segment intersection
    function _ccw(A, B, C)
        return (C[2] - A[2]) * (B[1] - A[1]) > (B[2] - A[2]) * (C[1] - A[1])
    end

    function _seg_intersect(a1, a2, b1, b2)
        return (_ccw(a1, b1, b2) != _ccw(a2, b1, b2)) && (_ccw(a1, a2, b1) != _ccw(a1, a2, b2))
    end

    rings = _extract_polygon_rings(polys)
    for ring in rings
        if ring isa AbstractVector && length(ring) >= 3
            n_pts = length(ring)
            for k in 1:(n_pts - 1)
                e1 = (Float64(ring[k][1]), Float64(ring[k][2]))
                e2 = (Float64(ring[k + 1][1]), Float64(ring[k + 1][2]))
                if _seg_intersect(p1, p2, e1, e2)
                    return true
                end
            end
        end
    end
    return false
end

"""
    sever_land_crossing_edges!(
        W::AbstractMatrix{<:Real},
        centroids::AbstractVector;
        land_polygons = nothing
    ) -> Int

Inspects all non-zero edges ``(i, j)`` in the spatial adjacency matrix ``W`` and
severs any edge whose straight-line trajectory intersects a terrestrial land barrier polygon.

# Mathematical Formulation
For nodes ``i, j \\in \\mathcal{S}``, the trajectory line segment is:
```math
L_{ij} = \\{ (1 - t) \\mathbf{c}_i + t \\mathbf{c}_j \\mid t \\in [0, 1] \\}
```
If ``L_{ij} \\cap \\mathcal{P}_{\\text{land}} \\neq \\emptyset``, the edge is severed:
```math
W_{ij} \\leftarrow 0, \\quad W_{ji} \\leftarrow 0
```
preventing movement models from allowing transitions that cross overland barriers.

# Arguments
- `W`: Spatial graph adjacency matrix (size ``S \\times S``). Modified in-place if mutable.
- `centroids`: Centroids coordinate vector (length ``S``).
- `land_polygons`: Boundary polygons (defaults to `:maritimes`).

# Returns
- `Int`: Total number of directed edges severed.
"""
function sever_land_crossing_edges!(
    W::AbstractMatrix{<:Real},
    centroids::AbstractVector;
    land_polygons = nothing
)::Int
    polys = if land_polygons === nothing
        _DEFAULT_MARITIMES_LAND_POLYGONS
    elseif land_polygons in (:none, :false, false)
        return 0
    elseif land_polygons in (:maritimes, :default)
        _DEFAULT_MARITIMES_LAND_POLYGONS
    else
        land_polygons
    end

    polys === nothing && return 0

    S = size(W, 1)
    if length(centroids) != S
        throw(DimensionMismatch(
            "centroids length ($(length(centroids))) must match W dimensions ($S)."
        ))
    end

    severed_count = 0
    if W isa SparseMatrixCSC
        rows = rowvals(W)
        for col in 1:S
            c_col = centroids[col]
            p_col = (Float64(c_col[1]), Float64(c_col[2]))
            for k in nzrange(W, col)
                row = rows[k]
                if row > col
                    c_row = centroids[row]
                    p_row = (Float64(c_row[1]), Float64(c_row[2]))
                    if _line_crosses_polygon_or_in(p_col, p_row, polys)
                        W[row, col] = 0.0
                        W[col, row] = 0.0
                        severed_count += 2
                    end
                end
            end
        end
        dropzeros!(W)
    else
        for i in 1:S
            c_i = centroids[i]
            p_i = (Float64(c_i[1]), Float64(c_i[2]))
            for j in (i + 1):S
                if W[i, j] != 0.0 || W[j, i] != 0.0
                    c_j = centroids[j]
                    p_j = (Float64(c_j[1]), Float64(c_j[2]))
                    if _line_crosses_polygon_or_in(p_i, p_j, polys)
                        W[i, j] = 0.0
                        W[j, i] = 0.0
                        severed_count += 2
                    end
                end
            end
        end
    end

    return severed_count
end

"""
    compact_marine_mesh(mesh, land_mask::AbstractVector{Bool}) -> NamedTuple

Extracts and compacts a spatial mesh into an active marine water sub-domain by
filtering out all terrestrial land units and reindexing spatial units from ``1`` to
``S_{\\text{water}}``.

# Mathematical Formulation
Given spatial unit set ``\\mathcal{S} = \\{1, \\dots, S\\}`` and boolean indicator
``\\text{land\\_mask}``, the navigable marine sub-domain is defined by:
```math
\\mathcal{M} = \\{ i \\in \\mathcal{S} \\mid \\neg \\text{land\\_mask}[i] \\}
```
The adjacency graph is subsetted to the induced sub-graph:
```math
W_{\\text{marine}} = W[\\mathcal{M}, \\mathcal{M}]
```
preserving all marine graph topology while reducing state dimensionality.

# Arguments
- `mesh`: Spatial mesh NamedTuple containing `:centroids`, `:polygons`, `:n_units`, and `:W`.
- `land_mask`: Boolean vector of length ``S`` (`true` for land units).

# Returns
- `NamedTuple`: Compacted marine mesh with updated `:n_units`, `:W`, `:centroids_lonlat`,
  `:polygons_lonlat`, and the mapping vector `:water_indices`.
"""
function compact_marine_mesh(mesh, land_mask::AbstractVector{Bool})::NamedTuple
    water_idx = findall(!, land_mask)
    S_water = length(water_idx)

    cents_ll = hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat[water_idx] :
               (hasproperty(mesh, :centroids) ? mesh.centroids[water_idx] : Tuple{Float64, Float64}[])
    polys_ll = hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat[water_idx] :
               (hasproperty(mesh, :polygons) ? mesh.polygons[water_idx] : Vector{Vector{Tuple{Float64, Float64}}}())
    cents_km = hasproperty(mesh, :centroids_km) ? mesh.centroids_km[water_idx] : nothing
    polys_km = hasproperty(mesh, :polygons_km) ? mesh.polygons_km[water_idx] : nothing

    W_sub = copy(mesh.W[water_idx, water_idx])
    if W_sub isa SparseMatrixCSC
        dropzeros!(W_sub)
    end

    res = Dict{Symbol, Any}(
        :n_units          => S_water,
        :centroids        => cents_ll,
        :centroids_lonlat => cents_ll,
        :polygons         => polys_ll,
        :polygons_lonlat  => polys_ll,
        :W                => W_sub,
        :water_indices    => water_idx
    )
    if cents_km !== nothing
        res[:centroids_km] = cents_km
    end
    if polys_km !== nothing
        res[:polygons_km] = polys_km
    end
    if hasproperty(mesh, :radius_km)
        res[:radius_km] = mesh.radius_km
    end
    if hasproperty(mesh, :areas_km2)
        res[:areas_km2] = mesh.areas_km2[water_idx]
    end

    return NamedTuple(res)
end


"""
    infill_spatial_hsi(
        hsi_raw::AbstractVector{<:Real},
        W::AbstractMatrix{<:Real},
        known_mask::AbstractVector{Bool};
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        centroids::Union{Nothing, AbstractVector} = nothing,
        rho::Real = 0.95,
        baseline_quantile::Real = 0.1,
        fallback_decay_km::Real = 50.0
    ) -> Vector{Float64}

Infills missing Habitat Suitability Index (HSI) values outside the primary survey domain
using a data-driven graph-Laplacian screened Dirichlet system constrained to marine water channels.

# Mathematical Formulation
The spatial domain is partitioned into known survey units ``\\mathcal{K}``, unknown marine units
``\\mathcal{U}``, and terrestrial land units ``\\mathcal{L}``. On the severed water graph
``W^{\\text{water}}``, the conditional expectation of unknown HSI values satisfies:
```math
(D_{\\mathcal{UU}} - \\rho W_{\\mathcal{UU}}) \\mathbf{h}_{\\mathcal{U}} = 
\\rho W_{\\mathcal{UK}} \\mathbf{h}_{\\mathcal{K}} + (1 - \\rho) D_{\\mathcal{UU}} \\mu_0 \\mathbf{1}
```
where:
- ``D_{\\mathcal{UU}} = \\operatorname{diag}\\left( \\sum_{j \\in \\mathcal{W}} W_{ij} \\right)_{i \\in \\mathcal{U}}``
  is the total degree of each unknown node in the water graph.
- ``W_{\\mathcal{UU}}`` is the internal adjacency submatrix among unknown water nodes.
- ``W_{\\mathcal{UK}}`` connects unknown water nodes to adjacent known survey nodes.
- ``\\rho \\in (0, 1)`` controls spatial autocorrelation strength (default 0.95).
- ``\\mu_0 = \\operatorname{quantile}(\\mathbf{h}_{\\mathcal{K}}, \\text{baseline\\_quantile})``
  is the empirical conservative baseline HSI for distant unsampled marine waters.

Because ``W^{\\text{water}}`` has all land edges severed, the Dirichlet diffusion flows strictly
around peninsulas and through marine straits (e.g. Cabot Strait), never jumping across land.
Isolated water components without graph connections to known units decay smoothly to ``\\mu_0``
or utilize an exponential spatial covariance fallback ``K(d) = \\exp(-d / \\ell)``.

# Arguments
- `hsi_raw`: Vector of observed HSI values (length ``S``).
- `W`: Spatial graph adjacency matrix (size ``S \\times S``).
- `known_mask`: Boolean vector of length ``S`` (`true` where HSI is observed).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units).
- `centroids`: Optional coordinates vector for spatial distance fallback on disconnected units.
- `rho`: Spatial autocorrelation strength parameter in ``(0, 1)`` (default 0.95).
- `baseline`: Optional scalar baseline prior mean for unobserved regions. If `nothing`,
  defaults to empirical quantile `baseline_quantile` of known values.
- `baseline_quantile`: Empirical quantile of known HSI used as prior baseline (default 0.1).
- `fallback_decay_km`: Spatial correlation length scale in km for disconnected units (default 50.0).

# Returns
- `Vector{Float64}`: Infilled HSI vector of length ``S`` bounded in ``[0, 1]`` with land units set to 0.0.
"""
function infill_spatial_hsi(
    hsi_raw::AbstractVector{<:Real},
    W::AbstractMatrix{<:Real},
    known_mask::AbstractVector{Bool};
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    centroids::Union{Nothing, AbstractVector} = nothing,
    rho::Real = 0.95,
    baseline::Union{Nothing, Real} = nothing,
    baseline_quantile::Real = 0.1,
    fallback_decay_km::Real = 50.0
)::Vector{Float64}
    S = size(W, 1)
    if length(hsi_raw) != S || length(known_mask) != S
        throw(DimensionMismatch("hsi_raw, W, and known_mask dimensions must match."))
    end

    mask_l = land_mask === nothing ? falses(S) : land_mask
    hsi_out = copy(Float64.(hsi_raw))

    # Land units are set to 0.0
    hsi_out[mask_l] .= 0.0

    # Unknown water units
    unknown_idx = findall(.!known_mask .& .!mask_l)
    known_idx   = findall(known_mask .& .!mask_l)

    if isempty(unknown_idx)
        return hsi_out
    end

    # Baseline prior mean from known survey values or explicit baseline
    known_vals = filter(isfinite, hsi_out[known_idx])
    mu0 = if baseline !== nothing
        Float64(baseline)
    elseif isempty(known_vals)
        0.2
    else
        Float64(quantile(known_vals, baseline_quantile))
    end

    # Graph degree vector (summing only water connections)
    W_sp = sparse(W)
    D = vec(sum(W_sp, dims=2))

    W_uu = W_sp[unknown_idx, unknown_idx]
    W_uk = W_sp[unknown_idx, known_idx]
    D_uu = Diagonal(D[unknown_idx])

    rho_val = clamp(Float64(rho), 0.5, 0.999)
    A_mat = D_uu - rho_val * W_uu
    b_vec = rho_val * (W_uk * hsi_out[known_idx]) + (1.0 - rho_val) * (D[unknown_idx] .* mu0)

    # For any isolated nodes (degree == 0), regularize diagonal to yield mu0
    for (k_sub, u) in enumerate(unknown_idx)
        if D[u] < 1e-12
            A_mat[k_sub, k_sub] = 1.0
            b_vec[k_sub] = mu0
        end
    end

    # Solve the screened Dirichlet system
    h_u = try
        A_mat \ b_vec
    catch e
        @warn "infill_spatial_hsi: direct solve failed ($e); using regularized solve."
        (A_mat + 1e-4 * I) \ b_vec
    end

    # Fallback distance decay for units with no direct path to survey area
    if centroids !== nothing && !isempty(known_idx)
        tree_known = KDTree(hcat([[Float64(c[1]), Float64(c[2])] for c in centroids[known_idx]]...))
        for (k_sub, u) in enumerate(unknown_idx)
            has_survey_conn = sum(W_uk[k_sub, :]) > 0.0
            if !has_survey_conn
                c_u = centroids[u]
                idx_nn, dists = knn(tree_known, [Float64(c_u[1]), Float64(c_u[2])], 1)
                nn_unit = known_idx[first(idx_nn)]
                d_val = first(dists)
                w_dist = exp(-d_val / fallback_decay_km)
                h_u[k_sub] = w_dist * hsi_out[nn_unit] + (1.0 - w_dist) * mu0
            end
        end
    end

    # Bound filled values in [0.0, 1.0]
    for (k_sub, u) in enumerate(unknown_idx)
        hsi_out[u] = clamp(h_u[k_sub], 0.0, 1.0)
    end

    return hsi_out
end


"""
    construct_full_movement_domain(
        lon_vec::AbstractVector{<:Real},
        lat_vec::AbstractVector{<:Real};
        radius_km::Real = 15.0,
        land_polygons = nothing,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest
    ) -> NamedTuple

Generates a unified data-driven spatial tessellation covering the complete movement domain
(including areas outside the core survey domain), identifies terrestrial land units, and
severs land edges to form contiguous marine movement channels.

# Mathematical Formulation
A regular planar hexagonal lattice of circumradius ``r`` is generated across the bounding
envelope of all spatial observations:
```math
\\Delta x = \\sqrt{3} r, \\quad \\Delta y = \\frac{3}{2} r
```
Centroids ``\\mathbf{c}_s`` are classified as land via `identify_land_units`. Land-water edges
in graph ``W`` are severed via `apply_land_barrier`, creating a topological state space where
marine passages remain fully connected while landmasses act as impenetrable barriers.

# Arguments
- `lon_vec, lat_vec`: Geographic coordinates encompassing the full observation domain.
- `radius_km`: Hexagon circumradius in km (default 15.0 km).
- `land_polygons`: Optional land boundary polygons (defaults to `:maritimes`).
- `depth`: Optional depth vector matching units.
- `depth_threshold`: Bathymetric cutoff below which units are classified as land (default 0.0).
- `crs`: Coordinate reference system (default local tangent projection).
- `datum`: Reference ellipsoid datum (default `WGS84Latest`).

# Returns
- `NamedTuple`:
  - `centroids_km`, `centroids_lonlat`: Centroid coordinates in km and degrees.
  - `polygons_km`, `polygons_lonlat`: Hexagonal cell boundary vertex rings.
  - `n_units::Int`: Total count of spatial units ``S``.
  - `W`: Cleaned adjacency matrix with land edges severed.
  - `W_raw`: Original geometric adjacency matrix prior to land barrier severing.
  - `land_mask`: Boolean vector of length ``S`` indicating land units.
  - `radius_km`: Hexagon radius in km.
  - `areas_km2`: Cell area per unit.
  - `center_lon`, `center_lat`: Projection origin.
"""
function construct_full_movement_domain(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    radius_km::Real = 15.0,
    land_polygons = nothing,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest
)::NamedTuple
    lon_min, lon_max = extrema(lon_vec)
    lat_min, lat_max = extrema(lat_vec)

    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0

    # Planar bounding coordinates
    xy_sw = lonlat_to_xy_km(
        lon_min, lat_min;
        center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
    )
    xy_ne = lonlat_to_xy_km(
        lon_max, lat_max;
        center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
    )

    r = Float64(radius_km)
    dx = sqrt(3.0) * r
    dy = 1.5 * r

    x_min = min(xy_sw[1], xy_ne[1]) - r
    x_max = max(xy_sw[1], xy_ne[1]) + r
    y_min = min(xy_sw[2], xy_ne[2]) - r
    y_max = max(xy_sw[2], xy_ne[2]) + r

    row_min = floor(Int, y_min / dy)
    row_max = ceil(Int, y_max / dy)

    centroids_km = Tuple{Float64, Float64}[]
    for row in row_min:row_max
        yk = row * dy
        xoff = isodd(row) ? (dx / 2.0) : 0.0
        col_min = floor(Int, (x_min - xoff) / dx)
        col_max = ceil(Int, (x_max - xoff) / dx)
        for col in col_min:col_max
            xk = col * dx + xoff
            push!(centroids_km, (xk, yk))
        end
    end

    S = length(centroids_km)
    centroids_lonlat = [
        xy_km_to_lonlat(
            c[1], c[2];
            center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
        )
        for c in centroids_km
    ]

    # Flat-top hexagon vertices
    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)
    polygons_km     = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    polygons_lonlat = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)

    for i in 1:S
        cx, cy = centroids_km[i]
        verts_km = [(cx + r * cos(a), cy + r * sin(a)) for a in hex_angles]
        push!(verts_km, verts_km[1])
        polygons_km[i] = verts_km
        polygons_lonlat[i] = [
            xy_km_to_lonlat(
                v[1], v[2];
                center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
            )
            for v in verts_km
        ]
    end

    # Adjacency graph W via KDTree
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = centroids_km[i][1]
        c_mat[2, i] = centroids_km[i][2]
    end
    tree = KDTree(c_mat)
    adj_thresh = sqrt(3.0) * r * 1.05

    rows_idx = Int[]
    cols_idx = Int[]
    for i in 1:S
        nbrs = inrange(tree, [centroids_km[i][1], centroids_km[i][2]], adj_thresh)
        for j in nbrs
            if j != i
                push!(rows_idx, i)
                push!(cols_idx, j)
            end
        end
    end
    W_init = sparse(rows_idx, cols_idx, ones(Float64, length(rows_idx)), S, S)
    W_init = max.(W_init, W_init')

    # Identify land units with area fraction check
    land_mask = identify_land_units(
        centroids_lonlat;
        polygons=polygons_lonlat,
        land_polygons=land_polygons,
        depth=depth,
        depth_threshold=depth_threshold
    )

    # Sever land edges in W and topological land-crossing links
    W_water, _ = apply_land_barrier(W_init, zeros(Float64, S), land_mask)
    sever_land_crossing_edges!(W_water, centroids_lonlat; land_polygons=land_polygons)

    area_km2 = (3.0 * sqrt(3.0) / 2.0) * r^2

    return (
        centroids        = centroids_lonlat,
        centroids_km     = centroids_km,
        centroids_lonlat = centroids_lonlat,
        polygons         = polygons_lonlat,
        polygons_km      = polygons_km,
        polygons_lonlat  = polygons_lonlat,
        n_units          = S,
        W                = W_water,
        W_raw            = W_init,
        land_mask        = land_mask,
        radius_km        = r,
        areas_km2        = fill(area_km2, S),
        center_lon       = center_lon,
        center_lat       = center_lat
    )
end


"""
    prepare_movement_data(
        tagging::DataFrame;
        hsi_file::Union{Nothing, AbstractString} = nothing,
        sppoly_file::Union{Nothing, AbstractString} = nothing,
        radius_km::Real = 15.0,
        time_interval::Symbol = :monthly,
        land_polygons = nothing,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest,
        ref_doy::Int = 182,
        verbose::Bool = true
    ) -> NamedTuple

High-level end-to-end data preparation pipeline for individual animal movement and telemetry.
Builds the full-domain tessellation, classifies and blocks terrestrial land barriers, reshards
and autocorrelates HSI into external marine areas, maps telemetry observations, and extracts
mark-recapture transition events with biological group stratifications.

# Arguments
- `tagging::DataFrame`: Raw telemetry records with `:tagid`, `:lon`, `:lat`, `:time`, `:tag`.
- `hsi_file`: Path to environmental Habitat Suitability Index JLD2 file.
- `sppoly_file`: Path to survey spatial polygons JLD2 file defining known HSI units.
- `radius_km`: Spatial hexagon circumradius in km (default 15.0 km).
- `time_interval`: Discrete transition interval (`:monthly`, `:weekly`, `:biweekly`, `:daily`).
- `land_polygons`: Terrestrial boundary polygons (defaults to `:maritimes`).
- `depth`: Optional depth vector matching units.
- `depth_threshold`: Bathymetric cutoff below which units are classified as land (default 0.0).
- `crs`: Target coordinate reference system (default local tangent projection).
- `datum`: Reference ellipsoid datum (default `WGS84Latest`).
- `ref_doy`: Annual reference survey day of year (default 182).
- `verbose`: Toggle progress logging to console.
- `pre_mapped`: Optional pre-computed mesh NamedTuple to turn off dynamic resharding.

# Returns
- `NamedTuple`:
  - `tagging`: Processed telemetry DataFrame with `:s_idx` and `:hsi`.
  - `mesh`: Full-domain tessellation NamedTuple.
  - `W`: Adjacency matrix with severed land boundaries.
  - `hsi_vec`: Infilled spatial HSI vector (length ``S``).
  - `monthly_hsi`: Monthly HSI matrix (size ``S \\times T``).
  - `month_lookup`: Mapping of `(year, month)` to column index.
  - `years`: Vector of survey years.
  - `obs`: Extracted mark-recapture event pairs DataFrame.
  - `group_lookup`: Biological stratum dictionary mapping.
  - `land_mask`: Boolean vector of length ``S`` denoting terrestrial barrier units.
"""
function prepare_movement_data(
    tagging::DataFrame;
    hsi_file::Union{Nothing, AbstractString} = nothing,
    sppoly_file::Union{Nothing, AbstractString} = nothing,
    radius_km::Real = 15.0,
    time_interval::Symbol = :daily,
    land_polygons = nothing,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest,
    ref_doy::Int = 182,
    verbose::Bool = true,
    pre_mapped = nothing
)::NamedTuple
    tag_df = copy(tagging)

    # Filter dead records
    if hasproperty(tag_df, :is_dead)
        filter!(:is_dead => d -> !coalesce(d, false), tag_df)
    end

    # Resolve time column (:time or :timestamp)
    time_col = hasproperty(tag_df, :time) ? :time :
               (hasproperty(tag_df, :timestamp) ? :timestamp : nothing)
    if time_col === nothing
        throw(ArgumentError(
            "tagging DataFrame must contain either :time or :timestamp column."
        ))
    end

    # Filter invalid records (missing/non-finite coordinates or identifiers)
    filter!(r -> !ismissing(r.lon) && !ismissing(r.lat) &&
                 !ismissing(r.tagid) && !ismissing(r.tag) &&
                 !ismissing(r[time_col]) &&
                 isfinite(Float64(r.lon)) && isfinite(Float64(r.lat)), tag_df)

    tag_df[!, :lon]   = Float64.(tag_df.lon)
    tag_df[!, :lat]   = Float64.(tag_df.lat)
    tag_df[!, :tag]   = Int.(tag_df.tag)
    tag_df[!, :tagid] = string.(tag_df.tagid)

    if time_col == :time
        tag_df[!, :time] = Float64.(tag_df.time)
    else
        t_vals = Vector{Float64}(undef, nrow(tag_df))
        for (i, row) in enumerate(eachrow(tag_df))
            raw_t = row[time_col]
            if raw_t isa Real
                t_vals[i] = Float64(raw_t)
            else
                parsed_d = _parse_flexible_date(raw_t)
                t_vals[i] = parsed_d !== nothing ?
                    _date_to_decimal_year(parsed_d) : 0.0
            end
        end
        tag_df[!, :time] = t_vals
    end

    # Require both release and recapture events
    valid_set = Set{String}()
    for sub in groupby(tag_df, :tagid)
        tags = sub.tag
        if any(==(0), tags) && any(>(0), tags)
            push!(valid_set, sub.tagid[1])
        end
    end
    filter!(:tagid => ∈(valid_set), tag_df)
    sort!(tag_df, [:tagid, :time])

    # 1. Full-domain tessellation with land barriers
    mesh = if pre_mapped !== nothing
        verbose && println("  [prepare] Using user pre-mapped domain mesh …")
        c_lon = hasproperty(pre_mapped, :center_lon) ?
            pre_mapped.center_lon : ((minimum(tag_df.lon) + maximum(tag_df.lon)) / 2.0)
        c_lat = hasproperty(pre_mapped, :center_lat) ?
            pre_mapped.center_lat : ((minimum(tag_df.lat) + maximum(tag_df.lat)) / 2.0)
        c_km = if hasproperty(pre_mapped, :centroids_km)
            pre_mapped.centroids_km
        elseif hasproperty(pre_mapped, :centroids_lonlat)
            [lonlat_to_xy_km(
                c[1], c[2];
                center_lon = c_lon, center_lat = c_lat,
                crs = crs, datum = datum
            ) for c in pre_mapped.centroids_lonlat]
        elseif hasproperty(pre_mapped, :centroids)
            [lonlat_to_xy_km(
                c[1], c[2];
                center_lon = c_lon, center_lat = c_lat,
                crs = crs, datum = datum
            ) for c in pre_mapped.centroids]
        else
            throw(ArgumentError(
                "pre_mapped must contain :centroids or :centroids_km."
            ))
        end
        n_u = hasproperty(pre_mapped, :n_units) ?
            pre_mapped.n_units : length(c_km)
        l_mask = hasproperty(pre_mapped, :land_mask) ?
            pre_mapped.land_mask : falses(n_u)
        c_ll = if hasproperty(pre_mapped, :centroids_lonlat)
            pre_mapped.centroids_lonlat
        elseif hasproperty(pre_mapped, :centroids)
            pre_mapped.centroids
        else
            [xy_km_to_lonlat(
                c[1], c[2];
                center_lon = c_lon, center_lat = c_lat,
                crs = crs, datum = datum
            ) for c in c_km]
        end
        (
            centroids_km     = c_km,
            centroids_lonlat = c_ll,
            n_units          = n_u,
            W                = pre_mapped.W,
            land_mask        = l_mask,
            center_lon       = c_lon,
            center_lat       = c_lat,
            radius_km        = hasproperty(pre_mapped, :radius_km) ?
                pre_mapped.radius_km : radius_km
        )
    else
        verbose && println("  [prepare] Constructing full movement domain (r=$(radius_km) km) …")
        construct_full_movement_domain(
            tag_df.lon, tag_df.lat;
            radius_km=radius_km, land_polygons=land_polygons,
            depth=depth, depth_threshold=depth_threshold,
            crs=crs, datum=datum
        )
    end
    verbose && println("    Total mesh units: $(mesh.n_units) " *
                       "(water: $(count(!, mesh.land_mask)), land: $(sum(mesh.land_mask)))")

    # 2. Map telemetry observations to hex units strictly on active water
    verbose && println("  [prepare] Mapping observations to domain units …")
    tag_df = map_point_to_units(
        tag_df, mesh.centroids_km,
        mesh.center_lon, mesh.center_lat;
        crs=crs, datum=datum,
        land_mask=mesh.land_mask, W=mesh.W
    )

    # 3. Load HSI and perform autocorrelation infilling outside core domain
    hsi_vec      = zeros(Float64, mesh.n_units)
    monthly_hsi  = Matrix{Float64}(undef, 0, 0)
    month_lookup = Dict{Tuple{Int, Int}, Int}()
    years_vec    = Int[]

    if hsi_file !== nothing && isfile(hsi_file)
        verbose && println("  [prepare] Loading and infilling HSI field …")
        h = load_hsi_jld2(hsi_file; ref_doy=ref_doy)
        years_vec    = h.years
        month_lookup = h.month_lookup
        S_src        = size(h.monthly_hsi, 1)

        src_coords_km = Tuple{Float64, Float64}[]
        known_mask = falses(mesh.n_units)

        if sppoly_file !== nothing && isfile(sppoly_file)
            au_src = try
                jldopen(sppoly_file, "r") do f
                    haskey(f, "au") ? f["au"] : nothing
                end
            catch err
                verbose && @warn "Failed reading :au from $sppoly_file: $err"
                nothing
            end

            if au_src !== nothing && hasproperty(au_src, :lon) && hasproperty(au_src, :lat)
                n_au = length(au_src.lon)
                src_coords_km = Vector{Tuple{Float64, Float64}}(undef, n_au)
                for i in 1:n_au
                    src_coords_km[i] = lonlat_to_xy_km(
                        au_src.lon[i], au_src.lat[i];
                        center_lon=mesh.center_lon, center_lat=mesh.center_lat,
                        crs=crs, datum=datum
                    )
                end

                # Identify mesh units lying within the known HSI survey domain
                tree_src = KDTree(hcat([[Float64(c[1]), Float64(c[2])] for c in src_coords_km]...))
                survey_radius_thresh = 1.5 * radius_km
                for i in 1:mesh.n_units
                    if !mesh.land_mask[i]
                        _, dists = knn(tree_src, [mesh.centroids_km[i][1], mesh.centroids_km[i][2]], 1)
                        if first(dists) <= survey_radius_thresh
                            known_mask[i] = true
                        end
                    end
                end
            end
        end

        n_mo = size(h.monthly_hsi, 2)
        if !isempty(src_coords_km) && n_mo > 0
            monthly_hsi = zeros(Float64, mesh.n_units, n_mo)
            for m in 1:n_mo
                raw_m = reshard_hsi_field(
                    h.monthly_hsi[:, m], (centroids=mesh.centroids_km,); hsi_coords=src_coords_km
                )
                monthly_hsi[:, m] = infill_spatial_hsi(
                    raw_m, mesh.W, known_mask;
                    land_mask=mesh.land_mask, centroids=mesh.centroids_km
                )
            end
            hsi_vec = vec(mean(monthly_hsi, dims=2))
        else
            base_vec = fill(0.2, mesh.n_units)
            hsi_vec = infill_spatial_hsi(
                base_vec, mesh.W, known_mask;
                land_mask=mesh.land_mask, centroids=mesh.centroids_km
            )
        end

        if !isempty(month_lookup) && size(monthly_hsi, 1) == mesh.n_units
            tag_df[!, :hsi] = match_telemetry_closest_month_hsi(
                tag_df, monthly_hsi, month_lookup, years_vec
            )
            # Climatology fallback: replace NaN observations
            # (missing year/month) with the spatial mean HSI
            n_nan = 0
            hsi_col = tag_df.hsi
            s_col   = tag_df.s_idx
            for i in eachindex(hsi_col)
                if isnan(hsi_col[i])
                    s = s_col[i]
                    hsi_col[i] = (1 <= s <= length(hsi_vec)) ?
                                 hsi_vec[s] : 0.5
                    n_nan += 1
                end
            end
            n_nan > 0 && verbose && println(
                "  [prepare] HSI climatology fallback applied " *
                "to $n_nan / $(nrow(tag_df)) observations."
            )
        else
            tag_df[!, :hsi] = fill(0.5, nrow(tag_df))
        end
    else
        verbose && println("  [prepare] No HSI file provided; using uniform marine HSI = 0.5 …")
        hsi_vec = [mesh.land_mask[i] ? 0.0 : 0.5 for i in 1:mesh.n_units]
        tag_df[!, :hsi] = fill(0.5, nrow(tag_df))
    end

    # 4. Extract mark-recapture event pairs
    dt_map = (monthly=1.0/12.0, weekly=1.0/52.0, biweekly=1.0/26.0, daily=1.0/365.25, raw=1.0)
    dt = hasproperty(dt_map, time_interval) ? getproperty(dt_map, time_interval) : 1.0 / 12.0

    has_sex = hasproperty(tag_df, :sex)
    has_mat = hasproperty(tag_df, :mat)

    sorted_df = sort(tag_df, [:tagid, :time])
    n_rows = nrow(sorted_df)

    tagids    = sorted_df.tagid
    times     = sorted_df.time
    s_idxs    = sorted_df.s_idx
    sexes_col = has_sex ? sorted_df.sex : nothing
    mats_col  = has_mat ? sorted_df.mat : nothing

    RecordType = NamedTuple{
        (:tagid, :release, :recapture, :k, :rel_time, :sex, :mat),
        Tuple{String, Int, Int, Int, Float64, String, String}
    }
    pair_records = Vector{RecordType}(undef, 0)

    if n_rows >= 2
        sizehint!(pair_records, n_rows)
        for i in 2:n_rows
            if tagids[i] == tagids[i-1]
                Δt = times[i] - times[i-1]
                k  = max(1, round(Int, Δt / dt))
                push!(pair_records, (
                    tagid     = string(tagids[i-1]),
                    release   = s_idxs[i-1],
                    recapture = s_idxs[i],
                    k         = k,
                    rel_time  = Float64(times[i-1]),
                    sex       = has_sex ? string(sexes_col[i-1]) : "unknown",
                    mat       = has_mat ? string(mats_col[i-1])  : "unknown"
                ))
            end
        end
    end

    obs = DataFrame(pair_records)

    # 5. Assign biological groupings
    n_obs = nrow(obs)
    labels = Vector{String}(undef, n_obs)
    if n_obs > 0
        obs_sexes = obs[!, :sex]
        obs_mats  = obs[!, :mat]
        @inbounds for i in 1:n_obs
            sx = string(obs_sexes[i])
            mt = string(obs_mats[i])
            if mt == "immature"
                labels[i] = "immature"
            elseif mt == "mature" && sx == "M"
                labels[i] = "male"
            elseif mt == "mature" && sx == "F"
                labels[i] = "female"
            else
                labels[i] = "unknown"
            end
        end
    end

    unique_labels = sort!(unique(labels))
    group_lookup  = Dict{String, Int}(lbl => i for (i, lbl) in enumerate(unique_labels))
    group_ids = Vector{Int}(undef, n_obs)
    @inbounds for i in 1:n_obs
        group_ids[i] = group_lookup[labels[i]]
    end
    obs[!, :group] = group_ids

    return (
        tagging      = tag_df,
        mesh         = mesh,
        W            = mesh.W,
        hsi_vec      = hsi_vec,
        monthly_hsi  = monthly_hsi,
        month_lookup = month_lookup,
        years        = years_vec,
        obs          = obs,
        group_lookup = group_lookup,
        land_mask    = mesh.land_mask
    )
end




# =============================================================================
# Movement Ecology Outputs & Trait Integration
# =============================================================================

"""
    compute_movement_statistics(paths_rich, path_results, loaded; params = nothing)

Computes comprehensive movement ecology metrics across all reconstructed
individual trajectories:
1. Net displacement: Straight-line distance between release and recapture
   locations using spherical Haversine distance.
2. Path efficiency: Ratio of net displacement to total cumulative path length:
   `Efficiency = displacement / total_distance ∈ [0, 1]`.
3. Directional bias: Mean bearing θ̄ and von Mises concentration parameter κ
   approximated using the Mardia-Jupp and Best-Fisher circular formulations:
   `R̄ = (1/N) * sqrt((∑ cos θ_i)^2 + (∑ sin θ_i)^2)`.
4. Residence time per patch: Total dwelling days allocated across each spatial
   mesh unit u ∈ {1, ..., S}.
5. Behavioral bout classification: Categorizes each individual trajectory into
   `"Directed / Migratory"`, `"Exploratory / Search"`, or
   `"Resident / Encamped"` based on path efficiency.

# Arguments
- `paths_rich::Vector{<:NamedTuple}`: Reconstructed paths with coordinates,
  distances, and durations.
- `path_results::NamedTuple`: Summary outputs from path reconstruction.
- `loaded::NamedTuple`: Spatial domain and observation dataset.
- `params`: Optional configuration named tuple or dictionary.

# Returns
`NamedTuple` with summary DataFrame, efficiency vectors, directional bias,
patch residence times, and behavioral bout breakdown.
"""
function compute_movement_statistics(
    paths_rich::Vector{<:NamedTuple},
    path_results::NamedTuple,
    loaded::NamedTuple;
    params = nothing
)::NamedTuple
    n_paths = length(paths_rich)
    n_spatial = loaded.n_spatial

    net_disp_v = Float64[p.displacement_km for p in paths_rich]
    path_len_v = Float64[p.total_dist_km for p in paths_rich]
    eff_v      = Float64[]
    tort_v     = Float64[]
    bearings_v = Float64[]
    bouts_v    = String[]

    for p in paths_rich
        eff = p.total_dist_km > 0.01 ?
              clamp(p.displacement_km / p.total_dist_km, 0.0, 1.0) : 1.0
        tort = p.displacement_km > 0.01 ?
               p.total_dist_km / p.displacement_km : 1.0
        push!(eff_v, eff)
        push!(tort_v, tort)

        if length(p.coords) >= 2
            d_lon = p.coords[end][1] - p.coords[1][1]
            d_lat = p.coords[end][2] - p.coords[1][2]
            brg = atand(d_lon, d_lat)
            brg < 0.0 && (brg += 360.0)
            push!(bearings_v, brg)
        else
            push!(bearings_v, 0.0)
        end

        bout = eff >= 0.70 ? "Directed / Migratory" :
               (eff >= 0.30 ? "Exploratory / Search" : "Resident / Encamped")
        push!(bouts_v, bout)
    end

    # Circular statistics for directional bias
    C_sum = 0.0
    S_sum = 0.0
    for b in bearings_v
        r = deg2rad(b)
        C_sum += cos(r)
        S_sum += sin(r)
    end
    N_circ = max(1, length(bearings_v))
    C_bar  = C_sum / N_circ
    S_bar  = S_sum / N_circ
    R_bar  = clamp(sqrt(C_bar^2 + S_bar^2), 0.0, 1.0)

    mean_dir_rad = atan(S_bar, C_bar)
    mean_dir_deg = mod(rad2deg(mean_dir_rad), 360.0)
    circ_var     = 1.0 - R_bar
    circ_sd_deg  = rad2deg(sqrt(max(0.0, -2.0 * log(max(1e-10, R_bar)))))

    # Von Mises kappa approximation (Mardia & Jupp 2000)
    kappa_hat = if R_bar < 0.53
        2.0 * R_bar + R_bar^3 + (5.0 / 6.0) * R_bar^5
    elseif R_bar < 0.85
        -0.4 + 1.39 * R_bar + 0.43 / (1.0 - R_bar)
    else
        denom = R_bar^3 - 4.0 * R_bar^2 + 3.0 * R_bar
        denom > 1e-6 ? 1.0 / denom : 100.0
    end

    # Rayleigh test of circular uniformity
    rayleigh_z = N_circ * R_bar^2
    rayleigh_p = exp(-rayleigh_z) * (
        1.0 + (2.0 * rayleigh_z - rayleigh_z^2) / (4.0 * N_circ) -
        (24.0 * rayleigh_z - 132.0 * rayleigh_z^2 + 76.0 * rayleigh_z^3 -
         9.0 * rayleigh_z^4) / (288.0 * N_circ^2)
    )
    rayleigh_p = clamp(rayleigh_p, 0.0, 1.0)

    directional_bias = (
        kappa              = kappa_hat,
        mean_vector_length = R_bar,
        mean_bearing_deg   = mean_dir_deg,
        circular_variance  = circ_var,
        circular_sd_deg    = circ_sd_deg,
        rayleigh_z         = rayleigh_z,
        rayleigh_p         = rayleigh_p,
    )

    # Residence time per patch
    residence_time = zeros(Float64, n_spatial)
    paths_dict = hasproperty(path_results, :paths) ?
        path_results.paths : Dict{String, Vector{Int}}()
    for p in paths_rich
        node_seq = if hasproperty(p, :path)
            p.path
        elseif haskey(paths_dict, string(p.tagid))
            paths_dict[string(p.tagid)]
        elseif haskey(paths_dict, Symbol(p.tagid))
            paths_dict[Symbol(p.tagid)]
        else
            Int[]
        end
        if length(node_seq) > 0
            dur = hasproperty(p, :duration_days) ?
                Float64(p.duration_days) : Float64(length(node_seq) - 1)
            time_per_step = dur / max(1, length(node_seq) - 1)
            for node in node_seq
                if 1 <= node <= n_spatial
                    residence_time[node] += time_per_step
                end
            end
        end
    end

    # Bout counts
    n_directed    = count(==("Directed / Migratory"), bouts_v)
    n_exploratory = count(==("Exploratory / Search"), bouts_v)
    n_resident    = count(==("Resident / Encamped"), bouts_v)

    behavioral_bouts = (
        counts = (
            directed    = n_directed,
            exploratory = n_exploratory,
            resident    = n_resident,
        ),
        proportions = (
            directed    = n_directed / max(1, n_paths),
            exploratory = n_exploratory / max(1, n_paths),
            resident    = n_resident / max(1, n_paths),
        ),
        classification = bouts_v,
    )

    summary_df = DataFrame(
        tagid           = String[string(p.tagid) for p in paths_rich],
        displacement_km = net_disp_v,
        total_dist_km   = path_len_v,
        efficiency      = eff_v,
        tortuosity      = tort_v,
        bearing_deg     = bearings_v,
        behavioral_bout = bouts_v,
    )

    return (
        summary_df               = summary_df,
        net_displacement_km      = net_disp_v,
        path_length_km           = path_len_v,
        path_efficiency          = eff_v,
        tortuosity               = tort_v,
        bearings_deg             = bearings_v,
        directional_bias         = directional_bias,
        residence_time_per_patch = residence_time,
        behavioral_bouts         = behavioral_bouts,
    )
end

"""
    analyze_seasonal_movement_phenology(obs_df, mov_stats, loaded)

Aggregates individual movement metrics across months and seasonal quarters to
reveal within-year movement phenology, migration pulses, and seasonal velocity
dynamics.

# Arguments
- `obs_df::DataFrame`: Movement observation records.
- `mov_stats::NamedTuple`: Output of `compute_movement_statistics`.
- `loaded::NamedTuple`: Loaded spatial and temporal metadata.

# Returns
`NamedTuple` with monthly metrics, quarterly statistics, and peak timing.
"""
function analyze_seasonal_movement_phenology(
    obs_df::DataFrame,
    mov_stats::NamedTuple,
    loaded::NamedTuple
)::NamedTuple
    n_obs = nrow(obs_df)
    n_obs == 0 && return (monthly = DataFrame(), quarters = DataFrame())

    months_v = Int[]
    for row in eachrow(obs_df)
        m = if hasproperty(row, :month) && !ismissing(row.month)
            Int(row.month)
        elseif hasproperty(row, :date) && !ismissing(row.date)
            month(Date(string(row.date)))
        elseif hasproperty(row, :k)
            mod(Int(row.k) - 1, 12) + 1
        else
            6
        end
        push!(months_v, clamp(m, 1, 12))
    end

    n_stat = length(mov_stats.net_displacement_km)
    disp_all = n_stat == n_obs ? mov_stats.net_displacement_km :
               fill(mean(mov_stats.net_displacement_km), n_obs)
    eff_all  = n_stat == n_obs ? mov_stats.path_efficiency :
               fill(mean(mov_stats.path_efficiency), n_obs)

    m_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_rows = []

    for m in 1:12
        idxs = findall(==(m), months_v)
        c = length(idxs)
        m_disp = c > 0 ? mean(disp_all[idxs]) : 0.0
        m_eff  = c > 0 ? mean(eff_all[idxs]) : 0.0
        push!(monthly_rows, (
            month       = m,
            month_name  = m_names[m],
            n_events    = c,
            mean_disp_km = m_disp,
            mean_eff    = m_eff,
        ))
    end
    monthly_df = DataFrame(monthly_rows)

    q_names = ["Q1 (Winter)", "Q2 (Spring)", "Q3 (Summer)", "Q4 (Autumn)"]
    quarterly_rows = []
    for q in 1:4
        q_m = ((q - 1) * 3 + 1):(q * 3)
        idxs = findall(m -> m in q_m, months_v)
        c = length(idxs)
        q_disp = c > 0 ? mean(disp_all[idxs]) : 0.0
        q_eff  = c > 0 ? mean(eff_all[idxs]) : 0.0
        push!(quarterly_rows, (
            quarter      = q,
            quarter_name = q_names[q],
            n_events     = c,
            mean_disp_km = q_disp,
            mean_eff     = q_eff,
        ))
    end
    quarterly_df = DataFrame(quarterly_rows)

    peak_row = argmax(monthly_df.mean_disp_km)
    peak_month = monthly_df.month_name[peak_row]

    return (
        monthly_df   = monthly_df,
        quarterly_df = quarterly_df,
        peak_month   = peak_month,
    )
end

"""
    model_trait_movement_associations(obs_df, mov_stats, loaded)

Fits Ordinary Least Squares (OLS) regression models assessing relationships
between biological traits (carapace width, sex, maturity, genomic cluster)
and movement parameters (speed, displacement, efficiency, tortuosity).

`Y_i = β_0 + β_1 * Trait_i + ε_i,  ε_i ~ N(0, σ²)`

# Arguments
- `obs_df::DataFrame`: Telemetry dataset with optional morphometric traits.
- `mov_stats::NamedTuple`: Output of `compute_movement_statistics`.
- `loaded::NamedTuple`: Loaded spatial and group metadata.

# Returns
`NamedTuple` with regression coefficients, t-statistics, p-values, and R².
"""
function model_trait_movement_associations(
    obs_df::DataFrame,
    mov_stats::NamedTuple,
    loaded::NamedTuple
)::NamedTuple
    n_paths = length(mov_stats.net_displacement_km)
    n_paths < 3 && return (models = Dict{String, Any}(), message = "Too few samples")

    trait_vec = Float64[]
    trait_name = "carapace_width"

    candidates = [:carapace_width, :cw, :size, :weight, :length, :size_mm]
    found_col = nothing
    for c in candidates
        if hasproperty(obs_df, c)
            found_col = c
            break
        end
    end

    if found_col !== nothing && nrow(obs_df) >= n_paths
        for v in obs_df[1:n_paths, found_col]
            push!(trait_vec, ismissing(v) ? 110.0 : Float64(v))
        end
        trait_name = string(found_col)
    else
        # Synthetic baseline trait for snow crab: carapace width ~ N(115, 18^2) mm
        rng = MersenneTwister(42)
        trait_vec = [115.0 + 18.0 * randn(rng) for _ in 1:n_paths]
        trait_name = "carapace_width_sim"
    end

    results = Dict{String, NamedTuple}()
    targets = [
        ("displacement", mov_stats.net_displacement_km),
        ("efficiency", mov_stats.path_efficiency),
        ("tortuosity", mov_stats.tortuosity),
    ]

    for (t_name, y_vec) in targets
        x_bar = mean(trait_vec)
        y_bar = mean(y_vec)
        ss_xx = sum((x - x_bar)^2 for x in trait_vec)
        ss_yy = sum((y - y_bar)^2 for y in y_vec)
        ss_xy = sum((trait_vec[i] - x_bar) * (y_vec[i] - y_bar) for i in 1:n_paths)

        beta_1 = ss_xx > 1e-10 ? ss_xy / ss_xx : 0.0
        beta_0 = y_bar - beta_1 * x_bar
        residuals = [y_vec[i] - (beta_0 + beta_1 * trait_vec[i]) for i in 1:n_paths]
        sse = sum(r^2 for r in residuals)
        r2 = ss_yy > 1e-10 ? clamp(1.0 - sse / ss_yy, 0.0, 1.0) : 0.0

        df_deg = max(1, n_paths - 2)
        s_err = sqrt(sse / df_deg)
        se_beta1 = ss_xx > 1e-10 ? s_err / sqrt(ss_xx) : 1.0
        t_stat = se_beta1 > 1e-10 ? beta_1 / se_beta1 : 0.0

        p_val = 2.0 * ccdf(TDist(df_deg), abs(t_stat))
        p_val = clamp(p_val, 0.0, 1.0)

        results[t_name] = (
            trait_name = trait_name,
            beta_0     = beta_0,
            beta_1     = beta_1,
            se_beta_1  = se_beta1,
            t_stat     = t_stat,
            p_val      = p_val,
            r2         = r2,
        )
    end

    return (
        trait_name = trait_name,
        trait_vals = trait_vec,
        models     = results,
    )
end

"""
    export_movement_summary_csv(filepath, paths_rich, mov_stats, obs_df)

Exports per-individual movement trajectory metrics and behavioral bout
classifications to a standardized CSV file.

# Arguments
- `filepath::String`: Target file path for the output CSV.
- `paths_rich::Vector{<:NamedTuple}`: Trajectories with coordinates and times.
- `mov_stats::NamedTuple`: Output of `compute_movement_statistics`.
- `obs_df::DataFrame`: Movement observation records.

# Returns
`String`: Output file path.
"""
function export_movement_summary_csv(
    filepath::String,
    paths_rich::Vector{<:NamedTuple},
    mov_stats::NamedTuple,
    obs_df::DataFrame
)::String
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        write(io, "tagid,group,release_date,recapture_date,start_lon,start_lat," *
                  "end_lon,end_lat,duration_days,total_dist_km,displacement_km," *
                  "efficiency,tortuosity,velocity_km_day,mean_hsi,bearing_deg," *
                  "primary_behavior\n")
        for (i, p) in enumerate(paths_rich)
            coords = hasproperty(p, :coords) ? p.coords : Tuple{Float64, Float64}[]
            c_start = length(coords) >= 1 ? coords[1] : (0.0, 0.0)
            c_end   = length(coords) >= 1 ? coords[end] : (0.0, 0.0)
            dur = hasproperty(p, :duration_days) ? Float64(p.duration_days) : 1.0
            tot_d = hasproperty(p, :total_dist_km) ? Float64(p.total_dist_km) : 0.0
            disp_d = hasproperty(p, :displacement_km) ? Float64(p.displacement_km) : 0.0
            vel = dur > 0.1 ? tot_d / dur : 0.0
            r_date = hasproperty(p, :release_date) ? string(p.release_date) : "N/A"
            c_date = hasproperty(p, :recapture_date) ? string(p.recapture_date) : "N/A"
            grp_val = hasproperty(p, :group) ? string(p.group) :
                (hasproperty(p, :group_label) ? string(p.group_label) : "1")
            m_hsi = hasproperty(p, :mean_hsi) ? Float64(p.mean_hsi) : 0.5
            eff_val = i <= length(mov_stats.path_efficiency) ?
                mov_stats.path_efficiency[i] : 1.0
            tort_val = i <= length(mov_stats.tortuosity) ?
                mov_stats.tortuosity[i] : 1.0
            bearing_val = i <= length(mov_stats.bearings_deg) ?
                mov_stats.bearings_deg[i] : 0.0
            bouts = mov_stats.behavioral_bouts
            bout_cls = hasproperty(bouts, :classification) ?
                bouts.classification : (bouts isa AbstractVector ? bouts : String[])
            bout_val = i <= length(bout_cls) ?
                bout_cls[i] : "Resident / Encamped"

            write(io, string(
                p.tagid, ",",
                grp_val, ",",
                r_date, ",",
                c_date, ",",
                round(c_start[1]; digits=4), ",",
                round(c_start[2]; digits=4), ",",
                round(c_end[1]; digits=4), ",",
                round(c_end[2]; digits=4), ",",
                round(dur; digits=1), ",",
                round(tot_d; digits=2), ",",
                round(disp_d; digits=2), ",",
                round(eff_val; digits=4), ",",
                round(tort_val; digits=2), ",",
                round(vel; digits=3), ",",
                round(m_hsi; digits=3), ",",
                round(bearing_val; digits=1), ",",
                bout_val, "\n"
            ))
        end
    end
    return filepath
end

# =============================================================================
# Posterior Uncertainty Propagation & Priority Analyses
# =============================================================================

"""
    path_credible_intervals(loaded, fitted, kernels, params = (;)) -> NamedTuple

Propagates posterior uncertainty from MCMC parameter draws through individual
trajectory reconstruction. Reconstructs path ensembles across draws to compute
per-node waypoint distributions and credible intervals on path length and
corridor width.

# Arguments
- `loaded::NamedTuple`: Loaded spatial domain, mesh, and observation records.
- `fitted::NamedTuple`: Model fitting output containing MCMC chains.
- `kernels::NamedTuple`: Transition kernels and parameter summaries.
- `params`: Optional parameter overrides.

# Returns
`NamedTuple` with path ensembles, summary credible intervals, and node
visitation matrices.
"""
function path_credible_intervals(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params = (;)
)::NamedTuple
    verbose   = get(params, :verbose, true)
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    land_mask = loaded.land_mask
    G         = kernels.G

    n_draws = if hasproperty(kernels, :alpha_samples) &&
                 !isempty(kernels.alpha_samples)
        size(kernels.alpha_samples, 1)
    else
        chains = fitted.chains
        active_chain = haskey(chains, :telemetry) ?
                       chains[:telemetry] : chains[:telemetry_and_survey]
        size(Array(active_chain), 1)
    end

    v_samples = zeros(n_draws, G)
    d_samples = zeros(n_draws, G)
    g_samples = zeros(n_draws, G)

    if hasproperty(kernels, :alpha_samples) && !isempty(kernels.alpha_samples)
        for g in 1:G
            v_samples[:, g] .= kernels.alpha_samples[:, min(g, size(kernels.alpha_samples, 2))]
            d_samples[:, g] .= kernels.rho_samples[:, min(g, size(kernels.rho_samples, 2))]
            g_samples[:, g] .= kernels.gamma_samples[:, min(g, size(kernels.gamma_samples, 2))]
        end
    else
        chains = fitted.chains
        active_chain = haskey(chains, :telemetry) ?
                       chains[:telemetry] : chains[:telemetry_and_survey]
        for draw in 1:n_draws, g in 1:G
            vk = Symbol("velocity[$g]")
            dk = Symbol("diffusion[$g]")
            gk = Symbol("gamma[$g]")
            v_samples[draw, g] = haskey(chains, vk) ? mean(active_chain[vk]) : 0.3
            d_samples[draw, g] = haskey(chains, dk) ? mean(active_chain[dk]) : 0.1
            g_samples[draw, g] = haskey(chains, gk) ? mean(active_chain[gk]) : 1.0
        end
    end

    all_tags = unique(obs_df.tagid)
    path_samples = Dict{String, Vector{Vector{Int}}}()
    path_stats = Dict{String, NamedTuple}()
    node_visit_counts = zeros(Int, n_spatial, n_spatial)

    n_eval_draws = min(n_draws, 50)
    eval_indices = round.(Int, range(1, n_draws, length=n_eval_draws))

    for tid in all_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue

        grp = hasproperty(sub_obs, :group) ? first(sub_obs.group) : 1
        grp = clamp(grp, 1, G)

        path_ens = Vector{Int}[]
        path_lengths = Float64[]

        for draw in eval_indices
            alpha_draw = clamp(v_samples[draw, grp] /
                               (v_samples[draw, grp] + d_samples[draw, grp] + 1e-6),
                               0.0, 1.0)
            rho_draw   = clamp(1.0 / (1.0 + v_samples[draw, grp] + d_samples[draw, grp]),
                               0.01, 0.95)
            gamma_draw = g_samples[draw, grp]

            P_draw = construct_stochastic_transition_kernel(
                loaded.W, loaded.hsi_vec;
                gamma     = fill(gamma_draw, G),
                residence = fill(rho_draw, G),
                advection = fill(alpha_draw, G),
                land_mask = land_mask
            )
            P_k = P_draw isa AbstractVector ? P_draw[grp] : P_draw

            full_path = Int[sub_obs.release[1]]
            for row in eachrow(sub_obs)
                seg = predict_path(
                    P_k, row.release, row.recapture, row.k;
                    method    = :astar,
                    land_mask = land_mask
                )
                append!(full_path, seg[2:end])
            end
            push!(path_ens, full_path)
            push!(path_lengths, Float64(length(full_path)))

            for i in 1:(length(full_path) - 1)
                u, v = full_path[i], full_path[i+1]
                if 1 <= u <= n_spatial && 1 <= v <= n_spatial
                    node_visit_counts[u, v] += 1
                end
            end
        end

        len_mean   = mean(path_lengths)
        len_lower  = quantile(path_lengths, 0.025)
        len_upper  = quantile(path_lengths, 0.975)
        len_sd     = std(path_lengths)

        waypoint_freq = zeros(n_spatial)
        for path in path_ens
            for node in path
                1 <= node <= n_spatial && (waypoint_freq[node] += 1)
            end
        end
        waypoint_freq ./= max(1, length(path_ens))

        path_samples[string(tid)] = path_ens
        path_stats[string(tid)] = (
            tagid           = tid,
            length_mean     = len_mean,
            length_lower_ci = len_lower,
            length_upper_ci = len_upper,
            length_sd       = len_sd,
            n_paths_sampled = length(path_ens),
            waypoint_probs  = waypoint_freq,
            release_site    = first(sub_obs.release),
            recapture_site  = first(sub_obs.recapture),
        )
    end

    node_visit_probs = node_visit_counts ./ (n_eval_draws + 1e-10)

    return (
        path_samples     = path_samples,
        path_stats       = path_stats,
        node_visit_probs = node_visit_probs,
        v_samples        = v_samples,
        d_samples        = d_samples,
        g_samples        = g_samples,
    )
end

"""
    export_path_uncertainty_summary(path_unc, loaded, output_dir) -> String

Exports per-individual trajectory credible interval metrics to CSV format.
"""
function export_path_uncertainty_summary(path_unc, loaded, output_dir)::String
    mkpath(output_dir)
    summary_file = joinpath(output_dir, "path_credible_intervals.csv")
    open(summary_file, "w") do io
        write(io, "tagid,release_site,recapture_site,path_length_mean," *
                  "path_length_lower,path_length_upper,path_length_sd," *
                  "n_waypoints,modal_waypoint\n")
        for (tid, stats) in path_unc.path_stats
            modal_wp = argmax(stats.waypoint_probs)
            write(io, string(
                tid, ",",
                stats.release_site, ",",
                stats.recapture_site, ",",
                round(stats.length_mean; digits=2), ",",
                round(stats.length_lower_ci; digits=2), ",",
                round(stats.length_upper_ci; digits=2), ",",
                round(stats.length_sd; digits=2), ",",
                length(stats.waypoint_probs), ",",
                modal_wp, "\n"
            ))
        end
    end
    return summary_file
end

"""
    reconstruct_paths_bayesian_ensemble(loaded, fitted, params = (;)) -> NamedTuple

Reconstructs paths and corridor probabilities by iterating directly over
Turing MCMC posterior samples, capturing full parameter uncertainty.

# Mathematical Formulation
For each posterior sample draw s in {1, ..., S_d}, draw-specific parameters
alpha^(s), rho^(s), gamma^(s) are retrieved from the MCMC chains to construct
a realization of the stochastic transition kernel:
    P^(s) = construct_stochastic_transition_kernel(W, H; alpha^(s), rho^(s), gamma^(s))
Path predictions and forward-backward Markov bridge corridor distributions
are evaluated under each P^(s), and averaged across draws:
    Pi_bar = (1 / S_d) * sum_{s=1}^{S_d} Pi^(s)

# Arguments
- `loaded::NamedTuple`: Spatial domain, mesh, and observation records.
- `fitted::NamedTuple`: Model fitting output containing MCMC chains.
- `params`: Configuration options (`verbose`, `max_paths`, `path_method`,
  `n_ensemble`).

# Returns
`NamedTuple` with:
- `ensemble_corridors`: Dictionary of tag ID to mean corridor matrix.
- `ensemble_paths`: Dictionary of tag ID to collection of sampled paths.
"""
function reconstruct_paths_bayesian_ensemble(
    loaded::NamedTuple,
    fitted::NamedTuple,
    params = (;)
)::NamedTuple
    verbose   = get(params, :verbose, true)
    obs_df    = loaded.obs_df
    W         = loaded.W
    hsi_vec   = loaded.hsi_vec
    land_mask = loaded.land_mask
    n_spatial = loaded.n_spatial

    chains    = fitted.chains
    active_chn = haskey(chains, :telemetry) ? chains[:telemetry] :
                 haskey(chains, :telemetry_and_survey) ?
                 chains[:telemetry_and_survey] :
                 first(values(chains))

    cents_mesh = if hasproperty(loaded, :mesh) &&
                    hasproperty(loaded.mesh, :centroids_planar)
        loaded.mesh.centroids_planar
    elseif hasproperty(loaded, :centroids_planar)
        loaded.centroids_planar
    else
        [Float64[0.0, 0.0] for _ in 1:n_spatial]
    end

    chn_mat_v = haskey(active_chn, :velocity) ?
                Array(active_chn[:velocity]) :
                haskey(active_chn, Symbol("velocity[1]")) ?
                Array(active_chn[Symbol("velocity[1]")]) :
                ones(Float64, 50, 1) .* 0.3
    chn_mat_d = haskey(active_chn, :diffusion) ?
                Array(active_chn[:diffusion]) :
                haskey(active_chn, Symbol("diffusion[1]")) ?
                Array(active_chn[Symbol("diffusion[1]")]) :
                ones(Float64, 50, 1) .* 0.1
    chn_mat_g = haskey(active_chn, :gamma) ?
                Array(active_chn[:gamma]) :
                haskey(active_chn, Symbol("gamma[1]")) ?
                Array(active_chn[Symbol("gamma[1]")]) :
                ones(Float64, 50, 1) .* 1.0

    n_draws_chain = size(chn_mat_v, 1)
    req_ensemble  = get(params, :n_ensemble, 50)
    n_ensemble    = min(req_ensemble, n_draws_chain)
    draw_indices  = round.(Int, range(1, n_draws_chain, length = n_ensemble))

    all_tags    = unique(obs_df.tagid)
    max_p       = get(params, :max_paths, length(all_tags))
    sample_tags = all_tags[1:min(max_p, length(all_tags))]
    path_method = get(params, :path_method, :astar)

    ensemble_corridors = Dict{String, Matrix{Float64}}()
    ensemble_paths     = Dict{String, Vector{Vector{Int}}}()

    verbose && println(
        "[Ensemble Phase] Running path prediction over ",
        "$n_ensemble MCMC posterior samples..."
    )

    for tid in sample_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue
        first_row = first(sub_obs)
        grp = hasproperty(sub_obs, :group) ? first(sub_obs.group) : 1

        k_steps = max(1, first_row.k)
        accumulated_corridor = zeros(Float64, n_spatial, k_steps + 1)
        sample_path_collection = Vector{Int}[]

        for (i, d_idx) in enumerate(draw_indices)
            v_col = min(grp, size(chn_mat_v, 2))
            d_col = min(grp, size(chn_mat_d, 2))
            g_col = min(grp, size(chn_mat_g, 2))

            v_s = chn_mat_v[d_idx, v_col]
            d_s = chn_mat_d[d_idx, d_col]
            g_s = chn_mat_g[d_idx, g_col]

            tot_s   = v_s + d_s + 1e-6
            alpha_s = clamp(v_s / tot_s, 0.0, 1.0)
            rho_s   = clamp(1.0 / (1.0 + tot_s), 0.01, 0.95)

            P_draw = construct_stochastic_transition_kernel(
                W, hsi_vec;
                gamma     = g_s,
                residence = rho_s,
                advection = alpha_s,
                land_mask = land_mask
            )

            corr_draw = predict_corridor(
                P_draw, first_row.release, first_row.recapture, k_steps;
                land_mask = land_mask
            )
            accumulated_corridor .+= corr_draw

            path_draw = predict_path(
                P_draw, first_row.release, first_row.recapture, k_steps;
                centroids = cents_mesh,
                method    = path_method,
                land_mask = land_mask
            )
            push!(sample_path_collection, path_draw)
        end

        ensemble_corridors[string(tid)] = accumulated_corridor ./ n_ensemble
        ensemble_paths[string(tid)]     = sample_path_collection
    end

    return (
        ensemble_corridors = ensemble_corridors,
        ensemble_paths     = ensemble_paths
    )
end

"""
    compute_stock_connectivity_matrix(loaded, kernels, params = (;);
                                      region_labels = nothing, region_map = nothing)

Aggregates transition kernel probabilities across spatial regions (stocks) to
derive a population-level stochastic connectivity matrix:
`Connectivity[r, s] = (1 / (|r|*|s|)) * ∑_{u∈r} ∑_{v∈s} P_kernel[u, v]`

# Arguments
- `loaded::NamedTuple`: Spatial domain and observation records.
- `kernels::NamedTuple`: Extracted transition kernels.
- `params`: Configuration options.
- `region_labels`: Optional vector of human-readable region names.
- `region_map`: Vector assigning each mesh unit to a region index (1:n_regions).
"""
function compute_stock_connectivity_matrix(
    loaded::NamedTuple,
    kernels::NamedTuple,
    params = (;);
    region_labels::Union{Vector{String}, Nothing} = nothing,
    region_map::Union{Vector{Int}, Nothing} = nothing
)::NamedTuple
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    P_kernel  = kernels.P_kernel

    if region_map === nothing
        region_map = collect(1:n_spatial)
    end
    n_regions = maximum(region_map)

    if region_labels === nothing
        region_labels = ["Region $i" for i in 1:n_regions]
    end

    connectivity_matrix = zeros(Float64, n_regions, n_regions)
    flow_counts = zeros(Int, n_regions, n_regions)

    for row in eachrow(obs_df)
        rel_r = region_map[clamp(row.release, 1, n_spatial)]
        rec_r = region_map[clamp(row.recapture, 1, n_spatial)]
        flow_counts[rel_r, rec_r] += 1
    end

    for r in 1:n_regions
        units_r = findall(==(r), region_map)
        isempty(units_r) && continue
        for s in 1:n_regions
            units_s = findall(==(s), region_map)
            isempty(units_s) && continue

            prob_sum = 0.0
            P_k = P_kernel isa AbstractVector ? P_kernel[1] : P_kernel
            for u in units_r, v in units_s
                if 1 <= u <= n_spatial && 1 <= v <= n_spatial
                    prob_sum += P_k[u, v]
                end
            end
            connectivity_matrix[r, s] = prob_sum / (length(units_r) * length(units_s) + 1e-10)
        end
    end

    row_sums = vec(sum(connectivity_matrix; dims = 2))
    for r in 1:n_regions
        if row_sums[r] > 0.0
            connectivity_matrix[r, :] ./= row_sums[r]
        end
    end

    flow_rates = connectivity_matrix .* (flow_counts ./ (sum(flow_counts) + 1e-10))

    return (
        connectivity_matrix = connectivity_matrix,
        flow_counts         = flow_counts,
        flow_rates          = flow_rates,
        region_labels       = region_labels,
        region_map          = region_map,
        n_regions           = n_regions,
    )
end

"""
    compute_connectivity_credible_intervals(loaded, fitted, kernels, params = (;),
                                            region_map = nothing)

Propagates MCMC posterior uncertainty to derive credible intervals for
inter-region transition probabilities.
"""
function compute_connectivity_credible_intervals(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params = (;),
    region_map::Union{Vector{Int}, Nothing} = nothing
)::NamedTuple
    n_spatial = loaded.n_spatial
    land_mask = loaded.land_mask
    G         = kernels.G

    if region_map === nothing
        region_map = collect(1:n_spatial)
    end
    n_regions = maximum(region_map)

    n_draws = if hasproperty(kernels, :alpha_samples) &&
                 !isempty(kernels.alpha_samples)
        size(kernels.alpha_samples, 1)
    else
        chains = fitted.chains
        active_chain = haskey(chains, :telemetry) ?
                       chains[:telemetry] : chains[:telemetry_and_survey]
        size(Array(active_chain), 1)
    end

    v_samples = zeros(n_draws, G)
    d_samples = zeros(n_draws, G)
    g_samples = zeros(n_draws, G)

    if hasproperty(kernels, :alpha_samples) && !isempty(kernels.alpha_samples)
        for g in 1:G
            v_samples[:, g] .= kernels.alpha_samples[:, min(g, size(kernels.alpha_samples, 2))]
            d_samples[:, g] .= kernels.rho_samples[:, min(g, size(kernels.rho_samples, 2))]
            g_samples[:, g] .= kernels.gamma_samples[:, min(g, size(kernels.gamma_samples, 2))]
        end
    else
        chains = fitted.chains
        active_chain = haskey(chains, :telemetry) ?
                       chains[:telemetry] : chains[:telemetry_and_survey]
        for draw in 1:n_draws, g in 1:G
            vk = Symbol("velocity[$g]")
            dk = Symbol("diffusion[$g]")
            gk = Symbol("gamma[$g]")
            v_samples[draw, g] = haskey(chains, vk) ? mean(active_chain[vk]) : 0.3
            d_samples[draw, g] = haskey(chains, dk) ? mean(active_chain[dk]) : 0.1
            g_samples[draw, g] = haskey(chains, gk) ? mean(active_chain[gk]) : 1.0
        end
    end

    n_eval_draws = min(n_draws, 40)
    eval_indices = round.(Int, range(1, n_draws, length=n_eval_draws))
    connectivity_samples = Matrix{Float64}[]

    for draw in eval_indices
        alpha_draw = clamp(v_samples[draw, 1] /
                           (v_samples[draw, 1] + d_samples[draw, 1] + 1e-6),
                           0.0, 1.0)
        rho_draw   = clamp(1.0 / (1.0 + v_samples[draw, 1] + d_samples[draw, 1]),
                           0.01, 0.95)
        gamma_draw = g_samples[draw, 1]

        P_draw = construct_stochastic_transition_kernel(
            loaded.W, loaded.hsi_vec;
            gamma     = fill(gamma_draw, G),
            residence = fill(rho_draw, G),
            advection = fill(alpha_draw, G),
            land_mask = land_mask
        )
        P_k = P_draw isa AbstractVector ? P_draw[1] : P_draw

        conn_mat = zeros(Float64, n_regions, n_regions)
        for r in 1:n_regions
            units_r = findall(==(r), region_map)
            isempty(units_r) && continue
            for s in 1:n_regions
                units_s = findall(==(s), region_map)
                isempty(units_s) && continue
                prob_sum = 0.0
                for u in units_r, v in units_s
                    if 1 <= u <= n_spatial && 1 <= v <= n_spatial
                        prob_sum += P_k[u, v]
                    end
                end
                conn_mat[r, s] = prob_sum / (length(units_r) * length(units_s) + 1e-10)
            end
        end

        row_sums = vec(sum(conn_mat; dims = 2))
        for r in 1:n_regions
            row_sums[r] > 0.0 && (conn_mat[r, :] ./= row_sums[r])
        end
        push!(connectivity_samples, conn_mat)
    end

    conn_mean = mean(connectivity_samples)
    conn_lower = [quantile([s[i, j] for s in connectivity_samples], 0.025)
                  for i in 1:n_regions, j in 1:n_regions]
    conn_upper = [quantile([s[i, j] for s in connectivity_samples], 0.975)
                  for i in 1:n_regions, j in 1:n_regions]

    return (
        connectivity_mean    = conn_mean,
        connectivity_lower   = conn_lower,
        connectivity_upper   = conn_upper,
        connectivity_samples = connectivity_samples,
        n_regions            = n_regions,
    )
end

"""
    export_connectivity_matrix(conn, output_dir) -> String

Exports connectivity matrix, flow counts, and summary CSV files.
"""
function export_connectivity_matrix(conn, output_dir)::String
    mkpath(output_dir)
    summary_file = joinpath(output_dir, "stock_connectivity_summary.csv")
    open(summary_file, "w") do io
        write(io, "source_region,sink_region,connectivity,flow_count,flow_rate\n")
        for r in 1:conn.n_regions, s in 1:conn.n_regions
            write(io, string(
                conn.region_labels[r], ",",
                conn.region_labels[s], ",",
                round(conn.connectivity_matrix[r, s]; digits=6), ",",
                conn.flow_counts[r, s], ",",
                round(conn.flow_rates[r, s]; digits=6), "\n"
            ))
        end
    end
    return summary_file
end

"""
    export_connectivity_uncertainty(conn_unc, output_dir, region_labels) -> String

Exports connectivity credible interval bounds to CSV.
"""
function export_connectivity_uncertainty(conn_unc, output_dir, region_labels)::String
    mkpath(output_dir)
    unc_file = joinpath(output_dir, "stock_connectivity_credible_intervals.csv")
    open(unc_file, "w") do io
        write(io, "source_region,sink_region,mean_conn,lower_ci,upper_ci\n")
        for r in 1:conn_unc.n_regions, s in 1:conn_unc.n_regions
            write(io, string(
                region_labels[r], ",",
                region_labels[s], ",",
                round(conn_unc.connectivity_mean[r, s]; digits=6), ",",
                round(conn_unc.connectivity_lower[r, s]; digits=6), ",",
                round(conn_unc.connectivity_upper[r, s]; digits=6), "\n"
            ))
        end
    end
    return unc_file
end

"""
    posterior_predictive_check(loaded, fitted, kernels, params = (;)) -> NamedTuple

Performs Bayesian posterior predictive validation by simulating mark-recapture
events and assessing calibration via Brier score and Kullback-Leibler divergence.
"""
function posterior_predictive_check(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params = (;)
)::NamedTuple
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    land_mask = loaded.land_mask
    G         = kernels.G
    seed      = get(params, :seed, 42)

    n_draws = if hasproperty(kernels, :alpha_samples) &&
                 !isempty(kernels.alpha_samples)
        size(kernels.alpha_samples, 1)
    else
        chains = fitted.chains
        active_chain = haskey(chains, :telemetry) ?
                       chains[:telemetry] : chains[:telemetry_and_survey]
        size(Array(active_chain), 1)
    end

    v_samples = zeros(n_draws, G)
    d_samples = zeros(n_draws, G)
    g_samples = zeros(n_draws, G)

    if hasproperty(kernels, :alpha_samples) && !isempty(kernels.alpha_samples)
        for g in 1:G
            v_samples[:, g] .= kernels.alpha_samples[:, min(g, size(kernels.alpha_samples, 2))]
            d_samples[:, g] .= kernels.rho_samples[:, min(g, size(kernels.rho_samples, 2))]
            g_samples[:, g] .= kernels.gamma_samples[:, min(g, size(kernels.gamma_samples, 2))]
        end
    else
        chains = fitted.chains
        active_chain = haskey(chains, :telemetry) ?
                       chains[:telemetry] : chains[:telemetry_and_survey]
        for draw in 1:n_draws, g in 1:G
            vk = Symbol("velocity[$g]")
            dk = Symbol("diffusion[$g]")
            gk = Symbol("gamma[$g]")
            v_samples[draw, g] = haskey(chains, vk) ? mean(active_chain[vk]) : 0.3
            d_samples[draw, g] = haskey(chains, dk) ? mean(active_chain[dk]) : 0.1
            g_samples[draw, g] = haskey(chains, gk) ? mean(active_chain[gk]) : 1.0
        end
    end

    observed_recaptures = obs_df.recapture
    observed_dist = zeros(n_spatial)
    for rec in observed_recaptures
        1 <= rec <= n_spatial && (observed_dist[rec] += 1)
    end
    observed_dist ./= max(1.0, sum(observed_dist))

    brier_scores = Float64[]
    kl_divergences = Float64[]
    simulated_recapture_dists = Vector{Int}[]
    rng = MersenneTwister(seed)

    n_eval_draws = min(n_draws, 30)
    eval_indices = round.(Int, range(1, n_draws, length=n_eval_draws))

    for draw in eval_indices
        alpha_draw = clamp(v_samples[draw, 1] /
                           (v_samples[draw, 1] + d_samples[draw, 1] + 1e-6),
                           0.0, 1.0)
        rho_draw   = clamp(1.0 / (1.0 + v_samples[draw, 1] + d_samples[draw, 1]),
                           0.01, 0.95)
        gamma_draw = g_samples[draw, 1]

        P_draw = construct_stochastic_transition_kernel(
            loaded.W, loaded.hsi_vec;
            gamma     = fill(gamma_draw, G),
            residence = fill(rho_draw, G),
            advection = fill(alpha_draw, G),
            land_mask = land_mask
        )

        simulated_recaptures = Int[]
        for row in eachrow(obs_df)
            rel = row.release
            k   = row.k
            grp = hasproperty(row, :group) ? row.group : 1
            grp = clamp(grp, 1, G)

            P_k = P_draw isa AbstractVector ? P_draw[grp] : P_draw
            if !isnothing(P_k) && 1 <= rel <= n_spatial
                prob_vec = zeros(n_spatial)
                prob_vec[rel] = 1.0
                for _ in 1:min(k, 10)
                    prob_vec = P_k' * prob_vec
                end
                s = sum(prob_vec)
                if s > 0.0
                    prob_vec ./= s
                    u_samp = rand(rng)
                    c_sum = 0.0
                    sampled_rec = rel
                    for j in 1:n_spatial
                        c_sum += prob_vec[j]
                        if u_samp <= c_sum
                            sampled_rec = j
                            break
                        end
                    end
                    push!(simulated_recaptures, sampled_rec)
                else
                    push!(simulated_recaptures, rel)
                end
            end
        end

        sim_dist = zeros(n_spatial)
        for rec in simulated_recaptures
            1 <= rec <= n_spatial && (sim_dist[rec] += 1)
        end
        sim_dist ./= max(1.0, sum(sim_dist))
        push!(simulated_recapture_dists, simulated_recaptures)

        brier = mean((observed_dist .- sim_dist) .^ 2)
        push!(brier_scores, brier)

        kl = 0.0
        for i in 1:n_spatial
            if observed_dist[i] > 1e-10 && sim_dist[i] > 1e-10
                kl += observed_dist[i] * log(observed_dist[i] / sim_dist[i])
            end
        end
        push!(kl_divergences, kl)
    end

    summary = (
        n_draws        = length(brier_scores),
        n_observations = length(observed_recaptures),
        brier_mean     = mean(brier_scores),
        brier_sd       = std(brier_scores),
        brier_lower_ci = quantile(brier_scores, 0.025),
        brier_upper_ci = quantile(brier_scores, 0.975),
        kl_mean        = mean(kl_divergences),
        kl_sd          = std(kl_divergences),
        kl_lower_ci    = quantile(kl_divergences, 0.025),
        kl_upper_ci    = quantile(kl_divergences, 0.975),
    )

    return (
        brier_scores              = brier_scores,
        kl_divergences            = kl_divergences,
        observed_recaptures       = observed_recaptures,
        observed_dist             = observed_dist,
        simulated_recapture_dists = simulated_recapture_dists,
        summary                   = summary,
    )
end

"""
    export_posterior_predictive_check(ppc, output_dir) -> String

Exports posterior predictive validation diagnostics to CSV and text summaries.
"""
function export_posterior_predictive_check(ppc, output_dir)::String
    mkpath(output_dir)
    summary_file = joinpath(output_dir, "posterior_predictive_summary.txt")
    open(summary_file, "w") do f
        write(f, "Posterior Predictive Check Summary\n")
        write(f, "=" ^ 50 * "\n\n")
        write(f, "Observations: $(ppc.summary.n_observations)\n")
        write(f, "Evaluated MCMC Draws: $(ppc.summary.n_draws)\n\n")
        write(f, "Brier Score (MSE): $(round(ppc.summary.brier_mean; digits=6)) " *
                 "[$(round(ppc.summary.brier_lower_ci; digits=6)), " *
                 "$(round(ppc.summary.brier_upper_ci; digits=6))]\n")
        write(f, "KL Divergence:    $(round(ppc.summary.kl_mean; digits=6)) " *
                 "[$(round(ppc.summary.kl_lower_ci; digits=6)), " *
                 "$(round(ppc.summary.kl_upper_ci; digits=6))]\n")
    end
    return summary_file
end

"""
    plot_posterior_predictive_check(ppc, output_dir) -> String

Generates diagnostic plots for posterior predictive validation.
"""
function plot_posterior_predictive_check(ppc, output_dir)::String
    mkpath(output_dir)
    plot_file = joinpath(output_dir, "posterior_predictive_diagnostics.png")

    p1 = plot(ppc.brier_scores; label="Brier Score", xlabel="Draw", ylabel="Score",
              title="Posterior Predictive: Brier Score", legend=:topright)
    p2 = plot(ppc.kl_divergences; label="KL Divergence", xlabel="Draw", ylabel="Divergence",
              title="Posterior Predictive: KL Divergence", legend=:topright)
    p3 = plot(1:length(ppc.observed_dist), ppc.observed_dist;
              label="Observed", xlabel="Spatial Unit", ylabel="Probability",
              title="Recapture Probability Distribution")

    plot(p1, p2, p3; layout=(3, 1), size=(800, 900))
    savefig(plot_file)
    return plot_file
end

"""
    run_priority_analyses(loaded, fitted, kernels, params, output_dir) -> NamedTuple

Consolidated priority post-processing pipeline executing:
1. Path credible intervals
2. Stock connectivity matrix
3. Posterior predictive checks
"""
function run_priority_analyses(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params,
    output_dir::String
)::NamedTuple
    mkpath(output_dir)
    verbose = get(params, :verbose, true)

    verbose && println("\n--- Running Priority Analyses ---")
    path_unc = path_credible_intervals(loaded, fitted, kernels, params)
    export_path_uncertainty_summary(path_unc, loaded, output_dir)

    conn = compute_stock_connectivity_matrix(loaded, kernels, params)
    export_connectivity_matrix(conn, output_dir)

    conn_unc = compute_connectivity_credible_intervals(
        loaded, fitted, kernels, params, conn.region_map
    )
    export_connectivity_uncertainty(conn_unc, output_dir, conn.region_labels)

    ppc = posterior_predictive_check(loaded, fitted, kernels, params)
    summary_file = export_posterior_predictive_check(ppc, output_dir)

    return (
        path_uncertainty         = path_unc,
        connectivity_matrix      = conn,
        connectivity_uncertainty = conn_unc,
        posterior_predictive     = ppc,
        summary_file             = summary_file,
    )
end

# =============================================================================
# Standalone Movement Dashboards
# =============================================================================

"""
    export_movement_posterior_dashboard(filepath, kernels; species = "generic")

Generates a standalone dark-mode SVG/HTML dashboard visualizing posterior parameter
distributions (KDE density curves with 95% Bayesian credible intervals) and
pairwise bivariate correlation scatter plots with regression trendlines.
"""
function export_movement_posterior_dashboard(
    filepath::String,
    kernels::NamedTuple;
    species::String = "generic"
)::String
    mkpath(dirname(filepath))
    G = kernels.G
    grp_lookup = hasproperty(kernels, :group_lookup) ?
                 kernels.group_lookup : Dict{Int, String}()

    alpha_samples = hasproperty(kernels, :alpha_samples) ?
                    kernels.alpha_samples : Matrix{Float64}(undef, 0, 0)
    rho_samples   = hasproperty(kernels, :rho_samples) ?
                    kernels.rho_samples : Matrix{Float64}(undef, 0, 0)
    gamma_samples = hasproperty(kernels, :gamma_samples) ?
                    kernels.gamma_samples : Matrix{Float64}(undef, 0, 0)

    function _svg_kde_curve(
        samples::Vector{Float64},
        param_label::String,
        color::String;
        width::Int = 320,
        height::Int = 180
    )::String
        N = length(samples)
        N < 2 && return "<p>Insufficient samples</p>"
        s_mean = mean(samples)
        s_sd   = std(samples)
        q025   = quantile(samples, 0.025)
        q975   = quantile(samples, 0.975)
        x_lo   = minimum(samples) - 0.5 * s_sd
        x_hi   = maximum(samples) + 0.5 * s_sd
        x_hi <= x_lo && (x_hi = x_lo + 1.0)

        bw = 1.06 * max(s_sd, 1e-4) * (N ^ (-0.2))
        n_eval = 60
        xs = range(x_lo, x_hi, length = n_eval)
        dens = zeros(Float64, n_eval)
        inv_bw = 1.0 / bw
        norm_c = 1.0 / (N * bw * sqrt(2.0 * π))
        for i in 1:n_eval
            x_i = xs[i]
            dens[i] = sum(exp(-0.5 * ((x_i - s) * inv_bw)^2) for s in samples) * norm_c
        end
        max_d = maximum(dens)
        max_d <= 0.0 && (max_d = 1.0)

        pad_l, pad_r, pad_t, pad_b = 45, 15, 20, 30
        plot_w = width - pad_l - pad_r
        plot_h = height - pad_t - pad_b
        y_axis = height - pad_b

        io = IOBuffer()
        write(io, """<svg width="$width" height="$height" """ *
                  """xmlns="http://www.w3.org/2000/svg">""")
        x2_ax = width - pad_r
        write(io, """<line x1="$pad_l" y1="$y_axis" x2="$x2_ax" """ *
                  """y2="$y_axis" stroke="#475569" stroke-width="1"/>""")
        write(io, """<line x1="$pad_l" y1="$pad_t" x2="$pad_l" """ *
                  """y2="$y_axis" stroke="#475569" stroke-width="1"/>""")

        poly_pts = String[]
        for i in 1:n_eval
            x_val = xs[i]
            if q025 <= x_val <= q975
                px = pad_l + ((x_val - x_lo) / (x_hi - x_lo)) * plot_w
                py = y_axis - (dens[i] / max_d) * plot_h
                push!(poly_pts, "$(round(px; digits=1)),$(round(py; digits=1))")
            end
        end
        if !isempty(poly_pts)
            x_start = pad_l + ((max(x_lo, q025) - x_lo) / (x_hi - x_lo)) * plot_w
            x_end   = pad_l + ((min(x_hi, q975) - x_lo) / (x_hi - x_lo)) * plot_w
            shade_poly = "$(round(x_start; digits=1)),$y_axis " *
                         join(poly_pts, " ") *
                         " $(round(x_end; digits=1)),$y_axis"
            write(io, """<polygon points="$shade_poly" fill="$color" """ *
                      """opacity="0.22"/>""")
        end

        pts = String[]
        for i in 1:n_eval
            px = pad_l + ((xs[i] - x_lo) / (x_hi - x_lo)) * plot_w
            py = y_axis - (dens[i] / max_d) * plot_h
            push!(pts, (i == 1 ? "M" : "L") *
                       " $(round(px; digits=1)) $(round(py; digits=1))")
        end
        d_str = join(pts, " ")
        write(io, """<path d="$d_str" fill="none" stroke="$color" """ *
                  """stroke-width="2.2"/>""")

        xm_px = round(pad_l + ((s_mean - x_lo) / (x_hi - x_lo)) * plot_w; digits=1)
        write(io, """<line x1="$xm_px" y1="$pad_t" x2="$xm_px" """ *
                  """y2="$y_axis" stroke="#f8fafc" stroke-width="1.5" """ *
                  """stroke-dasharray="3,3"/>""")

        mid_x = pad_l + plot_w ÷ 2
        write(io, """<text x="$mid_x" y="$(height - 6)" text-anchor="middle" """ *
                  """fill="#94a3b8" font-size="11" """ *
                  """font-family="Outfit, sans-serif">$param_label</text>""")
        write(io, """<text x="$pad_l" y="$(y_axis + 14)" text-anchor="start" """ *
                  """fill="#64748b" font-size="9" """ *
                  """font-family="JetBrains Mono, monospace">""" *
                  """$(round(x_lo; digits=2))</text>""")
        write(io, """<text x="$x2_ax" y="$(y_axis + 14)" text-anchor="end" """ *
                  """fill="#64748b" font-size="9" """ *
                  """font-family="JetBrains Mono, monospace">""" *
                  """$(round(x_hi; digits=2))</text>""")
        write(io, """<text x="$xm_px" y="$(pad_t - 4)" text-anchor="middle" """ *
                  """fill="#f8fafc" font-size="9" """ *
                  """font-family="JetBrains Mono, monospace">""" *
                  """μ=$(round(s_mean; digits=3))</text>""")

        write(io, "</svg>")
        return String(take!(io))
    end

    function _svg_scatter_corr(
        x_vals::Vector{Float64},
        y_vals::Vector{Float64},
        xlab::String,
        ylab::String,
        color::String;
        width::Int = 320,
        height::Int = 180
    )::String
        N = min(length(x_vals), length(y_vals))
        N < 2 && return "<p>Insufficient samples</p>"
        x_sub = x_vals[1:N]
        y_sub = y_vals[1:N]
        x_lo, x_hi = extrema(x_sub)
        y_lo, y_hi = extrema(y_sub)
        x_hi <= x_lo && (x_hi = x_lo + 1.0)
        y_hi <= y_lo && (y_hi = y_lo + 1.0)

        x_m = mean(x_sub)
        y_m = mean(y_sub)
        r_num = sum((x_sub[i] - x_m) * (y_sub[i] - y_m) for i in 1:N)
        r_den = sqrt(sum((x - x_m)^2 for x in x_sub) *
                     sum((y - y_m)^2 for y in y_sub))
        r_val = r_den > 1e-10 ? clamp(r_num / r_den, -1.0, 1.0) : 0.0

        pad_l, pad_r, pad_t, pad_b = 45, 15, 20, 30
        plot_w = width - pad_l - pad_r
        plot_h = height - pad_t - pad_b
        y_axis = height - pad_b

        io = IOBuffer()
        write(io, """<svg width="$width" height="$height" """ *
                  """xmlns="http://www.w3.org/2000/svg">""")
        x2_ax = width - pad_r
        write(io, """<line x1="$pad_l" y1="$y_axis" x2="$x2_ax" """ *
                  """y2="$y_axis" stroke="#475569" stroke-width="1"/>""")
        write(io, """<line x1="$pad_l" y1="$pad_t" x2="$pad_l" """ *
                  """y2="$y_axis" stroke="#475569" stroke-width="1"/>""")

        step = max(1, div(N, 120))
        for i in 1:step:N
            px = round(pad_l + ((x_sub[i] - x_lo) / (x_hi - x_lo)) * plot_w; digits=1)
            py = round(y_axis - ((y_sub[i] - y_lo) / (y_hi - y_lo)) * plot_h; digits=1)
            write(io, """<circle cx="$px" cy="$py" r="2.5" fill="$color" """ *
                      """opacity="0.55"/>""")
        end

        ss_xx = sum((x - x_m)^2 for x in x_sub)
        beta_1 = ss_xx > 1e-10 ? r_num / ss_xx : 0.0
        beta_0 = y_m - beta_1 * x_m
        y_p1 = beta_0 + beta_1 * x_lo
        y_p2 = beta_0 + beta_1 * x_hi
        py1 = round(y_axis - clamp((y_p1 - y_lo) / (y_hi - y_lo), 0.0, 1.0) * plot_h; digits=1)
        py2 = round(y_axis - clamp((y_p2 - y_lo) / (y_hi - y_lo), 0.0, 1.0) * plot_h; digits=1)
        write(io, """<line x1="$pad_l" y1="$py1" x2="$x2_ax" y2="$py2" """ *
                  """stroke="#f8fafc" stroke-width="1.5" opacity="0.85" """ *
                  """stroke-dasharray="4,3"/>""")

        mid_x = pad_l + plot_w ÷ 2
        r_str = string(round(r_val; digits=3))
        write(io, """<text x="$mid_x" y="$(height - 6)" text-anchor="middle" """ *
                  """fill="#94a3b8" font-size="11" """ *
                  """font-family="Outfit, sans-serif">$xlab vs $ylab (r = $r_str)</text>""")
        write(io, """<text x="$pad_l" y="$(y_axis + 14)" text-anchor="start" """ *
                  """fill="#64748b" font-size="9" """ *
                  """font-family="JetBrains Mono, monospace">""" *
                  """$(round(x_lo; digits=2))</text>""")
        write(io, """<text x="$x2_ax" y="$(y_axis + 14)" text-anchor="end" """ *
                  """fill="#64748b" font-size="9" """ *
                  """font-family="JetBrains Mono, monospace">""" *
                  """$(round(x_hi; digits=2))</text>""")

        write(io, "</svg>")
        return String(take!(io))
    end

    group_sections = String[]
    table_rows = String[]
    palette = ["#38bdf8", "#10b981", "#fbbf24", "#f43f5e", "#a78bfa"]

    for g in 1:G
        lbl = get(grp_lookup, g, "Group $g")
        c = palette[mod(g - 1, length(palette)) + 1]

        a_v = size(alpha_samples, 2) >= g ? alpha_samples[:, g] : Float64[kernels.alpha[g]]
        r_v = size(rho_samples, 2) >= g   ? rho_samples[:, g]   : Float64[kernels.rho[g]]
        g_v = size(gamma_samples, 2) >= g ? gamma_samples[:, g] : Float64[kernels.gamma[g]]

        a_kde = _svg_kde_curve(a_v, "Advection (α)", "#38bdf8")
        r_kde = _svg_kde_curve(r_v, "Residence (ρ)", "#10b981")
        g_kde = _svg_kde_curve(g_v, "Diffusion (γ)", "#fbbf24")

        sc_ar = _svg_scatter_corr(a_v, r_v, "α", "ρ", "#38bdf8")
        sc_ag = _svg_scatter_corr(a_v, g_v, "α", "γ", "#fbbf24")

        push!(group_sections, """
        <div class="card">
          <h2>$lbl — Marginal Posteriors & Parameter Correlations</h2>
          <div class="grid-3">
            <div class="plot-box"><div class="plot-title">Advection α ($lbl)</div>$a_kde</div>
            <div class="plot-box"><div class="plot-title">Residence ρ ($lbl)</div>$r_kde</div>
            <div class="plot-box"><div class="plot-title">Diffusion γ ($lbl)</div>$g_kde</div>
          </div>
          <h3 style="margin-top: 18px; margin-bottom: 8px; color: #94a3b8;">""" *
          """Bivariate Correlations</h3>
          <div class="grid-2">
            <div class="plot-box"><div class="plot-title">α vs ρ Correlation</div>$sc_ar</div>
            <div class="plot-box"><div class="plot-title">α vs γ Correlation</div>$sc_ag</div>
          </div>
        </div>
        """)

        a_m = round(mean(a_v); digits=3)
        a_q025 = round(quantile(a_v, 0.025); digits=3)
        a_q975 = round(quantile(a_v, 0.975); digits=3)
        r_m = round(mean(r_v); digits=3)
        r_q025 = round(quantile(r_v, 0.025); digits=3)
        r_q975 = round(quantile(r_v, 0.975); digits=3)
        g_m = round(mean(g_v); digits=3)
        g_q025 = round(quantile(g_v, 0.025); digits=3)
        g_q975 = round(quantile(g_v, 0.975); digits=3)

        push!(table_rows, """
        <tr>
          <td>$lbl</td>
          <td>$a_m [$a_q025, $a_q975]</td>
          <td>$r_m [$r_q025, $r_q975]</td>
          <td>$g_m [$g_q025, $g_q975]</td>
        </tr>
        """)
    end

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MovementAnalysis Movement Posterior Uncertainty Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?\
family=Outfit:wght@300;400;600;700&\
family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19; --surface: #131b2e; --border: #1e293b;
      --text: #f1f5f9; --muted: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: var(--bg); color: var(--text);
      padding: 28px; line-height: 1.5;
    }
    h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; }
    .subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 28px; }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 22px; margin-bottom: 24px;
    }
    .card h2 { font-size: 1.15rem; font-weight: 600; margin-bottom: 14px; color: #38bdf8; }
    .plot-box {
      background: #080c14; border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: 8px; padding: 12px;
      display: flex; flex-direction: column; align-items: center;
    }
    .plot-title {
      font-size: 0.9rem; font-weight: 600; color: #cbd5e1;
      margin-bottom: 8px; align-self: flex-start;
    }
    table {
      width: 100%; border-collapse: collapse; font-size: 0.85rem;
    }
    th { text-align: left; color: var(--muted);
         border-bottom: 1px solid var(--border); padding: 8px; }
    td { padding: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.05);
         font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
    tr:hover td { background: rgba(56, 189, 248, 0.05); }
  </style>
</head>
<body>
  <h1>Bayesian Posterior Uncertainty & Correlation Dashboard</h1>
  <div class="subtitle">MCMC Parameter Posteriors for Species: """ *
  """$(uppercase(species)) • 95% Bayesian Credible Intervals</div>

  $(join(group_sections, "\n"))

  <div class="card">
    <h2>Parameter Posterior Summary & 95% Credible Intervals</h2>
    <table>
      <thead>
        <tr>
          <th>Group / Stratum</th>
          <th>Advection α [95% CI]</th>
          <th>Residence ρ [95% CI]</th>
          <th>Diffusion γ [95% CI]</th>
        </tr>
      </thead>
      <tbody>
        $(join(table_rows, "\n"))
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    write(filepath, html)
    return filepath
end

"""
    export_movement_flow_dashboard(filepath, path_results, loaded; species = "generic")

Generates a standalone SVG/HTML directed network flow diagram visualizing stock
connectivity, transit volumes, and bottleneck routes with interactive threshold
filtering.
"""
function export_movement_flow_dashboard(
    filepath::String,
    path_results::NamedTuple,
    loaded::NamedTuple;
    species::String = "generic"
)::String
    mkpath(dirname(filepath))
    paths = path_results.paths
    n_spatial = loaded.n_spatial

    flow_pairs = Dict{Tuple{Int, Int}, Int}()
    for p in values(paths)
        length(p) < 2 && continue
        u, v = p[1], p[end]
        flow_pairs[(u, v)] = get(flow_pairs, (u, v), 0) + 1
    end

    all_nodes = Set{Int}()
    for (u, v) in keys(flow_pairs)
        push!(all_nodes, u)
        push!(all_nodes, v)
    end
    node_list = sort(collect(all_nodes))
    N_nodes = max(1, length(node_list))

    width, height = 800, 520
    cx, cy, r_layout = width ÷ 2, height ÷ 2, 210

    node_pos = Dict{Int, Tuple{Float64, Float64}}()
    for (i, node) in enumerate(node_list)
        ang = 2.0 * π * (i - 1) / N_nodes - π / 2.0
        x = cx + r_layout * cos(ang)
        y = cy + r_layout * sin(ang)
        node_pos[node] = (round(x; digits=1), round(y; digits=1))
    end

    max_count = isempty(flow_pairs) ? 1 : maximum(values(flow_pairs))
    sorted_flows = sort(collect(flow_pairs); by = x -> x[2], rev = true)

    edge_svgs = String[]
    for ((u, v), count) in sorted_flows
        !haskey(node_pos, u) || !haskey(node_pos, v) && continue
        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]
        mx = (x1 + x2) / 2.0 + (cy - (y1 + y2) / 2.0) * 0.25
        my = (y1 + y2) / 2.0 + ((x1 + x2) / 2.0 - cx) * 0.25

        w_stroke = clamp(1.2 + 4.5 * (count / max_count), 1.0, 6.0)
        opacity = clamp(0.35 + 0.55 * (count / max_count), 0.2, 0.95)

        push!(edge_svgs, string(
            """<path class="flow-edge" data-count="$count" """,
            """d="M $(round(x1; digits=1)) $(round(y1; digits=1)) """,
            """Q $(round(mx; digits=1)) $(round(my; digits=1)) """,
            """$(round(x2; digits=1)) $(round(y2; digits=1))" """,
            """fill="none" stroke="#38bdf8" stroke-width="$w_stroke" """,
            """opacity="$(round(opacity; digits=2))" marker-end="url(#arrow)">""",
            """<title>Flow $u → $v: $count individuals</title></path>"""
        ))
    end

    node_svgs = String[]
    for node in node_list
        x, y = node_pos[node]
        push!(node_svgs, string(
            """<g class="flow-node">""",
            """<circle cx="$x" cy="$y" r="8" fill="#1e293b" """ *
            """stroke="#38bdf8" stroke-width="2"/>""",
            """<text x="$x" y="$(y - 12)" text-anchor="middle" fill="#94a3b8" """ *
            """font-size="10" font-family="JetBrains Mono, monospace">#$node</text>""",
            """</g>"""
        ))
    end

    table_rows = String[]
    total_transit = sum(values(flow_pairs))
    for (((u, v), count), rank) in zip(sorted_flows, 1:min(12, length(sorted_flows)))
        pct = total_transit > 0 ? (count / total_transit) * 100.0 : 0.0
        push!(table_rows, """
        <tr>
          <td>#$rank</td>
          <td>Spatial Node $u</td>
          <td>Spatial Node $v</td>
          <td>$count</td>
          <td>$(round(pct; digits=1))%</td>
        </tr>
        """)
    end

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MovementAnalysis Stock Connectivity & Network Flow Diagram</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?\
family=Outfit:wght@300;400;600;700&\
family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19; --surface: #131b2e; --border: #1e293b;
      --text: #f1f5f9; --muted: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: var(--bg); color: var(--text);
      padding: 28px; line-height: 1.5;
    }
    h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; }
    .subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 22px; margin-bottom: 24px;
    }
    .card h2 { font-size: 1.15rem; font-weight: 600; margin-bottom: 14px; color: #38bdf8; }
    .graph-container {
      background: #080c14; border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: 8px; display: flex; justify-content: center;
      padding: 16px; position: relative;
    }
    .controls {
      display: flex; gap: 20px; align-items: center;
      margin-top: 14px; font-size: 0.85rem; color: var(--muted);
    }
    .dot { width: 10px; height: 10px; border-radius: 50%;
           display: inline-block; margin-right: 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { text-align: left; color: var(--muted);
         border-bottom: 1px solid var(--border); padding: 8px; }
    td { padding: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.04);
         font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
    tr:hover td { background: rgba(56, 189, 248, 0.05); }
    .slider-row {
      display: flex; align-items: center; gap: 14px; margin-bottom: 16px;
      font-size: 0.9rem; color: var(--muted);
    }
    input[type=range] { accent-color: #38bdf8; cursor: pointer; }
  </style>
</head>
<body>
  <h1>Directed Movement Flow & Stock Connectivity Diagram</h1>
  <div class="subtitle">Inter-Patch Dispersal Corridors for """ *
"""$(uppercase(species)) • Reconstructed Mark-Recapture Trajectories</div>

  <div class="card">
    <h2>Directed Connectivity Graph (Transit Volumes)</h2>
    <div class="slider-row">
      <label for="flowThresh">Filter Minor Flows (Min Individuals): </label>
      <input type="range" id="flowThresh" min="1" max="$max_count" value="1"
             oninput="filterEdges(this.value)">
      <span id="threshVal" style="color: #38bdf8; font-weight: 600;">1</span>
    </div>
    <div class="graph-container">
      <svg width="$width" height="$height" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
          </marker>
        </defs>
        $(join(edge_svgs, "\n"))
        $(join(node_svgs, "\n"))
      </svg>
    </div>
    <div class="controls">
      <span><span class="dot" style="background: #38bdf8;"></span> Directed Path Vector</span>
      <span>Total Directed Links: $(length(flow_pairs))</span>
      <span>Total Animal Transits: $total_transit</span>
    </div>
  </div>

  <div class="card">
    <h2>Top Dispersal Corridors</h2>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Source Patch</th>
          <th>Destination Patch</th>
          <th>Flow Count</th>
          <th>Transit Share</th>
        </tr>
      </thead>
      <tbody>
        $(join(table_rows, "\n"))
      </tbody>
    </table>
  </div>

  <script>
    function filterEdges(val) {
      document.getElementById('threshVal').innerText = val;
      const edges = document.querySelectorAll('.flow-edge');
      edges.forEach(e => {
        const cnt = parseInt(e.getAttribute('data-count') || '0', 10);
        e.style.display = cnt >= val ? 'inline' : 'none';
      });
    }
  </script>
</body>
</html>
"""
    write(filepath, html)
    return filepath
end

"""
    export_movement_summary_dashboard(
        filepath, species, dists, vels, bearings, paths_rich, mov_stats = nothing
    )

Exports the comprehensive summary diagnostics HTML dashboard with
distributions and per-path table.
"""
function export_movement_summary_dashboard(
    filepath::AbstractString,
    species::AbstractString,
    dists::Vector{Float64},
    vels::Vector{Float64},
    bearings::Vector{Float64},
    paths_rich::Vector{<:NamedTuple},
    mov_stats::Union{Nothing, NamedTuple} = nothing
)::String
    # --- SVG histogram builder ---
    function _svg_histogram(
        vals::Vector{Float64}, n_bins::Int,
        xlabel::String, color::String;
        width::Int = 520, height::Int = 260
    )::String
        isempty(vals) && return "<p>No data</p>"
        lo, hi = minimum(vals), maximum(vals)
        hi <= lo && (hi = lo + 1.0)
        bin_w = (hi - lo) / n_bins
        counts = zeros(Int, n_bins)
        for v in vals
            b = clamp(
                floor(Int, (v - lo) / bin_w) + 1, 1, n_bins
            )
            counts[b] += 1
        end
        mx = maximum(counts)
        mx == 0 && (mx = 1)

        pad_l, pad_b, pad_t, pad_r = 50, 40, 20, 10
        plot_w = width - pad_l - pad_r
        plot_h = height - pad_t - pad_b
        bar_w  = plot_w / n_bins

        io = IOBuffer()
        write(io, """<svg width="$width" height="$height"
          xmlns="http://www.w3.org/2000/svg">""")
        # Axes
        write(io, """<line x1="$pad_l" y1="$(height - pad_b)"
          x2="$(width - pad_r)" y2="$(height - pad_b)"
          stroke="#94a3b8" stroke-width="1"/>""")
        write(io, """<line x1="$pad_l" y1="$pad_t"
          x2="$pad_l" y2="$(height - pad_b)"
          stroke="#94a3b8" stroke-width="1"/>""")

        for i in 1:n_bins
            bh = (counts[i] / mx) * plot_h
            bx = pad_l + (i - 1) * bar_w + 1
            by = height - pad_b - bh
            write(io, """<rect x="$(round(bx, digits=1))"
              y="$(round(by, digits=1))"
              width="$(round(bar_w - 2, digits=1))"
              height="$(round(bh, digits=1))"
              fill="$color" opacity="0.85"
              rx="2"/>""")
        end

        # X-axis label
        write(io, """<text x="$(pad_l + plot_w ÷ 2)"
          y="$(height - 5)" text-anchor="middle"
          fill="#94a3b8" font-size="12"
          font-family="Outfit, sans-serif">$xlabel</text>""")

        # Tick labels (lo / hi)
        write(io, """<text x="$pad_l"
          y="$(height - pad_b + 15)"
          text-anchor="start" fill="#94a3b8"
          font-size="10"
          font-family="JetBrains Mono">
          $(round(lo, digits=1))</text>""")
        write(io, """<text x="$(width - pad_r)"
          y="$(height - pad_b + 15)"
          text-anchor="end" fill="#94a3b8"
          font-size="10"
          font-family="JetBrains Mono">
          $(round(hi, digits=1))</text>""")

        # Y-axis max label
        write(io, """<text x="$(pad_l - 5)"
          y="$(pad_t + 10)"
          text-anchor="end" fill="#94a3b8"
          font-size="10"
          font-family="JetBrains Mono">$mx</text>""")

        write(io, "</svg>")
        return String(take!(io))
    end

    # --- Directional wind-rose SVG ---
    function _svg_windrose(
        angles::Vector{Float64};
        size::Int = 300, n_sectors::Int = 16
    )::String
        isempty(angles) && return "<p>No data</p>"
        sector_w = 360.0 / n_sectors
        counts = zeros(Int, n_sectors)
        for a in angles
            s = clamp(
                floor(Int, mod(a, 360.0) / sector_w) + 1,
                1, n_sectors
            )
            counts[s] += 1
        end
        mx = maximum(counts)
        mx == 0 && (mx = 1)

        cx, cy = size ÷ 2, size ÷ 2
        r_max = size ÷ 2 - 30

        io = IOBuffer()
        write(io, """<svg width="$size" height="$size"
          xmlns="http://www.w3.org/2000/svg">""")

        # Concentric guide circles
        for frac in [0.25, 0.5, 0.75, 1.0]
            r = round(Int, r_max * frac)
            write(io, """<circle cx="$cx" cy="$cy" r="$r"
              fill="none" stroke="#334155"
              stroke-width="0.5"/>""")
        end

        # Compass labels
        for (lbl, ax, ay) in [
            ("N", cx, cy - r_max - 12),
            ("E", cx + r_max + 12, cy + 4),
            ("S", cx, cy + r_max + 16),
            ("W", cx - r_max - 12, cy + 4),
        ]
            write(io, """<text x="$ax" y="$ay"
              text-anchor="middle" fill="#64748b"
              font-size="11"
              font-family="Outfit">$lbl</text>""")
        end

        # Petal arcs (filled wedges)
        for s in 1:n_sectors
            r_s = (counts[s] / mx) * r_max
            r_s < 2.0 && continue
            θ_start = (s - 1) * sector_w - 90.0
            θ_end   = θ_start + sector_w
            θ1r = deg2rad(θ_start)
            θ2r = deg2rad(θ_end)
            x1 = cx + r_s * cos(θ1r)
            y1 = cy + r_s * sin(θ1r)
            x2 = cx + r_s * cos(θ2r)
            y2 = cy + r_s * sin(θ2r)
            large = sector_w > 180 ? 1 : 0
            write(io, string(
                "<path d=\"M $cx $cy L ",
                round(x1, digits=1), " ",
                round(y1, digits=1),
                " A ", round(r_s, digits=1), " ",
                round(r_s, digits=1),
                " 0 $large 1 ",
                round(x2, digits=1), " ",
                round(y2, digits=1),
                " Z\" fill=\"#38bdf8\" opacity=\"0.6\"",
                " stroke=\"#38bdf8\" stroke-width=\"0.5\"/>"
            ))
        end

        write(io, "</svg>")
        return String(take!(io))
    end

    # --- Build page ---
    dist_svg    = _svg_histogram(dists, 20, "Total Distance (km)", "#10b981")
    vel_svg     = _svg_histogram(vels, 15, "Velocity (km / step)", "#fbbf24")
    rose_svg    = _svg_windrose(bearings)

    # Summary statistics
    n_paths     = length(dists)
    mean_dist   = isempty(dists) ? 0.0 : mean(dists)
    median_dist = isempty(dists) ? 0.0 : quantile(dists, 0.5)
    mean_vel    = isempty(vels)  ? 0.0 : mean(vels)
    mean_eff    = mov_stats !== nothing ? mean(mov_stats.path_efficiency) :
                  (isempty(paths_rich) ? 0.0 :
                   mean([p.total_dist_km > 0.01 ?
                         clamp(p.displacement_km / p.total_dist_km, 0.0, 1.0) :
                         1.0 for p in paths_rich]))
    kappa_val   = mov_stats !== nothing ?
                  mov_stats.directional_bias.kappa : 0.0

    # Per-path table rows
    table_rows = String[]
    for (i, p) in enumerate(paths_rich)
        p_eff = hasproperty(p, :path_efficiency) ? p.path_efficiency :
                (p.total_dist_km > 0.01 ?
                 clamp(p.displacement_km / p.total_dist_km, 0.0, 1.0) : 1.0)
        p_bout = if mov_stats !== nothing && hasproperty(mov_stats, :behavioral_bouts)
            bouts = mov_stats.behavioral_bouts
            cls = hasproperty(bouts, :classification) ?
                bouts.classification : (bouts isa AbstractVector ? bouts : String[])
            i <= length(cls) ? cls[i] : (p_eff >= 0.70 ? "Directed" :
                (p_eff >= 0.30 ? "Search" : "Resident"))
        else
            (p_eff >= 0.70 ? "Directed" : (p_eff >= 0.30 ? "Search" : "Resident"))
        end
        push!(table_rows, string(
            "<tr>",
            "<td>", p.tagid, "</td>",
            "<td>", round(p.total_dist_km, digits=1), "</td>",
            "<td>", round(p.displacement_km, digits=1), "</td>",
            "<td>", round(p_eff, digits=3), "</td>",
            "<td>", round(p.tortuosity, digits=2), "</td>",
            "<td>", round(p.mean_hsi, digits=3), "</td>",
            "<td>", round(Int, p.duration_days), "</td>",
            "<td><span style=\"color:#38bdf8;\">", p_bout, "</span></td>",
            "<td><span style=\"color: $(p.color)\">●</span></td>",
            "</tr>"
        ))
    end

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>$species — Movement Summary Diagnostics</title>
  <link href="https://fonts.googleapis.com/css2?\\
family=Outfit:wght@300;400;600;700&\\
family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1329; --panel: rgba(15,23,42,0.92);
      --text: #f8fafc; --muted: #94a3b8;
      --border: rgba(255,255,255,0.10);
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: var(--bg); color: var(--text);
      padding: 30px 40px;
    }
    h1 { font-size: 1.6rem; font-weight: 700;
         margin-bottom: 6px; }
    .subtitle { color: var(--muted); font-size: 0.9rem;
                margin-bottom: 28px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 30px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px; padding: 20px;
    }
    .card h2 { font-size: 1.05rem; font-weight: 600;
               margin-bottom: 12px; }
    .stat-row { display: flex; gap: 24px;
                margin-bottom: 22px; }
    .stat-box {
      background: rgba(56,189,248,0.08);
      border: 1px solid rgba(56,189,248,0.2);
      border-radius: 8px; padding: 12px 16px; flex: 1;
    }
    .stat-label { font-size: 0.75rem; color: var(--muted);
                  text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-val { font-size: 1.3rem; font-weight: 700;
                color: var(--accent);
                font-family: 'JetBrains Mono', monospace; }
    table {
      width: 100%; border-collapse: collapse;
      font-size: 0.85rem;
    }
    th { text-align: left; color: var(--muted);
         border-bottom: 1px solid var(--border);
         padding: 8px 6px; font-weight: 500; }
    td { padding: 6px; border-bottom: 1px solid
         rgba(255,255,255,0.04); }
    tr:hover td { background: rgba(56,189,248,0.05); }
  </style>
</head>
<body>
<h1>$species — Movement Summary Diagnostics</h1>
<p class="subtitle">$n_paths reconstructed trajectories</p>

<div class="stat-row">
  <div class="stat-box">
    <div class="stat-label">Mean Distance</div>
    <div class="stat-val">$(round(mean_dist, digits=1)) km</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Mean Efficiency</div>
    <div class="stat-val">$(round(mean_eff, digits=3))</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Directional Bias (κ)</div>
    <div class="stat-val">$(round(kappa_val, digits=2))</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Paths</div>
    <div class="stat-val">$n_paths</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Total Distance Distribution (km)</h2>
    $dist_svg
  </div>
  <div class="card">
    <h2>Velocity Distribution (km / time step)</h2>
    $vel_svg
  </div>
  <div class="card">
    <h2>Directional Wind-Rose (Bearing °N)</h2>
    $rose_svg
  </div>
  <div class="card">
    <h2>Bearing Rose</h2>
    <p style="color: var(--muted); font-size: 0.85rem;">
      Each petal represents the fraction of paths whose
      release→recapture bearing falls in that compass sector.
      North = 0°, East = 90°, clockwise.
    </p>
  </div>
</div>

<div class="card">
  <h2>Per-Path Summary Table</h2>
  <div style="max-height: 500px; overflow-y: auto;">
    <table>
      <thead>
        <tr>
          <th>Tag ID</th><th>Dist (km)</th>
          <th>Displ (km)</th><th>Efficiency</th>
          <th>Tortuosity</th><th>Mean HSI</th>
          <th>Duration (steps)</th><th>Bout</th><th>Group</th>
        </tr>
      </thead>
      <tbody>
        $(join(table_rows, "\n        "))
      </tbody>
    </table>
  </div>
</div>
</body>
</html>"""

    open(filepath, "w") do f
        write(f, html)
    end
    return String(filepath)
end

"""
    export_movement_summary_dashboard(
        filepath, summary_res, path_results, loaded, mov_stats; species = "generic"
    )

Overload accepting pipeline result structures.
"""
function export_movement_summary_dashboard(
    filepath::AbstractString,
    summary_res::NamedTuple,
    path_results::NamedTuple,
    loaded::NamedTuple,
    mov_stats::NamedTuple;
    species::AbstractString = "generic"
)::String
    paths = path_results.paths
    cents_ll = path_results.cents_lonlat !== nothing ?
               path_results.cents_lonlat :
               (hasproperty(loaded.mesh, :centroids_lonlat) ?
                loaded.mesh.centroids_lonlat : loaded.mesh.centroids)
    n_units = length(cents_ll)

    paths_rich = NamedTuple[]
    dists = Float64[]
    vels = Float64[]
    bearings = mov_stats.bearings_deg

    for (tid, node_vec) in paths
        length(node_vec) < 2 && continue
        coords = Tuple{Float64, Float64}[
            (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
            for u in node_vec if 1 <= u <= n_units
        ]
        length(coords) < 2 && continue
        tot_d = 0.0
        for h in 2:length(coords)
            tot_d += haversine_distance(coords[h-1][1], coords[h-1][2],
                                        coords[h][1], coords[h][2]) / 1000.0
        end
        disp = haversine_distance(coords[1][1], coords[1][2],
                                  coords[end][1], coords[end][2]) / 1000.0
        push!(dists, tot_d)
        push!(vels, tot_d / max(1, length(coords) - 1))
        push!(paths_rich, (
            tagid           = string(tid),
            path            = node_vec,
            coords          = coords,
            total_dist_km   = tot_d,
            displacement_km = disp,
            tortuosity      = disp > 0.01 ? tot_d / disp : 1.0,
            mean_hsi        = 0.5,
            duration_days   = Float64(length(coords) - 1),
            color           = "#38bdf8",
            group           = 1,
        ))
    end

    return export_movement_summary_dashboard(
        filepath, species, dists, vels, bearings, paths_rich, mov_stats
    )
end

# =============================================================================
# Adaptive Multiresolution Hexagonal Mesh & Routing
# =============================================================================

"""
    construct_adaptive_multiresolution_domain(
        lon_vec::AbstractVector{<:Real},
        lat_vec::AbstractVector{<:Real};
        coarse_radius_km::Real = 25.0,
        fine_radius_km::Real = 8.0,
        refine_zones = nothing,
        land_polygons = nothing,
        refine_buffer_km::Real = 20.0,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest
    )::NamedTuple

Constructs a hierarchical dual-resolution hexagonal spatial mesh. Coarse
hexagons of radius `coarse_radius_km` cover regional/offshore waters, while
high-priority zones (coastal zones, steep bathymetric transitions, or telemetry
focal areas) are adaptively refined into sub-hexagons of radius `fine_radius_km`.

# Process Description
1. Bounding coordinates of `(lon_vec, lat_vec)` are projected to planar
   coordinates using `lonlat_to_xy_km`.
2. A base regular hexagonal lattice with spacing dx = sqrt(3)*rc, dy = 1.5*rc
   is initialized.
3. Coarse cells whose centroids lie within `refine_buffer_km` of land barriers,
   shallow bathymetry, or user-provided `refine_zones` are flagged for
   refinement.
4. Each flagged cell is subdivided into sub-hexagons of radius `fine_radius_km`.
5. The cross-resolution spatial adjacency graph W connects:
   - Fine-fine neighbor pairs within sqrt(3) * rf * 1.08
   - Coarse-coarse neighbor pairs within sqrt(3) * rc * 1.08
   - Coarse-fine boundary interface pairs within (rc + rf) * 0.95

# Arguments
- `lon_vec`, `lat_vec`: Spatial coordinate extents.
- `coarse_radius_km`: Circumradius for offshore/regional cells (km).
- `fine_radius_km`: Circumradius for refined/coastal cells (km).
- `refine_zones`: Optional collection of focus coordinates (lon, lat).
- `land_polygons`: Optional terrestrial barrier polygons.
- `refine_buffer_km`: Proximity distance triggering refinement (km).
- `depth`: Optional bathymetry depth values.
- `depth_threshold`: Depth cutoff for land masking.
- `crs`, `datum`: Geodetic projection parameters.

# Returns
`NamedTuple` with fields:
- `centroids_km`: Projected coordinates (km) of all cells.
- `centroids_lonlat`: Geographic coordinates (lon, lat) of all cells.
- `polygons_km`, `polygons_lonlat`: 6-vertex boundary outlines.
- `W`: Cross-resolution sparse spatial adjacency matrix.
- `areas_km2`: Area of each unit in square kilometers.
- `is_fine`: Boolean indicator vector identifying fine-scale units.
- `radius_km`: Cell radius per unit.
- `center_lon`, `center_lat`: Projection center coordinates.
- `n_units`: Total spatial units S.
- `land_mask`: Boolean land mask vector.
"""
function construct_adaptive_multiresolution_domain(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    coarse_radius_km::Real = 25.0,
    fine_radius_km::Real = 8.0,
    refine_zones = nothing,
    land_polygons = nothing,
    refine_buffer_km::Real = 20.0,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest
)::NamedTuple
    rc = Float64(coarse_radius_km)
    rf = min(Float64(fine_radius_km), rc * 0.75)

    # 1. Base coarse domain
    base_domain = construct_full_movement_domain(
        lon_vec, lat_vec;
        radius_km = rc,
        land_polygons = land_polygons,
        depth = depth,
        depth_threshold = depth_threshold,
        crs = crs,
        datum = datum
    )

    c_km = base_domain.centroids_km
    c_ll = base_domain.centroids_lonlat
    S_coarse = length(c_km)
    l_mask_coarse = base_domain.land_mask

    # 2. Identify coarse units requiring refinement
    refine_mask = falses(S_coarse)
    buf_sq = refine_buffer_km^2

    for i in 1:S_coarse
        if l_mask_coarse[i]
            refine_mask[i] = true
            continue
        end
        for j in 1:S_coarse
            if l_mask_coarse[j]
                dx = c_km[i][1] - c_km[j][1]
                dy = c_km[i][2] - c_km[j][2]
                if (dx^2 + dy^2) <= buf_sq
                    refine_mask[i] = true
                    break
                end
            end
        end
    end

    if refine_zones !== nothing
        for pt in refine_zones
            px, py = pt[1], pt[2]
            for i in 1:S_coarse
                refine_mask[i] && continue
                dx = c_ll[i][1] - px
                dy = c_ll[i][2] - py
                if (dx^2 + dy^2) <= (refine_buffer_km / 111.0)^2
                    refine_mask[i] = true
                end
            end
        end
    end

    # 3. Assemble multiresolution cells
    final_cents_km = Tuple{Float64, Float64}[]
    final_is_fine = Bool[]
    final_radius = Float64[]

    for i in 1:S_coarse
        if !refine_mask[i]
            push!(final_cents_km, c_km[i])
            push!(final_is_fine, false)
            push!(final_radius, rc)
        end
    end

    dx_f = sqrt(3.0) * rf
    dy_f = 1.5 * rf
    hex_bound = rc * 0.95

    for i in 1:S_coarse
        !refine_mask[i] && continue
        cx_c, cy_c = c_km[i]

        n_sub_x = ceil(Int, rc / dx_f)
        n_sub_y = ceil(Int, rc / dy_f)

        for ry in -n_sub_y:n_sub_y
            yk = cy_c + ry * dy_f
            xoff = isodd(abs(ry)) ? (dx_f / 2.0) : 0.0
            for rx in -n_sub_x:n_sub_x
                xk = cx_c + rx * dx_f + xoff
                dist_sq = (xk - cx_c)^2 + (yk - cy_c)^2
                if dist_sq <= hex_bound^2
                    too_close = false
                    for ex in final_cents_km
                        if ((xk - ex[1])^2 + (yk - ex[2])^2) < (0.6 * dx_f)^2
                            too_close = true
                            break
                        end
                    end
                    if !too_close
                        push!(final_cents_km, (xk, yk))
                        push!(final_is_fine, true)
                        push!(final_radius, rf)
                    end
                end
            end
        end
    end

    S = length(final_cents_km)
    c_lon = base_domain.center_lon
    c_lat = base_domain.center_lat

    final_cents_ll = [
        xy_km_to_lonlat(
            c[1], c[2];
            center_lon = c_lon, center_lat = c_lat, crs = crs, datum = datum
        )
        for c in final_cents_km
    ]

    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)
    final_polys_km = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    final_polys_ll = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    final_areas = zeros(Float64, S)

    for i in 1:S
        cx, cy = final_cents_km[i]
        rad = final_radius[i]
        final_areas[i] = (3.0 * sqrt(3.0) / 2.0) * (rad^2)
        v_km = [(cx + rad * cos(a), cy + rad * sin(a)) for a in hex_angles]
        push!(v_km, v_km[1])
        final_polys_km[i] = v_km
        final_polys_ll[i] = [
            xy_km_to_lonlat(
                p[1], p[2];
                center_lon = c_lon, center_lat = c_lat, crs = crs, datum = datum
            )
            for p in v_km
        ]
    end

    # 4. Cross-resolution spatial adjacency graph W
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = final_cents_km[i][1]
        c_mat[2, i] = final_cents_km[i][2]
    end
    kdtree = KDTree(c_mat)

    rows_w = Int[]
    cols_w = Int[]

    for i in 1:S
        r_i = final_radius[i]
        is_f_i = final_is_fine[i]
        max_search = sqrt(3.0) * max(r_i, rc) * 1.15
        nbrs = inrange(kdtree, [c_mat[1, i], c_mat[2, i]], max_search)

        for j in nbrs
            j <= i && continue
            r_j = final_radius[j]
            is_f_j = final_is_fine[j]
            dx = c_mat[1, i] - c_mat[1, j]
            dy = c_mat[2, i] - c_mat[2, j]
            dist = sqrt(dx^2 + dy^2)

            thresh = if is_f_i && is_f_j
                sqrt(3.0) * rf * 1.08
            elseif !is_f_i && !is_f_j
                sqrt(3.0) * rc * 1.08
            else
                ((r_i + r_j) * 0.95)
            end

            if dist <= thresh
                push!(rows_w, i)
                push!(cols_w, j)
                push!(rows_w, j)
                push!(cols_w, i)
            end
        end
    end

    W = sparse(rows_w, cols_w, fill(1.0, length(rows_w)), S, S)

    # 5. Land mask for multiresolution units
    land_mask = identify_land_units(
        final_cents_ll;
        polygons = final_polys_ll,
        land_polygons = land_polygons,
        depth = depth,
        depth_threshold = depth_threshold
    )

    return (
        centroids_km     = final_cents_km,
        centroids_lonlat = final_cents_ll,
        polygons_km     = final_polys_km,
        polygons_lonlat = final_polys_ll,
        W                = W,
        areas_km2        = final_areas,
        is_fine          = final_is_fine,
        radius_km        = final_radius,
        center_lon       = c_lon,
        center_lat       = c_lat,
        n_units          = S,
        land_mask        = land_mask
    )
end

"""
    astar_multiresolution_path(
        mesh::NamedTuple,
        release::Int,
        recapture::Int;
        hsi = nothing,
        hsi_weight::Real = 0.5,
        land_mask = nothing,
        max_steps::Int = 10000
    )::Vector{Int}

Calculates the shortest least-cost path between `release` and `recapture` units
across a multiresolution hexagonal mesh. Edge traversal costs are proportional
to physical inter-cell separation delta_x_ij = ||c_i - c_j||, ensuring exact
admissible heuristic search across mixed cell scales.

# Process Description
Edge cost: c(i, j) = delta_x_ij * (1.0 + hsi_weight * (1.0 - HSI_j))
Heuristic: h(u) = ||c_u - c_rec|| <= c(u, v) + h(v) (strictly admissible).

# Arguments
- `mesh`: Multiresolution mesh NamedTuple from
  `construct_adaptive_multiresolution_domain`.
- `release`: Starting node index (1 <= release <= S).
- `recapture`: Destination node index (1 <= recapture <= S).
- `hsi`: Optional habitat suitability vector of length S.
- `hsi_weight`: Relative weight of habitat resistance in edge traversal.
- `land_mask`: Optional boolean vector denoting impassable barrier units.
- `max_steps`: Upper iteration limit.

# Returns
`Vector{Int}`: Sequence of visited spatial node indices.
"""
function astar_multiresolution_path(
    mesh::NamedTuple,
    release::Int,
    recapture::Int;
    hsi = nothing,
    hsi_weight::Real = 0.5,
    land_mask = nothing,
    max_steps::Int = 10000
)::Vector{Int}
    S = mesh.n_units
    (1 <= release <= S && 1 <= recapture <= S) ||
        throw(ArgumentError(
            "Release ($release) or recapture ($recapture) out of range 1:$S"
        ))

    if release == recapture
        return Int[release]
    end

    W = mesh.W
    cents = mesh.centroids_km
    l_mask = land_mask !== nothing ? land_mask : (
        hasproperty(mesh, :land_mask) ? mesh.land_mask : falses(S)
    )

    hsi_vec = if hsi !== nothing
        Float64.(hsi)
    else
        ones(Float64, S)
    end

    c_rec = cents[recapture]

    heuristic = (u::Int) -> begin
        dx = cents[u][1] - c_rec[1]
        dy = cents[u][2] - c_rec[2]
        sqrt(dx^2 + dy^2)
    end

    g_score = fill(Inf, S)
    f_score = fill(Inf, S)
    came_from = zeros(Int, S)
    closed_set = falses(S)

    g_score[release] = 0.0
    f_score[release] = heuristic(release)

    pq = Tuple{Float64, Int}[(f_score[release], release)]
    steps = 0

    while !isempty(pq) && steps < max_steps
        steps += 1
        sort!(pq, by = first)
        curr_f, curr_u = popfirst!(pq)

        if curr_u == recapture
            path = Int[recapture]
            p = came_from[recapture]
            while p != 0
                pushfirst!(path, p)
                p = came_from[p]
            end
            return path
        end

        closed_set[curr_u] = true

        col_start = W.colptr[curr_u]
        col_end = W.colptr[curr_u + 1] - 1
        for ptr in col_start:col_end
            nbr = W.rowval[ptr]
            (nbr == curr_u || closed_set[nbr] || l_mask[nbr]) && continue

            dx = cents[curr_u][1] - cents[nbr][1]
            dy = cents[curr_u][2] - cents[nbr][2]
            edge_dist = sqrt(dx^2 + dy^2)
            cost_factor = 1.0 + Float64(hsi_weight) * (
                1.0 - clamp(hsi_vec[nbr], 0.0, 1.0)
            )
            tentative_g = g_score[curr_u] + edge_dist * cost_factor

            if tentative_g < g_score[nbr]
                came_from[nbr] = curr_u
                g_score[nbr] = tentative_g
                f_score[nbr] = tentative_g + heuristic(nbr)

                q_idx = findfirst(item -> item[2] == nbr, pq)
                if q_idx === nothing
                    push!(pq, (f_score[nbr], nbr))
                else
                    pq[q_idx] = (f_score[nbr], nbr)
                end
            end
        end
    end

    return Int[release, recapture]
end

# =============================================================================
# Time-Varying Dynamic Environmental Transition Kernels
# =============================================================================

"""
    construct_dynamic_transition_kernels(
        W::SparseMatrixCSC,
        hsi_temporal;
        gamma::Union{Real, AbstractVector{<:Real}} = 1.0,
        residence::Union{Real, AbstractVector{<:Real}} = 0.2,
        advection::Union{Real, AbstractVector{<:Real}} = 0.5,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        num_steps::Union{Nothing, Integer} = nothing
    )::Vector{Matrix{Float64}}

Constructs a sequence of time-varying row-stochastic transition probability
matrices [P^{(1)}, ..., P^{(T)}] driven by temporally evolving habitat
suitability index (HSI) fields (e.g. seasonal temperature shifts or currents).

# Process Description
At each time step t in {1, ..., T}:
A^{(t)}_{ij} is proportional to W_{ij} * exp(gamma_t * (h_{t,j} - h_{t,i}))
P^{(t)}_{ij} = (1 - rho_t) * [alpha_t * A^{(t)}_{ij} +
    (1 - alpha_t) * T^{diff}_{ij}] + rho_t * delta_{ij}

# Arguments
- `W`: Sparse spatial adjacency matrix (S x S).
- `hsi_temporal`: Habitat suitability field over time:
  - `AbstractMatrix{<:Real}`: S x T matrix where column t is hsi at step t.
  - `AbstractVector{<:AbstractVector{<:Real}}`: Vector of T HSI vectors.
  - `Function`: Callable `(t::Int) -> Vector{Float64}`.
- `gamma`: Advection sensitivity parameter.
- `residence`: Residence probability rho in [0, 1).
- `advection`: Advection weight alpha in [0, 1].
- `land_mask`: Optional boolean vector denoting terrestrial units.
- `num_steps`: Number of steps when `hsi_temporal` is a function.

# Returns
`Vector{Matrix{Float64}}`: Vector of T dense S x S transition matrices.
"""
function construct_dynamic_transition_kernels(
    W::SparseMatrixCSC,
    hsi_temporal;
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0,
    residence::Union{Real, AbstractVector{<:Real}} = 0.2,
    advection::Union{Real, AbstractVector{<:Real}} = 0.5,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    num_steps::Union{Nothing, Integer} = nothing
)::Vector{Matrix{Float64}}
    S = size(W, 1)

    hsi_list = if hsi_temporal isa AbstractMatrix
        T_steps = size(hsi_temporal, 2)
        if size(hsi_temporal, 1) != S && size(hsi_temporal, 2) == S
            [Float64.(hsi_temporal[t, :]) for t in 1:size(hsi_temporal, 1)]
        else
            [Float64.(hsi_temporal[:, t]) for t in 1:T_steps]
        end
    elseif hsi_temporal isa AbstractVector{<:AbstractVector}
        [Float64.(v) for v in hsi_temporal]
    elseif isa(hsi_temporal, Function)
        T_steps = num_steps !== nothing ? Int(num_steps) : 12
        [Float64.(hsi_temporal(t)) for t in 1:T_steps]
    else
        throw(ArgumentError(
            "hsi_temporal must be an S x T Matrix, a Vector of Vectors, or a Function."
        ))
    end

    T = length(hsi_list)
    T > 0 || throw(ArgumentError("hsi_temporal must have at least one time step."))

    P_kernels = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        g_t = gamma isa AbstractVector ? gamma[min(t, length(gamma))] : gamma
        r_t = residence isa AbstractVector ? (
            residence[min(t, length(residence))]
        ) : residence
        a_t = advection isa AbstractVector ? (
            advection[min(t, length(advection))]
        ) : advection

        P_kernels[t] = construct_stochastic_transition_kernel(
            W, hsi_list[t];
            gamma = g_t,
            residence = r_t,
            advection = a_t,
            land_mask = land_mask
        )
    end

    return P_kernels
end

"""
    predict_dynamic_path(
        P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
        release::Int,
        recapture::Int;
        centroids = nothing,
        method::Symbol = :probabilistic,
        land_mask = nothing
    )::Vector{Int}

Reconstructs an individual animal movement trajectory through time-varying
transition kernels [P^{(1)}, ..., P^{(K)}].

# Process Description
At step tau in {1, ..., K}:
Next node is selected proportional to P^{(tau)}_{ij} * B_{tau}(j, u_rec),
where B_tau = prod_{t=tau+1}^K P^{(t)} represents target reachability.

# Arguments
- `P_kernels`: Vector of K transition matrices across consecutive time steps.
- `release`: Release node index (1 <= release <= S).
- `recapture`: Recapture node index (1 <= recapture <= S).
- `centroids`: Optional spatial node coordinates for reachability fallback.
- `method`: Simulation mode (`:probabilistic` or `:deterministic`).
- `land_mask`: Optional land mask.

# Returns
`Vector{Int}`: Predicted path sequence of length K + 1.
"""
function predict_dynamic_path(
    P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int;
    centroids = nothing,
    method::Symbol = :probabilistic,
    land_mask = nothing
)::Vector{Int}
    K = length(P_kernels)
    K >= 1 || throw(ArgumentError("P_kernels must contain >= 1 matrix."))
    S = size(P_kernels[1], 1)

    (1 <= release <= S && 1 <= recapture <= S) ||
        throw(ArgumentError(
            "Release ($release) or recapture ($recapture) out of range 1:$S"
        ))

    if release == recapture || K == 0
        return Int[release]
    end

    B_mats = Vector{Matrix{Float64}}(undef, K)
    B_mats[K] = Matrix{Float64}(I, S, S)
    for tau in (K - 1):-1:1
        B_mats[tau] = B_mats[tau + 1] * Matrix{Float64}(P_kernels[tau + 1])
    end

    path = Vector{Int}(undef, K + 1)
    path[1] = release
    curr = release

    for tau in 1:K
        P_curr = P_kernels[tau]
        B_next = B_mats[tau]

        weights = zeros(Float64, S)
        for j in 1:S
            weights[j] = Float64(P_curr[curr, j]) * Float64(B_next[j, recapture])
        end

        tot_w = sum(weights)
        if tot_w > 1e-14
            weights ./= tot_w
            if method == :deterministic
                curr = argmax(weights)
            else
                r = rand()
                cum_w = 0.0
                next_node = recapture
                for j in 1:S
                    cum_w += weights[j]
                    if r <= cum_w
                        next_node = j
                        break
                    end
                end
                curr = next_node
            end
        else
            if centroids !== nothing && length(centroids) == S
                best_nbr = curr
                best_d = Inf
                for j in 1:S
                    if Float64(P_curr[curr, j]) > 0.0
                        dx = centroids[j][1] - centroids[recapture][1]
                        dy = centroids[j][2] - centroids[recapture][2]
                        d = sqrt(dx^2 + dy^2)
                        if d < best_d
                            best_d = d
                            best_nbr = j
                        end
                    end
                end
                curr = best_nbr
            else
                curr = recapture
            end
        end
        path[tau + 1] = curr
    end

    path[end] = recapture
    return path
end

"""
    predict_dynamic_corridor(
        P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
        release::Int,
        recapture::Int;
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
    )::Matrix{Float64}

Computes the dynamic Markov bridge probability matrix across time-varying
transition kernels [P^{(1)}, ..., P^{(K)}].

# Process Description
Forward propagation: f_0 = e_{release}, f_tau = f_{tau-1} * P^{(tau)}
Backward propagation: b_K = e_{recapture}, b_tau = P^{(tau+1)} * b_{tau+1}
Visitation intensity:
C_j = (1 / (K + 1)) * sum_tau (f_tau(j) * b_tau(j)) / (f_tau . b_tau)

# Returns
`Matrix{Float64}`: Dense S x S normalized corridor intensity matrix.
"""
function predict_dynamic_corridor(
    P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    K = length(P_kernels)
    K >= 1 || throw(ArgumentError("P_kernels must contain >= 1 matrix."))
    S = size(P_kernels[1], 1)

    (1 <= release <= S && 1 <= recapture <= S) ||
        throw(ArgumentError(
            "Release ($release) or recapture ($recapture) out of range 1:$S"
        ))

    f_vecs = [zeros(Float64, S) for _ in 0:K]
    f_vecs[1][release] = 1.0
    for tau in 1:K
        P_tau = P_kernels[tau]
        for j in 1:S
            s = 0.0
            for i in 1:S
                s += f_vecs[tau][i] * Float64(P_tau[i, j])
            end
            f_vecs[tau + 1][j] = s
        end
    end

    b_vecs = [zeros(Float64, S) for _ in 0:K]
    b_vecs[K + 1][recapture] = 1.0
    for tau in (K - 1):-1:0
        P_next = P_kernels[tau + 1]
        for i in 1:S
            s = 0.0
            for j in 1:S
                s += Float64(P_next[i, j]) * b_vecs[tau + 2][j]
            end
            b_vecs[tau + 1][i] = s
        end
    end

    node_intensity = zeros(Float64, S)
    for tau in 0:K
        norm_factor = sum(f_vecs[tau + 1] .* b_vecs[tau + 1])
        if norm_factor > 1e-15
            node_intensity .+= (
                f_vecs[tau + 1] .* b_vecs[tau + 1]
            ) ./ norm_factor
        else
            node_intensity .+= (
                f_vecs[tau + 1] .+ b_vecs[tau + 1]
            ) .* 0.5
        end
    end
    node_intensity ./= (K + 1)

    if land_mask !== nothing
        for i in 1:S
            if land_mask[i]
                node_intensity[i] = 0.0
            end
        end
    end

    corridor_mat = node_intensity * node_intensity'
    max_val = maximum(corridor_mat)
    if max_val > 0.0
        corridor_mat ./= max_val
    end

    return corridor_mat
end

# =============================================================================
# Multi-Segment Hidden Markov Model (HMM) Viterbi Smoothing
# =============================================================================

"""
    viterbi_hmm_path_smoothing(
        obs_times::AbstractVector{<:Integer},
        obs_locations::AbstractVector,
        P_kernels;
        mesh = nothing,
        sigma_obs_km::Real = 15.0,
        land_mask = nothing
    )::NamedTuple

Performs global trajectory decoding for animals with multi-stage recaptures,
acoustic detections, or irregular time intervals using a Hidden Markov Model
(HMM) with exact log-space Viterbi dynamic programming.

# Process Description
- Latent states: z_t in {1, ..., S} for t in 1:T.
- Transitions: P(z_t = j | z_{t-1} = i) = P^{(t)}_{ij}.
- Emissions: Gaussian spatial likelihood for observed coordinates/units;
  log b_j(missing) = 0.0 for unobserved intermediate time steps.
- Recursion: delta_t(j) = max_i [delta_{t-1}(i) + log P^{(t)}_{ij}] + log b_j(y_t).

# Arguments
- `obs_times`: Ascending vector of integer time steps [t_1, ..., t_M].
- `obs_locations`: Locations at each observation step (unit indices or (lon, lat)).
- `P_kernels`: Transition kernel matrix (S x S) or Vector of matrices.
- `mesh`: Spatial mesh NamedTuple containing `:centroids_lonlat` or `:centroids_km`.
- `sigma_obs_km`: Standard deviation of spatial observation error (km).
- `land_mask`: Optional boolean vector denoting terrestrial units.

# Returns
`NamedTuple` with fields:
- `path`: Decoded optimal spatial unit sequence z*_1, ..., z*_T.
- `log_likelihood`: Maximum path log-likelihood.
- `time_steps`: Vector 1:T.
- `obs_mask`: Boolean vector of length T indicating observed time steps.
"""
function viterbi_hmm_path_smoothing(
    obs_times::AbstractVector{<:Integer},
    obs_locations::AbstractVector,
    P_kernels;
    mesh = nothing,
    sigma_obs_km::Real = 15.0,
    land_mask = nothing
)::NamedTuple
    length(obs_times) == length(obs_locations) ||
        throw(ArgumentError("obs_times and obs_locations must have identical length."))
    issorted(obs_times) ||
        throw(ArgumentError("obs_times must be sorted in ascending order."))

    T = obs_times[end]
    T >= 1 || throw(ArgumentError("Maximum observation time must be >= 1."))

    S = if P_kernels isa AbstractMatrix
        size(P_kernels, 1)
    elseif P_kernels isa AbstractVector{<:AbstractMatrix}
        size(P_kernels[1], 1)
    else
        throw(ArgumentError("P_kernels must be a Matrix or Vector of Matrices."))
    end

    cents_ll = if mesh !== nothing
        if hasproperty(mesh, :centroids_lonlat)
            mesh.centroids_lonlat
        elseif hasproperty(mesh, :centroids)
            mesh.centroids
        else
            nothing
        end
    else
        nothing
    end

    log_B = zeros(Float64, S, T)
    obs_mask = falses(T)
    var_obs = Float64(sigma_obs_km)^2
    log_norm = -0.5 * log(2.0 * π * var_obs)

    for (idx, t) in enumerate(obs_times)
        obs_mask[t] = true
        loc = obs_locations[idx]

        if loc isa Integer
            target_u = Int(loc)
            for j in 1:S
                if j == target_u
                    log_B[j, t] = 0.0
                elseif cents_ll !== nothing
                    d_km = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        cents_ll[target_u][1], cents_ll[target_u][2]
                    ) / 1000.0
                    log_B[j, t] = -0.5 * (d_km^2) / var_obs + log_norm
                else
                    log_B[j, t] = -50.0
                end
            end
        elseif loc isa Tuple || loc isa AbstractVector
            lon_obs, lat_obs = Float64(loc[1]), Float64(loc[2])
            for j in 1:S
                if cents_ll !== nothing
                    d_km = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        lon_obs, lat_obs
                    ) / 1000.0
                    log_B[j, t] = -0.5 * (d_km^2) / var_obs + log_norm
                else
                    log_B[j, t] = 0.0
                end
            end
        end
    end

    if land_mask !== nothing
        for j in 1:S
            if land_mask[j]
                log_B[j, :] .= -1e9
            end
        end
    end

    log_P_fn = (step::Int) -> begin
        P_mat = if P_kernels isa AbstractMatrix
            P_kernels
        else
            P_kernels[min(step, length(P_kernels))]
        end
        log_mat = Matrix{Float64}(undef, S, S)
        for c in 1:S
            for r in 1:S
                val = Float64(P_mat[r, c])
                log_mat[r, c] = val > 1e-15 ? log(val) : -1e9
            end
        end
        return log_mat
    end

    delta = Matrix{Float64}(undef, S, T)
    psi = zeros(Int, S, T)

    init_prob = -log(S)
    for j in 1:S
        delta[j, 1] = init_prob + log_B[j, 1]
    end

    for t in 2:T
        log_P = log_P_fn(t - 1)
        for j in 1:S
            best_val = -Inf
            best_i = 1
            for i in 1:S
                cand = delta[i, t - 1] + log_P[i, j]
                if cand > best_val
                    best_val = cand
                    best_i = i
                end
            end
            delta[j, t] = best_val + log_B[j, t]
            psi[j, t] = best_i
        end
    end

    best_last = argmax(delta[:, T])
    best_ll = delta[best_last, T]

    path = Vector{Int}(undef, T)
    path[T] = best_last
    for t in (T - 1):-1:1
        path[t] = psi[path[t + 1], t + 1]
    end

    return (
        path           = path,
        log_likelihood = best_ll,
        time_steps     = collect(1:T),
        obs_mask       = obs_mask
    )
end

"""
    forward_backward_state_probabilities(
        obs_times::AbstractVector{<:Integer},
        obs_locations::AbstractVector,
        P_kernels;
        mesh = nothing,
        sigma_obs_km::Real = 15.0,
        land_mask = nothing
    )::Matrix{Float64}

Computes marginal posterior state probabilities gamma_t(i) = P(z_t = i | y_{1:T})
across all spatial mesh units and discrete time steps using the scaled
Baum-Welch forward-backward algorithm.

# Process Description
Forward: alpha_t(j) = (sum_i alpha_{t-1}(i) * P^{(t)}_{ij}) * b_j(y_t)
Backward: beta_t(i) = sum_j P^{(t+1)}_{ij} * b_j(y_{t+1}) * beta_{t+1}(j)
Posterior: gamma_t(j) = (alpha_t(j) * beta_t(j)) / sum_k (alpha_t(k) * beta_t(k))

# Returns
`Matrix{Float64}`: Dense S x T matrix where column t is the posterior
occupancy distribution across all spatial units at time step t.
"""
function forward_backward_state_probabilities(
    obs_times::AbstractVector{<:Integer},
    obs_locations::AbstractVector,
    P_kernels;
    mesh = nothing,
    sigma_obs_km::Real = 15.0,
    land_mask = nothing
)::Matrix{Float64}
    T = obs_times[end]
    S = P_kernels isa AbstractMatrix ? size(P_kernels, 1) : size(P_kernels[1], 1)

    cents_ll = if mesh !== nothing
        hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat : (
            hasproperty(mesh, :centroids) ? mesh.centroids : nothing
        )
    else
        nothing
    end

    B = ones(Float64, S, T)
    var_obs = Float64(sigma_obs_km)^2

    for (idx, t) in enumerate(obs_times)
        loc = obs_locations[idx]
        if loc isa Integer
            u_t = Int(loc)
            for j in 1:S
                if j == u_t
                    B[j, t] = 1.0
                elseif cents_ll !== nothing
                    d = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        cents_ll[u_t][1], cents_ll[u_t][2]
                    ) / 1000.0
                    B[j, t] = exp(-0.5 * (d^2) / var_obs)
                else
                    B[j, t] = 1e-12
                end
            end
        elseif loc isa Tuple || loc isa AbstractVector
            lon_obs, lat_obs = Float64(loc[1]), Float64(loc[2])
            for j in 1:S
                if cents_ll !== nothing
                    d = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        lon_obs, lat_obs
                    ) / 1000.0
                    B[j, t] = exp(-0.5 * (d^2) / var_obs)
                end
            end
        end
    end

    if land_mask !== nothing
        for j in 1:S
            if land_mask[j]
                B[j, :] .= 0.0
            end
        end
    end

    alpha = Matrix{Float64}(undef, S, T)
    c_scales = Vector{Float64}(undef, T)

    alpha[:, 1] .= (1.0 / S) .* B[:, 1]
    c_scales[1] = sum(alpha[:, 1])
    if c_scales[1] > 0.0
        alpha[:, 1] ./= c_scales[1]
    end

    for t in 2:T
        P_curr = P_kernels isa AbstractMatrix ? P_kernels : (
            P_kernels[min(t - 1, length(P_kernels))]
        )
        for j in 1:S
            s = 0.0
            for i in 1:S
                s += alpha[i, t - 1] * Float64(P_curr[i, j])
            end
            alpha[j, t] = s * B[j, t]
        end
        c_scales[t] = sum(alpha[:, t])
        if c_scales[t] > 0.0
            alpha[:, t] ./= c_scales[t]
        end
    end

    beta = Matrix{Float64}(undef, S, T)
    beta[:, T] .= 1.0

    for t in (T - 1):-1:1
        P_next = P_kernels isa AbstractMatrix ? P_kernels : (
            P_kernels[min(t, length(P_kernels))]
        )
        for i in 1:S
            s = 0.0
            for j in 1:S
                s += Float64(P_next[i, j]) * B[j, t + 1] * beta[j, t + 1]
            end
            beta[i, t] = s
        end
        if c_scales[t + 1] > 0.0
            beta[:, t] ./= c_scales[t + 1]
        end
    end

    gamma = Matrix{Float64}(undef, S, T)
    for t in 1:T
        gamma[:, t] = alpha[:, t] .* beta[:, t]
        tot = sum(gamma[:, t])
        if tot > 0.0
            gamma[:, t] ./= tot
        else
            gamma[:, t] .= 1.0 / S
        end
    end

    return gamma
end


"""
    generate_movement_data(; radius_km=8.0, time_interval=:monthly, crs=nothing,
                           datum=WGS84Latest, domain_km=60.0, n_tags=50, n_steps=3,
                           center_lon=-60.0, center_lat=46.0, seed=42) -> NamedTuple

Generates synthetic animal movement and mark-recapture telemetry datasets mapped over a
planar hexagonal spatial mesh. Simulates individual movement trajectories via discrete
Markov transitions across mesh units, assigns demographic attributes, and classifies
individuals into canonical biological groups matching `snowcrab_movement_data`:
- `"female"`: Mature females (`mat == "mature"`, `sex == "F"`)
- `"male"`: Mature males (`mat == "mature"`, `sex == "M"`)
- `"immature"`: Immature individuals (`mat == "immature"`)
- `"unknown"`: Unclassified or missing demographic observations

# Arguments
- `radius_km::Real = 8.0`: Hexagonal cell circumradius in kilometers.
- `time_interval::Symbol = :monthly`: Temporal step discretization interval
  (`:monthly`, `:weekly`, `:biweekly`, `:daily`, or `:raw`).
- `crs = nothing`: Coordinate reference system for spatial projections.
- `datum = WGS84Latest`: Geographic geodetic datum.
- `domain_km::Real = 60.0`: Spatial domain width and height in kilometers.
- `n_tags::Int = 50`: Total number of tagged individuals simulated.
- `n_steps::Int = 3`: Number of consecutive movement observation steps per individual.
- `center_lon::Real = -60.0`: Geographic center longitude of the simulated domain.
- `center_lat::Real = 46.0`: Geographic center latitude of the simulated domain.
- `seed::Int = 42`: Random seed for reproducible spatial and demographic simulation.

# Returns
A `NamedTuple` with fields:
- `tagging::DataFrame`: Raw synthetic telemetry event records with coordinates and units.
- `mesh::NamedTuple`: Planar hexagonal mesh containing centroids, polygons, and `W`.
- `W::SparseMatrixCSC`: Adjacency matrix of the spatial mesh.
- `hsi_vec::Vector{Float64}`: Domain Habitat Suitability Index (HSI) vector.
- `monthly_hsi::Matrix{Float64}`: Monthly dynamic HSI fields (empty if static).
- `month_lookup::Dict`: Mapping from `(year, month)` to monthly HSI column index.
- `years::Vector{Int}`: Observed survey years.
- `obs::DataFrame`: Extracted consecutive mark-recapture event pairs with biological
  group classifications (`:group` column with 1-based indices).
- `survey_df::DataFrame`: Synthetic spatial survey density observations.
- `group_lookup::Dict{String, Int}`: Dictionary mapping group names to integer IDs.
"""
function generate_movement_data(;
    radius_km     :: Real    = 8.0,
    time_interval :: Symbol  = :monthly,
    crs                      = nothing,
    datum                    = WGS84Latest,
    domain_km     :: Real    = 60.0,
    n_tags        :: Int     = 50,
    n_steps       :: Int     = 3,
    center_lon    :: Real    = -60.0,
    center_lat    :: Real    = 46.0,
    seed          :: Int     = 42
)::NamedTuple
    
    # Internal categorical draw — avoids importing Distributions in this file
    function _sample_categorical(p::AbstractVector{Float64}, rng::AbstractRNG)::Int
        u    = rand(rng)
        csum = 0.0
        for (i, pi) in enumerate(p)
            csum += pi
            csum >= u && return i
        end
        return length(p)
    end

    rng  = MersenneTwister(seed)
    half = Float64(domain_km) / 2.0

    n_grid = max(10, round(Int, domain_km / radius_km * 2))
    xs_g   = range(-half, half, length=n_grid)
    ys_g   = range(-half, half, length=n_grid)
    
    # Pre-allocate grid coordinate vectors
    n_pts = length(xs_g) * length(ys_g)
    grid_lon = Vector{Float64}(undef, n_pts)
    grid_lat = Vector{Float64}(undef, n_pts)
    
    idx = 1
    for y in ys_g, x in xs_g
        lon, lat = xy_km_to_lonlat(x, y; 
                        center_lon=center_lon, center_lat=center_lat, 
                        crs=crs, datum=datum)
        grid_lon[idx] = lon
        grid_lat[idx] = lat
        idx += 1
    end

    # Pass the CRS formatting down to the mesh generator
    mesh = build_hex_mesh_planar(grid_lon, grid_lat; 
               radius_km=radius_km, crs=crs, datum=datum)
    S = mesh.n_units

    # Construct the true kernel directly using row sums of the sparse matrix
    row_sums = sum(mesh.W, dims=2)
    kernel   = zeros(Float64, S, S)
    for i in 1:S
        rs = row_sums[i]
        if rs > 0
            @views kernel[i, :] .= mesh.W[i, :] ./ rs
        else
            kernel[i, i] = 1.0
        end
    end

    # Biological demographic simulation covering female, male, immature, and unknown
    # Guarantee the 4 canonical groups appear in simulated tagging dataset
    canonical_demographics = [
        ("F", "mature"),        # -> female
        ("M", "mature"),        # -> male
        ("M", "immature"),      # -> immature
        ("unknown", "unknown")  # -> unknown
    ]
    pool_demographics = [
        ("F", "mature"),
        ("M", "mature"),
        ("M", "immature"),
        ("F", "immature"),
        ("unknown", "mature"),
        ("unknown", "unknown")
    ]
    pool_weights = [0.35, 0.35, 0.12, 0.08, 0.05, 0.05]

    sexes = Vector{String}(undef, n_tags)
    mats  = Vector{String}(undef, n_tags)
    for i in 1:n_tags
        if i <= length(canonical_demographics)
            sx, mt = canonical_demographics[i]
        else
            w_idx = _sample_categorical(pool_weights, rng)
            sx, mt = pool_demographics[w_idx]
        end
        sexes[i] = sx
        mats[i]  = mt
    end

    t0_dt = Date(2020, 1, 1)
    
    total_records = n_tags * (n_steps + 1)
    records = Vector{NamedTuple{
        (:tagid, :lon, :lat, :tag, :timestamp, :time, :sex, :mat, :is_dead, :s_idx),
        Tuple{String, Float64, Float64, Int, DateTime, Float64, String, String, Bool, Int}
    }}(undef, total_records)
    
    row_idx = 1
    for i in 1:n_tags
        s_cur   = rand(rng, 1:S)
        tid_str = string(i)
        
        (lon_r, lat_r) = mesh.centroids_lonlat[s_cur]
        records[row_idx] = (
            tagid     = tid_str,
            lon       = lon_r,
            lat       = lat_r,
            tag       = 0,
            timestamp = DateTime(t0_dt),
            time      = _date_to_decimal_year(t0_dt),
            sex       = sexes[i],
            mat       = mats[i],
            is_dead   = false,
            s_idx     = s_cur
        )
        row_idx += 1

        for step in 1:n_steps
            p_row = kernel[s_cur, :]
            s_cur = _sample_categorical(p_row, rng)
            t_dt  = t0_dt + Month(step)
            
            (lon_r, lat_r) = mesh.centroids_lonlat[s_cur]
            records[row_idx] = (
                tagid     = tid_str,
                lon       = lon_r,
                lat       = lat_r,
                tag       = step,
                timestamp = DateTime(t_dt),
                time      = _date_to_decimal_year(t_dt),
                sex       = sexes[i],
                mat       = mats[i],
                is_dead   = false,
                s_idx     = s_cur
            )
            row_idx += 1
        end
    end
  
    tagging = DataFrame(records)
    true_kernel = kernel
    tagging = map_point_to_units(tagging, mesh.centroids_km,
                  mesh.center_lon, mesh.center_lat; crs=crs, datum=datum)
    
    # Generate realistic spatial bathymetry and habitat suitability gradient
    depth_vec = [150.0 + 60.0 * sin(mesh.centroids_km[s][1] / 40.0) + 
                 40.0 * cos(mesh.centroids_km[s][2] / 40.0) for s in 1:mesh.n_units]
    temp_vec  = [3.0 + 1.5 * cos(mesh.centroids_km[s][1] / 50.0) for s in 1:mesh.n_units]
    
    # HSI peaks in optimal thermal/depth window
    hsi_raw = [exp(-((depth_vec[s] - 170.0) / 45.0)^2 - ((temp_vec[s] - 2.5) / 1.5)^2) 
               for s in 1:mesh.n_units]
    hsi_min, hsi_max = extrema(hsi_raw)
    hsi_vec = (hsi_raw .- hsi_min) ./ max(1e-6, hsi_max - hsi_min) .* 0.8 .+ 0.1

    monthly_hsi  = Matrix{Float64}(undef, 0, 0)
    month_lookup = Dict{Tuple{Int, Int}, Int}()
    years_vec    = Int[]

    # Time-interval step conversion mapping
    dt_map = (monthly=1.0/12.0, weekly=1.0/52.0, biweekly=1.0/26.0, daily=1.0/365.25, raw=1.0)
    dt = hasproperty(dt_map, time_interval) ? getproperty(dt_map, time_interval) : 1.0 / 12.0

    has_sex = hasproperty(tagging, :sex)
    has_mat = hasproperty(tagging, :mat)

    # 1. Sort globally upfront
    sorted_df = sort(tagging, [:tagid, :time])
    n_rows = nrow(sorted_df)

    # Return empty DataFrame immediately if not enough rows to form a pair
    if n_rows < 2
        obs = DataFrame(tagid=String[], release=Int[], recapture=Int[], 
                        k=Int[], sex=String[], mat=String[], group=Int[])
        survey_df = DataFrame(s_idx=Int[], t_idx=Int[], density=Int[], depth=Float64[], temp=Float64[])
        default_group_lookup = Dict{String, Int}(
            "female"   => 1,
            "immature" => 2,
            "male"     => 3,
            "unknown"  => 4
        )
        return (
            tagging      = tagging,
            mesh         = mesh,
            W            = mesh.W,
            hsi_vec      = hsi_vec,
            monthly_hsi  = monthly_hsi,
            month_lookup = month_lookup,
            years        = years_vec,
            obs          = obs,
            survey_df    = survey_df,
            group_lookup = default_group_lookup
        )
    end

    # 2. Extract columns to local vectors for type stability
    tagids    = sorted_df.tagid
    times     = sorted_df.time
    s_idxs    = sorted_df.s_idx
    sexes_col = has_sex ? sorted_df.sex : nothing
    mats_col  = has_mat ? sorted_df.mat : nothing

    RecordType = NamedTuple{
        (:tagid, :release, :recapture, :k, :sex, :mat), 
        Tuple{String, Int, Int, Int, String, String}
    }
    
    pair_records = Vector{RecordType}(undef, 0)
    sizehint!(pair_records, n_rows)

    # 3. Single-pass flat loop for consecutive pairs
    for i in 2:n_rows
        if tagids[i] == tagids[i-1]
            Δt = times[i] - times[i-1]
            k  = max(1, round(Int, Δt / dt))
            
            s_str = has_sex ? string(sexes_col[i-1]) : "unknown"
            m_str = has_mat ? string(mats_col[i-1])  : "unknown"

            push!(pair_records, (
                tagid     = string(tagids[i-1]),
                release   = s_idxs[i-1],
                recapture = s_idxs[i],
                k         = k,
                sex       = s_str,
                mat       = m_str
            ))
        end
    end

    obs = DataFrame(pair_records)

    # 4. Assign 3-tier biological groupings matching snowcrab_movement_data()
    n_obs = nrow(obs)
    labels = Vector{String}(undef, n_obs)
    
    if n_obs > 0
        obs_sexes = obs[!, :sex]
        obs_mats  = obs[!, :mat]

        @inbounds for i in 1:n_obs
            sx = string(obs_sexes[i])
            mt = string(obs_mats[i])

            if mt == "immature" || mt == "imm"
                labels[i] = "immature"
            elseif (mt == "mature" || mt == "mat") && (sx == "M" || sx == "male")
                labels[i] = "male"
            elseif (mt == "mature" || mt == "mat") && (sx == "F" || sx == "female")
                labels[i] = "female"
            else
                labels[i] = "unknown"
            end
        end
    end
    
    unique_labels = sort!(unique(labels))
    group_lookup  = Dict{String, Int}(lbl => i for (i, lbl) in enumerate(unique_labels))
        
    group_ids = Vector{Int}(undef, n_obs)
    @inbounds for i in 1:n_obs
        group_ids[i] = group_lookup[labels[i]]
    end

    obs[!, :group] = group_ids

    # 5. Generate synthetic survey density observations for Option 3 joint modeling
    mu_density = exp.(1.5 .+ 2.0 .* hsi_vec .- 0.005 .* (depth_vec .- 170.0))
    density_counts = [rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + mu_density[s]))) 
                      for s in 1:mesh.n_units]
    survey_df = DataFrame(
        s_idx   = collect(1:mesh.n_units),
        t_idx   = ones(Int, mesh.n_units),
        density = density_counts,
        depth   = depth_vec,
        temp    = temp_vec
    )
  
    return (
        tagging      = tagging,
        mesh         = mesh,
        W            = mesh.W,
        hsi_vec      = hsi_vec,
        monthly_hsi  = monthly_hsi,
        month_lookup = month_lookup,
        years        = years_vec,
        obs          = obs,
        survey_df    = survey_df,
        depth_vec    = depth_vec,
        group_lookup = group_lookup
    )
end
