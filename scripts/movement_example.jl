
# Framework to fit the Eulerian telemetry model and then use those posterior
# parameters to run Individual-Based Model (IBM) simulations.
#
# Workflow:
#   Step 0  -- Load data and unpack domain variables
#   Step 1  -- Fit mark-recapture telemetry model via MCMC
#   Step 2  -- Build transition kernel P from posterior means
#   Step 3  -- IBM simulations (forward and conditional Markov bridge)
#   Step 4  -- Visualize IBM paths as interactive HTML dashboards
#   Step 5  -- Full pipeline (fit + reconstruct + dashboards)

using Turing
using MovementAnalysis

# ============================================================
# Step 0: Data Preparation
# ============================================================

params = movement_parameters_snowcrab()
loaded = load_movement_data(params)

# Unpack domain variables
W         = loaded.W           # SparseMatrixCSC mesh adjacency
hsi       = loaded.hsi_vec     # Habitat Suitability Index per node (length S)
land_mask = loaded.land_mask   # BitVector: true = impassable (land/out-of-depth)
obs_df    = loaded.obs_df      # Mark-recapture observations DataFrame
n_spatial = loaded.n_spatial   # Total number of spatial units S

# Extract telemetry observation vectors from obs_df
releases   = Int.(obs_df.release)
recaptures = Int.(obs_df.recapture)
ks         = Int.(obs_df.k)
groups     = Int.(obs_df.group)
G          = maximum(values(loaded.group_map))   # number of groups

println("\nObservations: $(length(releases)) events, $G group(s)")

# ============================================================
# Step 1: Fit the Telemetry Model using MCMC
# ============================================================
# pure_telemetry_turing_model estimates group-stratified advection (alpha),
# residence (rho), and HSI gradient response (gamma) via NUTS MCMC.
#
#   P_g = (1-rho_g)[(1-alpha_g) T_diff + alpha_g A_g(gamma)] + rho_g I
#
# where T_diff is the symmetric diffusion operator and
# A_g is the HSI-biased directed adjacency.

model = pure_telemetry_turing_model(
    releases, recaptures, ks, groups,
    W, hsi, land_mask,
    G
)

n_samples = get(params, :n_samples, 500)
chain = sample(model, NUTS(200, 0.65), n_samples)

# Posterior means for Group 1
# The model samples velocity (advection rate) and diffusion rate; alpha and
# rho are derived as:
#   alpha = velocity / (velocity + diffusion)    (advection fraction)
#   rho   = 1 / (1 + velocity + diffusion)       (residence probability)
velocity_est  = mean(chain[Symbol("velocity[1]")])
diffusion_est = mean(chain[Symbol("diffusion[1]")])
gamma_est     = mean(chain[Symbol("gamma[1]")])
tot_est       = velocity_est + diffusion_est + 1e-6
alpha_est     = clamp(velocity_est / tot_est, 0.0, 1.0)
rho_est       = clamp(1.0 / (1.0 + tot_est),  0.01, 0.95)

println("\nPosterior means (group 1):")
println("  velocity  (raw advection rate) : $(round(velocity_est;  digits=4))")
println("  diffusion (raw diffusion rate) : $(round(diffusion_est; digits=4))")
println("  gamma     (HSI grad.)          : $(round(gamma_est;     digits=4))")
println("  → alpha   (advection fraction) : $(round(alpha_est;     digits=4))")
println("  → rho     (residence prob.)    : $(round(rho_est;       digits=4))")

# ============================================================
# Step 2: Build the Stochastic Transition Kernel P
# ============================================================
# Row-stochastic S x S matrix. Each row gives the probability
# distribution over neighbours (and self) for one time step.

P = construct_stochastic_transition_kernel(
    W, hsi;
    gamma     = gamma_est,
    residence = rho_est,
    advection = alpha_est,
    land_mask = land_mask
)
println("\nTransition kernel P: $(size(P,1)) x $(size(P,2))")

# ============================================================
# Step 2b: Propagate full posterior uncertainty (optional)
# ============================================================
# Build one P per posterior draw to assess path uncertainty.

n_post_draws = min(50, length(chain))
P_draws = Vector{Matrix{Float64}}(undef, n_post_draws)
for i in 1:n_post_draws
    v_i = chain[Symbol("velocity[1]")][i]
    d_i = chain[Symbol("diffusion[1]")][i]
    g_i = chain[Symbol("gamma[1]")][i]
    tot_i = Float64(v_i) + Float64(d_i) + 1e-6
    a_i = clamp(Float64(v_i) / tot_i, 0.0, 1.0)
    r_i = clamp(1.0 / (1.0 + tot_i), 0.01, 0.95)
    P_draws[i] = construct_stochastic_transition_kernel(
        W, hsi;
        gamma     = Float64(g_i),
        residence = r_i,
        advection = a_i,
        land_mask = land_mask
    )
end
println("  Built $n_post_draws posterior-draw kernels.")

# ============================================================
# Step 3a: Forward IBM Simulation (unconstrained random walk)
# ============================================================
# Drop a virtual individual at start_node; sample transitions
# from P for `steps` discrete time steps.

function simulate_forward_ibm(
    P::AbstractMatrix{<:Real},
    start_node::Int,
    steps::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    path = Int[start_node]
    node = start_node
    for _ in 1:steps
        row = copy(vec(P[node, :]))
        land_mask !== nothing && (row[land_mask] .= 0.0)
        rs = sum(row)
        rs <= 0.0 && break
        row ./= rs
        node = rand(Categorical(row))
        push!(path, node)
    end
    return path
end

start_node = releases[1]
ibm_path   = simulate_forward_ibm(P, start_node, 20; land_mask = land_mask)
println("\nForward IBM: $(ibm_path[1]) -> $(ibm_path[end])  ($(length(ibm_path)) nodes)")

# ============================================================
# Step 3b: Conditional IBM Simulation (Markov bridge)
# ============================================================
# Given known release and recapture, sample stochastic A* paths.
# Each realization is one plausible individual trajectory that
# connects the two endpoints through the in-depth marine domain.

release_node   = releases[1]
recapture_node = recaptures[1]
time_steps     = ks[1]
n_realizations = 100

cents_planar = if hasproperty(loaded.mesh, :centroids_km)
    loaded.mesh.centroids_km
elseif hasproperty(loaded.mesh, :centroids)
    loaded.mesh.centroids
else
    nothing
end

println("\nConditional IBM: $release_node -> $recapture_node over $time_steps steps")
stochastic_result = astar_stochastic_least_cost_path(
    cents_planar, W,
    release_node, recapture_node;
    hsi_mean       = hsi,
    hsi_se         = fill(params.hsi_se, n_spatial),
    n_draws        = n_realizations,
    friction_power = 2.0,
    land_mask      = land_mask
)
println("  $(length(stochastic_result.all_paths)) IBM paths sampled")
println("  Mean distance : $(round(stochastic_result.mean_distance; digits=1)) km")
println("  95% CI        : $(round.(stochastic_result.ci_distance; digits=1)) km")

# ============================================================
# Step 4: Visualize IBM Paths
# ============================================================

cents_ll = hasproperty(loaded.mesh, :centroids_lonlat) ?
           loaded.mesh.centroids_lonlat :
           loaded.mesh.centroids
polys_ll = hasproperty(loaded.mesh, :polygons_lonlat) ?
           loaded.mesh.polygons_lonlat :
           loaded.mesh.polygons

au_mesh = (
    centroids        = cents_ll,
    centroids_lonlat = cents_ll,
    polygons         = polys_ll,
    polygons_lonlat  = polys_ll,
    W                = W,
    n_units          = n_spatial,
)

# 4a. Interactive corridor explorer (two-click for any node pair)
corr_html = joinpath(params.output_dir, "example_corridor.html")
corr_map  = leaflet_interactive_corridor_dashboard(
    P, au_mesh;
    hsi   = hsi,
    title = "Snow Crab Movement Corridor (Example)"
)
save_html(corr_map, corr_html)
println("\nCorridor dashboard : $corr_html")

# 4b. IBM realizations on tracks map
# Two types of paths are visualized:
#   - ibm_path         : unconditioned forward IBM (free walk from release)
#   - stochastic paths : conditioned IBM realizations (Markov bridge A*)
#   - empirical        : the observed release→recapture segment (overlay)

ibm_rich = NamedTuple[]

# -- Unconditioned forward IBM path ----------------------------------------
if length(ibm_path) >= 2
    fwd_coords = [
        (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
        for u in ibm_path if 1 <= u <= length(cents_ll)
    ]
    if length(fwd_coords) >= 2
        push!(ibm_rich, (
            tagid           = "ibm_forward",
            path            = ibm_path,
            coords          = fwd_coords,
            n_steps         = length(ibm_path) - 1,
            total_dist_km   = 0.0,
            displacement_km = 0.0,
            tortuosity      = 1.0,
            mean_hsi        = mean(hsi[u] for u in ibm_path if 1 <= u <= length(hsi)),
            color           = "#f97316",    # orange: unconditioned walk
            duration_days   = Float64(20),
            group           = 2,
            group_label     = "IBM Forward (unconditioned)",
        ))
    end
end

# -- Conditioned stochastic A* IBM paths -----------------------------------
for (i, path) in enumerate(stochastic_result.all_paths)
    length(path) < 2 && continue
    coords = [
        (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
        for u in path if 1 <= u <= length(cents_ll)
    ]
    length(coords) < 2 && continue
    push!(ibm_rich, (
        tagid           = "ibm_$i",
        path            = path,
        coords          = coords,
        n_steps         = length(path) - 1,
        total_dist_km   = 0.0,
        displacement_km = 0.0,
        tortuosity      = 1.0,
        mean_hsi        = mean(hsi[u] for u in path if 1 <= u <= length(hsi)),
        color           = "#7dd3fc",    # blue: conditioned Markov bridge
        duration_days   = Float64(time_steps),
        group           = 1,
        group_label     = "IBM Realization",
    ))
end

# -- Observed release→recapture segment (empirical overlay) ----------------
empirical_entry = if 1 <= release_node <= length(cents_ll) &&
                     1 <= recapture_node <= length(cents_ll)
    [(
        tagid  = "observed_1",
        coords = [
            (Float64(cents_ll[release_node][1]),   Float64(cents_ll[release_node][2])),
            (Float64(cents_ll[recapture_node][1]), Float64(cents_ll[recapture_node][2])),
        ],
        color  = "#facc15",   # yellow: observed datum
    )]
else
    nothing
end

tracks_html = joinpath(params.output_dir, "example_ibm_paths.html")
tracks_map  = leaflet_tracks_map(
    ibm_rich, au_mesh;
    hsi              = hsi,
    empirical_paths  = empirical_entry,
    title            = "Snow Crab IBM Paths: node $release_node → $recapture_node\n" *
                       "(orange=unconditioned IBM; blue=conditioned Markov bridge; yellow=observed)"
)
save_html(tracks_map, tracks_html)
println("IBM tracks dashboard: $tracks_html")
println("  Forward IBM     : $(length(ibm_path)) nodes")
println("  Conditioned IBM : $(length(stochastic_result.all_paths)) paths")


# ============================================================
# Step 5: Full Pipeline (fit + reconstruct + dashboards)
# ============================================================
fitted     = fit_movement_models(loaded, params)
kernels    = extract_transition_kernels(loaded, fitted, params)
path_res   = reconstruct_paths_and_diagnostics(loaded, kernels, params)
diags      = compute_advanced_diagnostics(loaded, path_res, params)
dashboards = export_dashboards(loaded, kernels, path_res, diags, params)

println("\n" * "=" ^ 72)
println("  Pipeline Completed Successfully!")
println("  Output : $(params.output_dir)")
println("=" ^ 72)

out = (
    data               = loaded.data,
    models             = fitted.models,
    chains             = fitted.chains,
    P_kernel           = kernels.P_kernel,
    paths              = path_res.paths,
    corridors          = path_res.corridors,
    stochastic_paths   = path_res.stochastic_paths,
    domain_bottlenecks = path_res.domain_bottlenecks,
    circuit            = diags.circuit,
    wavelets           = diags.wavelets,
    parameters         = (
        alpha     = kernels.alpha_hat,
        residence = kernels.rho_hat,
        gamma     = kernels.gamma_hat,
    ),
    depth_range = loaded.parsed_depth_range,
)
