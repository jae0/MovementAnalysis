
# Framework to fit the Eulerian telemetry model 
# and then use those posterior parameters to run 
# Individual-Based Model (IBM) simulations.

# Step 0: Data Preparation

using Turing
using MovementAnalysis
 
params = movement_parameters_snowcrab()

loaded = load_movement_data(params)

# W: Adjacency matrix of your mesh (SparseMatrixCSC)
# hsi: Vector of Habitat Suitability Index per mesh node
# land_mask: Boolean vector of impermeable nodes
# releases, recaptures: Vector of node indices where crabs were tagged/caught
# ks: Vector of elapsed time steps for each recapture
# groups: Vector of group assignments (e.g., [1, 1, 2...]) for stratified parameters



# Step 1: Fit the Telemetry Model using MCMC
# First, you pass your mark-recapture observations into the pure_telemetry_turing_model to infer the underlying movement parameters.

# Instantiate the Turing Model
model = pure_telemetry_turing_model(
    releases, recaptures, ks, groups, 
    W, hsi, land_mask, 
    1 # Total number of groups (G)
)

# 3. Sample the posterior using NUTS (or MH)
chain = sample(model, NUTS(1000, 0.65), 500)

# 4. Extract the mean estimated parameters for Group 1
alpha_est = mean(chain[:"alpha[1]"]) # Advection / drift weight
rho_est   = mean(chain[:"rho[1]"])   # Residence / fidelity weight
gamma_est = mean(chain[:"gamma[1]"]) # HSI gradient responsiveness


# Step 2: Construct the Stochastic Transition Kernel ($P$)
# Use the estimated parameters to build the exact continuous-time 
# Advection-Diffusion operator (transition matrix), which represents the environment.


# Build the row-stochastic transition probability matrix P
# Size: (Number of spatial nodes) x (Number of spatial nodes)
P = construct_stochastic_transition_kernel(
    W, hsi; 
    gamma = gamma_est, 
    residence = rho_est, 
    advection = alpha_est, 
    land_mask = land_mask
)



### --- STOP :: this next section needs to be checked. 
## --- note, the above is for a point estimate of the posterior
## --- below, using the posterior samples to build all possible paths and
## --- the transition matrix P for each posterior sample

# 1. Initialize containers to store the results
# Assuming you know the number of draws (n_draws) and simulations per draw (n_sims)
# This example creates nested vectors: Vector of Draws -> Vector of Simulations -> Vector of Paths -> Vector of Nodes
# You might want to use Arrays if sizes are fixed, or a more sophisticated data structure.
simulated_paths = Vector{Vector{Vector{Int}}}(undef, n_draws)

# Assuming 'sample' is your Turing.jl chain object and n_draws is the number of posterior draws you want to use
for i in 1:n_draws
    alpha_sample = sample.alpha[i]   # e.g., 0.15
    beta_sample  = sample.beta[i]   # e.g., 0.65

    P_sample = build_transition_matrix(W, alpha_sample, beta_sample)
    
    stochastic_paths = simulate_stochastic_paths(P_sample, start_node, k, n_sims; land_mask=land_mask)

    # 4. Save the stochastic paths (e.g., in a DataFrame or Array)
    push!(simulated_paths, stochastic_paths)
end

# You now have a collection of stochastic paths that respect the posterior uncertainty. 
# They are guaranteed to connect the start and end points.

# Let's assume you already have:
# 1. A graph or mesh structure (Nodes + Edges)
# 2. A matrix of advection strengths (alpha_field) for each node
# 3. A matrix of diffusion strengths (beta_field) for each node

function build_transition_matrix(W, alpha_field, beta_field)
    n_nodes = size(W, 1)
    # Initialize sparse matrix
    P = spzeros(Float64, n_nodes, n_nodes)
    
    for i in 1:n_nodes
        # Get neighbors from your adjacency matrix W
        # (Assuming W is a sparse matrix where W[i,j] > 0 means connected)
        neighbors = findnz(W[i, :])[2]
        
        # Extract local values
        alpha_i = alpha_field[i]
        beta_i  = beta_field[i]
        
        # Self-diffusion (staying put)
        # Note: Diffusion usually depends on neighbors. 
        # Here we use a simple form: sum(beta_j for neighbors) / 2
        # Or if beta is already the total diffusion probability from node i:
        self_prob = 1.0 - sum(beta_field[j] for j in neighbors if j in neighbors)
        
        P[i, i] = self_prob
        
        # Transition to neighbors
        for j in neighbors
            if i != j
                # The probability is split equally among neighbors if symmetric
                P[i, j] = beta_field[j]
            end
        end
    end
    
    # Normalize rows to ensure they sum to 1 (Stochastic Matrix)
    for i in 1:n_nodes
        row_sum = sum(P[i, :])
        if row_sum > 0
            P[i, :] ./= row_sum
        end
    end
    
    return P
end

### STOP ::  the above section needs to be checked 
### ---

# Step 3: Simulate individual crabs moving around the mesh using the fitted parameters.

# Scenario 3.1: Forward IBM Simulation (Random Walk)
# Drop a crab at a node and simulate its path forward for 20 time steps based on the advection/diffusion parameters:


using Distributions

function simulate_forward_ibm(P, start_node, steps)
    path = Int[start_node]
    current_node = start_node
    
    for t in 1:steps
        # Extract transition probabilities from current node to neighbors
        probs = P[current_node, :]
        
        # IBM step: Stochastically sample the next node
        next_node = rand(Categorical(probs))
        push!(path, next_node)
        
        current_node = next_node
    end
    return path
end

# Drop a crab at node 150 and simulate 20 steps
ibm_path = simulate_forward_ibm(P, 150, 20)



# Scenario 3.2: Conditional IBM Simulation (Markov Bridges)
# Movement from A to B is know, but the intermediate locations are not known: 
# simulate the most probable IBM stochastic paths it took between them 
# (incorporating the physical coastal boundaries and HSI):

release_node = 150
recapture_node = 320
time_steps = 20

# Use A* Stochastic algorithm to sample N individual paths
# that successfully connect the release and recapture nodes
n_realizations = 100
stochastic_result = astar_stochastic_least_cost_path(
    P, release_node, recapture_node; 
    k = time_steps,
    n_sims = n_realizations,
    land_mask = land_mask
)

# stochastic_result.simulated_paths contains the IBM trajectories


# Step 4: Visualizing the IBM Paths (Corridor Dashboard)
# Interactive HTML map that overlays all the sampled paths (IBM trajectories) directly onto the geographic domain.

# Load the mesh object. It should have `mesh_data` containing centroids and polygons
# For example: mesh_data = loaded_data.mesh

# Extract the edge counts (how many times each edge was traversed across all IBM realizations)
simulated_edges = stochastic_result.simulated_edges

# Generate the interactive Leaflet HTML dashboard
html_file = "C:/path/to/save/ibm_simulation_dashboard.html"

leaflet_interactive_corridor_dashboard(
    P, release_node, recapture_node, time_steps;
    mesh = mesh_data,                 # Required: Your spatial polygons/centroids
    hsi = hsi,                        # Optional: Will plot HSI as the background heatmap
    corridor_prob = stochastic_result.prob_corridor, # Heatmap of where the crabs went
    edge_counts = simulated_edges,    # The specific path lines drawn from the IBM
    land_mask = land_mask,
    output_path = html_file,
    title = "IBM Simulation: Crab Movement"
)

println("Dashboard saved to: ", html_file)



# Step 5: Visualizing the Underlying Advection/Diffusion Fields
# Visualize the parameters (Advection vs. Diffusion) you extracted from the Turing model to see if the crabs are primarily drifting (advection) or spreading randomly (diffusion) using the built-in diagnostic plot:


# Create a vector of the advection strengths per node
# (In this example, it's uniform if alpha_est is a scalar, but could be a vector if spatially varying)
advection_field = fill(alpha_est, length(hsi))

# Create a vector of the diffusion strengths
# (Diffusion is typically proportional to (1 - alpha_est) * (1 - rho_est))
diffusion_field = fill((1 - alpha_est) * (1 - rho_est), length(hsi))

# Generate the Advection-to-Diffusion (Péclet-like) ratio distribution
plot_ad_ratio_distribution(advection_field, diffusion_field; mode=:plots)



### ---------------

# Step 6: Run the full MovementAnalysis pipeline
fitted      = fit_movement_models(loaded, params)

# 3. Extract the posterior parameters (Advection, Diffusion, HSI gradient responsiveness)
kernels     = extract_transition_kernels(loaded, fitted, params)



agent_trajectories = nothing
if params.model_mode == "agent"
    if params.verbose
        println("\n[Phase 2b] Simulating Agent-Based Movement Alternative Model...")
    end
# broken:
    n_sim_agents = min(100, size(loaded.obs_df, 1))
    # Use observed release sites to start agents
    start_nodes = loaded.obs_df.release_unit[1:n_sim_agents]
    groups = [loaded.group_map[g] for g in loaded.obs_df.group[1:n_sim_agents]]
    
    agent_trajectories = simulate_agent_trajectories(
        n_sim_agents, start_nodes, groups, kernels.P_kernel, 50; seed=params.seed
    )
    if params.verbose
        println("  Simulated $(n_sim_agents) agents for 50 steps.")
    end
end

path_res    = reconstruct_paths_and_diagnostics(loaded, kernels, params)
diagnostics = compute_advanced_diagnostics(loaded, path_res, params)
priority    = execute_priority_analyses(loaded, fitted, kernels, params)
dashboards  = export_dashboards(loaded, kernels, path_res, diagnostics, params)

if params.verbose
    println("\n" * "=" ^ 72)
    println("  Pipeline Completed Successfully!")
    println("=" ^ 72)
end

out = (
        data               = loaded.data,
        models             = fitted.models,
        chains             = fitted.chains,
        P_kernel           = kernels.P_kernel,
        paths              = path_res.paths,
        corridors          = path_res.corridors,
        stochastic_paths   = path_res.stochastic_paths,
        domain_bottlenecks = path_res.domain_bottlenecks,
        circuit            = diagnostics.circuit,
        wavelets           = diagnostics.wavelets,
        priority_analyses  = priority,
        agent_trajectories = agent_trajectories,
        movement_stats     = !isnothing(dashboards) && hasproperty(dashboards, :movement_stats) ?
                             dashboards.movement_stats : nothing,
        phenology          = !isnothing(dashboards) && hasproperty(dashboards, :phenology) ?
                             dashboards.phenology : nothing,
        trait_models       = !isnothing(dashboards) && hasproperty(dashboards, :trait_models) ?
                             dashboards.trait_models : nothing,
        parameters         = (
            alpha     = kernels.alpha_hat,
            residence = kernels.rho_hat,
            gamma     = kernels.gamma_hat,
        ),
        depth_range        = loaded.parsed_depth_range,
    )
