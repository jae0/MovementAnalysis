# BSTM Movement Analysis: Pipeline Guide & Technical Reference

This document provides a comprehensive technical guide to the Bayesian
Spatio-Temporal Movement (BSTM) mark-recapture analysis pipeline implemented in
`movement_analysis.jl` and `src/movement.jl`. It consolidates the workflow
architecture, priority post-processing analyses, advanced multi-scale routing,
mathematical formulations, configuration parameters, and CLI usage.

---

## 1. Overview & Architecture

The BSTM movement pipeline reconstructs individual animal trajectories,
evaluates population-level spatial connectivity, quantifies multi-scale
migratory corridors, and propagates Bayesian posterior parameter uncertainty.

The framework is organized into six cohesive phases:

```
[Phase 1] Data Ingestion & Mesh Preparation
   │      - Telemetry observation loading (empirical or simulated)
   │      - Hexagonal domain resharding & depth barrier masking
   │      - Optional adaptive multiresolution domain generation
   ▼
[Phase 2] Bayesian Model Fitting (@bstm)
   │      - Categorical pure-telemetry Markov likelihood
   │      - Joint Negative Binomial survey density & telemetry model
   ▼
[Phase 3] Stochastic Transition Kernel Construction
   │      - Group-stratified advection-diffusion-taxis parameter extraction
   │      - Multi-epoch non-stationary transition kernels P^(t)
   ▼
[Phase 4] Trajectory & Corridor Reconstruction
   │      - A* least-cost paths, Viterbi dynamic programming, HMM smoothing
   │      - Forward-backward Markov bridge corridor probability heatmaps
   │      - Domain-wide transit bottlenecks B(u) = C(u) / deg(u)
   ▼
[Phase 4b] Priority Mark-Recapture Analyses
   │      - Individual trajectory credible intervals (length, waypoints)
   │      - Regional stock connectivity matrix & posterior credible intervals
   │      - Posterior predictive checks (Brier score, KL divergence)
   │      - Full Bayesian MCMC ensemble path & corridor propagation
   ▼
[Phase 5] Advanced Physical & Spectral Diagnostics
   │      - Circuit theory electrical current density & pinch-points
   │      - Chebyshev Spectral Graph Wavelets (SGWT) with BayesShrink
   ▼
[Phase 6] Export & Leaflet Dashboards
          - Interactive HTML/SVG maps, animated tracks, and CSV summaries
```

---

## 2. Pipeline Execution Phases

### Phase 1: Data Ingestion, Resharding, and Traversal Barriers
Function: `load_movement_data(params)`

- Loads empirical mark-recapture data (e.g. Scotian Shelf snow crab) or
  generates synthetic multi-segment telemetry histories with known ground truth.
- **Hexagonal Resharding**: When `reshard_hex = true`, resamples coarse areal units
  onto a regular hexagonal lattice of radius `hex_radius_km` using LibGEOS polygon
  clipping and area-weighted centroid interpolation.
- **Traversal Barriers**: When `depth_range = (min_d, max_d)` is specified, bathymetric
  depth fields are evaluated to tag units outside the viable physiological range
  as impermeable land/depth barriers (`land_mask[u] = true`).
- **Adaptive Multiresolution**: When `adaptive_mesh = true`, synthesizes a dual-scale
  domain with coarse offshore cells ($r_{\text{coarse}}$) and refined coastal
  cells ($r_{\text{fine}}$) joined by cross-scale adjacency edges.

### Phase 2: Convex Bayesian Model Fitting
Function: `fit_movement_models(loaded, params)`

Fits convex movement parameters via Turing.jl and `@bstm()`:
- **Telemetry Likelihood**:
  $$\log \mathcal{L}(\mathbf{y}_{1:T} \mid \theta) =
    \sum_{t=1}^{T-1} \log P_{y_t, y_{t+1}}(\theta)$$
  where $\theta = (\alpha_g, \rho_g, \gamma_g)$ denote per-group advection
  weight, site fidelity (residence probability), and habitat gradient responsiveness.
- **Joint Density-Movement Model**: Integrates scientific survey counts $C_s$ via
  Negative Binomial observation likelihood coupled to habitat suitability $H_s$.

### Phase 3: Transition Kernel Construction
Function: `extract_transition_kernels(loaded, fitted, params)`

Constructs group-specific stochastic transition matrices $P_g \in \mathbb{R}^{S \times S}$:
$$P_g = (1 - \rho_g) \left[ (1 - \alpha_g) T_{\text{diff}} +
    \alpha_g A_g(\eta) \right] + \rho_g I$$
where:
- $T_{\text{diff}, ij} = W_{ij} / \sum_k W_{ik}$ is isotropic diffusion over adjacency $W$.
- $A_{g, ij} = W_{ij} \exp(\gamma_g (H_j - H_i)) / \sum_k W_{ik} \exp(\gamma_g (H_k - H_i))$
  is directional advective-taxis driven by habitat suitability index $H$.
- $\rho_g \in [0, 1)$ governs local patch residence (diagonal persistence).

### Phase 4: Trajectory & Corridor Reconstruction
Function: `reconstruct_paths_and_diagnostics(loaded, kernels, params)`

- **Least-Cost Path Routing**: Evaluates shortest movement routes via:
  - A* search (`:astar`) with Euclidean admissible heuristic.
  - Multiresolution A* search across mixed cell scales.
  - Dynamic programming Viterbi decoding (`:viterbi`).
- **Markov Bridge Corridors**: Computes space-time transit probability fields between
  release $u$ and recapture $v$ across $k$ discrete steps:
  $$\mathbb{P}(X_\tau = j \mid X_0 = u, X_k = v) =
    \frac{[P^\tau]_{uj} [P^{k-\tau}]_{jv}}{[P^k]_{uv}}$$
- **Domain-Wide Bottleneck Index**: Quantifies geographic migration pinch-points:
  $$B(u) = \frac{C_{\text{domain}}(u)}{\max(1, \deg_{\text{marine}}(u))}$$
  where $C_{\text{domain}}(u)$ aggregates transit density across all mark-recapture
  pairs and $\deg_{\text{marine}}(u)$ is the marine node degree.

### Phase 4b: Priority Mark-Recapture Analyses
Function: `execute_priority_analyses(loaded, fitted, kernels, params)`

Consolidated priority post-processing executing:
1. Individual path credible intervals (`path_credible_intervals`).
2. Regional stock connectivity matrix and credible intervals
   (`compute_stock_connectivity_matrix`, `compute_connectivity_credible_intervals`).
3. Posterior predictive validation checks (`posterior_predictive_check`).
4. Optional full Bayesian posterior trajectory & corridor propagation
   (`reconstruct_paths_bayesian_ensemble`).

### Phase 5: Advanced Physical & Spectral Diagnostics
Function: `compute_advanced_diagnostics(loaded, path_res, params)`

- **Circuit Theory Resistance Networks**: Models the spatial graph as an electrical
  resistor grid where conductances $C_{ij} = W_{ij} \sqrt{H_i H_j}$. Solves Poisson
  system $L v = I_{\text{ext}}$ to compute current densities and identify migratory
  pinch-points without pre-specifying travel duration $k$.
- **Spectral Graph Wavelets (SGWT)**: Employs Chebyshev polynomial expansions of the
  graph Laplacian $L = D - W$ across multiple spatial scales, combined with
  BayesShrink adaptive soft thresholding for spatial habitat denoising.

### Phase 6: Interactive Dashboard Export
Function: `export_dashboards(loaded, kernels, path_res, diagnostics, params)`

Generates standalone interactive HTML/SVG dashboards:
- Reconstructed movement tracks with animated particle playback.
- Forward-backward Markov bridge corridor probability heatmaps.
- Domain-wide bottleneck and pinch-point maps.
- Electrical current density and stochastic circuit flow maps.
- Multi-scale Chebyshev spectral graph wavelet dashboards.
- Posterior parameter distributions with bivariate correlation scatter plots.
- Directed network flow graphs with dynamic volume sliders.

---

## 3. Priority Analyses (Consolidated Reference)

### 1. Per-Path Credible Intervals (`path_credible_intervals`)

#### Purpose
Quantifies spatial and topological uncertainty in individual movement paths
by drawing parameter vectors from the posterior MCMC chain and reconstructing
an ensemble of trajectories for each individual animal.

#### Algorithm
1. Extract MCMC posterior parameter samples $(\alpha^{(s)}, \rho^{(s)}, \gamma^{(s)})$.
2. For each posterior draw $s \in \{1, \dots, S_d\}$:
   - Reconstruct draw-specific transition kernel $P^{(s)}$.
   - Evaluate trajectory from release to recapture via A* routing.
   - Record path length and node visitation indicators.
3. Compute empirical quantiles (2.5%, 50%, 97.5%) across draws.

#### Key Outputs
- `path_credible_intervals.csv`:
  - `tagid`: Animal identifier.
  - `release_site`, `recapture_site`: Spatial boundary units.
  - `path_length_mean`: Mean path length in hops/km.
  - `path_length_lower`, `path_length_upper`: 95% Bayesian credible interval.
  - `path_length_sd`: Standard deviation across posterior draws.
  - `modal_waypoint`: Most-visited intermediate mesh unit.
- `node_visit_probs`: $S \times S$ matrix of pairwise transition frequencies.

### 2. Bayesian Ensemble Trajectory & Corridor Propagation
Function: `reconstruct_paths_bayesian_ensemble(loaded, fitted, params)`

#### Purpose
Propagates parameter uncertainty directly into both path trajectories and Markov
bridge corridor probability fields by integrating across MCMC samples rather than
conditioning on the posterior mean parameter point estimate.

#### Mathematical Formulation
For each draw $s$:
$$P^{(s)} =
    \text{construct_stochastic_transition_kernel}(W, H;
    \alpha^{(s)}, \rho^{(s)}, \gamma^{(s)})$$
$$\bar{\Pi}_i = \frac{1}{S_d} \sum_{s=1}^{S_d} \Pi_i^{(s)}$$
where $\Pi_i^{(s)}$ is the forward-backward Markov bridge intensity under draw $s$.

#### Returns
- `ensemble_corridors`: Dict mapping tag ID to mean corridor field $\bar{\Pi}$.
- `ensemble_paths`: Dict mapping tag ID to collection of sampled paths across draws.

### 3. Regional Stock Connectivity Matrix
Functions:
- `compute_stock_connectivity_matrix(loaded, kernels, params; region_labels, region_map)`
- `compute_connectivity_credible_intervals(loaded, fitted, kernels, params, region_map)`

#### Purpose
Aggregates fine-scale mesh transition probabilities into a population-level
stochastic connectivity matrix between biologically distinct spatial regions
(e.g., Fishery Management Areas, depth strata, spawning vs. nursery grounds).

#### Mathematical Formulation
Given region assignments $r, s \in \{1, \dots, N_{\text{regions}}\}$:
$$\text{Connectivity}[r, s] = \frac{1}{|r| \cdot |s|} \sum_{u \in r} \sum_{v \in s} P[u, v]$$
Rows are normalized to enforce row-stochastic conservation: $\sum_s \text{Connectivity}[r, s] = 1$.

Posterior uncertainty is propagated across MCMC draws to derive 95% credible intervals
$[\text{lower}_{rs}, \text{upper}_{rs}]$ for every pairwise inter-regional flow.

#### Key Outputs
- `stock_connectivity_summary.csv`: Point estimates of connectivity probabilities,
  observed mark-recapture transition counts, and flow rates.
- `stock_connectivity_credible_intervals.csv`: Lower (2.5%) and upper (97.5%) credible
  bounds for all donor-recipient region pairs.

### 4. Posterior Predictive Validation Checks
Functions:
- `posterior_predictive_check(loaded, fitted, kernels, params)`
- `export_posterior_predictive_check(ppc, output_dir)`
- `plot_posterior_predictive_check(ppc, output_dir)`

#### Purpose
Assesses whether the fitted movement model reproduces empirical recapture distributions
by forward-simulating recapture locations from observed release sites across posterior
draws.

#### Quantitative Metrics
1. **Brier Score (Mean Squared Error)**:
   $$\text{Brier} = \frac{1}{S} \sum_{j=1}^S
    \left( P_{\text{obs}}(j) - P_{\text{sim}}(j) \right)^2$$
   Measures calibration quality ($0.0 = \text{perfect}$, $< 0.01 = \text{excellent}$).
2. **Kullback-Leibler Divergence**:
   $$D_{\text{KL}}(P_{\text{obs}} \parallel P_{\text{sim}}) =
    \sum_{j=1}^S P_{\text{obs}}(j) \log \left(
    \frac{P_{\text{obs}}(j)}{P_{\text{sim}}(j) + \epsilon} \right)$$
   Quantifies information-theoretic discrepancy ($< 0.5 \text{ nats} = \text{good}$).
3. **Rank Histogram**: Assesses uncertainty calibration across posterior draws. A uniform
   histogram indicates well-calibrated posterior spread; U-shaped indicates overconfidence.

#### Key Outputs
- `posterior_predictive_summary.txt`: Summary of Brier score, KL divergence, entropy.
- `posterior_predictive_diagnostics.csv`: Per-draw metric trace records.
- `posterior_predictive_distributions.csv`: Observed vs. predicted recapture vectors.
- `posterior_predictive_rank_histogram.csv`: Uniformity calibration counts.

---

## 4. Advanced Movement Enhancements

### 1. Adaptive Multiresolution Hexagonal Mesh Routing
Functions:
- `construct_adaptive_multiresolution_domain`:
  `(coarse_mesh, coastal_polygons; coarse_radius_km, fine_radius_km)`
- `astar_multiresolution_path(adaptive_domain, release, recapture; hsi, land_mask)`

Combines coarse offshore cells ($r_{\text{coarse}} = 25\text{ km}$) with fine coastal
cells ($r_{\text{fine}} = 8\text{ km}$) to dramatically reduce the state space dimension
$S$ while maintaining fine-scale resolution along convoluted shorelines and telemetry arrays.

The cross-scale graph links boundary nodes via distance-scaled edge costs:
$$\Delta g_{ij} = \frac{\Delta x_{ij}}{c_i \cdot c_j}$$
where $c_i, c_j$ are habitat conductance values and $\Delta x_{ij}$ is the physical
great-circle distance between cell centroids.

### 2. Time-Varying Dynamic Environmental Covariates
Functions:
- `construct_dynamic_transition_kernels(W, hsi_matrices; land_mask, delta_km)`
- `predict_dynamic_path(P_kernels, release, recapture, times; centroids, land_mask)`
- `predict_dynamic_corridor(P_kernels, release, recapture; land_mask)`

Accommodates non-stationary environmental dynamics (e.g. seasonal warming, shifting
thermoclines, dynamic chlorophyll blooms) by constructing an epoch-indexed sequence
of transition matrices $[P^{(1)}, \dots, P^{(T)}]$ from an $S \times T$ habitat matrix.
Forward-backward Markov bridges evaluate time-dependent corridor likelihoods
$\Pi_i^{(t)} = \alpha_t(i) \beta_t(i)$.

### 3. Multi-Segment Hidden Markov Model (HMM) Viterbi Smoothing
Functions:
- `viterbi_hmm_path_smoothing(obs_times, obs_locations, P_kernel; mesh, land_mask, sigma_obs_km)`
- `forward_backward_state_probabilities`:
  `(obs_times, obs_locations, P_kernel; mesh, land_mask, sigma_obs_km)`

Globally decodes the entire multi-stage capture-recapture history in log-space:
$$\max_{z_{1:T}} \left[ \log \pi(z_1) +
    \sum_{t=1}^{T-1} \log P^{(t)}(z_t, z_{t+1}) +
    \sum_{t=1}^T \log f(y_t \mid z_t) \right]$$
where $f(y_t \mid z_t) = \mathcal{N}(y_t; z_t, \sigma_y^2)$ is the continuous spatial
emission density. Avoids segment-isolation artifacts and guarantees continuous pathing.

---

## 5. Configuration Parameters

Parameter NamedTuples are generated via `movement_parameters_default()` or
`movement_parameters_snowcrab()`, and can be customized by merging overrides:

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `data_source` | `Symbol` | `:simulate` | `:simulate` or `:snowcrab` empirical data |
| `model_mode` | `String` | `"telemetry"` | `"telemetry"`, `"telemetry_and_survey"`, or `"both"` |
| `reshard_hex` | `Bool` | `false` | Reshard domain to fine regular hexagons via LibGEOS |
| `hex_radius_km` | `Float64` | `10.0` | Cell radius for resharded hexagons (km) |
| `use_hydrodynamics` | `Bool` | `false` | Ingest 3D hydrodynamic velocity and bathymetry |
| `depth_range` | `Tuple / Nothing` | `nothing` | Depth barrier `(min, max)` (m) |
| `max_paths` | `Int` | `25` | Maximum number of individual trajectories to reconstruct |
| `path_method` | `Symbol` | `:astar` | Routing algorithm (`:astar` or `:viterbi`) |
| `smooth_paths` | `Bool` | `false` | Apply line-of-sight raycast path smoothing |
| `compute_priority` | `Bool` | `true` | Execute priority uncertainty and connectivity analyses |
| `run_bayesian_ensemble` | `Bool` | `false` | Run full posterior MCMC path/corridor propagation |
| `compute_circuit` | `Bool` | `false` | Compute electrical circuit resistance current density |
| `compute_stochastic` | `Bool` | `false` | Reconstruct Monte Carlo stochastic least-cost paths |
| `compute_bottlenecks` | `Bool` | `false` | Evaluate domain-wide migration bottleneck index B(u) |
| `compute_wavelets` | `Bool` | `false` | Multi-scale Chebyshev graph wavelet decomposition |
| `adaptive_mesh` | `Bool` | `false` | Use dual-resolution coarse/fine hexagonal mesh |
| `coarse_radius_km` | `Float64` | `25.0` | Cell radius for coarse offshore units (km) |
| `fine_radius_km` | `Float64` | `8.0` | Cell radius for refined coastal units (km) |
| `dynamic_kernels` | `Bool` | `false` | Use time-varying dynamic transition kernels |
| `hmm_smoothing` | `Bool` | `false` | Apply multi-segment HMM Viterbi trajectory smoothing |
| `n_samples` | `Int` | `200` | Turing MCMC posterior draw count |
| `n_warmup` | `Int` | `100` | Turing MCMC warmup iterations |
| `seed` | `Int` | `42` | Random number generator seed |
| `hsi_se` | `Float64` | `0.08` | Observation standard error on habitat suitability |
| `propagate_hsi_error` | `Bool` | `true` | Propagate HSI uncertainty |
| `render_html` | `Bool` | `true` | Export interactive Leaflet HTML maps and dashboards |
| `output_dir` | `String` | `"output"` | Directory for all generated tables and HTML dashboards |
| `verbose` | `Bool` | `true` | Enable detailed step-by-step progress logging |
| `species_name` | `String` | `"Animal"` | Display name in reports/dashboards |

---

## 6. Command-Line Interface (CLI)

The pipeline can be executed directly from the terminal with modular flags:

```bash
# Display help and all available switches
julia --project=. docs/movement/movement_analysis.jl --help

# Run default simulated telemetry analysis
julia --project=. docs/movement/movement_analysis.jl --simulate

# Run Scotian Shelf snow crab analysis with all diagnostics
julia --project=. docs/movement/movement_analysis.jl --snowcrab --all-diagnostics

# Snow crab with custom depth constraint and fine hexagonal lattice
julia --project=. docs/movement/movement_analysis.jl \
  --snowcrab --depth-range 50,450 --hex --hex-radius 5.0

# Run with full Bayesian posterior ensemble propagation
julia --project=. docs/movement/movement_analysis.jl --snowcrab --bayesian-ensemble --samples 300

# Run with adaptive multiresolution mesh and HMM smoothing
julia --project=. docs/movement/movement_analysis.jl --snowcrab --adaptive-mesh --hmm-smoothing
```

### CLI Switch Table

| CLI Flag | Long Form | Parameter Mapping |
|:---------|:----------|:------------------|
| `-d` | `--data-source <src>` | `data_source = Symbol(src)` |
| | `--simulate` | `data_source = :simulate` |
| | `--snowcrab` | Snow crab preset configuration |
| `-m` | `--model-mode <mode>` | `model_mode = "telemetry"` or `"both"` |
| | `--reshard-hex`, `--hex` | `reshard_hex = true` |
| | `--hex-radius <km>` | `hex_radius_km = Float64(km)` |
| | `--depth-range <min,max>`| `depth_range = (min, max)` |
| `-p` | `--max-paths <N>` | `max_paths = Int(N)` |
| | `--astar` | `path_method = :astar` |
| | `--viterbi` | `path_method = :viterbi` |
| | `--smooth-paths` | `smooth_paths = true` |
| | `--priority` / `--no-priority` | `compute_priority = true / false` |
| | `--bayesian-ensemble` | `run_bayesian_ensemble = true` |
| | `--adaptive-mesh` | `adaptive_mesh = true` |
| | `--dynamic-kernels` | `dynamic_kernels = true` |
| | `--hmm-smoothing` | `hmm_smoothing = true` |
| | `--circuit` | `compute_circuit = true` |
| | `--stochastic` | `compute_stochastic = true` |
| | `--bottlenecks` | `compute_bottlenecks = true` |
| | `--wavelets`, `--sgwt` | `compute_wavelets = true` |
| | `--all-diagnostics` | Enables circuit, stochastic, bottlenecks, wavelets |
| `-n` | `--samples <N>` | `n_samples = Int(N)` |
| `-w` | `--warmup <N>` | `n_warmup = Int(N)` |
| `-o` | `--output-dir <path>` | `output_dir = String(path)` |
| `-q` | `--quiet` | `verbose = false` |

---

## 7. Scripting Examples & Workflows

### Example 1: Basic Analysis with Default Settings

```julia
using bstm
include("docs/movement/movement_analysis.jl")

# Run default simulated analysis pipeline
results = run_movement_analysis()

println("Reconstructed paths: ", length(results.paths))
println("Fitted velocity alpha: ", results.parameters.alpha)
println("Fitted residence rho:  ", results.parameters.residence)
```

### Example 2: Snow Crab with Depth Barriers & Priority Uncertainty

```julia
using bstm
include("docs/movement/movement_analysis.jl")

# Configure snow crab analysis with physiological depth boundaries
params = merge(movement_parameters_snowcrab(), (
    depth_range           = (60.0, 450.0),
    n_samples             = 300,
    n_warmup              = 100,
    compute_priority      = true,
    run_bayesian_ensemble = true,
    max_paths             = 40,
))

# Execute complete workflow
results = run_movement_analysis(params)

# Inspect priority outputs
pa = results.priority_analyses
println("Path uncertainty summary: ", pa.summary_file)
println("Posterior predictive Brier: ", pa.posterior_predictive.summary.brier_mean)
println("Stock connectivity matrix size: ", size(pa.connectivity_matrix.connectivity_matrix))
```

### Example 3: Adaptive Mesh Routing & Dynamic Corridors

```julia
using bstm
include("docs/movement/movement_analysis.jl")

# Configure high-resolution coastal refinement
params = merge(movement_parameters_default(), (
    adaptive_mesh    = true,
    coarse_radius_km = 20.0,
    fine_radius_km   = 6.0,
    dynamic_kernels  = true,
    hmm_smoothing    = true,
    max_paths        = 15,
))

results = run_movement_analysis(params)
```

---

## 8. Output Directory Structure

Executing the pipeline populates `output_dir` with standardized deliverables:

```text
output/
├── path_credible_intervals.csv              # Per-path length CIs & modal waypoints
├── stock_connectivity_summary.csv           # Regional connectivity point estimates
├── stock_connectivity_credible_intervals.csv# Regional connectivity 95% CIs
├── posterior_predictive_summary.txt         # Brier score & KL divergence summary
├── posterior_predictive_diagnostics.csv     # Per-draw PPC metrics trace
├── posterior_predictive_distributions.csv   # Observed vs. predicted recapture dists
├── posterior_predictive_rank_histogram.csv  # Uncertainty calibration histogram
├── movement_path_metrics.csv                # Path displacement, tortuosity, bearing
├── movement_tracks_animated.html            # Animated Leaflet trajectory playback
├── movement_corridors_heatmap.html          # Markov bridge transition corridors
├── movement_domain_bottlenecks.html         # Regional pinch-points and bottleneck index
├── movement_current_density.html            # Circuit theory electrical current flux
├── movement_stochastic_circuit.html         # Posterior current density with HSI error
├── movement_wavelet_dashboard.html          # Multi-scale Chebyshev SGWT wavelets
├── movement_posterior_uncertainty.html      # MCMC parameter KDEs & correlations
├── movement_network_flow.html               # Directed Bézier network flow graph
└── movement_summary_diagnostics.html        # Comprehensive diagnostics overview
```

---

## 9. Scientific References

1. **BSTM Framework**: Choi, J. (2025). *Bayesian Spatio-Temporal Models for Marine Ecology*.
2. **Circuit Theory in Ecology**: McRae, B. H., Dickson, B. G., Keitt, T. H., & Shah, V. B.
   (2008). Using circuit theory to model connectivity in ecology, evolution, and
   conservation. *Ecology*, 89(10), 2712–2724.
3. **Spectral Graph Wavelets**: Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011).
   Wavelets on graphs via spectral graph theory. *Applied and Computational Harmonic Analysis*,
   30(2), 129–150.
4. **BayesShrink Wavelet Denoising**: Chang, S. G., Yu, B., & Vetterli, M. (2000). Adaptive
   wavelet thresholding for image denoising and compression. *IEEE Trans. Image Processing*,
   9(9), 1532–1546.
5. **Forecast Verification (Brier Score)**: Brier, G. W. (1950). Verification of forecasts
   expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
6. **Information Divergence**: Kullback, S., & Leibler, R. A. (1951). On information and
   sufficiency. *Annals of Mathematical Statistics*, 22(1), 79–86.
7. **Bayesian Posterior Predictive Checks**: Gelman, A., Carlin, J. B., Stern, H. S.,
   Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.).
   Chapman and Hall/CRC.
