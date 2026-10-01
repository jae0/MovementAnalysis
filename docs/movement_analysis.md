# Movement Analysis: Pipeline Guide & Technical Reference

---

## 1. Overview & Architecture

The movement pipeline reconstructs individual animal trajectories, evaluates
population-level spatial connectivity, quantifies migratory corridors, and
propagates Bayesian posterior parameter uncertainty across a discrete hexagonal
spatial domain.

The framework is organized into six cohesive phases:

```
[Phase 1] Data Ingestion & Mesh Preparation
   │      - Telemetry observation loading (empirical or simulated)
   │      - Hexagonal domain resharding & depth preference encoding
   │      - Optional adaptive multiresolution domain
   ▼
[Phase 2] Bayesian Model Fitting
   │      - Categorical pure-telemetry Markov likelihood
   │      - Joint Negative Binomial survey density & telemetry model
   ▼
[Phase 3] Stochastic Transition Kernel Construction
   │      - Pooled advection-diffusion-taxis extraction
   │      - Time-varying (monthly) kernel construction from seasonal HSI
   ▼
[Phase 4] Trajectory & Corridor Reconstruction
   │      - A* least-cost paths, Viterbi dynamic programming, HMM smoothing
   │      - Forward-backward Markov bridge corridor probability heatmaps
   │      - Domain-wide transit bottlenecks B(u) = C(u) / deg(u)
   ▼
[Phase 4b] Validation Mark-Recapture Analyses
   │      - Individual trajectory credible intervals
   │      - Regional stock connectivity matrix & credible intervals
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

### Phase 1: Data Ingestion, Resharding, and Depth Encoding

Function: `load_movement_data(params)`

#### 1a. Observation Loading

Loads empirical mark-recapture data (e.g. Scotian Shelf snow crab) or generates
synthetic multi-segment telemetry histories with known ground truth. Each
observation pair records:

JLD2 inputs work without R. R serialized inputs (`.rds`, `.rda`, `.rdz`, `.qs`)
require the optional `RCall` dependency and an available R installation; load
`RCall` in the active Julia environment to activate R-format support.

- `release`, `recapture`: mesh node indices.
- `k`: elapsed time steps (derived from `Δt / dt` where `dt` is the time
  resolution, e.g. `1/365.25` for daily).
- `rel_time`: decimal-year timestamp of release (e.g. `2018.583 ≈ Aug 2018`),
  used to select the correct monthly HSI slice during path reconstruction.
- `sex`, `mat`: descriptive demographic observations, not model strata.

#### 1b. Hexagonal Resharding

When `reshard_hex = true`, resamples coarse areal units onto a regular hexagonal
lattice of radius `hex_radius_km` using LibGEOS polygon clipping and
area-weighted centroid interpolation. Observation endpoints are remapped to
the nearest active marine unit in the fine mesh.

#### 1c. Depth Range Encoding

When `depth_range = (min_d, max_d)` is specified, bathymetric depth fields are
evaluated to identify out-of-depth marine nodes. Behaviour is controlled by
`depth_barrier_mode`:

| Mode | W connectivity | Depth treatment | Observations dropped |
|:-----|:--------------|:----------------|:--------------------|
| `:hsi_only` **(default)** | Land-only barrier | HSI clamped to `hsi_ood_floor` (default 0.01) for out-of-depth nodes | Only genuine coordinate errors |
| `:hard` | Depth + land combined barrier | Hard structural exclusion | Observations spanning disconnected depth corridors |
| `:bridge` | Depth + land barrier + local bridge edges | Bridge edges connect in-depth neighbours separated by thin transit corridors | Only coordinate errors |

**Rationale for `:hsi_only`**: applying depth as a hard structural constraint
on the adjacency matrix $W$ fragments physically contiguous marine regions at
fine mesh resolutions (5 km hexagons), creating many disconnected basins where
none truly exist. The `:hsi_only` approach encodes depth preference entirely
through the habitat suitability index — the advective taxis term
$\exp(\gamma(H_j - H_i))$ strongly discourages transit through low-HSI
out-of-depth nodes without severing connectivity.

**Bridge method** (when `:bridge` is selected): for each out-of-depth marine
transit node $t$, bridge edges are added between every pair of its in-depth
marine neighbours $\{u, v\} \subseteq N_t$ not already adjacent in $W$. Cost is
$O(\text{nnz}(W_{\text{marine}}))$ — a single pass over existing edges, at most
$\binom{\deg(t)}{2}$ new edges per transit node. This replaces the prior
component-wide BFS that produced $O(n^2)$ edges per disconnected component.

#### Reachability Check

After all remapping, a final BFS on the land-only $W$ detects observations
whose endpoints cannot be connected via any marine route. These are reported
by `tagid` and original lon/lat coordinates and are likely GPS/datum errors:

```
  Dropped N obs unreachable via any marine route (likely coordinate/datum errors):
    tagid=SC_XXXX  k=7  rel=(-63.21, 45.18)  rec=(-63.22, 45.19)
```

---

### Phase 2: Bayesian Model Fitting

Function: `fit_movement_models(loaded, params)`

Fits movement parameters via Turing.jl NUTS (No-U-Turn Sampler) using
**ForwardDiff automatic differentiation**. The transition kernel
`build_sparse_transition_kernel` is fully generic over the element type,
accepting `ForwardDiff.Dual` numbers during gradient evaluation. All
intermediate computations use the promoted type `T = promote_type(Float64,
typeof(gamma), ...)` to preserve dual-number partials.

#### Model Parameters

The model samples three pooled physical parameters:

- `velocity` $\sim \text{Truncated-Normal}(0.3, 0.2; 0, 0.95)$: raw
  advection rate.
- `diffusion` $\sim \text{Truncated-Normal}(0.1, 0.2; 0, \infty)$: raw
  isotropic diffusion rate.
- `gamma` $\sim \text{Normal}(1.0, 1.0)$: habitat gradient responsiveness.

There is no demographic stratification: a single parameter set is shared by
every observation, and `sex` / `mat` are carried only as descriptive metadata.

Derived parameters:

$$\alpha = \frac{v}{v + D + \epsilon}, \qquad
  \rho = \frac{1}{1 + v + D + \epsilon}$$

where $\epsilon = 10^{-6}$ avoids division by zero.

#### Likelihood (Discrete-Time Telemetry)

$$\log \mathcal{L}(\mathbf{y}_{1:T} \mid \theta) =
  \sum_{t=1}^{T-1} \log [T^{k_t}]_{y_t, y_{t+1}}$$

where $T^k$ is the $k$-step matrix power of the pooled transition kernel
and $(y_t, y_{t+1})$ are the release/recapture node indices of observation $t$.

**HSI in MCMC**: the Turing model uses the climatological mean `hsi_vec`
(temporal average across all months). This is intentional — the parameters
$(v, D, \gamma)$ are global; conditioning on a time-specific HSI during
HMC would require a different kernel per observation, making gradient
evaluation infeasible. Time-varying HSI is applied post-estimation during
path reconstruction and kernel construction (Phase 3–4).

#### Joint Density-Movement Model

Integrates scientific survey counts $C_s$ via Negative Binomial observation
likelihood coupled to habitat suitability $H_s$.

---

### Phase 3: Transition Kernel Construction

Function: `extract_transition_kernels(loaded, fitted, params)`

Constructs a single pooled stochastic transition matrix
$T \in \mathbb{R}^{S \times S}$:

$$T = (1 - \rho) \left[ (1 - \alpha) T_{\text{diff}} +
    \alpha A(H) \right] + \rho I$$

where:

- $T_{\text{diff}, ij} = W_{ij} / \sum_k W_{ik}$ is isotropic diffusion over
  adjacency $W$.
- $A_{ij} = W_{ij} \exp(\gamma (H_j - H_i)) /
  \sum_k W_{ik} \exp(\gamma (H_k - H_i))$ is directional habitat-taxis.
- $\rho \in [0, 1)$ governs local patch residence.

The posterior mean kernel uses `loaded.hsi_vec` (climatological mean). When
`loaded.monthly_hsi` is non-empty, Phase 4 (path reconstruction) builds
observation-specific kernels using the appropriate monthly HSI slice selected
by `_resolve_hsi_for_time(loaded, rel_time)`.

---

### Phase 4: Trajectory & Corridor Reconstruction

Function: `reconstruct_paths_and_diagnostics(loaded, kernels, params)`

#### Time-Varying HSI in Path Reconstruction

For each observation segment, the release time `rel_time` (stored in `obs_df`)
is used to select the appropriate monthly HSI slice:

```julia
hsi_seg = _resolve_hsi_for_time(loaded, row.rel_time)
```

If `monthly_hsi` is available and contains a column for the observation's
(year, month), `hsi_seg = monthly_hsi[:, col]`. Otherwise, falls back to
`hsi_vec`. A segment-specific kernel is then built from the posterior mean
parameters and this seasonal HSI, ensuring paths reflect the habitat
conditions present at the time of observed movement.

The same logic applies to Markov bridge corridor computation and HSI error
propagation — perturbations are centred on the seasonally correct `hsi_first`
rather than the temporal mean.

#### Path Methods

- **A\* search** (`:astar`): shortest-cost route with Euclidean admissible heuristic.
- **Viterbi** (`:viterbi`): dynamic programming decoding.
- **Multiresolution A\*** : for adaptive dual-scale mesh domains.
- **HMM Viterbi smoothing** (`hmm_smoothing = true`): globally decodes the
  entire multi-segment capture history in log-space.

#### Markov Bridge Corridors

Transit probability field between release $u$ and recapture $v$ across $k$ steps:

$$\mathbb{P}(X_\tau = j \mid X_0 = u, X_k = v) =
  \frac{[T^\tau]_{uj} [T^{k-\tau}]_{jv}}{[T^k]_{uv}}$$

#### Domain-Wide Bottleneck Index

$$B(u) = \frac{C_{\text{domain}}(u)}{\max(1, \deg_{\text{marine}}(u))}$$

where $C_{\text{domain}}(u)$ aggregates transit density across all mark-recapture
pairs.

---

### Phase 4b: Validation Mark-Recapture Analyses

Function: `execute_validation_analyses(loaded, fitted, kernels, params)`

1. Per-path credible intervals (`path_credible_intervals`).
2. Regional stock connectivity matrix and credible intervals.
3. Posterior predictive validation (Brier score, KL divergence).
4. Optional full Bayesian posterior trajectory & corridor ensemble.

---

### Phase 5: Advanced Physical & Spectral Diagnostics

Function: `compute_advanced_diagnostics(loaded, path_res, params)`

- **Circuit Theory**: models the spatial graph as a resistor grid with
  conductances $C_{ij} = W_{ij} \sqrt{H_i H_j}$. Solves Poisson system
  $L v = I_{\text{ext}}$ to identify migratory pinch-points.
- **Spectral Graph Wavelets (SGWT)**: Chebyshev polynomial expansions of the
  graph Laplacian across multiple spatial scales, with BayesShrink adaptive
  soft thresholding.

---

### Phase 6: Interactive Dashboard Export

Function: `export_dashboards(loaded, kernels, path_res, diagnostics, params)`

Generates standalone interactive HTML/SVG dashboards: reconstructed tracks,
Markov bridge corridors, bottleneck maps, circuit density, wavelet dashboards,
posterior parameter distributions, and directed network flow graphs.

---

## 3. HSI: Static vs. Time-Varying

The pipeline supports both a static HSI field and a time-varying (annual/monthly)
field. The distinction affects different pipeline stages:

| Stage | Static HSI | Monthly HSI |
|:------|:-----------|:------------|
| MCMC estimation (Phase 2) | `hsi_vec` | `hsi_vec` (climatological mean, by design) |
| Kernel construction (Phase 3, posterior mean) | `hsi_vec` | `hsi_vec` |
| Path reconstruction (Phase 4, per observation) | `hsi_vec` | `monthly_hsi[:, col]` for release (year, month) |
| Corridor / HSI error propagation | `hsi_vec` | `monthly_hsi[:, col]` for release time |

**`_resolve_hsi_for_time(loaded, t_decimal)`**: resolves the appropriate HSI
vector for decimal-year timestamp `t_decimal`. Decomposes as:

```
year  = floor(Int, t_decimal)
month = clamp(ceil(Int, (t_decimal - year) × 12), 1, 12)
```

and looks up `month_lookup[(year, month)]` in the column index of `monthly_hsi`.
Falls back to `hsi_vec` if no monthly data is present or the lookup fails.

The `:hsi_only` depth mode also uses HSI as the sole depth encoding mechanism:
out-of-depth marine nodes receive `hsi_vec[u] = min(hsi_vec[u], hsi_ood_floor)`
(default `hsi_ood_floor = 0.01`). Monthly HSI slices are floored similarly
during path reconstruction.

---

## 4. Configuration Parameters

Loaded from `configs/default.toml` or `configs/snowcrab.toml` by
`movement_parameters_default()` and `movement_parameters_snowcrab()`. Snow-crab
settings overlay the generic defaults. CLI options override the selected config.
TOML keys use the exact internal parameter names; unknown names are rejected.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `data_source` | `Symbol` | `:simulate` | `:simulate` or `:snowcrab` |
| `model_mode` | `String` | `"telemetry"` | `"telemetry"`, `"telemetry_and_survey"`, `"agent"`, `"both"` |
| `reshard_hex` | `Bool` | `false` | Reshard domain to fine hexagons via LibGEOS |
| `hex_radius_km` | `Float64` | `10.0` | Cell radius for resharded hexagons (km) |
| `use_hydrodynamics` | `Bool` | `false` | Ingest 3D hydrodynamic velocity & bathymetry |
| `depth_range` | `Tuple / Nothing` | `nothing` | Depth window `(min, max)` (m) |
| `depth_barrier_mode` | `Symbol` | `:hsi_only` | `:hsi_only`, `:bridge`, or `:hard` |
| `hsi_ood_floor` | `Float64` | `0.01` | HSI floor for out-of-depth nodes (`:hsi_only` mode) |
| `max_paths` | `Int` | `25` | Maximum individual trajectories to reconstruct |
| `path_method` | `Symbol` | `:astar` | Routing algorithm (`:astar` or `:viterbi`) |
| `smooth_paths` | `Bool` | `false` | Apply line-of-sight path smoothing |
| `compute_validation` | `Bool` | `true` | Execute validation uncertainty & connectivity analyses |
| `run_bayesian_ensemble` | `Bool` | `false` | Full posterior MCMC path/corridor propagation |
| `compute_circuit` | `Bool` | `false` | Electrical circuit resistance current density |
| `compute_stochastic` | `Bool` | `false` | Monte Carlo stochastic least-cost paths |
| `compute_bottlenecks` | `Bool` | `false` | Domain-wide migration bottleneck index $B(u)$ |
| `compute_wavelets` | `Bool` | `false` | Multi-scale Chebyshev graph wavelet decomposition |
| `adaptive_mesh` | `Bool` | `false` | Dual-resolution coarse/fine hexagonal mesh |
| `coarse_radius_km` | `Float64` | `25.0` | Cell radius for coarse offshore units (km) |
| `fine_radius_km` | `Float64` | `8.0` | Cell radius for refined coastal units (km) |
| `dynamic_kernels` | `Bool` | `false` | Time-varying dynamic transition kernels |
| `hmm_smoothing` | `Bool` | `false` | Multi-segment HMM Viterbi trajectory smoothing |
| `n_agent_projections` | `Int` | `200` | Synthetic agents to forward-project in `"agent"` mode |
| `persistence` | `Float64` | `0.0` | Directional persistence for agents, $\kappa$ (0 = memoryless) |
| `rest_coupling` | `Float64` | `0.0` | Couple the stay probability to local HSI advantage |
| `rest_advantage_form` | `Symbol` | `:difference` | Advantage measure, see [Section 7.3](#73-habitat-coupled-residency) |
| `n_samples` | `Int` | `200` | Turing MCMC posterior draw count |
| `n_warmup` | `Int` | `100` | Turing MCMC warmup (NUTS adaptation) iterations |
| `seed` | `Int` | `42` | Random number generator seed |
| `hsi_se` | `Float64` | `0.08` | Observation standard error on habitat suitability |
| `propagate_hsi_error` | `Bool` | `true` | Propagate HSI uncertainty into corridors |
| `render_html` | `Bool` | `true` | Export interactive Leaflet HTML dashboards |
| `output_dir` | `String` | `"output"` | Directory for generated tables and dashboards |
| `verbose` | `Bool` | `true` | Detailed step-by-step progress logging |
| `species_name` | `String` | `"Animal"` | Display name in reports/dashboards |

---

## 5. Command-Line Interface (CLI)

```bash
# Display help
julia --project=. scripts/run_movement.jl --help

# Default simulated analysis
julia --project=. scripts/run_movement.jl --simulate

# Snow crab analysis
julia --project=. scripts/run_movement.jl --snowcrab

# Select a TOML configuration explicitly
julia --project=. scripts/run_movement.jl --config configs/snowcrab.toml

# Snow crab with depth encoding (hsi_only default) and fine hexagonal mesh
julia --project=. scripts/run_movement.jl \
  --snowcrab --depth-range 25,400 --hex --hex-radius 5.0

# Use hard depth barrier
julia --project=. scripts/run_movement.jl \
  --snowcrab --depth-range 25,400 --depth-barrier-mode hard

# Full Bayesian ensemble with all diagnostics
julia --project=. scripts/run_movement.jl \
  --snowcrab --bayesian-ensemble --all-diagnostics --samples 500
```

### CLI Switch Table

| CLI Flag | Parameter |
|:---------|:----------|
| `--simulate` | `data_source = :simulate` |
| `--snowcrab` | Snow crab preset |
| `--config <path>` | TOML config file layered over the selected preset |
| `--model-mode <mode>` | `model_mode` |
| `--agent` | `model_mode = "agent"` |
| `--n-agents <N>` | `n_agent_projections` |
| `--persistence <k>` | `persistence` (directional persistence, $\kappa$) |
| `--rest-coupling <b>` | `rest_coupling` (habitat-coupled stay probability) |
| `--rest-advantage-form <f>` | `rest_advantage_form` (`difference`, `ratio`, `log_ratio`, `exp_difference`, `exp_ratio`, `exp_log_ratio`) |
| `--hex` / `--reshard-hex` | `reshard_hex = true` |
| `--hex-radius <km>` | `hex_radius_km` |
| `--depth-range <min,max>` | `depth_range = (min, max)` |
| `--depth-barrier-mode <mode>` | `depth_barrier_mode` (`:hsi_only`, `:bridge`, `:hard`) |
| `--hsi-ood-floor <v>` | `hsi_ood_floor` |
| `--max-paths <N>` | `max_paths` |
| `--astar` / `--viterbi` | `path_method` |
| `--smooth-paths` | `smooth_paths = true` |
| `--bayesian-ensemble` | `run_bayesian_ensemble = true` |
| `--adaptive-mesh` | `adaptive_mesh = true` |
| `--dynamic-kernels` | `dynamic_kernels = true` |
| `--hmm-smoothing` | `hmm_smoothing = true` |
| `--circuit` | `compute_circuit = true` |
| `--stochastic` | `compute_stochastic = true` |
| `--bottlenecks` | `compute_bottlenecks = true` |
| `--wavelets` / `--sgwt` | `compute_wavelets = true` |
| `--all-diagnostics` | Enables circuit, stochastic, bottlenecks, wavelets |
| `--samples <N>` | `n_samples` |
| `--warmup <N>` | `n_warmup` |
| `--output-dir <path>` | `output_dir` |
| `--quiet` / `-q` | `verbose = false` |

---

## 6. Scripting Examples

### Example 1: Basic Simulated Analysis

```julia
using MovementAnalysis

results = run_movement_analysis()
println("Paths reconstructed: ", length(results.paths))
```

### Example 2: Snow Crab with Seasonal HSI

```julia
using MovementAnalysis

params = merge(movement_parameters_snowcrab(), (
    depth_range      = (25.0, 400.0),
    depth_barrier_mode = :hsi_only,    # default; depth encoded via HSI
    n_samples        = 500,
    n_warmup         = 100,
    compute_validation = true,
    max_paths        = 40,
))

results = run_movement_analysis(params)
pa = results.validation_analyses
println("Brier score: ", pa.posterior_predictive.summary.brier_mean)
println("Connectivity matrix: ", size(pa.connectivity_matrix.connectivity_matrix))
```

### Example 3: Hard Depth Barrier with Bridge Fallback

```julia
using MovementAnalysis

# Use hard structural depth barrier with bridge edges for shallow corridors
params = merge(movement_parameters_snowcrab(), (
    depth_barrier_mode = :bridge,
    depth_range        = (25.0, 400.0),
))

results = run_movement_analysis(params)
```

### Example 4: Adaptive Mesh & Dynamic Corridors

```julia
using MovementAnalysis

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

### Example 5: Forward Projection with Habitat-Coupled Residency

```julia
using MovementAnalysis

params = merge(movement_parameters_default(), (
    model_mode         = "agent",
    n_agent_projections = 400,
    # Agents carry a heading, so they commit to a direction instead of turning
    # at every step.
    persistence        = 3.0,
    # Stay probability rises where a unit is locally better than its neighbours.
    rest_coupling      = 0.6,
    rest_advantage_form = :log_ratio,
))

results = run_movement_analysis(params)

# Unconditioned space use: which units an animal is likely to occupy at all,
# as opposed to the recapture-conditioned bridge distribution.
su = results.agent_space_use
top = sortperm(su.visit_probability; rev = true)[1:10]

# Interactive explorer: click a unit to project from it and accumulate
# visit frequency.
map_path = joinpath(params.output_dir, "forward_projection_map.html")
```

---

## 7. Agent-Based Model (ABM)

### 7.1 What it is, and what it is not

Despite the historical name, this component is a **forward projector**, not an
alternative model. It consumes the same pooled kernel as everything else,
introduces no parameters, and adds no likelihood term. The one thing it does that
nothing else in the codebase does is project **forward**:

- `reconstruct_paths_and_diagnostics` bridges release $\to$ recapture conditioned
  on **both** observed endpoints.
- The projector starts at a release unit and conditions on **neither** endpoint.

A projected track may therefore visit units a bridge is forced to avoid, which
makes it the only source of *unconditioned* space-use estimates. Agents are
independent: no interaction, memory, mortality, or agent-level state beyond
position. Section 7.4 lists what a genuinely agent-based formulation could add.

### 7.2 Functions

```text
simulate_agent_trajectories(n_agents, start_nodes, n_steps, transition_kernel;
                            seed, centroids, persistence, coord_space)
forward_project_agents(release_nodes, durations; n_agents, transition_kernel,
                        seed, centroids, persistence)
forward_space_use(trajectories, n_spatial) -> NamedTuple
```

`simulate_agent_trajectories` accepts either a single horizon or a **per-agent
vector** of horizons, so a run can inherit the empirical duration distribution
instead of an arbitrary fixed step count.

`forward_project_agents` draws start units **with replacement** from an empirical
release pool and horizons from an empirical duration pool. The agent count is
therefore a free design choice rather than a by-product of how many animals were
observed, and increasing it reduces Monte Carlo error without changing the
underlying distribution.

`forward_space_use` returns per-unit `visit_probability` and `mean_dwell_steps`.
These are the unconditioned counterparts to the bridge-conditioned residence
distribution, and are the natural companion to it.

### 7.3 Habitat-Coupled Residency

By default the stay probability is a single global constant `residence` applied
to every unit. That means an animal on a uniformly good patch has no more reason
to stay than one on a poor patch, and an animal whose entire neighbourhood is
worse than where it stands faces the same stay probability as one with much
better options.

`rest_coupling` makes the stay probability a function of **local habitat
advantage** over the unit's own neighbourhood. With `rest_coupling = 0.0` the
kernel is bit-for-bit unchanged, so this is a strict generalisation with an off
switch.

Let $\bar{h}_{N(i)}$ be the connectivity-weighted mean suitability of unit $i$'s
neighbours. Six measures are available through `rest_advantage_form`:

| Form | Advantage $A_i$ | How it enters the stay probability |
|:-----|:-----------------|:----------------------------------|
| `:difference` | $h_i - \bar{h}_{N(i)}$ | $\rho_i = \mathrm{clamp}(\rho + \beta A_i,\, 0,\, 0.999)$ |
| `:ratio` | $h_i / \bar{h}_{N(i)} - 1$ | as above |
| `:log_ratio` | $\log h_i - \log \bar{h}_{N(i)}$ | as above |
| `:exp_difference` | $h_i - \bar{h}_{N(i)}$ | $\rho_i = \mathrm{logistic}(\mathrm{logit}(\rho) + \beta A_i)$ |
| `:exp_ratio` | $h_i / \bar{h}_{N(i)} - 1$ | as above |
| `:exp_log_ratio` | $\log h_i - \log \bar{h}_{N(i)}$ | as above |

The additive forms are more interpretable; the exponential forms apply the coupling
as a bounded logit shift and so do not saturate at the $0.999$ ceiling, which
matters when $\beta$ is large. The ratio and log-ratio forms are scale-free and so
preferable when HSI is scaled differently between meshes.

**Guards.** Habitat is first passed through `sanitise_hsi`, which forces finite
values into $[0, 1]$ and replaces `NaN` / $\pm\infty$ with the mean of the finite
entries (or $0.0$ if none are finite). A unit with no usable neighbourhood gets
**zero** advantage, so its stay probability reverts to the global `residence`
rather than to an arbitrary number — a ratio is never formed against a zero or
non-finite denominator. `rest_coupling == 0` short-circuits *before* any
arithmetic, because $0 \times \infty$ is `NaN`. The logit shift is clamped to
$\pm 20$ and the result clamped to $[0, 0.999]$ as a final backstop.

### 7.4 Directional Persistence

The pooled kernel is *memoryless*: $P_{ij}$ depends only on the current unit, the
candidate, and the habitat field, so an animal that has just moved north-east is
given the same step distribution as one arriving from the south-west. Real
trajectories are directionally autocorrelated, and no first-order kernel can
represent that however its parameters are tuned.

`build_persistent_transition_kernel` builds a second-order chain over
(position, heading) states of size $S H$:

```math
T[(i,h),(j,h')] \propto P_{ij} \exp\left(\kappa \cos(\beta_{ij} - \theta_h)\right)
```

where $\beta_{ij}$ is the bearing of the step, $\theta_h$ the centre of heading
bin $h$, and $\kappa$ the `persistence` parameter. Heading turning and movement are
coupled through geometry, so a heading is only reachable if a neighbour lies in
that direction. $\kappa = 0$ makes the factor constant and the heading-uniform
unit marginal *exactly* the first-order kernel, so this too is a strict
generalisation. `simulate_agent_trajectories(persistence = ...)` applies the same
reweighting per agent, which is what makes individual trajectories actually turn.

`persistence_gain_report` scores held-out events over a grid of $\kappa$ so the
extra parameter can be judged rather than assumed; it forms $T^k$ explicitly and
is therefore limited to small graphs.

Note that integrating $\kappa$ into the fitted telemetry likelihood at full mesh
resolution is not currently affordable — the state space grows by a factor $H$ and
each likelihood evaluation would pay for $T^k$ — and is not attempted.

### 7.5 Interactive forward-projection explorer

`leaflet_forward_projection_map(P, au; hsi, n_paths, n_steps, seed, ...)` emits a
self-contained Leaflet application that embeds the kernel and walks it
client-side. Clicking a unit projects trajectories forward from it, and visit
counts accumulate across clicks into a per-unit expected visit probability, with
the polygon fill switching to that ramp. This is the interactive form of the
unconditioned space-use estimate, and it is the complement of the two-click
corridor explorer: every route that map draws is conditioned on arriving at the
chosen recapture unit, whereas this one conditions on nothing. Projection
randomness comes from a seeded in-page generator, so a given `seed`, `n_paths`,
and `n_steps` reproduce the same trajectories.

### 7.6 What a genuinely agent-based formulation could add

Assessed and deliberately not implemented, because each changes the model rather
than the plumbing:

- **Particle filter over the latent track.** `predict_path` returns a single
  Viterbi MAP route, so corridors currently rest on point estimates. Propagating
  a posterior over intermediate positions would give corridor uncertainty MAP
  decoding cannot. Partially overlaps the existing ensemble reconstruction.
- **Individual-level random effects.** A movement-type mixture is a real answer
  to the single pooled parameter set and is distinct from the demographic
  stratification removed in section 1, but it is weakly identifiable from a few
  hundred animals and can absorb model misspecification.
- **Density dependence, energetics, mortality, stage structure.** Not
  identifiable with this dataset.

---

## 8. Output Directory Structure

```text
output/
├── path_credible_intervals.csv               # Per-path length CIs & modal waypoints
├── stock_connectivity_summary.csv            # Regional connectivity point estimates
├── stock_connectivity_credible_intervals.csv # Regional connectivity 95% CIs
├── posterior_predictive_summary.txt          # Brier score & KL divergence
├── posterior_predictive_diagnostics.csv      # Per-draw PPC metrics
├── posterior_predictive_distributions.csv    # Observed vs. predicted recaptures
├── posterior_predictive_rank_histogram.csv   # Calibration histogram
├── movement_path_metrics.csv                 # Displacement, tortuosity, bearing
├── movement_tracks_animated.html             # Animated Leaflet trajectory playback
├── movement_corridors_heatmap.html           # Markov bridge transition corridors
├── forward_projection_map.html               # Click-to-project explorer ("agent" mode)
├── agent_projection_summary.csv              # Per-agent projection metrics
├── movement_domain_bottlenecks.html          # Pinch-points and bottleneck index
├── movement_current_density.html             # Circuit theory current flux
├── movement_stochastic_circuit.html          # Posterior current density with HSI error
├── movement_wavelet_dashboard.html           # Multi-scale Chebyshev SGWT wavelets
├── movement_posterior_uncertainty.html       # MCMC parameter KDEs & correlations
├── movement_network_flow.html                # Directed network flow graph
└── movement_summary_diagnostics.html         # Comprehensive diagnostics overview
```

---

## 9. Scientific References

1. **MovementAnalysis**: Choi, J. (2025). *Bayesian Spatio-Temporal Models for
   Marine Ecology*.
2. **Circuit Theory in Ecology**: McRae, B. H., Dickson, B. G., Keitt, T. H.,
   & Shah, V. B. (2008). Using circuit theory to model connectivity in ecology,
   evolution, and conservation. *Ecology*, 89(10), 2712–2724.
3. **Spectral Graph Wavelets**: Hammond, D. K., Vandergheynst, P., & Gribonval,
   R. (2011). Wavelets on graphs via spectral graph theory. *Applied and
   Computational Harmonic Analysis*, 30(2), 129–150.
4. **BayesShrink**: Chang, S. G., Yu, B., & Vetterli, M. (2000). Adaptive
   wavelet thresholding for image denoising and compression. *IEEE Trans. Image
   Processing*, 9(9), 1532–1546.
5. **Brier Score**: Brier, G. W. (1950). Verification of forecasts expressed in
   terms of probability. *Monthly Weather Review*, 78(1), 1–3.
6. **KL Divergence**: Kullback, S., & Leibler, R. A. (1951). On information and
   sufficiency. *Annals of Mathematical Statistics*, 22(1), 79–86.
7. **Bayesian Posterior Predictive Checks**: Gelman, A., Carlin, J. B., Stern,
   H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data
   Analysis* (3rd ed.). Chapman and Hall/CRC.
8. **Habitat-Corridor Theory**: McRae, B. H. (2006). A signed graph weighting
   function for biasing movement towards better habitat. *Ecography*, 29(1),
   51–60. Notes the *resistance* form $r_{ij} = r_0 + r_b \exp(-\beta h_{ij})$ used
   here for habitat-coupled residency, contrasted with the *conductance* form.
9. **Circuit Conductance**: Keitt, T. H. (2004). A landscape basis for resistance
   and conductance in ecological network analysis. *Proceedings of the National
   Academy of Sciences*, 101(17), 6174–6179.
10. **Correlated Random Walks**: Bergman, E. J., et al. (1999). Using multivector
    methods to analyze animal movement and habitat selection. *Ecography*, 22(4),
    370–384. Motivates the second-order (position, heading) chain used for
    directional persistence.
11. **Bugs and Recurrences in Stochastic Simulation**: Numerical Recipes / the
    standard guard practice of sanitising non-finite inputs before forming ratios;
    applied here in `sanitise_hsi` and the zero-coupling short circuit that avoids
    $0 \times \infty$.
