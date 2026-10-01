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
   â”‚      - Telemetry observation loading (empirical or simulated)
   â”‚      - Hexagonal domain resharding & depth preference encoding
   â”‚      - Optional adaptive multiresolution domain
   â–¼
[Phase 2] Bayesian Model Fitting
   â”‚      - Categorical pure-telemetry Markov likelihood (discrete-time)
   •      - Prior-scaled random-walk MH sampler (movement_sampler)
   â–¼
[Phase 3] Stochastic Transition Kernel Construction
   â”‚      - Group-stratified advection-diffusion-taxis extraction
   â”‚      - Time-varying (monthly) kernel construction from seasonal HSI
   â”‚      - Intermediate checkpoint written: movement_checkpoint.jld2
   â–¼
[Phase 4] Trajectory & Corridor Reconstruction
   â”‚      - A* least-cost paths, Viterbi dynamic programming, HMM smoothing
   â”‚      - Forward-backward Markov bridge corridor probability heatmaps
   â”‚      - Domain-wide transit bottlenecks B(u) = C(u) / deg(u)
   â–¼
[Phase 4b] Agent-Based Model
   â”‚      - Discrete-time agent simulation using fitted transition kernels
   â–¼
[Phase 5] Advanced Diagnostics & Validation
   â”‚      - Circuit theory electrical current density & pinch-points
   â”‚      - Individual path credible intervals
   â”‚      - Regional stock connectivity matrix & credible intervals
   â”‚      - Posterior predictive checks (Brier score, KL divergence)
   â”‚      - Optional full Bayesian MCMC ensemble path & corridor propagation
   â–¼
[Phase 6] Export & Leaflet Dashboards
          - Interactive HTML/SVG maps, animated tracks, and CSV summaries
          - Full results checkpoint written: movement_results_checkpoint.jld2
```

---

## 2. Pipeline Execution Phases

### Phase 1: Data Ingestion, Resharding, and Depth Encoding

Function: `load_movement_data(params)`

#### 1a. Observation Loading

Loads empirical mark-recapture data (e.g. Scotian Shelf snow crab) or generates
synthetic multi-segment telemetry histories with known ground truth. Each
observation pair records:

- `release`, `recapture`: mesh node indices.
- `k`: elapsed time steps (derived from `Î”t / dt` where `dt` is the time
  resolution, e.g. `1/365.25` for daily).
- `rel_time`: decimal-year timestamp of release (e.g. `2018.583 â‰ˆ Aug 2018`),
  used to select the correct monthly HSI slice during path reconstruction.
- `sex`, `mat`: biological group covariates.

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
through the habitat suitability index â€” the advective taxis term
$\exp(\gamma(H_j - H_i))$ strongly discourages transit through low-HSI
out-of-depth nodes without severing connectivity.

**Bridge method** (when `:bridge` is selected): for each out-of-depth marine
transit node $t$, bridge edges are added between every pair of its in-depth
marine neighbours $\{u, v\} \subseteq N_t$ not already adjacent in $W$. Cost is
$O(\text{nnz}(W_{\text{marine}}))$ â€” a single pass over existing edges, at most
$\binom{\deg(t)}{2}$ new edges per transit node.

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

Fits movement parameters via Turing.jl using a **prior-scaled random-walk MH
sampler** (`movement_sampler`). The proposal width is `mh_proposal_scale Ã— prior_sd`.
Standard `MH()` in Turing 0.49 proposes from the prior (zero acceptance under
sharp posteriors) and is not used. The transition kernel
`build_sparse_transition_kernel` is fully generic over its element type,
accepting `ForwardDiff.Dual` numbers for gradient-based sampling if needed.

The k-step transition cache (`kstep_transition_cache`) groups all observations
by release node and evaluates the iterate sequence $P^\top e_{\text{rel}}$
in a single sweep per node, reducing matrix-vector products by ~9Ã— over the
naive per-pair formulation.

#### Model Parameters

The model samples three physical parameters per group $g$:

- `velocity[g]` $\sim \text{Truncated-Normal}(0.3, 0.2; 0, 0.95)$
- `diffusion[g]` $\sim \text{Truncated-Normal}(0.1, 0.2; 0, \infty)$
- `gamma[g]` $\sim \text{Normal}(1.0, 1.0)$: habitat gradient responsiveness

Derived via `movement_alpha_rho(velocity, diffusion)`:

$$\alpha_g = \frac{v_g}{v_g + D_g + \epsilon}, \qquad
  \rho_g = \frac{1}{1 + v_g + D_g + \epsilon}$$

This is the single shared derivation used by all model variants and by
`extract_transition_kernels`.

#### Chain Health

`report_chain_health` prints acceptance rate and distinct-value count after
every model fit:

```
telemetry: acceptance=0.4, distinct values=6/15
```

A chain with one distinct value is frozen and all downstream results are
unreliable. ESS and R-hat are not yet computed â€” tracked in `todo.md` Â§0.2.

#### Likelihood (Discrete-Time Telemetry)

$$\log \mathcal{L}(\mathbf{y}_{1:T} \mid \theta) =
  \sum_{t=1}^{T-1} \log [T_g^{k_t}]_{y_t, y_{t+1}}$$

**HSI in MCMC**: uses climatological mean `hsi_vec`. Time-varying HSI is
applied post-estimation during path reconstruction (Phase 4).

#### Joint Density-Movement Model

The survey density likelihood is **not yet implemented** â€” both joint models
reduce to the telemetry-only model with unused arguments. A `@warn` is emitted
on construction. See `todo.md` Â§1.6.

---

### Phase 3: Transition Kernel Construction

Function: `extract_transition_kernels(loaded, fitted, params)`

$$T_g = (1 - \rho_g) \left[ (1 - \alpha_g) T_{\text{diff}} +
    \alpha_g A_g(H) \right] + \rho_g I$$

where $T_{\text{diff}, ij} = W_{ij} / \sum_k W_{ik}$ and
$A_{g, ij} \propto W_{ij} \exp(\gamma_g (H_j - H_i))$.

An intermediate checkpoint (`movement_checkpoint.jld2`) is written here,
storing `loaded`, `fitted`, and `kernels`. `--resume=true` loads this
checkpoint to skip Phases 1â€“3.

---

### Phase 4: Trajectory & Corridor Reconstruction

Function: `reconstruct_paths_and_diagnostics(loaded, kernels, params)`

For each observation, the release time `rel_time` selects the monthly HSI
slice via `_resolve_hsi_for_time(loaded, row.rel_time)`, falling back to
`hsi_vec` when no monthly data is available.

**Path methods**: `:astar` (A* with Euclidean heuristic), `:viterbi` (dynamic
programming), HMM smoothing over multi-segment capture histories.

**Markov bridge corridors**:

$$\mathbb{P}(X_\tau = j \mid X_0 = u, X_k = v) =
  \frac{[T^\tau]_{uj} [T^{k-\tau}]_{jv}}{[T^k]_{uv}}$$

Corridor matrices are summed into an aggregate visitation heatmap
(`movement_corridors_heatmap.html`). The bottleneck SE field is rendered
separately as `movement_bottleneck_uncertainty.html`.

---

### Phase 4b: Agent-Based Model

Activated by including `:agent` in `model_modes`. Runs up to 100 agents for
50 steps from observed release nodes using fitted transition kernels. Exports:

- `movement_agent_trajectories.html`: Leaflet track map per agent.
- `movement_agent_visit_frequency.csv`: Mean node visit frequency.

---

### Phase 5: Advanced Diagnostics & Validation

Functions: `compute_advanced_diagnostics`, `execute_validation_analyses`

**Circuit theory** (`--diagnostics=circuit`): solves $Lv = I_{\text{ext}}$
on the graph with conductances $C_{ij} = W_{ij} \sqrt{H_i H_j}$ to identify
migratory pinch-points.

**Validation** (`--diagnostics=validation`):
1. Per-path credible intervals â†’ `path_credible_intervals.csv`
2. Regional stock connectivity â†’ `stock_connectivity_summary.csv`,
   `stock_connectivity_credible_intervals.csv`,
   `movement_stock_connectivity.html`
3. Posterior predictive checks â†’ `posterior_predictive_summary.txt`,
   `movement_ppc_summary.html`

**Bayesian ensemble** (`--diagnostics=bayesian_ensemble`): MCMC ensemble path
& corridor propagation over posterior parameter draws.

Spectral graph wavelets were removed from the package.

---

### Phase 6: Interactive Dashboard Export

Function: `export_dashboards(...)`

Generates standalone HTML/SVG dashboards. After Phase 6 completes, writes
`movement_results_checkpoint.jld2` containing all Phase 4â€“5 results, enabling
`--figures-only=true` to regenerate all dashboards without recomputing anything.

---

## 3. HSI: Static vs. Time-Varying

| Stage | Static HSI | Monthly HSI |
|:------|:-----------|:------------|
| MCMC estimation (Phase 2) | `hsi_vec` | `hsi_vec` (climatological mean, by design) |
| Kernel construction (Phase 3, posterior mean) | `hsi_vec` | `hsi_vec` |
| Path reconstruction (Phase 4, per observation) | `hsi_vec` | `monthly_hsi[:, col]` for release (year, month) |
| Corridor / HSI error propagation | `hsi_vec` | `monthly_hsi[:, col]` for release time |

`_resolve_hsi_for_time(loaded, t_decimal)` decomposes `t_decimal` into
`(year, month)` and looks up the monthly column, falling back to `hsi_vec`.

---

## 4. Configuration Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `data_source` | `Symbol` | `:simulate` | `:simulate` or `:snowcrab` |
| `model_modes` | `Vector{Symbol}` | `[:telemetry]` | Active model types (`:telemetry`, `:telemetry_and_survey`, `:agent`) |
| `reshard_hex` | `Bool` | `false` | Reshard domain to fine hexagons |
| `hex_radius_km` | `Float64` | `10.0` | Cell radius for resharded hexagons (km) |
| `depth_range` | `Tuple / Nothing` | `nothing` | Depth window `(min, max)` (m) |
| `depth_barrier_mode` | `Symbol` | `:hsi_only` | `:hsi_only`, `:bridge`, or `:hard` |
| `hsi_ood_floor` | `Float64` | `0.01` | HSI floor for out-of-depth nodes |
| `max_paths` | `Int` | `25` | Maximum trajectories to reconstruct |
| `path_methods` | `Vector{Symbol}` | `[:astar]` | `:astar`, `:viterbi` |
| `diagnostics` | `Vector{Symbol}` | see config | `:circuit`, `:stochastic`, `:bottlenecks`, `:validation`, `:bayesian_ensemble` |
| `n_samples` | `Int` | `200` | MCMC posterior draw count |
| `n_warmup` | `Int` | `100` | MCMC warmup iterations |
| `mh_proposal_scale` | `Float64` | `0.05` | Random-walk proposal width (Ã— prior SD) |
| `seed` | `Int` | `42` | RNG seed |
| `n_ensemble` | `Int` | `50` | Posterior draws for ensemble corridor propagation |
| `hsi_se` | `Float64` | `0.08` | HSI observation standard error |
| `propagate_hsi_error` | `Bool` | `true` | Propagate HSI uncertainty into corridors |
| `render_html` | `Bool` | `true` | Export Leaflet HTML dashboards |
| `output_dir` | `String` | `"output"` | Output directory |
| `resume_from_checkpoint` | `Bool` | `false` | Load `movement_checkpoint.jld2`, skip Phases 1â€“3 |
| `figures_only` | `Bool` | `false` | Load `movement_results_checkpoint.jld2`, skip to Phase 6 |
| `dark_mode` | `Bool` | `false` | Dark theme in dashboards |
| `cmap` | `Symbol` | `:viridis` | Colour palette for choropleth maps |
| `species_name` | `String` | `"Animal"` | Display name in reports |
| `verbose` | `Bool` | `true` | Progress logging |
| `region_labels` | `Vector{String}` | `[]` | Region names for connectivity analysis |

---

## 5. Command-Line Interface (CLI)

```powershell
# Full run
julia --project=. scripts/run_movement.jl --config=configs/snowcrab.toml

# Skip MCMC â€” load movement_checkpoint.jld2, run Phases 4â€“6
julia --project=. scripts/run_movement.jl --config=configs/snowcrab.toml --resume=true

# Regenerate all HTML dashboards only (requires prior successful run)
julia --project=. scripts/run_movement.jl --config=configs/snowcrab.toml --figures-only=true

# Tweak styling and regenerate
julia --project=. scripts/run_movement.jl --config=configs/snowcrab.toml `
    --figures-only=true --dark-mode=true --cmap=plasma

# Write results checkpoint without rendering HTML
julia --project=. scripts/run_movement.jl --config=configs/snowcrab.toml `
    --resume=true --render-html=false
```

### CLI Flag Table

| CLI Flag | Config field |
|:---------|:-------------|
| `--config=<path>` | â€” (TOML config file) |
| `--model-modes=<list>` | `model_modes` |
| `--diagnostics=<list>` | `diagnostics` |
| `--path-methods=<list>` | `path_methods` |
| `--max-paths=<N>` | `max_paths` |
| `--samples=<N>` | `n_samples` |
| `--warmup=<N>` | `n_warmup` |
| `--mh-proposal-scale=<v>` | `mh_proposal_scale` |
| `--n-draws=<N>` | `n_ensemble` |
| `--output-dir=<path>` | `output_dir` |
| `--resume=<bool>` | `resume_from_checkpoint` |
| `--figures-only=<bool>` | `figures_only` |
| `--render-html=<bool>` | `render_html` |
| `--dark-mode=<bool>` | `dark_mode` |
| `--cmap=<name>` | `cmap` |
| `--species-name=<name>` | `species_name` |
| `--quiet=true` | `verbose` |

---

## 6. Checkpointing

| File | Contents | Written | Used by |
|:-----|:---------|:--------|:--------|
| `movement_checkpoint.jld2` | `loaded`, `fitted`, `kernels` | End of Phase 3 | `--resume=true` |
| `movement_results_checkpoint.jld2` | All Phase 4â€“5 results | End of Phase 6 | `--figures-only=true` |

To bootstrap the full results checkpoint from an existing MCMC run:

```powershell
julia --project=. scripts/run_movement.jl --config=configs/snowcrab.toml --resume=true
```

---

## 7. Scripting Examples

### Example 1: Basic Analysis

```julia
using MovementAnalysis
results = run_movement_analysis()
println("Paths reconstructed: ", length(results.paths))
```

### Example 2: Snow Crab with Validation

```julia
using MovementAnalysis

params = merge(movement_parameters_snowcrab(), (;
    depth_range  = (25.0, 400.0),
    n_samples    = 500,
    n_warmup     = 100,
    max_paths    = 40,
    diagnostics  = [:validation, :bottlenecks, :circuit],
))

results = run_movement_analysis(params)
pa = results.validation_analyses
println("Brier score: ", pa.posterior_predictive.summary.brier_mean)
```

### Example 3: Regenerate Figures Only

```julia
using MovementAnalysis

params = merge(movement_parameters_snowcrab(), (;
    figures_only = true,
    dark_mode    = true,
    cmap         = :plasma,
))
run_movement_analysis(params)
```

---

## 8. Output Directory Structure

```text
output/
â”œâ”€â”€ movement_checkpoint.jld2                   # Intermediate: loaded + fitted + kernels
â”œâ”€â”€ movement_results_checkpoint.jld2           # Full results: all Phase 4â€“5 outputs
â”‚
â”œâ”€â”€ path_credible_intervals.csv
â”œâ”€â”€ stock_connectivity_summary.csv
â”œâ”€â”€ stock_connectivity_credible_intervals.csv
â”œâ”€â”€ posterior_predictive_summary.txt
â”œâ”€â”€ movement_path_metrics.csv
â”œâ”€â”€ movement_agent_visit_frequency.csv
â”‚
â”œâ”€â”€ movement_tracks_animated.html
â”œâ”€â”€ movement_corridors_heatmap.html            # Aggregate Markov bridge corridor visitation
â”œâ”€â”€ movement_bottleneck_uncertainty.html       # Bottleneck index SE
â”œâ”€â”€ movement_domain_bottlenecks.html
â”œâ”€â”€ movement_current_density.html
â”œâ”€â”€ movement_stochastic_circuit.html
â”œâ”€â”€ movement_posterior_uncertainty.html
â”œâ”€â”€ movement_posterior_path_ensemble.html
â”œâ”€â”€ movement_network_flow.html
â”œâ”€â”€ movement_agent_trajectories.html
â”œâ”€â”€ movement_stock_connectivity.html
â”œâ”€â”€ movement_ppc_summary.html                  # PPC Brier/KL traces + distribution overlay
â””â”€â”€ movement_summary_diagnostics.html
```

---

## 9. Scientific References

1. **MovementAnalysis**: Choi, J. (2025). *Bayesian Spatio-Temporal Models for
   Marine Ecology*.
2. **Circuit Theory**: McRae, B. H. et al. (2008). Using circuit theory to model
   connectivity in ecology. *Ecology*, 89(10), 2712â€“2724.
3. **Brier Score**: Brier, G. W. (1950). *Monthly Weather Review*, 78(1), 1â€“3.
4. **KL Divergence**: Kullback, S. & Leibler, R. A. (1951). *Annals of
   Mathematical Statistics*, 22(1), 79â€“86.
5. **Bayesian Data Analysis**: Gelman, A. et al. (2013). *Bayesian Data Analysis*
   (3rd ed.). Chapman and Hall/CRC.
6. **Uniformization**: Grassmann, W. K. (1977). Transient solutions in Markovian
   queuing systems. *Computers & Operations Research*, 4(1), 47â€“53.

