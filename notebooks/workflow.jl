### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 3f2c8a41-6b57-4d1e-9a83-2c5b7e1d0f44
# ╠═╡
using Pkg

# ╔═╡ 9d4e1b72-8c05-4a63-b7e1-6f0d3a8c5e21
# ╠═╡
md"""
# MovementAnalysis — the workflow, its assumptions, and its references

This notebook walks the whole analysis pipeline once, one panel at a time, on the Snow Crab
configuration in `configs/snowcrab.toml`.

Each step is an ordinary package call, so a cell can be re-run on its own while you inspect what
it returned — that is what Pluto buys you over a script. The markdown between the cells is the
documentation: **what the panel does, which assumptions it bakes in, which choices the
configuration makes, and where each assumption comes from**.

### What the pipeline is

Mark-recapture telemetry and a habitat-suitability time series are turned into a **hexagonal
lattice**, a **transition kernel** between lattice cells, **least-cost paths** through that
kernel, **electrical-circuit diagnostics** of connectivity, and finally **HTML dashboards**. The
scientific chain is: an animal released in one cell is recaptured in another, so the measured
transitions estimate a probability of moving between any two cells; a path through that landscape
is a plausible route; and resistance to flow identifies where movement is squeezed.

### Runtime

The heavy cells are marked. With the Snow Crab configuration the mesh construction, the Turing fit
(`n_samples = 500`, `n_warmup = 100`), and the circuit diagnostics dominate. To go fast, lower
`n_samples` in the cell that builds `params`, or set `model_modes` to a smaller list — the panel
skips whatever was not asked for and says so at the end of the run.
"""

# ╔═╡ 2b7f5d93-1e48-4c0a-8f6d-9a1c4e7b0d35
# ╠═╡
md"""
## Running this notebook

Launch Pluto from an environment that already has it (the global environment, or one where you
have added `Pluto`) and open this file from the notebook list:

```
julia -e 'using Pluto; Pluto.run()'
```

The first three cells pin the working directory, activate **this package's environment**, and
load the package, so the `data/tagging.jld2` and `configs/snowcrab.toml` paths in the
configuration resolve relative to the package rather than to wherever Pluto was started.

`Pluto` is deliberately **not** a dependency of this package: a notebook is how you look at the
work, not something the run needs. Its version and HTTP server have historically conflicted with
this package's pinned `Zarr`/`YAXArrays` stack, so it is installed only where the notebooks run.

One consequence of the format matters here: a cell defines something and every later cell sees
it, and re-running a cell re-runs everything downstream. So changing `n_samples` and re-running
re-fits the model and re-derives the kernels, the paths and the figures, without re-reading the
input data.
"""

# ╔═╡ 5c1d8e64-2a79-4b5f-9e07-3d6a1c8b4f29
# ╠═╡
cd(joinpath(dirname(@__DIR__), "MovementAnalysis"))

# ╔═╡ 7a3f0c15-9b42-4e8d-a6f1-2b8c5d0e7a93
# ╠═╡
Pkg.activate(@__DIR__)

# ╔═╡ 8e2b6d47-3f19-4c7b-8d5a-1f4c9b2e0a36
# ╠═╡
using MovementAnalysis

# ╔═╡ 1d9c4a28-7e63-4b0f-9c52-8a3d6f1b5e04
# ╠═╡
md"""
## The six panels

`MovementAnalysis` runs as a sequence of panels, each a function in `src/pipeline.jl`, each
taking what the previous one produced. Running them one at a time here rather than calling
`run_movement_analysis` end to end is what makes the intermediate state inspectable.

| # | Function | What it does | Assumption it encodes |
|---|---|---|---|
| 1 | `load_movement_dataset` | reads telemetry, HSI and the spatial-unit file; builds the lattice and maps observations onto it | the domain is the tagging extent plus `bbox_padding_deg`; land is where the raster says land |
| 2 | `fit_movement_models` | fits the categorical mark-recapture transitions (Turing MH), or the agent model | a recapture is an independent draw from a transition kernel |
| 3 | `extract_transition_kernels` | turns the posterior into a cell-to-cell transition matrix | dispersal is log-normal in distance with a residence term |
| 4 | `reconstruct_paths_and_diagnostics` | A* least-cost paths, stochastic ensembles, corridors, bottlenecks | movement cost is −log(transition probability) |
| 5 | `compute_advanced_diagnostics` | electrical-circuit theory: current density and pinch points | the landscape is a resistor network; current ∝ probability of passage |
| 6 | `execute_validation_analyses`, `export_dashboards` | credible intervals, connectivity matrices, posterior predictive checks, HTML | a model that cannot predict held-out transitions is not reported as if it could |

The configuration below is the one that drives every panel. It is `configs/snowcrab.toml`, parsed
through the one canonical path: package defaults, then this file, then any command-line flags.
"""

# ╔═╡ 4f8a2b61-0c37-4d5e-8b9a-6c2f7d1e0b53
# ╠═╡
config = MovementAnalysis.load_config(config_path = "configs/snowcrab.toml")

# ╔═╡ 6b0e3d92-5a18-4c6f-8d04-7e1b9c4a2f68
# ╠═╡
md"""
### What the configuration fixes, and why

Reading the resolved values next to their meaning is the point of the notebook.

**The lattice.** `hex_radius_km = 5.0` with `reshard_hex = false`: cells are 5 km hexagonal
prisms. Hexagons rather than squares because a hexagon has equal distance to all neighbours, so
"adjacent" does not depend on which way you look — a square lattice distinguishes diagonal and
orthogonal neighbours, and that anisotropy propagates into the kernel. The cost is that the
species-polygon units must be resharded onto the new lattice, and resharding is only approximate:
a unit boundary that falls inside a prism is split, and the split is where the HSI average is
least trustworthy.

**The domain.** Not declared: it is derived from the extent of the tagging data plus
`bbox_padding_deg = 0.25`. Two assumptions follow. First, the animals were observed everywhere
they go, so the observed extent is the occupied extent — a tag released outside the tagging
area is not in the data by construction. Second, the padding is large enough that no observation
lands on a lattice edge, so `map_point_to_units` never has to decide what a recapture outside the
mesh means.

**Land.** `land_source = "landmask"` reads a global land/sea raster at
`land_mask_grid_minutes = 5.0`, and `configs/geo/sable_island.geojson` is *added* to it. Sable
Island is a 42 km sandbar under 1.5 km wide, and every bundled GMT raster classifies it as
water — probed at 5, 2.5 and 1.25 arc-minutes, all three read water — so no resolution of that
mask resolves it. A cell classified as water that is land matters twice: larvae cannot be released
there, and a least-cost path may cross it. The geojson is the documented workaround, not a
convenience.

**Depth.** `depth_range = [25.0, 350.0]` is the principal benthic habitat band of the snow crab
stock, and `depth_barrier_mode = "hsi_only"` encodes depth preference entirely through the HSI
rather than as a second mask. That is a deliberate choice: a hard depth mask would make cells
outside the band impassable, which reads as a cliff rather than as preference, and a cliff in a
transition kernel produces paths that hug the boundary of allowed water.

**The models.** `model_modes = ["telemetry", "agent"]`. The survey-dependent approach is absent
because this dataset carries no paired survey-density file, so it is listed out rather than left
to fail — asking for one without the file is skipped and reported, not an error.

**The habitat.** `use_geodata_habitat = true` queries real physical habitat through **GeoData**
instead of the static HSI alone: bottom temperature from Copernicus GLORYS12V1
(`temperature_source`), bathymetry from ETOPO 2022 (`bathymetry_source`), and
`hsi_dynamic_method = "glorys_monthly"` for the time-varying series. The assumption is that the
Copernicus reanalysis is a valid description of the animals' thermal environment — it is a
physical model assimilating observations, not an observation, and its resolution is coarser than
the lattice.

**Inference.** `n_samples = 500`, `n_warmup = 100`, `mh_proposal_scale = 0.05`. The sampler is an
explicit prior-scaled random walk, not the default Turing `MH()`: on a dataset this size the
default draws proposals from the prior and accepts none of them, so an unmodified sampler would
return the prior and look like a posterior. `hsi_se = 0.08` propagates HSI uncertainty forward
with `propagate_hsi_error = true` and `n_stochastic_draws = 100`.
"""

# ╔═╡ 9c5b8e13-2d74-4a6f-8e05-1b3c7f0d9a62
# ╠═╡
md"""
## Panel 1 — ingestion and the lattice

`load_movement_dataset` is the only panel that touches the input files. It reads the
mark-recapture events, derives the domain, builds the hex prism lattice, classifies each cell as
land or water, and maps every observation onto a cell.

Everything downstream is built on `n_units` cells and a sparse weight matrix `W` describing which
cells are neighbours. Two cells are neighbours if their hexagons share an edge, and `W` holds
either the distance or a weight between them — which one is what the kernel step converts into a
probability of movement.
"""

# ╔═╡ 3e7c1f95-8b04-4d2a-9f6c-5a0d3b8e1c74
# ╠═╡
loaded = MovementAnalysis.load_movement_dataset(
    radius_km       = 15.0,
    time_interval   = :daily,
    verbose         = true,
    tagging_file    = "data/tagging.jld2",
    hsi_file        = "data/hsi.jld2",
    sppoly_file     = "data/sppoly.jld2",
)

# ╔═╡ 8f2d6c40-1b93-4e7a-8305-6d9a4f1c0e27
# ╠═╡
md"""
### What came back, and what to check

`loaded` is a `NamedTuple`. The fields that matter downstream:

- `obs_df` — the recapture events, each with the cell index it was mapped to. If this is empty,
  every later panel is fitting nothing.
- `centroids_km`, `centroids_lonlat` — where each cell is, in the two coordinate systems the
  paths are computed in. `lonlat_to_xy_km` converts between them about one centre, which is why
  the conversion is only accurate near the centre of the domain.
- `n_units`, `land_mask`, `W` — the lattice itself.
- `hsi_vec` — suitability per cell, flattened from the monthly field.

The land/water split is the one to look at: a cell the raster calls land is never a source, a
destination, or a step on a path. An error here is invisible in the plots — it removes habitat,
and paths route around the missing region exactly as they would around a real coastline.
"""

# ╔═╡ 2a9e5b73-4c08-4f6d-9b2a-7e1c5d0a3f86
# ╠═╡
md"""
## Panel 2 — fitting the transition model

`fit_movement_models` is where the posterior comes from. For `model_modes` containing
`telemetry` this is a categorical model over cell-to-cell transitions, fitted with Turing's MH
sampler: the likelihood of a recapture in cell *j* given a release in cell *i* is the transition
probability, and the parameters of the dispersal and residence terms are inferred from the
observed pairs.

The assumption worth stating plainly: **a recapture is an exchangeable draw from the kernel**. If
the animals were observed on a schedule, or released as cohorts, the transitions are not
independent and the posterior is optimistic about how well-determined the kernel is. The
`ref_doy`/`time_interval` options handle seasonal structure by treating each day (or month) as
its own transition, which weakens that assumption to *conditionally* exchangeable given season.
"""

# ╔═╡ 7d1c4e98-3b52-4a8f-8c6d-0f4b9e2a1c53
# ╠═╡
params = config

# ╔═╡ 4b6a0d27-9e81-4c5a-8b7f-3f2e8a1d5c04
# ╠═╡
fitted = MovementAnalysis.fit_movement_models(loaded, params)

# ╔═╡ 6e3b8a15-2c47-4d9f-8a5b-1c0f7d4e2a83
# ╠═╡
md"""
## Panel 3 — the transition kernel

`extract_transition_kernels` converts the posterior into the object the paths are built on: a
sparse *n_units × n_units* matrix whose entry *(i, j)* is the probability that an animal in cell
*i* is in cell *j* next.

This is the panel where the science is decided. The kernel is an **α-stable dispersal term with a
residence term** — `alpha` controls how heavy-tailed the dispersal distance distribution is, and
`rho` is the probability of staying put. A kernel with a heavy tail allows the long movements that
connect distant parts of the shelf and is what makes the connectivity matrix non-zero at range;
a kernel that is mostly residence produces paths that wander locally and never leave the home
range. `posterior_predictive_check` in panel 6 is the test of whether the fitted pair actually
predicts the transitions that were not used to fit it.
"""

# ╔═╡ 9a4c0b83-6d15-4e7b-8f3a-2b1d8e5c0a47
# ╠═╡
kernels = MovementAnalysis.extract_transition_kernels(loaded, fitted, params)

# ╔═╡ 1f8d3b64-0a29-4c6f-9e8b-5a2c7d0b4e13
# ╠═╡
md"""
## Panel 4 — paths and corridors

`reconstruct_paths_and_diagnostics` answers "if an animal moved from here to there, how did it
get there". Three families of path, and the difference between them is the difference between a
cheapest route and a distribution of routes:

- **A\*** least-cost paths (`astar_least_cost_path`) — the single cheapest route given a cost
  derived from −log(transition probability). Deterministic, and the cheapest route in a landscape
  with two similar corridors is an arbitrary choice between them.
- **Stochastic ensembles** (`astar_stochastic_least_cost_path`, `identify_stochastic_pinchpoints`)
  — many paths drawn with the uncertainty of the kernel propagated in. The spread is the answer
  to "how much do I believe this specific route".
- **Corridors** (`predict_corridor`, `smooth_marine_path`) — the envelope of those paths rather
  than any one of them, smoothed by line-of-sight against the land mask, because a least-cost
  path through a hex lattice follows cell centres and kinks at every boundary crossing.
"""

# ╔═╡ 5c2f9a47-8b61-4d3e-9a0c-7e4b1d8f5a26
# ╠═╡
path_res = MovementAnalysis.reconstruct_paths_and_diagnostics(loaded, kernels, params)

# ╔═╡ 3a8e1d95-7c04-4b6a-8f2d-9c5a0e3b1d68
# ╠═╡
md"""
## Panel 5 — circuit diagnostics

`compute_advanced_diagnostics` treats the landscape as a resistor network: each cell is a node,
each neighbour connection is a resistor whose conductance is the transition probability, and the
result is a map of **current density**. Where current is forced through a narrow passage, the
current density is high and that passage is a **pinch point** — the assumption being that
probability of passage behaves like electrical current, which is the circuit-theory result
(McRae et al. 2008) and holds because both are governed by a harmonic function on a graph.

Two things the map does *not* say: it does not say how many animals pass through a pinch point,
only how much of the flow is obliged to. And it depends on where nodes are grounded — a
"source" node and a "sink" node are chosen, so the current map is conditional on that pair.
"""

# ╔═╡ 8b4c7f12-5e93-4a6d-8c0f-2d9e5b3a1c74
# ╠═╡
diagnostics = MovementAnalysis.compute_advanced_diagnostics(loaded, path_res, params)

# ╔═╡ 2f6a9d38-1b47-4c8a-9e1d-7b3c0f5a2e85
# ╠═╡
md"""
## Panel 6 — validation and the dashboards

`execute_validation_analyses` produces the credible intervals, the stock connectivity matrices,
and the posterior predictive check. The predictive check is the one to read first: it compares the
distribution of summary statistics of held-out transitions against the same statistics simulated
from the fitted kernel. If they disagree, the kernel is wrong in a way the posterior's own width
does not reveal.

`export_dashboards` writes the HTML. It is a separate function rather than a side effect of the
panels because rendering is the slow part of re-plotting, and the run can be re-rendered from the
results checkpoint without re-fitting — which is the `--figures-only` route in the CLI.
"""

# ╔═╡ 6c0b3e81-4a25-4d9f-8b6c-1e8f2a7d0b53
# ╠═╡
validation = MovementAnalysis.execute_validation_analyses(loaded, fitted, kernels, params)

# ╔═╡ 9e5d1a47-2c38-4b0a-8f7d-5a3e9c6b1d02
# ╠═╡
md"""
## Assumptions, in one list

Everything below is a choice this pipeline makes, not a fact it discovers. Each is visible in a
resolved configuration value, so changing one is a line in the TOML rather than an edit to the
code.

1. **The observed extent is the occupied extent** — the domain is the tagging extent plus
   padding, so the analysis says nothing about animals outside the tagging area.
2. **Hexagonal cells are an adequate lattice** — 5 km prisms average habitat, release sites and
   recapture sites within a cell. Any structure finer than 5 km is not resolved, and a
   resharded boundary is the least reliable part of the HSI field.
3. **Land is a binary mask** — from a global raster at 5 arc-minutes, with the Sable Island
   overlay. A cell is water or land; there is no partial coast, and no seasonal drying or
   sea-ice.
4. **Depth is a preference, not a wall** — `depth_barrier_mode = "hsi_only"`, so unsuitable depth
   is discouraged through the HSI. Crossing the band is possible, at a cost.
5. **Recaptures are exchangeable given season** — the categorical likelihood treats each
   transition as an independent draw. Tag design, cohort release and observation schedules are
   not modelled.
6. **The kernel is α-stable dispersal plus residence** — a parametric form for how far animals
   move. A fat tail is what connects distant regions; if the true movement has a different shape,
   the connectivity matrix inherits the wrong shape.
7. **GLORYS12V1 is the animals' thermal environment** — suitability is queried from a Copernicus
   reanalysis, which is a physical model assimilating observations rather than an observation, at
   resolution coarser than the lattice.
8. **Cost is −log(transition probability)** — a path is cheap where movement is probable, and
   there is no third axis of cost (e.g. no wind, no predation risk) beyond depth and suitability.
9. **Current is proportional to probability of passage** — the circuit-theory analogy, exact for
   harmonic functions on a graph and conditional on the grounding nodes chosen.
10. **The posterior is only as good as the kernel** — the predictive check in panel 6 is the
    guard, and it is a diagnostic rather than a gate: the dashboard is exported either way.

## References

1. **MovementAnalysis** — Choi, J. (2025). *Bayesian Spatio-Temporal Models for Marine Ecology*.
2. **Circuit theory for connectivity** — McRae, B. H. et al. (2008). Using circuit theory to
   model connectivity in ecology. *Ecology*, 89(10), 2712–2724.
3. **Brier score** — Brier, G. W. (1950). Verification of forecasts expressed in terms of
   probability. *Monthly Weather Review*, 78(1), 1–3. (Used by the posterior predictive check.)
4. **KL divergence** — Kullback, S. & Leibler, R. A. (1951). On information and sufficiency.
   *Annals of Mathematical Statistics*, 22(1), 79–86. (Used to compare simulated against
   observed transition distributions.)
5. **Uniformization** — Grassmann, W. K. (1977). Transient solutions in Markovian queuing
   systems. *Computers & Operations Research*, 4(1), 47–53. (Used for multi-step transition
   probabilities.)
6. **Bayesian data analysis** — Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.).
   Chapman and Hall/CRC.
7. **Hexagonal lattices in ecology** — Sahr, K. (2011). Hexagonal discrete global grid systems
   for geospatial computing. *Archives of Photogrammetry, Cartography and Remote Sensing*, 22,
   363–376. (Why hexagons rather than squares.)
8. **Least-cost paths** — Etherington, T. R. (2016). Least-cost modelling and landscape
   ecology. *Springer*.
"""

# ╔═╡ 4a7c2e95-0b18-4e6d-9f3a-8c1b5d0e3a72
# ╠═╡
md"""
## Where the numbers live

`output_dir` (default `output`) holds the artefacts of a run: the checkpoints, the connectivity
matrices as CSV, and the HTML dashboards. The names are listed in the pipeline guide
(`docs/movement_analysis.md` §8). Two things are worth knowing about them:

- **The checkpoints are the resume route.** `movement_checkpoint.jld2` and
  `movement_results_checkpoint.jld2` let a run start after the expensive panels, and
  `--figures-only` re-renders dashboards from the results checkpoint without re-fitting.
- **`report_panel_skips()`** prints whatever was skipped. A skipped panel is not a failure —
  asking for an approach this dataset cannot support, such as the survey-dependent model with no
  survey file, is reported rather than silently dropped.
"""

# ╔═╡ 3f5a8c27-1e94-4b7d-8a0c-6d3b2e9f1a58
# ╠═╡
md"""
_Published by Pluto — see [the Pluto documentation](https://featured.plutojl.org/basics/notebooks)
for how notebooks are structured and shared._
"""

# ╔═╡ Cell order:
# ╠═╡ 3f2c8a41-6b57-4d1e-9a83-2c5b7e1d0f44 => 1
# ╠═╡ 9d4e1b72-8c05-4a63-b7e1-6f0d3a8c5e21 => 2
# ╠═╡ 2b7f5d93-1e48-4c0a-8f6d-9a1c4e7b0d35 => 3
# ╠═╡ 5c1d8e64-2a79-4b5f-9e07-3d6a1c8b4f29 => 4
# ╠═╡ 7a3f0c15-9b42-4e8d-a6f1-2b8c5d0e7a93 => 5
# ╠═╡ 8e2b6d47-3f19-4c7b-8d5a-1f4c9b2e0a36 => 6
# ╠═╡ 1d9c4a28-7e63-4b0f-9c52-8a3d6f1b5e04 => 7
# ╠═╡ 4f8a2b61-0c37-4d5e-8b9a-6c2f7d1e0b53 => 8
# ╠═╡ 6b0e3d92-5a18-4c6f-8d04-7e1b9c4a2f68 => 9
# ╠═╡ 9c5b8e13-2d74-4a6f-8e05-1b3c7f0d9a62 => 10
# ╠═╡ 3e7c1f95-8b04-4d2a-9f6c-5a0d3b8e1c74 => 11
# ╠═╡ 8f2d6c40-1b93-4e7a-8305-6d9a4f1c0e27 => 12
# ╠═╡ 2a9e5b73-4c08-4f6d-9b2a-7e1c5d0a3f86 => 13
# ╠═╡ 7d1c4e98-3b52-4a8f-8c6d-0f4b9e2a1c53 => 14
# ╠═╡ 4b6a0d27-9e81-4c5a-8b7f-3f2e8a1d5c04 => 15
# ╠═╡ 6e3b8a15-2c47-4d9f-8a5b-1c0f7d4e2a83 => 16
# ╠═╡ 9a4c0b83-6d15-4e7b-8f3a-2b1d8e5c0a47 => 17
# ╠═╡ 1f8d3b64-0a29-4c6f-9e8b-5a2c7d0b4e13 => 18
# ╠═╡ 5c2f9a47-8b61-4d3e-9a0c-7e4b1d8f5a26 => 19
# ╠═╡ 3a8e1d95-7c04-4b6a-8f2d-9c5a0e3b1d68 => 20
# ╠═╡ 8b4c7f12-5e93-4a6d-8c0f-2d9e5b3a1c74 => 21
# ╠═╡ 2f6a9d38-1b47-4c8a-9e1d-7b3c0f5a2e85 => 22
# ╠═╡ 6c0b3e81-4a25-4d9f-8b6c-1e8f2a7d0b53 => 23
# ╠═╡ 9e5d1a47-2c38-4b0a-8f7d-5a3e9c6b1d02 => 24
# ╠═╡ 4a7c2e95-0b18-4e6d-9f3a-8c1b5d0e3a72 => 25
# ╠═╡ 3f5a8c27-1e94-4b7d-8a0c-6d3b2e9f1a58 => 26
