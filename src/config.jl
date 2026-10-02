# Canonical parameter definition and configuration loading for MovementAnalysis.
#
# Every entry point in this package -- library call, CLI invocation, or TOML file
# -- resolves to the single `MovementAnalysisConfig` struct defined below. There
# are no dataset presets, no flag aliases, and no per-call-site defaults: a
# dataset is selected by its configuration file, not by a special code path.
#
# Layering, lowest to highest precedence:
#
#   1. Field defaults declared in `MovementAnalysisConfig`
#   2. TOML config file   (e.g. `configs/my_dataset.toml`)
#   3. Command-line flags (declared once in `create_argparse_settings`)
#   4. Programmatic overrides passed to `load_config(overrides = ...)`
#
# See the files under `configs/` for documented examples.

using ArgParse
using TOML

# =============================================================================
# Configuration Struct
# =============================================================================

"""
    DIAGNOSTIC_CHOICES, MODEL_MODE_CHOICES, PATH_METHOD_CHOICES

The complete set of accepted entries in the three list-valued settings.

These are the only places the allowed values are written down, so the CLI help,
the default and the load-time validation cannot drift apart.
"""
const DIAGNOSTIC_CHOICES = Set([:circuit, :stochastic, :bottlenecks,
                                :validation, :bayesian_ensemble])
  const MODEL_MODE_CHOICES = Set([:telemetry, :telemetry_and_survey, :agent])
const PATH_METHOD_CHOICES = Set([:astar, :viterbi])

"""
    MovementAnalysisConfig

Canonical container for every analysis parameter.

All fields are typed and carry a default, so `run_movement_analysis` can read any
parameter directly as `params.field_name` without existence checks or fallbacks.

    # Three settings are lists and are the complete specification each -- there are no
    # aggregate sentinels and no parallel boolean switches:
  model_modes    which inference approaches to fit
  path_methods   which routing methods to reconstruct paths with
  diagnostics    which analyses to run afterwards

    Their accepted values are fixed by `MODEL_MODE_CHOICES`, `PATH_METHOD_CHOICES`
    and `DIAGNOSTIC_CHOICES`, and anything else is rejected at load time rather
    than silently doing less than asked.


`land_source`, `land_mask_resolution` and `depth_barrier_mode` are `Symbol`s so a
typo is a load error rather than a silently unmatched string comparison.
"""
Base.@kwdef struct MovementAnalysisConfig
    # --- Approaches to fit ---------------------------------------------------
      model_modes::Vector{Symbol} = Symbol[
          :telemetry, :telemetry_and_survey, :agent
      ]

    # --- Input files ---------------------------------------------------------
    tagging_file::Union{Nothing,String} = nothing
    hsi_file::Union{Nothing,String} = nothing
    sppoly_file::Union{Nothing,String} = nothing
    surveydata_file::Union{Nothing,String} = nothing

    # --- Spatial domain ------------------------------------------------------
    bbox::Union{Nothing,Vector{Float64}} = nothing
    bbox_padding_deg::Float64 = 0.25
    land_source::Symbol = :landmask
    land_mask_resolution::String = "l"
    land_mask_grid_minutes::Float64 = 5.0
    land_polygon_files::Vector{String} = String[]
    reshard_hex::Bool = false
    hex_radius_km::Float64 = 10.0
    use_hydrodynamics::Bool = false
    depth_barrier_mode::Symbol = :hsi_only
    depth_range::Union{Nothing,Vector{Float64}} = nothing
    hsi_ood_floor::Float64 = 0.01

    # --- Path reconstruction -------------------------------------------------
    max_paths::Int = 1000
    path_methods::Vector{Symbol} = Symbol[:astar]
    smooth_paths::Bool = true

    # --- Advanced routing ----------------------------------------------------
    adaptive_mesh::Bool = false
    coarse_radius_km::Float64 = 25.0
    fine_radius_km::Float64 = 8.0
    dynamic_kernels::Bool = false
    hmm_smoothing::Bool = false

    # --- Habitat-coupled residency --------------------------------------------
    # Strength of the coupling between local habitat advantage and how long an
    # animal stays put. Zero makes every unit share the fitted scalar rho, which
    # is the historical behaviour and the default. Positive values raise
    # residency where the local habitat is favourable relative to its neighbours.
    rest_advantage::Float64 = 0.0
    # How local advantage is measured: :difference, :ratio, :log_ratio,
    # :exp_difference, :exp_ratio, or :exp_log_ratio.
    rest_advantage_form::Symbol = :difference

    # --- Agent forward projection --------------------------------------------
    # Heading persistence for the agent projection, kappa in
    # exp(kappa * cos(bearing - heading)). Zero is the memoryless chain, where a
    # step depends only on current position. Only read by :agent runs.
    persistence::Float64 = 0.0
    # Number of synthetic animals projected. Deliberately decoupled from the
    # number of observations: starts are drawn with replacement from the observed
    # release set, so raising this reduces Monte Carlo error without changing the
    # distribution being estimated.
    n_agent_projections::Int = 200
    # Horizon used when the telemetry carries no usable duration column.
    agent_horizon::Int = 50

    # --- Posterior inference -------------------------------------------------
    n_samples::Int = 200
    n_warmup::Int = 100
    seed::Int = 42
    n_ensemble::Int = 50
    # Multiplier on the prior scale that sets the random-walk proposal width.
    # See `movement_sampler` for why `MH()` itself cannot be used.
    mh_proposal_scale::Float64 = 0.05

    # --- Habitat suitability uncertainty -------------------------------------
    hsi_se::Float64 = 0.08
    propagate_hsi_error::Bool = true
    n_stochastic_draws::Int = 10
    add_structural_uncertainty::Bool = false
    structural_uncertainty_scale::Float64 = 0.05

    # --- Diagnostics ---------------------------------------------------------
    diagnostics::Vector{Symbol} = Symbol[
        :circuit, :stochastic, :bottlenecks, :validation, :bayesian_ensemble
    ]

    # --- Output and reporting ------------------------------------------------
    species_name::String = "Animal"
    render_html::Bool = true
    output_dir::String = "output"
    resume_from_checkpoint::Bool = false
    dark_mode::Bool = false
    cmap::Symbol = :viridis
    overlay_hsi::Bool = false
    # NOT YET APPLIED. The dashboard CSS sets `--font-main` to a hardcoded stack
    # inside `_generate_leaflet_html_document`, and no panel forwards this value
    # there, so setting it has no effect. Wiring it means adding a keyword to all
    # 19 `leaflet_*` functions or threading it through their `kwargs`. Recorded in
    # `todo.md` rather than silently left looking functional.
    font::String = "Inter, system-ui, sans-serif"
    verbose::Bool = true

    # --- Stock connectivity regions ------------------------------------------
    region_labels::Vector{String} = String[]
    region_polygon_files::Vector{String} = String[]
    region_map::Union{Nothing,Dict{Int,Int}} = nothing

    # --- CLI-only ------------------------------------------------------------
    # Not part of the analysis; set by the parser and never read by the pipeline.
    show_help::Bool = false
    # When true, skip all computation and regenerate dashboards from the saved
    # full results checkpoint (movement_results_checkpoint.jld2).  Implies
    # render_html=true.  Requires a prior successful run.
    figures_only::Bool = false
end

# =============================================================================
# Coercion
# =============================================================================

"""
    _coerce_field(name::Symbol, value) -> Any

Convert a raw TOML or command-line value into the declared type of the
corresponding struct field, so `apply_overrides` can hand the result straight to
the keyword constructor.

TOML has no symbol or tuple type, and the command line delivers everything as a
string, so list-valued and enum-valued settings need explicit conversion. A
value that cannot be converted raises here rather than surfacing as a type error
far from the file or flag that produced it.

A nullable field is coerced to the same target as its non-nullable form, so
`bbox` behaves exactly like a `Vector{Float64}` field except that an empty value
is accepted as "not configured".
"""
function _coerce_field(name::Symbol, value)
    value === nothing && return nothing
    return _coerce_value(_strip_nothing(fieldtype(MovementAnalysisConfig, name)), value)
end

# `Union{Nothing,T}` is coerced as `T`, so nullable and required fields share one
# implementation instead of duplicating every case.
_strip_nothing(::Type{Union{Nothing,T}}) where {T} = _strip_nothing(T)
_strip_nothing(::Type{T}) where {T} = T

# Anything already of the declared type passes through untouched: a TOML array, or
# an already-typed programmatic override. Anything else falls through unchanged so
# the keyword constructor raises a `convert` error naming the real target type,
# which is a better diagnostic than a silent pass-through.
_coerce_value(::Type, value) = value

_coerce_value(::Type{Symbol}, value::AbstractString) = Symbol(value)
_coerce_value(::Type{Symbol}, value::Symbol) = value
_coerce_value(::Type{String}, value::AbstractString) = String(value)
_coerce_value(::Type{String}, value::Symbol) = String(value)
_coerce_value(::Type{Bool}, value) = _coerce_bool(value)
_coerce_value(::Type{Int}, value::Real) = Int(value)
_coerce_value(::Type{Float64}, value::Real) = Float64(value)

# A command-line list arrives as one comma-separated string; a TOML list arrives
# as a vector. Both have to end up as a vector of the declared element type.
_coerce_value(::Type{Vector{Symbol}}, value::AbstractString) =
    Symbol[Symbol(strip(v)) for v in split(value, ',') if !isempty(strip(v))]
_coerce_value(::Type{Vector{String}}, value::AbstractString) =
    String[String(strip(v)) for v in split(value, ',') if !isempty(strip(v))]
_coerce_value(::Type{Vector{Float64}}, value::AbstractString) =
    Float64[parse(Float64, strip(v)) for v in split(value, ',') if !isempty(strip(v))]

_coerce_value(::Type{Vector{Symbol}}, value::AbstractVector) =
    Symbol[Symbol(v) for v in value]
_coerce_value(::Type{Vector{String}}, value::AbstractVector) =
    String[String(v) for v in value]
_coerce_value(::Type{Vector{Float64}}, value::AbstractVector) =
    Float64[Float64(v) for v in value]

# TOML tables always have string keys, so an integer-keyed map is rebuilt from them.
_coerce_value(::Type{Dict{Int,Int}}, value::AbstractDict) =
    Dict{Int,Int}(parse(Int, string(k)) => parse(Int, string(v)) for (k, v) in value)
_coerce_value(::Type{Dict{Int,Int}}, value::AbstractVector) =
    Dict{Int,Int}(parse(Int, string(first(p))) => parse(Int, string(last(p)))
                  for p in value)

# An empty string for a nullable field means "not configured", so a hand-edited
# file can blank out a path without having to delete the line.
function _coerce_field(name::Symbol, value::AbstractString)
    T = fieldtype(MovementAnalysisConfig, name)
    if isempty(strip(value)) && Base.nonmissingtype(T) === Union{Nothing}
        return nothing
    end
    return _coerce_value(_strip_nothing(T), value)
end

"""
    _coerce_bool(value) -> Bool

Accept the spellings a human actually types for a boolean. Julia's `Bool("false")`
is a `MethodError`, so `--render-html false` would otherwise be rejected.
"""
_coerce_bool(value::Bool) = value
_coerce_bool(value::Real) = value != 0
function _coerce_bool(value::AbstractString)
    s = lowercase(strip(value))
    s in ("true", "t", "yes", "y", "1", "on") && return true
    s in ("false", "f", "no", "n", "0", "off") && return false
    throw(ArgumentError("cannot read \"$value\" as a boolean"))
end

# =============================================================================
# Overlays
# =============================================================================

"""
    apply_overrides(base::MovementAnalysisConfig, overlay::Dict{Symbol,Any})

Return a copy of `base` with every present key of `overlay` replaced by its
coerced value. A key whose value is `nothing` or `missing` is skipped, so an
absent setting leaves the lower layer untouched rather than erasing it.
"""
function apply_overrides(
    base::MovementAnalysisConfig,
    overlay::Dict{Symbol,Any}
)::MovementAnalysisConfig
    fields = fieldnames(MovementAnalysisConfig)
    unknown = setdiff(keys(overlay), fields)
    isempty(unknown) || throw(ArgumentError(
        "configuration contains unknown key(s): " *
        join(sort!(string.(collect(unknown))), ", ")
    ))
    values = Any[getfield(base, f) for f in fields]
    for (i, f) in enumerate(fields)
        haskey(overlay, f) || continue
        v = overlay[f]
        (v === nothing || ismissing(v)) && continue
        values[i] = _coerce_field(f, v)
    end
    _validate_enum_settings(MovementAnalysisConfig(;
        NamedTuple{fields}(Tuple(values))...))
end

"""
    _validate_enum_settings(config) -> config

Reject unrecognised entries in the three list-valued settings.

`wants_diagnostic` is a membership test, so a typo like `--diagnostics=valdiation`
previously produced a run that "succeeded" having silently done less than asked —
no error, no warning, just a missing analysis. The same held for `model_modes` and
`path_methods`. A load-time check turns each of those into an immediate error that
names the valid values.
"""
function _validate_enum_settings(config::MovementAnalysisConfig)
    for (field, valid) in (
        (:diagnostics, DIAGNOSTIC_CHOICES),
        (:model_modes, MODEL_MODE_CHOICES),
        (:path_methods, PATH_METHOD_CHOICES),
    )
        got = getfield(config, field)
        bad = setdiff(got, valid)
        isempty(bad) || throw(ArgumentError(
            "unknown $(field) entr(y/ies): " *
            join(sort!(string.(collect(bad))), ", ") *
            ". Valid values are: " * join(string.(valid), ", ")
        ))
    end
    return config
end

"""
    MovementAnalysisConfig(path::AbstractString) -> MovementAnalysisConfig

Load a configuration from a TOML file. The resulting object is a fully populated
config, so it is used directly as a base layer rather than merged field-by-field.
"""
function MovementAnalysisConfig(path::AbstractString)::MovementAnalysisConfig
    parsed = TOML.parsefile(path)
    overlay = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in parsed)
    return apply_overrides(MovementAnalysisConfig(), overlay)
end

"""
    _get(params::MovementAnalysisConfig, field::Symbol, default)

Read a field that a caller may not have configured. Kept for the call sites that
predate the struct becoming exhaustive.
"""
_get(params::MovementAnalysisConfig, field::Symbol, default) =
    isdefined(params, field) ? getfield(params, field) : default

"""
    _get_active_chain(chains::Dict) -> Union{Nothing,MCMCChains.Chains}

Select the first available chain from the fitted models.

  Priority order: Telemetry > Telemetry+Survey, because the survey model
  conditions on the same observations with a strictly richer likelihood, so its
  posterior is the better one to summarise whenever it was actually fitted.
  Returns `nothing` when no model produced a chain, so a caller can report
  "nothing was fitted" rather than indexing an empty dict.
  """
  function _get_active_chain(chains::Dict)
      priority = [:telemetry, :telemetry_and_survey]
    for key in priority
        haskey(chains, key) && return chains[key]
    end
    return nothing
end

# =============================================================================
# TOML serialization
# =============================================================================

"""
    _toml_value(x) -> Any

Reduce a field value to something TOML can print. TOML has no symbol type, so a
`Symbol` is written as its name.
"""
_toml_value(x::Symbol) = String(x)
_toml_value(x::AbstractVector) = [_toml_value(v) for v in x]
_toml_value(x::AbstractDict) = Dict{String,Any}(String(k) => _toml_value(v)
                                                for (k, v) in x)
_toml_value(x) = x

# =============================================================================
# Command-line interface
# =============================================================================

"""
    create_argparse_settings() -> ArgParseSettings

Single declaration of every supported command-line flag. Each option has exactly
one long name and no aliases, so a flag is never accepted under two spellings.
The dataset is not selected by a flag at all: it is identified by the input
files named in the TOML configuration.

Every flag is bound to a real `MovementAnalysisConfig` field except `--config`,
which names the config file itself; a check at the end of this function enforces
that, so a flag can never parse successfully and then be silently discarded.
"""
function create_argparse_settings()::ArgParseSettings
    s = ArgParseSettings(
        description = "MovementAnalysis: mark-recapture movement and connectivity analysis",
        usage = "run_movement.jl [OPTIONS]",
        version = "1.0.0",
        add_version = true,
        add_help = false,
        autofix_names = true,
        suppress_warnings = true,
    )

    @add_arg_table! s begin
        # ArgParse's own `--help` prints and exits from inside `parse_args`, which
        # would bypass the config layering and make the flag untestable. It is
        # declared here as an ordinary flag instead, and the entry point decides
        # what to print.
        "-h"
            dest_name = "show_help"
            help = "Show this help message and exit."
            action = :store_true
        "--help"
            dest_name = "show_help"
            help = "Show this help message and exit."
            action = :store_true
        "--config"
            dest_name = "config"
            help = "Path to a TOML configuration file."
            arg_type = String
            metavar = "PATH"

        # --- Approaches to fit ------------------------------------------------
        "--model-modes"
            dest_name = "model_modes"
            help = "Comma-separated approaches: telemetry, telemetry_and_survey, agent."
            arg_type = String
            metavar = "MODES"

        # --- Input files ------------------------------------------------------
        "--tagging-file"
            dest_name = "tagging_file"
            help = "Mark-recapture telemetry file (.jld2, .rds or .rdz)."
            arg_type = String
            metavar = "PATH"
        "--hsi-file"
            dest_name = "hsi_file"
            help = "Habitat suitability time series file (.jld2)."
            arg_type = String
            metavar = "PATH"
        "--sppoly-file"
            dest_name = "sppoly_file"
            help = "Spatial unit definition file (.jld2)."
            arg_type = String
            metavar = "PATH"
        "--surveydata-file"
            dest_name = "surveydata_file"
            help = "Paired survey density file. Required by the survey-dependent approaches."
            arg_type = String
            metavar = "PATH"

        # --- Spatial domain ---------------------------------------------------
        "--bbox"
            dest_name = "bbox"
            help = "Analysis domain as 'west,south,east,north' in degrees. Derived from the data when omitted."
            arg_type = String
            metavar = "W,S,E,N"
        "--bbox-padding"
            dest_name = "bbox_padding_deg"
            help = "Padding in degrees added around a data-derived bounding box."
            arg_type = Float64
            metavar = "DEG"
        "--land-source"
            dest_name = "land_source"
            help = "How land is identified: landmask, depth, polygons or none."
            arg_type = String
            metavar = "MODE"
        "--land-mask-resolution"
            dest_name = "land_mask_resolution"
            help = "Global land/sea mask resolution: c, l, i, h or f."
            arg_type = String
            metavar = "LEVEL"
        "--land-mask-grid"
            dest_name = "land_mask_grid_minutes"
            help = "Global land/sea mask grid spacing in arc minutes."
            arg_type = Float64
            metavar = "MIN"
        "--land-polygon-files"
            dest_name = "land_polygon_files"
            help = "Comma-separated land polygon files (csv, jld2, rds/RData, shp, geojson)."
            arg_type = String
            metavar = "PATHS"
        "--reshard-hex"
            dest_name = "reshard_hex"
            help = "Reshard the domain onto a finer hexagonal lattice."
            arg_type = Bool
        "--hex-radius"
            dest_name = "hex_radius_km"
            help = "Radius of the resharded hexagonal cells, in kilometres."
            arg_type = Float64
            metavar = "KM"
        "--use-hydrodynamics"
            dest_name = "use_hydrodynamics"
            help = "Ingest 3D hydrodynamic fields and bathymetry."
            arg_type = Bool
        "--depth-barrier-mode"
            dest_name = "depth_barrier_mode"
            help = "Depth restriction basis: hsi_only, bathy_only or intersection."
            arg_type = String
            metavar = "MODE"
        "--depth-range"
            dest_name = "depth_range"
            help = "Depth range constraint as 'min,max' in metres."
            arg_type = String
            metavar = "MIN,MAX"
        "--hsi-ood-floor"
            dest_name = "hsi_ood_floor"
            help = "Habitat suitability value below which a unit is out-of-domain."
            arg_type = Float64
            metavar = "VALUE"

        # --- Path reconstruction ----------------------------------------------
        "--max-paths"
            dest_name = "max_paths"
            help = "Maximum number of trajectories to reconstruct."
            arg_type = Int
            metavar = "N"
        "--path-methods"
            dest_name = "path_methods"
            help = "Comma-separated routing methods: astar, viterbi."
            arg_type = String
            metavar = "METHODS"
        "--smooth-paths"
            dest_name = "smooth_paths"
            help = "Apply marine line-of-sight smoothing to reconstructed paths."
            arg_type = Bool

        # --- Advanced routing -------------------------------------------------
        "--adaptive-mesh"
            dest_name = "adaptive_mesh"
            help = "Use an adaptive multiresolution hexagonal mesh."
            arg_type = Bool
        "--coarse-radius"
            dest_name = "coarse_radius_km"
            help = "Coarse cell radius for the adaptive mesh, in kilometres."
            arg_type = Float64
            metavar = "KM"
        "--fine-radius"
            dest_name = "fine_radius_km"
            help = "Fine cell radius for the adaptive mesh, in kilometres."
            arg_type = Float64
            metavar = "KM"
        "--dynamic-kernels"
            dest_name = "dynamic_kernels"
            help = "Use time-varying environmental covariates in the kernel."
            arg_type = Bool
        "--hmm-smoothing"
            dest_name = "hmm_smoothing"
            help = "Apply global multi-segment HMM Viterbi smoothing."
            arg_type = Bool

        # --- Posterior inference ----------------------------------------------
        "--samples"
            dest_name = "n_samples"
            help = "Number of posterior draws."
            arg_type = Int
            metavar = "N"
        "--warmup"
            dest_name = "n_warmup"
            help = "Number of warmup iterations."
            arg_type = Int
            metavar = "N"
        "--seed"
            dest_name = "seed"
            help = "Random seed for reproducibility."
            arg_type = Int
            metavar = "N"
        "--n-ensemble"
            dest_name = "n_ensemble"
            help = "Posterior draws used by the Bayesian path ensemble."
            arg_type = Int
            metavar = "N"

        # --- Habitat suitability uncertainty -----------------------------------
        "--hsi-se"
            dest_name = "hsi_se"
            help = "Observation standard error on habitat suitability."
            arg_type = Float64
            metavar = "SIGMA"
        "--propagate-hsi-error"
            dest_name = "propagate_hsi_error"
            help = "Propagate habitat suitability error into path sampling."
            arg_type = Bool
        "--n-draws"
            dest_name = "n_stochastic_draws"
            help = "Monte Carlo draws per stochastic path."
            arg_type = Int
            metavar = "N"
        "--add-structural-uncertainty"
            dest_name = "add_structural_uncertainty"
            help = "Jitter each resistance draw by a log-normal factor."
            arg_type = Bool
        "--structural-uncertainty-scale"
            dest_name = "structural_uncertainty_scale"
            help = "Log-normal standard deviation for the resistance jitter."
            arg_type = Float64
            metavar = "SIGMA"

        # --- Diagnostics ------------------------------------------------------
        "--diagnostics"
            dest_name = "diagnostics"
            help = "Comma-separated analyses: circuit, stochastic, bottlenecks, validation, bayesian_ensemble."
            arg_type = String
            metavar = "ANALYSES"

        # --- Output and reporting ---------------------------------------------
        "--species-name"
            dest_name = "species_name"
            help = "Species name shown in dashboard titles."
            arg_type = String
            metavar = "NAME"
        "--render-html"
            dest_name = "render_html"
            help = "Render interactive HTML dashboards."
            arg_type = Bool
        "--output-dir"
            dest_name = "output_dir"
            help = "Output directory for generated artefacts."
            arg_type = String
            metavar = "DIR"
        "--resume"
            dest_name = "resume_from_checkpoint"
            help = "Resume from an intermediate checkpoint if available."
            arg_type = Bool
        "--dark-mode"
            dest_name = "dark_mode"
            help = "Use dark mode in rendered dashboards."
            arg_type = Bool
        "--cmap"
            dest_name = "cmap"
            help = "Colour palette for maps (e.g. viridis, plasma, turbo)."
            arg_type = String
            metavar = "NAME"
        "--overlay-hsi"
            dest_name = "overlay_hsi"
            help = "Overlay Habitat Suitability Index on maps (default: false)."
            arg_type = Bool
        "--font"
            dest_name = "font"
            help = "NOT YET APPLIED: main font stack for dashboards. The dashboard " *
                   "CSS currently hardcodes its own, so this has no effect."
            arg_type = String
            metavar = "CSS"
        "--region-labels"
            dest_name = "region_labels"
            help = "Comma-separated region of interest names."
            arg_type = String
            metavar = "NAMES"
        "--region-polygon-files"
            dest_name = "region_polygon_files"
            help = "Comma-separated polygon files, one per region label."
            arg_type = String
            metavar = "PATHS"
        "--figures-only"
            dest_name = "figures_only"
            help = "Regenerate all HTML dashboards from a saved results " *
                   "checkpoint (movement_results_checkpoint.jld2) without " *
                   "re-running MCMC, path reconstruction, or diagnostics. " *
                   "Implies --render-html=true. Requires a prior successful run."
            arg_type = Bool
        "--quiet"
            dest_name = "verbose"
            help = "Suppress progress output. Takes an explicit value: --quiet=true."
            arg_type = Bool
    end

    # Every flag must set a real field, otherwise it parses successfully and is
    # then discarded without trace. `--config` is the one flag that addresses
    # something other than a field: it names the config file to load.
    known = Set(fieldnames(MovementAnalysisConfig))
    for fld in s.args_table.fields
        dest = Symbol(fld.dest_name)
        dest === :config && continue
        dest in known || error(
            "CLI flag --$(join(fld.long_opt_name, ", --")) is bound to " *
            ":$dest, which is not a MovementAnalysisConfig field"
        )
    end

    return s
end

# =============================================================================
# Public entry point
# =============================================================================

"""
    load_config(;
        config_path::Union{Nothing,String} = nothing,
        cli_args::Union{Nothing,Vector{String}} = nothing,
        overrides::Union{Nothing,Dict{Symbol,Any}} = nothing
    ) -> MovementAnalysisConfig

Resolve the effective configuration from the layered sources described at the top
of this file. Only flags actually present on the command line override the TOML
file, so a config file's value is never displaced by an absent flag's default.
"""
function load_config(;
    config_path::Union{Nothing,String} = nothing,
    cli_args::Union{Nothing,Vector{String}} = nothing,
    overrides::Union{Nothing,Dict{Symbol,Any}} = nothing
)::MovementAnalysisConfig
    config = config_path === nothing ?
        MovementAnalysisConfig() :
        MovementAnalysisConfig(config_path)

    if cli_args !== nothing
        config = apply_overrides(config, _explicit_cli_overrides(cli_args))
    end

    if overrides !== nothing
        config = apply_overrides(config, overrides)
    end

    return config
end

"""
    _explicit_cli_overrides(args::Vector{String}) -> Dict{Symbol,Any}

Parse command-line arguments and return only the flags the caller actually
supplied. ArgParse fills unspecified options with their declared defaults, which
would otherwise silently overwrite values coming from a config file, so the
default-valued entries are dropped. `show_help` is a real field, so it rides the
same rule: present-and-true is kept, absent-and-false is dropped.
"""
function _explicit_cli_overrides(args::Vector{String})::Dict{Symbol,Any}
    isempty(args) && return Dict{Symbol,Any}()
    settings = create_argparse_settings()
    parsed = parse_args(args, settings; as_symbols = true)
    defaults = MovementAnalysisConfig()
    out = Dict{Symbol,Any}()
    for (k, v) in pairs(parsed)
        k === :config && continue
        v === nothing && continue
        v == "" && continue
        v == getfield(defaults, k) && continue
        out[k] = v
    end
    return out
end

# =============================================================================
# Approach availability
# =============================================================================

"""
    survey_dependent_modes() -> Set{Symbol}

Approaches whose likelihood is conditioned on paired survey density
observations, and therefore have nothing to fit without them.
"""
survey_dependent_modes() = Set([:telemetry_and_survey, :ssa_and_survey])

"""
    has_survey_data(config) -> Bool

Whether a readable paired survey file is configured.
"""
has_survey_data(config::MovementAnalysisConfig) =
    !isnothing(config.surveydata_file) && isfile(config.surveydata_file)

"""
    wants_diagnostic(config, name::Symbol) -> Bool

Whether a named analysis is in the requested `diagnostics` list.
"""
wants_diagnostic(config::MovementAnalysisConfig, name::Symbol) =
    name in config.diagnostics

"""
    effective_model_modes(config) -> Vector{Symbol}

The approaches that will actually be fitted.

`model_modes` is the request; this is what survives contact with the data. A
survey-dependent approach is dropped when no readable survey file is configured,
so asking for one is not an error -- it is simply not run, and `skipped_modes`
reports which were dropped and why.
"""
function effective_model_modes(config::MovementAnalysisConfig)
    has_survey_data(config) || return setdiff(config.model_modes, survey_dependent_modes())
    return copy(config.model_modes)
end

"""
    skipped_modes(config) -> Vector{Symbol}

Requested approaches that will not run because their required input is absent.
"""
function skipped_modes(config::MovementAnalysisConfig)
    has_survey_data(config) && return Symbol[]
    return intersect(config.model_modes, survey_dependent_modes())
end

# =============================================================================
# Convenience
# =============================================================================

"""
    save_config(config::MovementAnalysisConfig, path::AbstractString)

Write the effective configuration to a TOML file. Useful for recording the exact
settings behind a particular run, so it round-trips through
`MovementAnalysisConfig(path)`.

A field set to `nothing` is omitted rather than written as an empty string: TOML
has no null, and an empty string is a real value for a path field, so writing one
would turn "not configured" into "configured as the current directory". An absent
key loads back as `nothing`, which keeps the round trip exact.
"""
function save_config(config::MovementAnalysisConfig, path::AbstractString)
    parsed = Dict{String,Any}()
    for f in fieldnames(MovementAnalysisConfig)
        # `show_help` is a command-line artifact, not analysis state. It lives on
        # the struct only so `load_config` can surface it; recording it in a saved
        # config would claim a flag is part of the analysis.
        f === :show_help && continue
        v = getfield(config, f)
        v === nothing && continue
        parsed[String(f)] = _toml_value(v)
    end
    open(path, "w") do io
        TOML.print(io, parsed)
    end
    return path
end
