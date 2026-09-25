# Configuration module for MovementAnalysis
# This is included directly into MovementAnalysis main module

using ArgParse
using TOML

# -----------------------------------------------------------------------------
# Configuration Struct
# -----------------------------------------------------------------------------
"""
    MovementAnalysisConfig

Centralized configuration for the movement analysis pipeline.
All fields have defaults that can be overridden via TOML config file,
CLI arguments, or programmatic construction.
"""
Base.@kwdef struct MovementAnalysisConfig
    # Data & Model
    data_source::String = "simulate"
    model_mode::String = "telemetry"

    # Domain
    reshard_hex::Bool = false
    hex_radius_km::Float64 = 10.0
    use_hydrodynamics::Bool = false
    depth_range::Union{Vector{Float64},Nothing} = nothing

    # Path Reconstruction
    max_paths::Int = 1000
    path_method::String = "astar"
    smooth_paths::Bool = true

    # Validation
    compute_validation::Bool = true
    run_bayesian_ensemble::Bool = false

    # Advanced Routing
    adaptive_mesh::Bool = false
    coarse_radius_km::Float64 = 25.0
    fine_radius_km::Float64 = 8.0
    dynamic_kernels::Bool = false
    hmm_smoothing::Bool = false

    # Diagnostics
    compute_circuit::Bool = true
    compute_stochastic::Bool = true
    compute_bottlenecks::Bool = true
    n_stochastic_draws::Int = 10

    # HSI Error
    hsi_se::Float64 = 0.08
    propagate_hsi_error::Bool = true

    # MCMC
    n_samples::Int = 200
    n_warmup::Int = 100
    seed::Int = 42

    # Output
    render_html::Bool = true
    output_dir::String = "output"
    verbose::Bool = true

    # Groups
    group_labels::Vector{String} = String["All"]
    group_alpha::Vector{Float64} = Float64[0.40]
    group_rho::Vector{Float64} = Float64[0.25]
    group_gamma::Vector{Float64} = Float64[1.00]

    # Species
    species_name::String = "Animal"

    # Resume
    resume_from_checkpoint::Bool = false

    # Data file paths
    tagging_file::Union{Nothing,String} = nothing
    hsi_file::Union{Nothing,String} = nothing
    sppoly_file::Union{Nothing,String} = nothing

    # Region mapping
    region_labels::Union{Nothing,Vector{String}} = nothing
    region_map::Union{Nothing,Dict{Int,Int}} = nothing
end

# -----------------------------------------------------------------------------
# TOML Serialization
# -----------------------------------------------------------------------------
function MovementAnalysisConfig(config_path::AbstractString)::MovementAnalysisConfig
    dict = TOML.parsefile(config_path)
    # Convert string keys to symbols for struct constructor
    sym_dict = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in dict)
    return MovementAnalysisConfig(; sym_dict...)
end

function TOML.print(config::MovementAnalysisConfig, io::IO = stdout)
    fields = Dict{String,Any}()
    for field in fieldnames(MovementAnalysisConfig)
        val = getfield(config, field)
        fields[string(field)] = _toml_safe(val)
    end
    TOML.print(io, fields)
end

_toml_safe(x) = x
_toml_safe(x::Tuple) = collect(x)
_toml_safe(x::Nothing) = ""
_toml_safe(x::Vector{<:Real}) = Float64.(x)
_toml_safe(x::Vector{String}) = x

# -----------------------------------------------------------------------------
# ArgParse CLI Parsing
# -----------------------------------------------------------------------------
function create_argparse_settings()::ArgParseSettings
    s = ArgParseSettings(
        description = "MovementAnalysis Pipeline - Animal movement modeling on spatial graphs",
        usage = "run_movement.jl [OPTIONS]",
        version = "0.1.0",
        add_version = true,
        add_help = true,
    )

    @add_arg_table! s begin
        "--config"
            help = "Path to TOML config file (optional)."
            arg_type = String
        "--data-source"
            help = "Data source: 'simulate' or path to data config TOML."
            arg_type = String
            default = "simulate"
        "--model-mode"
            help = "Model mode: 'telemetry', 'telemetry_and_survey', 'ssa', 'ssa_and_survey', 'agent', 'both'."
            arg_type = String
            default = "telemetry"
        "--reshard-hex"
            help = "Reshard domain to finer hexagonal lattice (true/false)."
            arg_type = Bool
            default = false
        "--hex-radius"
            help = "Fine hexagon cell radius in km."
            arg_type = Float64
            default = 10.0
        "--use-hydrodynamics"
            help = "Ingest 3D hydrodynamic fields and bathymetry (true/false)."
            arg_type = Bool
            default = false
        "--depth-range"
            help = "Depth range constraint as 'min,max' in meters."
            arg_type = String
        "--max-paths"
            help = "Maximum trajectories to reconstruct."
            arg_type = Int
            default = 1000
        "--path-method"
            help = "Path method: 'astar' or 'viterbi'."
            arg_type = String
            default = "astar"
        "--smooth-paths"
            help = "Apply marine line-of-sight smoothing (true/false)."
            arg_type = Bool
            default = true
        "--validation"
            help = "Run validation analyses (path CIs, connectivity, PPC) (true/false)."
            arg_type = Bool
            default = true
        "--bayesian-ensemble"
            help = "Full posterior MCMC ensemble path propagation (true/false)."
            arg_type = Bool
            default = false
        "--adaptive-mesh"
            help = "Adaptive multiresolution hexagonal mesh routing (true/false)."
            arg_type = Bool
            default = false
        "--coarse-radius"
            help = "Coarse hexagon radius for adaptive mesh (km)."
            arg_type = Float64
            default = 25.0
        "--fine-radius"
            help = "Fine hexagon radius for adaptive mesh (km)."
            arg_type = Float64
            default = 8.0
        "--dynamic-kernels"
            help = "Time-varying dynamic environmental covariates (true/false)."
            arg_type = Bool
            default = false
        "--hmm-smoothing"
            help = "Multi-segment HMM Viterbi trajectory smoothing (true/false)."
            arg_type = Bool
            default = false
        "--stochastic"
            help = "Stochastic least-cost path ensembles (true/false)."
            arg_type = Bool
            default = true
        "--bottlenecks"
            help = "Domain-wide bottleneck index B(u) (true/false)."
            arg_type = Bool
            default = true
        "--circuit"
            help = "Circuit current density & pinch-points (true/false)."
            arg_type = Bool
            default = true
        "--all-diagnostics"
            help = "Enable all optional diagnostics (true/false)."
            arg_type = Bool
            default = false
        "--samples"
            help = "Posterior MCMC draws."
            arg_type = Int
            default = 200
        "--warmup"
            help = "MCMC warmup iterations."
            arg_type = Int
            default = 100
        "--seed"
            help = "Random seed."
            arg_type = Int
            default = 42
        "--hsi-se"
            help = "HSI observation standard error."
            arg_type = Float64
            default = 0.08
        "--propagate-hsi-error"
            help = "Propagate HSI error (true/false)."
            arg_type = Bool
            default = true
        "--n-draws"
            help = "MC draws per stochastic path."
            arg_type = Int
            default = 10
        "--render-html"
            help = "Render HTML export (true/false)."
            arg_type = Bool
            default = true
        "--output-dir"
            help = "Output directory."
            arg_type = String
        "--tagging-file"
            help = "Explicit tagging data file path."
            arg_type = String
        "--hsi-file"
            help = "Explicit HSI data file path."
            arg_type = String
        "--sppoly-file"
            help = "Explicit spatial polygon data file path."
            arg_type = String
        "--resume"
            help = "Resume from intermediate checkpoint if available (true/false)."
            arg_type = Bool
            default = false
        "--dark-mode"
            help = "Use dark mode for visualizations (true/false)."
            arg_type = Bool
            default = false
        "--palette"
            help = "Color palette for maps (e.g. viridis, plasma, turbo)."
            arg_type = String
        "--font"
            help = "Main font stack to use in dashboards."
            arg_type = String
        "--quiet"
            help = "Suppress progress output (true/false)."
            arg_type = Bool
            default = false
            help = "Display this help and exit."
            action = :store_true
    end

    return s
end

# -----------------------------------------------------------------------------
# Configuration Merging
# -----------------------------------------------------------------------------
"""
    load_config(;
        config_path::Union{Nothing,String} = nothing,
        cli_args::Union{Nothing,Vector{String}} = nothing,
        overrides::Union{Nothing,Dict{Symbol,Any}} = nothing
    )::MovementAnalysisConfig

Load configuration with precedence: defaults < TOML config < CLI < overrides.
"""
function load_config(;
    config_path::Union{Nothing,String} = nothing,
    cli_args::Union{Nothing,Vector{String}} = nothing,
    overrides::Union{Nothing,Dict{Symbol,Any}} = nothing
)::MovementAnalysisConfig
    # Start with defaults
    config = MovementAnalysisConfig()

    # Layer 1: TOML config file
    if config_path !== nothing && isfile(config_path)
        toml_config = MovementAnalysisConfig(config_path)
        config = merge_configs(config, toml_config)
    end

    # Layer 2: CLI arguments (via ArgParse)
    if cli_args !== nothing
        cli_config = parse_cli_args(cli_args)
        config = merge_configs(config, cli_config)
    end

    # Layer 3: Programmatic overrides (highest precedence)
    if overrides !== nothing
        config = merge_configs(config, overrides)
    end

    return config
end

function merge_configs(base::MovementAnalysisConfig, overlay::MovementAnalysisConfig)::MovementAnalysisConfig
    fields = fieldnames(MovementAnalysisConfig)
    default = MovementAnalysisConfig()
    values = []
    for f in fields
        ov = getfield(overlay, f)
        if (ov !== nothing) && (ov != getfield(default, f))
            push!(values, ov)
        else
            push!(values, getfield(base, f))
        end
    end
    return MovementAnalysisConfig(; NamedTuple{fields}(Tuple(values))...)
end

function merge_configs(base::MovementAnalysisConfig, overlay::Dict{Symbol,Any})::MovementAnalysisConfig
    fields = fieldnames(MovementAnalysisConfig)
    values = []
    for f in fields
        if haskey(overlay, f) && overlay[f] !== nothing
            push!(values, overlay[f])
        else
            push!(values, getfield(base, f))
        end
    end
    return MovementAnalysisConfig(; NamedTuple{fields}(Tuple(values))...)
end

function parse_cli_args(args::Vector{String})::MovementAnalysisConfig
    s = create_argparse_settings()
    parsed = parse_args(args, s; as_symbols = true)

    config = MovementAnalysisConfig()
    for (k, v) in pairs(parsed)
        if hasfield(MovementAnalysisConfig, k) && v !== nothing
            if k == :depth_range && v isa String
                parts = split(v, ',')
                if length(parts) == 2
                    v = (parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2])))
                else
                    @warn "Invalid depth-range format: '$v'. Expected 'min,max'."
                    v = nothing
"""
    _get(params::MovementAnalysisConfig, field::Symbol, default)

Get field from config struct with fallback to default.
"""
_get(params::MovementAnalysisConfig, field::Symbol, default) =
    isdefined(params, field) ? getfield(params, field) : default

end
            elseif k in (:group_alpha, :group_rho, :group_gamma) && v isa String
                v = parse.(Float64, split(v, ','))
            elseif k in (:group_labels,) && v isa String
                v = String.(split(v, ','))
            end
            config = merge_configs(config, Dict{Symbol,Any}(k => v))
        end
    end

    if get(parsed, :all_diagnostics, false)
        config = merge_configs(config, Dict(
            :compute_circuit => true,
            :compute_stochastic => true,
            :compute_bottlenecks => true,
        ))
    end

    return config
end

# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------
"""
    _get(params::MovementAnalysisConfig, field::Symbol, default)

Get field from config struct with fallback to default.
"""
_get(params::MovementAnalysisConfig, field::Symbol, default) =
    isdefined(params, field) ? getfield(params, field) : default

"""
    save_config(config::MovementAnalysisConfig, path::String)

Save configuration to TOML file.
"""
function save_config(config::MovementAnalysisConfig, path::String)
    open(path, "w") do io
        TOML.print(config, io)
    end
end

end