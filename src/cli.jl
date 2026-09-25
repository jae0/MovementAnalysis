# Command Line Interface arguments parser
"""
    print_movement_help()

Prints CLI usage instructions for `run_movement.jl`.
"""
function print_movement_help()
    println("""
MovementAnalysis Pipeline
=========================
Usage:
  julia --project=movement scripts/run_movement.jl [OPTIONS]

Configuration:
  --config=<path>           Path to TOML config file (optional).

Data & Model:
  --data-source=<src>       Data source: 'simulate' or path to data config TOML.
  --model-mode=<mode>       Model mode: 'telemetry', 'telemetry_and_survey', 'ssa', 'ssa_and_survey', 'agent', 'both'.

Domain Resharding:
  --reshard-hex             Reshard domain to finer hexagons via LibGEOS (true/false).
  --hex-radius=<km>         Fine hexagon cell radius in km (default: 10.0).
  --use-hydrodynamics       Extract 3D hydrodynamic fields and bathymetry (true/false).
  --depth-range=<m,M>       Restrict movement to depth range [m, M] m.

Path Reconstruction:
  --max-paths=<N>           Maximum trajectories to reconstruct (default: 25).
  --path-method=<method>    Path method: 'astar' or 'viterbi' (default: astar).
  --smooth-paths            Apply line-of-sight raycast smoothing (true/false).

Validation Analyses & Uncertainty:
  --validation              Run validation analyses (path CIs, connectivity, PPC) (true/false).
  --bayesian-ensemble       Full posterior MCMC ensemble path/corridor propagation (true/false).

Advanced Routing & Dynamics:
  --adaptive-mesh           Adaptive multiresolution hexagonal mesh routing (true/false).
  --dynamic-kernels         Time-varying dynamic environmental covariates (true/false).
  --hmm-smoothing           Multi-segment HMM Viterbi trajectory smoothing (true/false).

Optional Diagnostics:
  --stochastic              Stochastic least-cost path ensembles (true/false).
  --bottlenecks             Domain-wide bottleneck index B(u) (true/false).
  --circuit                 Circuit current density & pinch-points (true/false).
  --all-diagnostics         Enable all optional diagnostics (true/false).

MCMC:
  --samples=<N>             Posterior draws (default: 200).
  --warmup=<N>              Warmup iterations (default: 100).
  --seed=<N>                Random seed (default: 42).
  --hsi-se=<sigma>          HSI observation error (default: 0.08).
  --propagate-hsi-error     Propagate HSI error (true/false).
  --n-draws=<N>             MC draws per stochastic path (default: 10).

Output & UI:
  --render-html             Render HTML export (true/false).
  --output-dir=<path>       Output directory (default: <repo>/output).
  --tagging-file=<path>     Explicit tagging data file path (.jld2, .rds, .rdz).
  --hsi-file=<path>         Explicit HSI data file path.
  --sppoly-file=<path>      Explicit spatial polygon data file path.
  --resume                  Resume analysis from intermediate checkpoint if available (true/false).
  --dark-mode               Use dark mode for visualizations (true/false).
  --palette=<cmap>          Color palette for maps (e.g. viridis, plasma, turbo).
  --font=<font>             Main font stack to use in dashboards.
  --quiet                   Suppress progress output (true/false).
  --help                    Display this help and exit.

Examples:
  julia --project=movement scripts/run_movement.jl --data-source=simulate
  julia --project=movement scripts/run_movement.jl --config=configs/snowcrab.toml
  julia --project=movement scripts/run_movement.jl --data-source=simulate --depth-range=50,400
  julia --project=movement scripts/run_movement.jl --help
""")
    return nothing
end

_is_cli_flag(token::AbstractString) =
    startswith(token, "--") ||
    (startswith(token, "-") && tryparse(Float64, token) === nothing)

function parse_movement_cli_args(args = ARGS)::NamedTuple
    opts = Dict{Symbol, Any}()
    idx  = 1
    N    = length(args)

    _is_false(s::AbstractString) =
        lowercase(strip(s)) in ("0", "false", "f", "no", "n", "off")

    while idx <= N
        raw = args[idx]
        has_eq = occursin('=', raw)
        key, inline_val = if has_eq
            k, v = split(raw, '='; limit = 2)
            (lowercase(strip(k)), strip(v))
        else
            (lowercase(strip(raw)), "")
        end
        has_inline = has_eq && !isempty(inline_val)

        function fv()
            if has_inline
                return inline_val
            elseif idx < N && !_is_cli_flag(args[idx + 1])
                idx += 1
                return args[idx]
            else
                error("Option '$raw' requires an argument.")
            end
        end

        if key == "--help"
            opts[:help] = true
        elseif key == "--config"
            opts[:config_path] = fv()
        elseif key == "--data-source"
            opts[:data_source] = fv()
        elseif key == "--model-mode"
            opts[:model_mode] = fv()
        elseif key == "--reshard-hex"
            opts[:reshard_hex] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--hex-radius"
            opts[:hex_radius_km] = parse(Float64, fv())
        elseif key == "--use-hydrodynamics"
            opts[:use_hydrodynamics] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--depth-range"
            val_str = fv()
            parts = split(val_str, ',')
            if length(parts) == 2
                opts[:depth_range] = (
                    parse(Float64, strip(parts[1])),
                    parse(Float64, strip(parts[2]))
                )
            else
                @warn "Invalid depth-range format: '$val_str'. Expected 'min,max'."
            end
        elseif key == "--max-paths"
            opts[:max_paths] = parse(Int, fv())
        elseif key == "--path-method"
            opts[:path_method] = Symbol(fv())
        elseif key == "--smooth-paths"
            opts[:smooth_paths] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--validation"
            opts[:compute_validation] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--bayesian-ensemble"
            opts[:run_bayesian_ensemble] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--adaptive-mesh"
            opts[:adaptive_mesh] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--dynamic-kernels"
            opts[:dynamic_kernels] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--hmm-smoothing"
            opts[:hmm_smoothing] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--stochastic"
            opts[:compute_stochastic] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--bottlenecks"
            opts[:compute_bottlenecks] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--circuit"
            opts[:compute_circuit] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--all-diagnostics"
            opts[:compute_circuit]     = true
            opts[:compute_stochastic]  = true
            opts[:compute_bottlenecks] = true
        elseif key == "--samples"
            opts[:n_samples] = parse(Int, fv())
        elseif key == "--warmup"
            opts[:n_warmup] = parse(Int, fv())
        elseif key == "--seed"
            opts[:seed] = parse(Int, fv())
        elseif key == "--hsi-se"
            opts[:hsi_se] = parse(Float64, fv())
        elseif key == "--propagate-hsi-error"
            opts[:propagate_hsi_error] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--n-draws"
            opts[:n_stochastic_draws] = parse(Int, fv())
        elseif key == "--render-html"
            opts[:render_html] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--output-dir"
            opts[:output_dir] = fv()
        elseif key == "--tagging-file"
            opts[:tagging_file] = fv()
        elseif key == "--hsi-file"
            opts[:hsi_file] = fv()
        elseif key == "--sppoly-file"
            opts[:sppoly_file] = fv()
        elseif key == "--resume"
            opts[:resume_from_checkpoint] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--dark-mode"
            opts[:dark_mode] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--palette"
            opts[:cmap] = Symbol(fv())
        elseif key == "--font"
            opts[:font] = fv()
        elseif key == "--quiet"
            opts[:verbose] = false
        else
            @warn "Unrecognized CLI flag: '$raw' -- ignoring."
        end

        idx += 1
    end

    return (; opts...)
end

Base.@ccallable function julia_main()::Cint
    try
        cli = parse_movement_cli_args(ARGS)
        if get(cli, :help, false)
            print_movement_help()
            return 0
        end

        # Load config file if provided
        config_path = get(cli, :config_path, nothing)
        if config_path !== nothing && isfile(config_path)
            config = MovementAnalysisConfig(config_path)
        else
            config = MovementAnalysisConfig()
        end

        # Merge CLI overrides
        overrides_dict = Dict{Symbol,Any}(pairs(cli))
        merged_config = load_config(config_path=nothing, cli_args=ARGS, overrides=overrides_dict)
        run_movement_analysis(merged_config)
    catch e
        @error "Pipeline failed" exception=(e, catch_backtrace())
        return 1
    end
    return 0
end