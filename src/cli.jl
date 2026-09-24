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

Data & Model:
  -d, --data-source=<src>   'simulate' (default) or 'snowcrab'.
      --simulate             Shorthand for --data-source=simulate.
      --snowcrab             Shorthand: snowcrab preset + all diagnostics.
  -m, --model-mode=<mode>   'telemetry' (default), 'telemetry_and_survey',
                             'ssa', 'ssa_and_survey', or 'both'.
      --telemetry            Shorthand for --model-mode=telemetry.
      --joint, --survey      Shorthand for telemetry_and_survey.
      --ssa                  Shorthand for continuous-time SSA model.
      --ssa-joint            Shorthand for joint survey + SSA model.
      --agent                Shorthand for agent-based alternative model.
      --both                 Shorthand for --model-mode=both.

Domain Resharding:
      --reshard-hex, --hex   Reshard to finer hexagons via LibGEOS.
      --hex-radius=<km>      Fine hexagon cell radius in km (default: 10.0).
      --hydro                Extract 3D hydrodynamic fields and bathymetry.
      --depth-range=<m,M>    Restrict movement to depth range [m, M] m.

Path Reconstruction:
  -p, --max-paths=<N>        Maximum trajectories (default: 25).
      --astar                Use A* path method (default).
      --viterbi              Use Viterbi path method.
      --smooth-paths         Apply line-of-sight raycast smoothing.

Validation Analyses & Uncertainty:
      --validation           Run validation analyses (path CIs, connectivity, PPC).
      --no-validation        Skip validation mark-recapture analyses.
      --bayesian-ensemble    Full posterior MCMC ensemble path/corridor propagation.

Advanced Routing & Dynamics:
      --adaptive-mesh        Adaptive multiresolution hexagonal mesh routing.
      --dynamic-kernels      Time-varying dynamic environmental covariates.
      --hmm-smoothing        Multi-segment HMM Viterbi trajectory smoothing.

Optional Diagnostics:
      --stochastic           Stochastic least-cost path ensembles.
      --bottlenecks          Domain-wide bottleneck index B(u).
      --circuit              Circuit current density & pinch-points.
      --wavelets, --sgwt     Chebyshev spectral graph wavelets.
      --all-diagnostics      Enable all four optional diagnostics.

MCMC:
  -n, --samples=<N>          Posterior draws (default: 200).
  -w, --warmup=<N>           Warmup iterations (default: 100).
      --seed=<N>             Random seed (default: 42).
      --hsi-se=<sigma>       HSI observation error (default: 0.08).
      --n-draws=<N>          MC draws per stochastic path (default: 10).

Output & UI:
      --no-html              Disable HTML export.
  -o, --output-dir=<path>    Output directory (default: <repo>/output).
      --tagging-file=<path>  Explicit tagging data file path (.jld2, .rds, .rdz).
      --hsi-file=<path>      Explicit HSI data file path.
      --sppoly-file=<path>   Explicit spatial polygon data file path.
      --resume               Resume analysis from intermediate checkpoint if available.
      --dark-mode            Use dark mode for visualizations.
      --light-mode           Use light mode for visualizations (default).
      --palette=<cmap>       Color palette for maps (e.g. viridis, plasma, turbo).
      --font=<font>          Main font stack to use in dashboards.
  -q, --quiet                Suppress progress output.
  -h, --help                 Display this help and exit.

Examples:
  julia --project=movement scripts/run_movement.jl --simulate
  julia --project=movement scripts/run_movement.jl --snowcrab
  julia --project=movement scripts/run_movement.jl --depth-range=50,400
  julia --project=movement scripts/run_movement.jl --snowcrab --wavelets
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

        if key in ("-h", "--help")
            opts[:help] = true
        elseif key in ("-d", "--data-source", "--data")
            opts[:data_source] = Symbol(fv())
        elseif key == "--simulate"
            opts[:data_source] = :simulate
        elseif key == "--snowcrab"
            opts[:data_source]         = :snowcrab
            opts[:compute_circuit]     = true
            opts[:compute_stochastic]  = true
            opts[:compute_bottlenecks] = true
            opts[:compute_wavelets]    = true
        elseif key in ("-m", "--model-mode", "--mode")
            opts[:model_mode] = String(fv())
        elseif key == "--telemetry"
            opts[:model_mode] = "telemetry"
        elseif key in ("--joint", "--survey")
            opts[:model_mode] = "telemetry_and_survey"
        elseif key == "--ssa"
            opts[:model_mode] = "ssa"
        elseif key in ("--ssa-joint", "--joint-ssa")
            opts[:model_mode] = "ssa_and_survey"
        elseif key == "--agent"
            opts[:model_mode] = "agent"
        elseif key == "--both"
            opts[:model_mode] = "both"
        elseif key in ("--reshard-hex", "--reshard", "--hex")
            opts[:reshard_hex] = has_inline ? !_is_false(inline_val) : true
        elseif key in ("--no-reshard-hex", "--no-reshard", "--no-hex")
            opts[:reshard_hex] = false
        elseif key in ("--hex-radius", "--radius", "-r")
            opts[:hex_radius_km] = parse(Float64, fv())
        elseif key == "--coarse-radius"
            opts[:coarse_radius_km] = parse(Float64, fv())
        elseif key == "--fine-radius"
            opts[:fine_radius_km] = parse(Float64, fv())
        elseif key == "--species-name"
            opts[:species_name] = String(fv())
        elseif key == "--group-labels"
            opts[:group_labels] = String.(split(fv(), ','))
        elseif key == "--group-alpha"
            opts[:group_alpha] = parse.(Float64, split(fv(), ','))
        elseif key == "--group-rho"
            opts[:group_rho] = parse.(Float64, split(fv(), ','))
        elseif key == "--group-gamma"
            opts[:group_gamma] = parse.(Float64, split(fv(), ','))
        elseif key in ("--hydro", "--use-hydrodynamics")
            opts[:use_hydrodynamics] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--no-hydro"
            opts[:use_hydrodynamics] = false
        elseif key in ("--depth-range", "--depths", "--depth")
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
        elseif key in ("-p", "--max-paths", "--paths")
            opts[:max_paths] = parse(Int, fv())
        elseif key == "--astar"
            opts[:path_method] = :astar
        elseif key == "--viterbi"
            opts[:path_method] = :viterbi
        elseif key == "--smooth-paths"
            opts[:smooth_paths] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--no-smooth-paths"
            opts[:smooth_paths] = false
        elseif key == "--validation"
            opts[:compute_validation] = true
        elseif key == "--no-validation"
            opts[:compute_validation] = false
        elseif key == "--bayesian-ensemble"
            opts[:run_bayesian_ensemble] = true
        elseif key == "--adaptive-mesh"
            opts[:adaptive_mesh] = true
        elseif key == "--dynamic-kernels"
            opts[:dynamic_kernels] = true
        elseif key == "--hmm-smoothing"
            opts[:hmm_smoothing] = true
        elseif key in ("--circuit", "--compute-circuit")
            opts[:compute_circuit] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--no-circuit"
            opts[:compute_circuit] = false
        elseif key in ("--stochastic", "--compute-stochastic")
            opts[:compute_stochastic] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--no-stochastic"
            opts[:compute_stochastic] = false
        elseif key in ("--bottlenecks", "--compute-bottlenecks")
            opts[:compute_bottlenecks] = has_inline ? !_is_false(inline_val) : true
        elseif key == "--no-bottlenecks"
            opts[:compute_bottlenecks] = false
        elseif key in ("--wavelets", "--sgwt", "--compute-wavelets")
            opts[:compute_wavelets] = has_inline ? !_is_false(inline_val) : true
        elseif key in ("--no-wavelets", "--no-sgwt")
            opts[:compute_wavelets] = false
        elseif key == "--all-diagnostics"
            opts[:compute_circuit]     = true
            opts[:compute_stochastic]  = true
            opts[:compute_bottlenecks] = true
            opts[:compute_wavelets]    = true
        elseif key in ("-n", "--samples")
            opts[:n_samples] = parse(Int, fv())
        elseif key in ("-w", "--warmup")
            opts[:n_warmup] = parse(Int, fv())
        elseif key == "--seed"
            opts[:seed] = parse(Int, fv())
        elseif key in ("--hsi-se", "--hsi-error")
            opts[:hsi_se] = parse(Float64, fv())
        elseif key == "--propagate-hsi-error"
            opts[:propagate_hsi_error] = true
        elseif key == "--no-propagate-hsi-error"
            opts[:propagate_hsi_error] = false
        elseif key in ("--n-draws", "--stochastic-draws")
            opts[:n_stochastic_draws] = parse(Int, fv())
        elseif key in ("--render-html", "--html")
            opts[:render_html] = has_inline ? !_is_false(inline_val) : true
        elseif key in ("--no-render-html", "--no-html")
            opts[:render_html] = false
        elseif key == "--resume"
            opts[:resume_from_checkpoint] = true
        elseif key in ("--dark-mode")
            opts[:dark_mode] = true
        elseif key in ("--light-mode")
            opts[:dark_mode] = false
        elseif key in ("--palette", "--cmap")
            opts[:cmap] = Symbol(fv())
        elseif key in ("--font")
            opts[:font] = String(fv())
        elseif key == "--tagging-file"
            opts[:tagging_file] = String(fv())
        elseif key == "--hsi-file"
            opts[:hsi_file] = String(fv())
        elseif key == "--sppoly-file"
            opts[:sppoly_file] = String(fv())
        elseif key in ("--output-dir", "--output", "-o")
            opts[:output_dir] = String(fv())
        elseif key in ("--verbose", "-v")
            opts[:verbose] = true
        elseif key in ("--quiet", "-q", "--silent")
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
        else
            base_params = (
                haskey(cli, :data_source) && cli.data_source == :snowcrab ?
                movement_parameters_snowcrab() :
                movement_parameters_default()
            )
            run_movement_analysis(merge(base_params, cli))
        end
    catch e
        @error "Pipeline failed" exception=(e, catch_backtrace())
        return 1
    end
    return 0
end
