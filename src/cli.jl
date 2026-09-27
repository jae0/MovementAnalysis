"""
    julia_main() -> Cint

Package entry point. Resolves parameters through the single canonical path
`MovementAnalysis.load_config` (defaults < TOML file < CLI flags), then runs
`run_movement_analysis`.

There are no dataset presets and no flag aliases. The dataset is identified
purely by the input files named in the TOML file, and every dataset-specific
value (depth range, model modes, MCMC budget, species name) belongs there too.
Each flag is declared exactly once in `MovementAnalysis.create_argparse_settings`.

Arguments are parsed once, here, to learn the config file path and whether help
was requested. `load_config` then re-parses to build the override layer, so the
two can never disagree about how a flag is spelled.
"""
function julia_main()::Cint
    try
        parsed = parse_args(ARGS, create_argparse_settings(); as_symbols = true)

        if get(parsed, :show_help, false)
            # `exit_when_done = false` so the entry point returns an exit code
            # instead of ArgParse calling `exit` from inside the library.
            ArgParse.show_help(stdout, create_argparse_settings(); exit_when_done = false)
            return Cint(0)
        end

        config = load_config(
            config_path = get(parsed, :config, nothing),
            cli_args = String.(ARGS),
        )
        run_movement_analysis(config)
    catch e
        @error "Pipeline failed" exception = (e, catch_backtrace())
        return Cint(1)
    end
    return Cint(0)
end
