#!/usr/bin/env julia

import Pkg

let
    curr_dir = @__DIR__
    proj_dir = normpath(joinpath(curr_dir, ".."))
    Pkg.activate(proj_dir)
end

using MovementAnalysis

if abspath(PROGRAM_FILE) == @__FILE__
    # `julia_main` returns a process exit code (0 on success, 1 on failure).
    # Propagating it matters for batch runs: a pipeline that dies partway through
    # otherwise looks like a clean run to any caller checking only `$?`, and the
    # partial outputs it left behind get mistaken for a result.
    exit(MovementAnalysis.julia_main())
end
