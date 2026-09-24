#!/usr/bin/env julia

import Pkg

let
    curr_dir = @__DIR__
    proj_dir = normpath(joinpath(curr_dir, ".."))
    Pkg.activate(proj_dir)
end

using MovementAnalysis

if abspath(PROGRAM_FILE) == @__FILE__
    MovementAnalysis.julia_main()
end
