using Pkg

println("Adding PackageCompiler to the default environment...")
Pkg.activate()
Pkg.add("PackageCompiler")
using PackageCompiler

curr_dir = @__DIR__
proj_dir = normpath(joinpath(curr_dir, ".."))
compiled_dir = joinpath(proj_dir, "compiled")

println("Creating optimized, standalone compiled application in $compiled_dir...")
create_app(
    proj_dir,
    compiled_dir;
    executables = ["movement" => "julia_main"],
    filter_stdlibs = true,
    force = true,
    incremental = false,
    cpu_target = "native"
)
println("Compilation successful! Executable is at $(joinpath(compiled_dir, "bin", "movement"))")
