"""
    MovementAnalysisRCallExt

Optional R backend for MovementAnalysis.

R was previously a hard dependency, which meant the package could not precompile
or load on any machine without an R installation — even for analyses that read
nothing but Arrow, CSV, GeoJSON, or shapefiles. It is now a weak dependency: this
extension loads automatically when both `RCall` and a working R are present, and
the base package raises a clear, actionable error when an `.rds`, `.rda`, or
`.qs` input is encountered without it.

The R packages `qs` and `arrow` must be installed in R for `r_to_ipc`; they are
not Julia dependencies and cannot be checked from here.
"""
module MovementAnalysisRCallExt

using MovementAnalysis
using RCall
using Arrow
using DataFrames

# -- tabular R data -----------------------------------------------------------

function MovementAnalysis._r_ipc_convert(filepath::AbstractString, temp_ipc::AbstractString)
    R"""
    local({
        filepath <- $(filepath)
        outpath <- $(temp_ipc)

        ext <- tolower(tools::file_ext(filepath))

        # 1. Parse the R serialization format
        if (ext %in% c("rdz", "qs")) {
            obj <- qs::qread(filepath)
        } else if (ext == "rds") {
            obj <- readRDS(filepath)
        } else if (ext %in% c("rda", "rdata")) {
            env <- new.env()
            load(filepath, envir = env)
            vars <- ls(env)
            if (length(vars) == 0) stop("No objects found in .rda file")
            obj <- env[[vars[1]]]
        } else {
            stop(paste("Unsupported file extension:", ext))
        }

        # 2. Validate tabular structure
        if (!is.data.frame(obj)) {
            stop("Object is not a data.frame. IPC requires tabular data.")
        }

        # 3. Export to intermediate Arrow IPC. zstd compresses blocks well, which
        # keeps the intermediate write small.
        arrow::write_ipc_file(obj, outpath, compression = "zstd")
    })
    """
    return nothing
end

# -- R data for polygon rings -------------------------------------------------

function MovementAnalysis._r_load_object(path::AbstractString)
    val = RCall.reval(R"load($path)")
    # Unwrap the single object an .rds/.RData file normally holds.
    if val isa RCall.RObject && length(RCall.rcopy(val)) == 1
        inner = first(RCall.rcopy(val))
        inner isa RCall.RObject && return RCall.rcopy(inner)
        return inner
    end
    return RCall.rcopy(val)
end

end # module
