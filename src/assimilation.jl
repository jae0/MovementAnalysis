# ==============================================================================
#  assimilation.jl -- the external inputs, converted into the GeoData store
# ==============================================================================
#
#  Movement analysis starts from three files someone else produced: a mark-recapture
#  telemetry table, a habitat-suitability time series, and a table of spatial units.
#  Nothing in this package can read them until they have been converted, and until now
#  that conversion happened *inside* the analysis, once per run, with JLD2 as both the
#  transport and the cache.
#
#  The separation this file makes:
#
#      data/*.jld2   the externally supplied inputs, read-only, never written to
#      lakehouse/GeoMotion/   the assimilated store, GeoData-native, built once
#
#  The store is a directory with three artefacts plus a manifest:
#
#      hsi.zarr          the habitat-suitability series, (unit x month x draw)
#      tagging.arrow     the telemetry events, one row per release/recapture
#      sppoly.arrow      the spatial units and their coordinates
#      manifest.toml   what was assimilated, from where, and from which bytes
#
#  Arrow for the tables and Zarr for the array, because those are what GeoData's own
#  backends read: `geoload` on the array, `Arrow.Table` on the tables. Nothing here
#  interprets the science -- it moves bytes, records where they came from, and refuses
#  to overwrite a newer artefact.
#
#  Assimilation is idempotent. A second call with no changes is a no-op that reports
#  what it skipped; `force = true` rebuilds. That is the whole point of the manifest: a
#  long run must not silently rebuild a 770 MB array, and a changed input must not be
#  silently reused.
#
#  Nothing here needs a package this workspace does not already have, which is why the
#  manifest is TOML and not JSON, and why freshness is size-and-mtime rather than a
#  digest: adding a dependency would force a re-resolve of a pinned stack that cannot be
#  re-resolved (Zarr 0.10.2 against Blink's WebSockets), so the conversion runs on what is
#  installed.
# ==============================================================================

using DataFrames
using Arrow
using TOML

"""
    movement_store_root() -> String

Where the assimilated movement store lives: `projects/lakehouse/GeoMotion`, beside
the other halves' stores, so the lakehouse layout is one directory per package.

The default is a path relative to this file Ã¢â‚¬â€ `src/` is two levels below the workspace root,
where the lakehouse sits beside each package Ã¢â‚¬â€ so it resolves the same way whether the package
is loaded from the working tree or from Pluto. `MOVEMENT_STORE_ROOT` overrides it for a machine
with the lakehouse somewhere else.
"""
function movement_store_root()::String
    override = get(ENV, "MOVEMENT_STORE_ROOT", "")
    isempty(override) || return abspath(override)
    return normpath(joinpath(@__DIR__, "..", "..", "lakehouse", "GeoMotion"))
end

"""
    movement_store_status(; store_dir = movement_store_root(), input_dir = "data",
                          tagging_file, hsi_file, sppoly_file) -> NamedTuple

What assimilation would do, without doing it: which artefacts are present, which of them
are up to date against their source, and which would be rebuilt. This is what a caller
checks before deciding to run the conversion, and what the notebook shows before asking
the reader to spend the time.
"""
function movement_store_status(;
    store_dir::AbstractString = movement_store_root(),
    input_dir::AbstractString = "data",
    tagging_file::Union{Nothing, AbstractString} = nothing,
    hsi_file::Union{Nothing, AbstractString} = nothing,
    sppoly_file::Union{Nothing, AbstractString} = nothing
)::NamedTuple
    specs = _movement_input_specs(; input_dir = input_dir, tagging_file = tagging_file,
                                  hsi_file = hsi_file, sppoly_file = sppoly_file)
    rows = NamedTuple[]
    for s in specs
        target = joinpath(store_dir, s.target)
        hit = isfile(s.source)
        stored_present = s.kind == :array ? isdir(target) : isfile(target)
        up_to_date = stored_present && hit && (mtime(target) >= mtime(s.source))
        push!(rows, (name = s.name, source = s.source, target = target,
                     source_present = hit, store_present = stored_present,
                     up_to_date = up_to_date))
    end
    return (
        store_dir = abspath(store_dir),
        store_exists = isdir(store_dir),
        artefacts = rows,
        missing_sources = [r.name for r in rows if !r.source_present],
        stale = [r.name for r in rows if r.store_present && !r.up_to_date],
        absent = [r.name for r in rows if !r.store_present],
        needs_work = any(r -> !r.up_to_date, rows)
    )
end

"""
    assimilate_movement_inputs(; ...) -> NamedTuple

Convert the externally supplied inputs into the GeoData store, skipping anything already
up to date.

Three conversions, one per input:

| input | artefact | what moves |
|---|---|---|
| `tagging.jld2` (a `DataFrame`) | `tagging.arrow` | the event table, one row per observation |
| `hsi.jld2` (`sims`, `years`, `auids`) | `hsi.zarr` | the suitability series as a `GeoDataset` |
| `sppoly.jld2` (a `DataFrame`) | `sppoly.arrow` | the unit table |

The array is written through `geosave` with declared dimensions rather than as a blob, so
a reader can ask for `unit 207` or `month 3` without knowing how the array was laid out.
The tables go to Arrow, which handles `PooledArray`/`InlineString` columns that a naive
write would rebuild wrongly Ã¢â‚¬â€ the reason an earlier attempt to read `sppoly.jld2` outside
its original session raised `MethodError` instead of returning its rows.

`manifest.toml` records, per artefact, the source path, its size, its modification time and
the time of assimilation. Freshness is size-and-mtime rather than a checksum: a digest would
need a dependency the pinned stack cannot take, and mtime is what the skip test compares
against, so the manifest records the same thing the test uses.
"""
function assimilate_movement_inputs(;
    input_dir::AbstractString = "data",
    store_dir::AbstractString = movement_store_root(),
    tagging_file::Union{Nothing, AbstractString} = nothing,
    hsi_file::Union{Nothing, AbstractString} = nothing,
    sppoly_file::Union{Nothing, AbstractString} = nothing,
    verbose::Bool = true,
    force::Bool = false
)::NamedTuple
    specs = _movement_input_specs(; input_dir = input_dir, tagging_file = tagging_file,
                                  hsi_file = hsi_file, sppoly_file = sppoly_file)
    missing = [s.name for s in specs if !isfile(s.source)]
    isempty(missing) || error(
        "Input file(s) not found: $(missing). Nothing can be assimilated without its " *
        "source, and a partial store is worse than none because the analysis would run " *
        "on what happened to be there.")

    mkpath(store_dir)
    written = String[]
    skipped = String[]

    for s in specs
        target = joinpath(store_dir, s.target)
        if !force && isfile(s.source)
            present = s.kind == :array ? isdir(target) : isfile(target)
            if present && mtime(target) >= mtime(s.source)
                push!(skipped, s.name)
                verbose && println("  [assimilate] $(s.name): store is up to date, skipped")
                continue
            end
        end
        verbose && println("  [assimilate] $(s.name): $(s.source) -> $(s.target)")
        t0 = time()
        if s.name == "hsi"
            _assimilate_hsi(s.source, target; verbose = verbose)
        else
            # The key of the table this input is documented to hold. A file with several
            # DataFrames in it cannot be selected by count, and a wrong one is silently the
            # wrong analysis.
            _assimilate_table(s.source, target, s.key; verbose = verbose)
        end
        push!(written, s.name)
        verbose && println("  [assimilate] $(s.name): done in " *
                           "$(round(time() - t0, digits = 1))s")
    end

    manifest = Dict{String, Any}(
        "assimilated_utc" => string(now(Dates.UTC)),
        "store_root" => abspath(store_dir),
        "freshness" => "size and mtime of the source; a digest would need a dependency " *
                       "the pinned stack cannot take",
        "artefacts" => [Dict{String, Any}(
            "name" => s.name,
            "source" => abspath(s.source),
            "target" => joinpath(store_dir, s.target),
            "source_bytes" => filesize(s.source),
            "source_modified_utc" => string(unix2datetime(mtime(s.source))),
        ) for s in specs],
    )
    open(joinpath(store_dir, "manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    verbose && println("  [assimilate] manifest.toml written")

    return (
        store_dir = abspath(store_dir),
        written = written,
        skipped = skipped,
        manifest = joinpath(store_dir, "manifest.toml"),
        status = movement_store_status(; store_dir = store_dir, input_dir = input_dir,
                                       tagging_file = tagging_file, hsi_file = hsi_file,
                                       sppoly_file = sppoly_file)
    )
end

# -----------------------------------------------------------------------------
#  internals
# -----------------------------------------------------------------------------

"""
    _movement_input_specs(; ...) -> Vector{NamedTuple}

The three inputs, as (name, source, target, kind). The source resolution is the same
convention `load_movement_dataset` uses Ã¢â‚¬â€ an explicit path wins, otherwise `data/` Ã¢â‚¬â€ so
what a run reads and what assimilation converts cannot drift apart.
"""
function _movement_input_specs(;
    input_dir::AbstractString = "data",
    tagging_file::Union{Nothing, AbstractString} = nothing,
    hsi_file::Union{Nothing, AbstractString} = nothing,
    sppoly_file::Union{Nothing, AbstractString} = nothing
)::Vector{@NamedTuple{name::String, source::String, target::String, kind::Symbol,
                      key::Union{Nothing, String}}}
    pick(explicit, fallback) = isnothing(explicit) ? joinpath(input_dir, fallback) :
        explicit
    return [
        (name = "tagging", source = pick(tagging_file, "tagging.jld2"),
         target = "tagging.arrow", kind = :table, key = "tagging"),
        (name = "hsi", source = pick(hsi_file, "hsi.jld2"),
         target = "hsi.zarr", kind = :array, key = nothing),
        (name = "sppoly", source = pick(sppoly_file, "sppoly.jld2"),
         target = "sppoly.arrow", kind = :table, key = "sppoly"),
    ]
end

"""
    _assimilate_table(source, target, key)

A `DataFrame` in JLD2, moved to Arrow.

The object is selected **by JLD2 key**, not by position. `sppoly.jld2` holds nine objects — a
`DataFrame`, a polygon table, an adjacency matrix, a graph, two centroid arrays and two WKT
string arrays — and `first(values(f))` happened to return the DataFrame. That is not a
contract: a JLD2 file's key order is a hash order, so the same expression picks a different
object in the next release and the conversion would silently move the wrong bytes.

The key is the one the file is documented to hold. A missing key is an error listing what the
file *does* hold, because a name that drifted means either the input changed or the reader
did, and those need different fixes.
"""
function _assimilate_table(source::AbstractString, target::AbstractString,
                           key::AbstractString;
                           verbose::Bool = false)
    df = JLD2.jldopen(source, "r") do f
        # The column types must be in the workspace before the file is opened: a
        # `PooledArray`/`InlineString` column read without them reconstructs as an unusable
        # object, and the failure is a MethodError about `convert` rather than about a
        # missing type.
        haskey(f, key) || error(
            "$(source) holds no '$(key)' key. It holds $(sort(collect(keys(f)))). Either " *
            "the input changed or its key did, and those need different fixes.")
        v = f[key]
        v isa DataFrame || error(
            "$(source) holds '$(key)' as $(typeof(v).name.name), not a DataFrame. " *
            "Assimilation converts the documented layout.")
        v
    end
    isdir(dirname(target)) || mkpath(dirname(target))
    # The existing artefact is removed before the new one is written, because `Arrow.write`
    # refuses an existing file ("Invalid argument" on Windows) and the whole point of
    # `force`, or of a source that changed, is to replace it.
    #
    # The removal can itself fail with permission denied: `Arrow.Table` memory-maps the file
    # it opens, and on Windows a mapped file cannot be unlinked while the mapping is alive.
    # A reader that has finished with the table has copies, so the mapping is garbage --
    # it simply has not been collected. One collection and one retry is therefore enough,
    # and the reason for it is stated rather than the retry looking like superstition.
    _unlink_for_rewrite(target)
    Arrow.write(target, df)
    verbose && println("    $(nrow(df)) rows x $(ncol(df)) columns -> " *
                       "$(basename(target))")
    return target
end

"""
    _unlink_for_rewrite(path)

Remove `path` so a new artefact can be written over it, tolerating the Windows memory-mapping
lock by collecting once.

`Arrow.Table` maps the file it reads, and on Windows a mapped file cannot be unlinked until
the mapping is collected. A caller that has finished with the table holds copies of its data,
so the mapping is garbage by then -- it has merely not been collected. A single collection and
one retry is enough, and the reason is recorded here because a bare retry looks like
superstition.
"""
function _unlink_for_rewrite(path::AbstractString)
    isfile(path) || return nothing
    for attempt in 1:2
        try
            rm(path; force = true)
            return nothing
        catch err
            if attempt == 2
                msg = first(split(sprint(showerror, err), "\n"))
                error("Cannot replace $(path): $(msg). On Windows this usually means " *
                      "another process holds the file open.")
            end
        end
        GC.gc()
        sleep(0.2)
    end
    return nothing
end

"""
    _assimilate_hsi(source, target)

The suitability series, moved to a `GeoDataset`.

The array is `(units x months x draws)` Ã¢â‚¬â€ 707 spatial units, 27 monthly steps, 5000 draws Ã¢â‚¬â€
so the dimensions are spatial, temporal and ensemble. `auids` are the unit identities and
become the spatial coordinate; `years` are recorded as an attribute because they are not a
dimension of this array, they are what `build_monthly_hsi_matrix` turns the month axis into
a calendar from. Storing them together is what makes the store self-describing: a reader
that wanted the annual cycle could not rebuild it from a bare 707 x 27 x 5000 array.
"""
function _assimilate_hsi(source::AbstractString, target::AbstractString; verbose::Bool = false)
    bundle = JLD2.load(source)
    hsi_key = haskey(bundle, "sims") ? "sims" :
              haskey(bundle, "hsi") ? "hsi" :
              haskey(bundle, "predictions") ? "predictions" : error(
                  "$(source) holds none of sims/hsi/predictions; it holds " *
                  "$(sort(collect(keys(bundle)))).")
    sims = bundle[hsi_key]
    years = something(getkey(bundle, "years"), collect(1:size(sims, 2)))
    auids = something(getkey(bundle, "auids"), collect(1:size(sims, 1)))
    length(auids) == size(sims, 1) || error(
        "$(source) has $(length(auids)) unit ids for $(size(sims, 1)) units.")
    length(years) == size(sims, 2) || error(
        "$(source) has $(length(years)) years for $(size(sims, 2)) month steps.")

    crs = GeoData.CoordinateSystem(crs = "EPSG:4326")
    nx, nm, nd = size(sims)
    d_unit = _dim(:unit, collect(Float64, auids), "1", "spatial unit id", GeoData.DIM_SPATIAL)
    d_month = _dim(:month, collect(1.0:nm), "1", "monthly step index", GeoData.DIM_TEMPORAL)
    d_draw = _dim(:draw, collect(1.0:nd), "1", "posterior draw", GeoData.DIM_PARAMETRIC)

    ds = GeoDataset(
        Dict{String, GeoArray}("sims" => GeoArray(Array{Float64}(sims),
                                                  (d_unit, d_month, d_draw), crs,
                                                  Dict{String, Any}())),
        Dict{Symbol, GeoArray}(
            :auids => GeoArray(collect(Float64, auids), (d_unit,), crs, Dict{String, Any}()),
            :month => GeoArray(collect(1.0:nm), (d_month,), crs, Dict{String, Any}()),
            :draw => GeoArray(collect(1.0:nd), (d_draw,), crs, Dict{String, Any}())),
        Dict{Symbol, GeoData.Dimension}(:unit => d_unit, :month => d_month, :draw => d_draw),
        crs,
        Dict{String, Any}(
            "years" => collect(Int, years),
            "layout" => "sims[unit, month, draw]; month is the step index the monthly " *
                        "matrix is built from, and `years` maps it to calendar years",
            "title" => "Assimilated habitat suitability series"),
        GeoData.ZarrBackend(), target)

    GeoData.geosave(target, ds; backend = :zarr, overwrite = true)
    verbose && println("    sims$(size(sims)) + $(length(years)) years, " *
                       "$(length(auids)) units -> $(basename(target))")
    return target
end

_dim(name, coords, units, standard_name, dim_type) = GeoData.Dimension(
    name = name, size = length(coords), coords = coords,
    units = units, standard_name = standard_name, dim_type = dim_type)

getkey(d, k) = haskey(d, k) ? d[k] : nothing

"""
    read_assimilated_movement_inputs(; store_dir = movement_store_root())

The assimilated store, read back through GeoData: the `GeoDataset` for the array and
`Arrow.Table` for the tables.

The HSI dataset is self-describing — the store records the dimensions it was written with, so
`geoload` returns `:unit` (DIM_SPATIAL), `:month` (DIM_TEMPORAL) and `:draw`
(DIM_PARAMETRIC) with their coordinate arrays attached, rather than handing back three
unnamed axes and letting the reader guess which is the spatial one.

`auids` is returned separately because the store names its coordinate array after the
dimension it labels (`:unit`), which is what makes the axis identifiable at all; the name in
the source file is not the name that makes the store readable.
"""
function read_assimilated_movement_inputs(;
    store_dir::AbstractString = movement_store_root()
)::NamedTuple
    hsi_path = joinpath(store_dir, "hsi.zarr")
    tagging_path = joinpath(store_dir, "tagging.arrow")
    sppoly_path = joinpath(store_dir, "sppoly.arrow")
    isdir(store_dir) || error(
        "No assimilated movement store at $(store_dir). Run " *
        "`assimilate_movement_inputs()` first.")
    for (name, path) in (("hsi", hsi_path), ("tagging", tagging_path), ("sppoly", sppoly_path))
        (isdir(path) || isfile(path)) || error(
            "The store at $(store_dir) has no $(name). Run `assimilate_movement_inputs()`.")
    end
    hsi = GeoData.geoload(hsi_path; backend = :zarr)
    auids = haskey(hsi.coords, :unit) ? collect(Int, hsi.coords[:unit].data) : Int[]
    return (
        hsi = hsi,
        auids = auids,
        years = something(get(hsi.attrs, "years", nothing), Int[]),
        tagging = _read_arrow(tagging_path),
        sppoly = _read_arrow(sppoly_path)
    )
end

"""
    _read_arrow(path) -> DataFrame

An Arrow table, read through a buffer rather than a mapping of the file.

`Arrow.Table(path)` memory-maps the file, and on Windows a mapped file cannot be unlinked
while any table built from it is alive -- so a run that read the store could not rewrite it,
and could not delete its scratch either. Reading the bytes and decoding from memory costs a
copy of a 21 MB table and removes the whole class of problem: after this call the file is
closed and nothing refers to it.
"""
function _read_arrow(path::AbstractString)
    isfile(path) || error("The Arrow table at $(path) is missing.")
    return DataFrame(Arrow.Table(IOBuffer(read(path))))
end
