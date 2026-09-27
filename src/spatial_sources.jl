# =============================================================================
# Spatial data sources: domain extent, land classification, and region polygons.
#
# Nothing here knows about any particular study area. The domain comes from an
# explicit bounding box or, failing that, from the extent of the input data; land
# comes from a global land/sea mask or from bathymetry; regions of interest come
# from polygon files the user supplies. Species- and region-specific knowledge
# lives in the TOML configuration, not in this file.
# =============================================================================

using CSV
using DataFrames
using GeoDatasets
using GeoInterface
using GeoJSON
using JLD2
using RCall

# =============================================================================
# Bounding box
# =============================================================================

"""
    resolve_bbox(configured, padding_deg, lons, lats) -> NTuple{4, Float64}

Determine the analysis domain as `(west, south, east, north)` in degrees.

A configured `bbox` is authoritative and is only validated. When it is absent the
extent is derived from the supplied coordinates -- the telemetry positions and, if
available, the spatial-unit polygons -- and padded so the mesh is not clipped
flush against the outermost detections.
"""
function resolve_bbox(
    configured::Union{Nothing, AbstractVector{<:Real}},
    padding_deg::Real,
    lons::AbstractVector{<:Real},
    lats::AbstractVector{<:Real}
)::NTuple{4, Float64}

    if configured !== nothing
        length(configured) == 4 || throw(ArgumentError(
            "bbox must be [west, south, east, north]; got $(length(configured)) value(s)"
        ))
        w, s, e, n = (Float64(x) for x in configured)
        all(isfinite, (w, s, e, n)) || throw(ArgumentError(
            "bbox must be finite; got ($w, $s, $e, $n)"
        ))
        (w < e && s < n) || throw(ArgumentError(
            "bbox must satisfy west < east and south < north; got ($w, $s, $e, $n)"
        ))
        return (w, s, e, n)
    end

    isempty(lons) && throw(ArgumentError(
        "Cannot derive a bounding box: no bbox configured and no coordinates available."
    ))

    w = minimum(lons) - padding_deg
    e = maximum(lons) + padding_deg
    s = minimum(lats) - padding_deg
    n = maximum(lats) + padding_deg
    # Clamp to the valid geographic range; padding must not run off the globe.
    return (max(w, -180.0), max(s, -90.0), min(e, 180.0), min(n, 90.0))
end

"""
    bbox_of(points) -> Union{Nothing, NTuple{4, Float64}}

Smallest `(west, south, east, north)` containing the given `(lon, lat)` points,
or `nothing` when there are none.
"""
function bbox_of(points)::Union{Nothing, NTuple{4, Float64}}
    isempty(points) && return nothing
    lons = [Float64(p[1]) for p in points]
    lats = [Float64(p[2]) for p in points]
    return (minimum(lons), minimum(lats), maximum(lons), maximum(lats))
end

# =============================================================================
# Land classification
# =============================================================================

"""
    _basemap_grid_token(grid_minutes) -> String

Render a grid spacing the way matplotlib's basemap names its mask files, so
`5.0` becomes `5` and `10.0` becomes `10`. Whole-minute grids carry no decimal
part in the filename; the fractional grids keep their point (`1.25`, `2.5`).
"""
function _basemap_grid_token(grid_minutes::Real)
    g = Float64(grid_minutes)
    g > 0 || throw(ArgumentError("land mask grid spacing must be positive, got $g"))
    return isinteger(g) ? string(Int(g)) : string(g)
end

const _BASEMAP_MASK_URL = "https://raw.githubusercontent.com/matplotlib/basemap/" *
                          "v1.2.2rel/lib/mpl_toolkits/basemap/data"

# Resolutions basemap ships, and the grids each one is published at.
const _BASEMAP_RESOLUTIONS = Set(['c', 'l', 'i', 'h', 'f'])
const _BASEMAP_GRIDS = (1.25, 2.5, 5.0, 10.0)

# Parsed masks, keyed by (resolution, grid token), so a run pays the gunzip once.
const _LAND_MASK_CACHE = Dict{Tuple{Char,String},Tuple{Vector{Float64},Vector{Float64},Matrix{UInt8}}}()

"""
    _basemap_mask_data(resolution, grid_minutes) -> (lon, lat, data)

Load a basemap land/sea mask as a gzipped, headerless, row-major `UInt8` raster
where 0 is ocean, 1 is land and 2 is lake.

This reads the file directly rather than going through
`GeoDatasets.landseamask`, for two reasons: that function derives its download URL
by interpolating the grid verbatim, so a `Float64` `5.0` requests
`lsmask_5.0min_l.bin` while the published file is `lsmask_5min_l.bin` -- the
mismatched request 404s, and the 14-byte error page it leaves behind then fails
to decompress with a `ZlibError`. The same call also rejects a grid passed as a
string, so the filename cannot be corrected through its API. The indexing below
reproduces GeoDatasets' `nearest_point` arithmetic exactly, so classification is
unchanged.
"""
function _basemap_mask_data(resolution::AbstractString, grid_minutes::Real)
    res = resolution[1]
    res in _BASEMAP_RESOLUTIONS || throw(ArgumentError(
        "land_mask_resolution must be one of c, l, i, h, f; got \"$resolution\""
    ))
    grid = Float64(grid_minutes)
    any(g -> isapprox(g, grid; atol = 1e-9), _BASEMAP_GRIDS) || throw(ArgumentError(
        "land_mask_grid_minutes must be one of 1.25, 2.5, 5.0, 10.0; got $grid"
    ))

    token = _basemap_grid_token(grid)
    key = (res, token)
    haskey(_LAND_MASK_CACHE, key) && return _LAND_MASK_CACHE[key]

    file = joinpath(_basemap_data_dir(), "lsmask_$(token)min_$(res).bin")
    raw = _read_gzip_bytes(file, token, res)

    arcsec = round(Int, grid * 60)
    nlon = length(-180 * 3600 + arcsec / 2:arcsec:180 * 3600 - arcsec / 2)
    nlat = length(-90 * 3600 + arcsec / 2:arcsec:90 * 3600 - arcsec / 2)
    length(raw) == nlon * nlat || throw(ArgumentError(
        "land mask $(basename(file)) holds $(length(raw)) bytes but a " *
        "$(nlon)x$(nlat) raster needs $(nlon * nlat)"
    ))

    lon = (-180 * 3600 + arcsec / 2:arcsec:180 * 3600 - arcsec / 2) ./ 3600
    lat = (-90 * 3600 + arcsec / 2:arcsec:90 * 3600 - arcsec / 2) ./ 3600
    data = reshape(raw, length(lon), length(lat))

    _LAND_MASK_CACHE[key] = (collect(lon), collect(lat), data)
    return _LAND_MASK_CACHE[key]
end

"Directory GeoDatasets keeps its bundled basemap data in."
function _basemap_data_dir()
    dir = joinpath(dirname(dirname(pathof(GeoDatasets))), "data")
    return dir
end

"""
    _read_gzip_bytes(file, token, resolution) -> Vector{UInt8}

Return the decompressed contents of `file`, downloading it into place if absent.

A previous failed request can leave a short error page where the mask belongs, so
a file that is implausibly small is treated as absent and fetched again.
"""
function _read_gzip_bytes(file::AbstractString, token::AbstractString, resolution::Char)
    if !isfile(file) || filesize(file) <= 1024
        url = "$_BASEMAP_MASK_URL/$(basename(file))"
        @info "Fetching global land/sea mask" resolution token url
        mkpath(dirname(file))
        download(url, file)
    end
    return open(GzipDecompressorStream, file) do io
        read(io)
    end
end

"""
    land_mask_from_global_mask(centroids_lonlat; resolution, grid_minutes) -> BitVector

Classify each unit centroid as land using the global land/sea mask from
matplotlib's basemap (originally GMT). The mask is a coarse global raster, so it
answers "is this on a continent or island" without any study-area coastline being
bundled with the package.
"""
function land_mask_from_global_mask(
    centroids_lonlat::AbstractVector;
    resolution::AbstractString = "l",
    grid_minutes::Real = 5.0
)::BitVector
    lon, lat, data = _basemap_mask_data(resolution, grid_minutes)
    lon0, dlon = first(lon), (last(lon) - first(lon)) / (length(lon) - 1)
    lat0, dlat = first(lat), (last(lat) - first(lat)) / (length(lat) - 1)
    mask = falses(length(centroids_lonlat))
    for (k, c) in enumerate(centroids_lonlat)
        x, y = Float64(c[1]), Float64(c[2])
        i = mod1(1 + round(Int, (x - lon0) / dlon), length(lon))
        j = 1 + round(Int, (y - lat0) / dlat)
        mask[k] = data[i, j] == 1
    end
    return mask
end

"""
    land_mask_from_polygon_files(files, centroids_lonlat) -> BitVector

Classify centroids against user-supplied land polygons, one polygon ring per
element of `files` (or every ring found within each file).
"""
function land_mask_from_polygon_files(
    files::AbstractVector{<:AbstractString},
    centroids_lonlat::AbstractVector
)::BitVector
    rings = reduce(vcat, read_polygon_file.(files))
    return BitVector([
        any(r -> point_in_ring(Float64(c[1]), Float64(c[2]), r), rings)
        for c in centroids_lonlat
    ])
end

"""
    point_in_ring(x::Real, y::Real, ring) -> Bool

Even-odd ray casting. `ring` is a closed or unclosed sequence of `(x, y)` vertices.
"""
function point_in_ring(x::Real, y::Real, ring)::Bool
    n = length(ring)
    n < 3 && return false
    inside = false
    j = n
    @inbounds for i in 1:n
        xi, yi = Float64(ring[i][1]), Float64(ring[i][2])
        xj, yj = Float64(ring[j][1]), Float64(ring[j][2])
        if ((yi > y) != (yj > y))
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            x < xint && (inside = !inside)
        end
        j = i
    end
    return inside
end

# =============================================================================
# Region-of-interest polygons
# =============================================================================

"""
    read_polygon_file(path) -> Vector{Vector{NTuple{2, Float64}}}

Read polygon rings from a region or land definition file. Format is chosen by
extension:

  - `.csv`          long format with `lon`/`x` and `lat`/`y` columns, and
                    optional `region`/`id`/`ring` column separating polygons
  - `.jld2`         a `DataFrame` in the same long format, or a vector of rings
  - `.rds`, `.RData`, `.rdata`  read through R; a list of rings or a data frame
                    in the same long format
  - `.shp`          an ESRI shapefile read with `Shapefile`
  - `.geojson`, `.json`  a FeatureCollection, read with `GeoJSON`

Every ring is returned as a vector of `(lon, lat)` vertices. Rings are not closed
for you; the containment test does not require closure.
"""
function read_polygon_file(path::AbstractString)::Vector{Vector{NTuple{2, Float64}}}
    isfile(path) || throw(ArgumentError("Polygon file not found: $path"))
    ext = lowercase(splitext(path)[2])

    if ext == ".csv"
        return _rings_from_table(CSV.read(DataFrame, path))
    elseif ext == ".jld2"
        return _rings_from_object(JLD2.load(path), path)
    elseif ext in (".rds", ".rdata")
        return _rings_from_object(_read_rdata(path), path)
    elseif ext == ".shp"
        return _rings_from_shapefile(path)
    elseif ext in (".geojson", ".json")
        return _rings_from_geojson(path)
    end

    throw(ArgumentError(
        "Unsupported polygon file '$path'. Supported extensions: " *
        ".csv, .jld2, .rds, .RData, .shp, .geojson, .json"
    ))
end

# -- shared table interpretation ---------------------------------------------

const LON_NAMES = (:lon, :long, :longitude, :x)
const LAT_NAMES = (:lat, :latitude, :y)
const GROUP_NAMES = (:region, :region_id, :id, :ring, :polygon, :part, :feature)

function _coord_column(df::DataFrame, candidates, what::AbstractString)
    syms = Symbol.(propertynames(df))
    for c in candidates
        c in syms && return c
    end
    throw(ArgumentError(
        "Polygon table is missing a $what column; looked for " *
        join(candidates, ", ") * " among " * join(Symbol.(propertynames(df)), ", ")
    ))
end

"""
    _rings_from_table(df) -> Vector{Vector{NTuple{2,Float64}}}

Interpret a long-format polygon table: one vertex per row, with optional group
columns separating one polygon from the next. When no group column is present the
whole table is treated as a single ring.
"""
function _rings_from_table(df::DataFrame)::Vector{Vector{NTuple{2, Float64}}}
    nrow(df) == 0 && return Vector{Vector{NTuple{2, Float64}}}()
    loncol = _coord_column(df, LON_NAMES, "longitude")
    latcol = _coord_column(df, LAT_NAMES, "latitude")

    syms = Symbol.(propertynames(df))
    groupcol = findfirst(c -> c in GROUP_NAMES, syms)
    isnothing(groupcol) && return [Vector{NTuple{2, Float64}}([
        (Float64(df[i, loncol]), Float64(df[i, latcol])) for i in 1:nrow(df)
    ])]

    rings = Vector{Vector{NTuple{2, Float64}}}()
    sub = df[!, groupcol]
    for key in unique(sub)
        idx = findall(isequal(key), sub)
        # Preserve file order within each group so ring vertices stay sequential.
        sort!(idx)
        ring = Vector{NTuple{2, Float64}}()
        for i in idx
            push!(ring, (Float64(df[i, loncol]), Float64(df[i, latcol])))
        end
        push!(rings, ring)
    end
    return rings
end

# -- JLD2 / RData -------------------------------------------------------------

function _rings_from_object(obj, path::AbstractString)
    if obj isa DataFrame
        return _rings_from_table(obj)
    elseif obj isa AbstractDict
        # A JLD2 file written with named variables loads as a Dict; use the sole
        # DataFrame or ring list it holds.
        found = nothing
        for v in values(obj)
            (v isa DataFrame || v isa AbstractVector) && (found = v; break)
        end
        found === nothing && throw(ArgumentError(
            "No DataFrame or polygon list found in $path; expected region " *
            "vertices with longitude/latitude columns"
        ))
        return _rings_from_object(found, path)
    elseif obj isa AbstractVector
        isempty(obj) && return Vector{Vector{NTuple{2, Float64}}}[]
        all(v -> v isa Union{Tuple, AbstractVector}, obj) || throw(ArgumentError(
            "Unsupported vector payload in $path; expected a list of polygon rings."
        ))
        return [Vector{NTuple{2, Float64}}(_as_ring(r)) for r in obj]
    end
    throw(ArgumentError(
        "Unsupported payload in $path of type $(typeof(obj)); expected a DataFrame " *
        "in long format, or a list of polygon rings."
    ))
end

function _as_ring(r)
    out = Vector{NTuple{2, Float64}}()
    isempty(r) && return out
    # A single vertex given as [x, y].
    if all(v -> v isa Real, r) && length(r) >= 2
        push!(r, (Float64(r[1]), Float64(r[2])))
        return out
    end
    for v in r
        (v isa Tuple || v isa AbstractVector) && length(v) >= 2 &&
            push!(out, (Float64(v[1]), Float64(v[2])))
    end
    return out
end

function _read_rdata(path::AbstractString)
    val = RCall.reval(R"load($path)")
    # Unwrap the single object an .rds/.RData file normally holds.
    if val isa RCall.RObject && length(RCall.rcopy(val)) == 1
        inner = first(RCall.rcopy(val))
        inner isa RCall.RObject && return RCall.rcopy(inner)
        return inner
    end
    return RCall.rcopy(val)
end

# -- vector formats -----------------------------------------------------------

function _rings_from_shapefile(path::AbstractString)
    handle = Shapefile.Handle(path)
    rings = Vector{Vector{NTuple{2, Float64}}}[]
    for shp in handle.shapes
        geom = GeoInterface.geometry(shp)
        coords = GeoInterface.coordinates(geom)
        for part in _flatten_coords(coords)
            push!(rings, Vector{NTuple{2, Float64}}(part))
        end
    end
    return rings
end

function _rings_from_geojson(path::AbstractString)
    return _collect_geojson(JSON.parsefile(path), Vector{Vector{NTuple{2, Float64}}}[])
end

function _collect_geojson(node, rings)
    node isa AbstractDict || return rings
    t = get(node, "type", "")
    if t == "Polygon"
        # coordinates: [ring, ring, ...], each a list of [lon, lat] positions.
        for ring in get(node, "coordinates", [])
            push!(rings, _flatten_coords(ring))
        end
    elseif t == "MultiPolygon"
        # coordinates: [polygon, ...], each polygon = [ring, ring, ...]
        for polygon in get(node, "coordinates", [])
            for ring in polygon
                push!(rings, _flatten_coords(ring))
            end
        end
    elseif t == "Feature"
        _collect_geojson(get(node, "geometry", nothing), rings)
    elseif t == "FeatureCollection"
        for f in get(node, "features", [])
            _collect_geojson(f, rings)
        end
    elseif t == "GeometryCollection"
        for g in get(node, "geometries", [])
            _collect_geojson(g, rings)
        end
    end
    return rings
end

# Flatten a GeoJSON coordinate nest down to a flat sequence of (x, y) pairs.
function _flatten_coords(c)
    out = Vector{NTuple{2, Float64}}()
    _walk_coords(c, out)
    return out
end

function _walk_coords(c, out)
    if c isa AbstractVector
        if !isempty(c) && all(v -> v isa Real, c) && length(c) >= 2
            push!(out, (Float64(c[1]), Float64(c[2])))
        else
            for v in c
                _walk_coords(v, out)
            end
        end
    end
    return out
end

# =============================================================================
# Region assignment
# =============================================================================

"""
    region_map_from_polygons(centroids_lonlat, region_polygons) -> Vector{Int}

Assign each unit to the first region whose polygons contain its centroid, or `0`
when it falls in none. Index `i` corresponds to `region_polygons[i]`.
"""
function region_map_from_polygons(
    centroids_lonlat::AbstractVector,
    region_polygons::AbstractVector
)::Vector{Int}
    map = zeros(Int, length(centroids_lonlat))
    for (i, c) in enumerate(centroids_lonlat)
        x, y = Float64(c[1]), Float64(c[2])
        for (j, rings) in enumerate(region_polygons)
            any(r -> point_in_ring(x, y, r), rings) && (map[i] = j; break)
        end
    end
    return map
end

"""
    load_region_polygons(files, labels) -> Vector{Vector{Vector{NTuple{2,Float64}}}}

Read one polygon file per region label. The returned vector is parallel to
`labels`, so `result[i]` holds the rings of `labels[i]`.
"""
function load_region_polygons(
    files::AbstractVector{<:AbstractString},
    labels::AbstractVector{<:AbstractString}
)::Vector{Vector{Vector{NTuple{2, Float64}}}}
    length(files) == length(labels) || throw(ArgumentError(
        "region_polygon_files has $(length(files)) entries but region_labels has $(length(labels)); supply exactly one polygon file per region."
    ))
    return [read_polygon_file(f) for f in files]
end