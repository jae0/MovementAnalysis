"""
    spatial_utils.jl

Self-contained spatial partitioning, hexagonal mesh generation, bathymetric
extraction, hydrodynamic modeling, and field resharding for MovementAnalysis.
"""

using LinearAlgebra
using SparseArrays
using NearestNeighbors
using LibGEOS
using DataFrames
using Statistics
using Dates
using CoordRefSystems
using Unitful

# --- Column Detection Utilities ---
const STANDARD_SPATIAL_COORDINATE_PAIRS = [
    (:s_x, :s_y),
    (:plon, :plat),
    (:lon, :lat),
    (:longitude, :latitude),
    (:easting, :northing),
    (:x, :y),
    (:coord_x, :coord_y),
    (:coords_x, :coords_y),
    (:spatial_x, :spatial_y),
    (:s1, :s2)
]

"""
Standard candidate column names for temporal indexing, ordered by preference.
"""
const STANDARD_TEMPORAL_CANDIDATES = [
    :t_idx,
    :year,
    :time,
    :timestamp,
    :date,
    :datetime,
    :t,
    :day,
    :week,
    :month,
    :hour,
    :step
]

"""
    _detect_xy_columns(df; x=nothing, y=nothing)::Tuple{Symbol, Symbol}

Automatically identifies or validates 2D spatial coordinate columns in a tabular dataset
or NamedTuple `df`.

# Process & Mathematical Context
Spatial models and continuous smooths operate on spatial point coordinates
\$\\mathbf{s}_i = (x_i, y_i) \\in \\mathbb{R}^2\$. Users commonly label their coordinate
axes with domain-specific conventions (e.g. `plon`/`plat` for oceanographic bathymetry,
`easting`/`northing` for projected UTM cartesian grids, or `lon`/`lat` for geodetic
data). This function systematically resolves the coordinate column names:
1. If both `x` and `y` are explicitly provided, it validates that they exist in `df`.
2. If either `x` or `y` is unspecified, it matches against known coordinate pairs
   defined in `STANDARD_SPATIAL_COORDINATE_PAIRS`.
3. If no recognized coordinate columns are present, it raises a descriptive `ArgumentError`.

# Inputs
- `df`: Tabular dataset (`DataFrame`, `NamedTuple`, or table-like object).
- `x`: Optional explicit name for the horizontal/x coordinate column.
- `y`: Optional explicit name for the vertical/y coordinate column.

# Outputs
- `Tuple{Symbol, Symbol}`: Resolved `(col_x, col_y)` symbols.
"""
function _detect_xy_columns(
    df; x=nothing, y=nothing, allow_nothing::Bool=false
)::Union{Tuple{Symbol, Symbol}, Nothing}
    # 1. Explicitly provided both x and y
    if !isnothing(x) && !isnothing(y)
        sx, sy = Symbol(x), Symbol(y)
        if !hasproperty(df, sx)
            allow_nothing && return nothing
            error(
                "Specified coordinate column x=:$sx not found in data: " *
                "$(propertynames(df))."
            )
        end
        if !hasproperty(df, sy)
            allow_nothing && return nothing
            error(
                "Specified coordinate column y=:$sy not found in data: " *
                "$(propertynames(df))."
            )
        end
        return (sx, sy)
    end

    # 2. If only one is specified, find matching partner in standard pairs
    if !isnothing(x) && isnothing(y)
        sx = Symbol(x)
        if !hasproperty(df, sx)
            allow_nothing && return nothing
            error(
                "Specified coordinate column x=:$sx not found in data: " *
                "$(propertynames(df))."
            )
        end
        for (cx, cy) in STANDARD_SPATIAL_COORDINATE_PAIRS
            if cx == sx && hasproperty(df, cy)
                return (sx, cy)
            elseif cy == sx && hasproperty(df, cx)
                return (cx, sx)
            end
        end
        for cy in [:s_y, :northing, :lat, :latitude, :y, :plat, :coords_y]
            if hasproperty(df, cy) && cy != sx
                return (sx, cy)
            end
        end
        allow_nothing && return nothing
        error(
            "Coordinate column x=:$sx found, but could not detect matching " *
            "y coordinate column."
        )
    end

    if isnothing(x) && !isnothing(y)
        sy = Symbol(y)
        if !hasproperty(df, sy)
            allow_nothing && return nothing
            error(
                "Specified coordinate column y=:$sy not found in data: " *
                "$(propertynames(df))."
            )
        end
        for (cx, cy) in STANDARD_SPATIAL_COORDINATE_PAIRS
            if cy == sy && hasproperty(df, cx)
                return (cx, sy)
            elseif cx == sy && hasproperty(df, cy)
                return (sy, cy)
            end
        end
        for cx in [:s_x, :plon, :lon, :longitude, :easting, :x, :coords_x]
            if hasproperty(df, cx) && cx != sy
                return (cx, sy)
            end
        end
        allow_nothing && return nothing
        error(
            "Coordinate column y=:$sy found, but could not detect matching " *
            "x coordinate column."
        )
    end

    # 3. Neither specified: iterate standard candidate pairs
    for (cx, cy) in STANDARD_SPATIAL_COORDINATE_PAIRS
        if hasproperty(df, cx) && hasproperty(df, cy)
            return (cx, cy)
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect spatial coordinate columns in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify coordinate columns explicitly " *
        "(e.g., x=:lon, y=:lat or x=:plon, y=:plat)."
    )
end

"""
    _detect_time_column(df; time_var=nothing, allow_nothing=false)::Union{Symbol, Nothing}

Automatically identifies or validates the temporal index or date column in a dataset.

# Inputs
- `df`: Tabular dataset.
- `time_var`: Optional explicit name for the temporal column.
- `allow_nothing`: If true, returns `nothing` when not detected instead of error.

# Outputs
- `Union{Symbol, Nothing}`: Resolved temporal column name or nothing.
"""
function _detect_time_column(
    df; time_var=nothing, allow_nothing::Bool=false
)::Union{Symbol, Nothing}
    if !isnothing(time_var)
        st = Symbol(time_var)
        if !hasproperty(df, st)
            allow_nothing && return nothing
            error(
                "Specified temporal column time_var=:$st not found in data: " *
                "$(propertynames(df))."
            )
        end
        return st
    end

    for t_cand in STANDARD_TEMPORAL_CANDIDATES
        if hasproperty(df, t_cand)
            return t_cand
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect temporal column in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify temporal column explicitly " *
        "(e.g., time_var=:year or time_var=:time)."
    )
end

const STANDARD_SPATIAL_UNIT_CANDIDATES = [
    :s_idx,
    :region,
    :district,
    :county,
    :area,
    :area_id,
    :zone,
    :unit,
    :unit_id,
    :spatial_unit,
    :au,
    :au_idx,
    :polygon_id,
    :id,
    :site,
    :location,
    :station
]

"""
    _detect_spatial_unit_column(df; s_idx_var=nothing, allow_nothing=false)::Union{Symbol, Nothing}

Automatically identifies or validates the spatial unit or areal index column in a dataset.

# Process
1. If `s_idx_var` is explicitly specified, validates that it exists in `df` and returns it.
2. Otherwise, matches against `STANDARD_SPATIAL_UNIT_CANDIDATES`.
3. If not found and `allow_nothing=false`, raises a descriptive `error`.

# Inputs
- `df`: Tabular dataset.
- `s_idx_var`: Optional explicit column name.
- `allow_nothing`: If true, returns `nothing` when not detected instead of error.

# Outputs
- `Union{Symbol, Nothing}`: Resolved column symbol or nothing.
"""
function _detect_spatial_unit_column(
    df; s_idx_var=nothing, allow_nothing::Bool=false
)::Union{Symbol, Nothing}
    if !isnothing(s_idx_var)
        sym = Symbol(s_idx_var)
        if !hasproperty(df, sym)
            allow_nothing && return nothing
            error(
                "Specified spatial unit column s_idx_var=:$sym not found in data: " *
                "$(propertynames(df))."
            )
        end
        return sym
    end

    for cand in STANDARD_SPATIAL_UNIT_CANDIDATES
        if hasproperty(df, cand)
            return cand
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect spatial unit column in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify spatial unit variable explicitly " *
        "(e.g., `random(region, model=:icar)` or s_idx_var=:region)."
    )
end

# --- Coordinate Conversion Utilities ---
const _HAS_COORDSYS = true
"""
    lonlat_to_xy_km(lon, lat; center_lon=nothing, center_lat=nothing, crs=nothing, datum=WGS84Latest)
    -> Tuple{Float64, Float64}

Projects geographic coordinates (longitude, latitude in decimal degrees) into a planar
Cartesian coordinate space in kilometers, optionally relative to a local tangent origin.

# Mathematical Formulation
When `crs` is supplied and `CoordRefSystems` is available, coordinates are converted
via geodesic projection. Otherwise, a local equirectangular tangent plane approximation
is computed:
```math
x = R \\cdot (\\lambda - \\lambda_0) \\cdot \\cos(\\phi_0) \\cdot \\frac{\\pi}{180^\\circ}
```
```math
y = R \\cdot (\\phi - \\phi_0) \\cdot \\frac{\\pi}{180^\\circ}
```
where \$R = 6371.0\\text{ km}\$ is Earth's mean radius, \$(\\lambda_0, \\phi_0)\$ is the
tangent projection origin (`center_lon`, `center_lat`), and \$(\\lambda, \\phi)\$ are
the input longitude and latitude.

# Arguments
- `lon::Real`: Longitude in decimal degrees.
- `lat::Real`: Latitude in decimal degrees.
- `center_lon`: Tangent plane projection center longitude (default `nothing`).
- `center_lat`: Tangent plane projection center latitude (default `nothing`).
- `crs`: Optional target Coordinate Reference System from `CoordRefSystems`.
- `datum`: Geographic datum (default `WGS84Latest`).

# Returns
- `Tuple{Float64, Float64}`: Planar coordinates `(x_km, y_km)` in kilometers.
"""
function lonlat_to_xy_km(
    lon::Real, lat::Real;
    center_lon = nothing,
    center_lat = nothing,
    crs = nothing,
    datum = WGS84Latest
)::Tuple{Float64, Float64}
    if !isnothing(crs) && _HAS_COORDSYS
        try
            source_pt = LatLon{datum}(lat, lon)
            proj_pt = convert(crs, source_pt)

            x_km = Float64(ustrip(u"km", proj_pt.x))
            y_km = Float64(ustrip(u"km", proj_pt.y))

            # If a local custom center is provided, subtract it to get relative offsets
            if !isnothing(center_lon) && !isnothing(center_lat)
                center_pt = LatLon{datum}(center_lat, center_lon)
                proj_center = convert(crs, center_pt)
                x_km -= Float64(ustrip(u"km", proj_center.x))
                y_km -= Float64(ustrip(u"km", proj_center.y))
            end

            return (x_km, y_km)
        catch
        end
    end
    
    # Fallback to local tangent plane (requires a center to function, defaults to 0.0)
    c_lon = isnothing(center_lon) ? 0.0 : Float64(center_lon)
    c_lat = isnothing(center_lat) ? 0.0 : Float64(center_lat)
    
    R  = 6371.0
    ϕ0 = c_lat * (π / 180.0)
    x  = (Float64(lon) - c_lon) * cos(ϕ0) * (π / 180.0) * R
    y  = (Float64(lat) - c_lat) * (π / 180.0) * R
    return (x, y)
end

"""
    xy_km_to_lonlat(x_km, y_km; center_lon=nothing, center_lat=nothing, crs=nothing, datum=WGS84Latest)
    -> Tuple{Float64, Float64}

Inverts planar Cartesian coordinates in kilometers back to geographic coordinates
(longitude, latitude in decimal degrees).

# Mathematical Formulation
For the local equirectangular tangent plane fallback:
```math
\\lambda = \\lambda_0 + \\frac{x}{R \\cdot \\cos(\\phi_0) \\cdot (\\pi / 180^\\circ)}
```
```math
\\phi = \\phi_0 + \\frac{y}{R \\cdot (\\pi / 180^\\circ)}
```
where \$R = 6371.0\\text{ km}\$ is Earth's mean radius and \$(\\lambda_0, \\phi_0)\$ is
the tangent plane projection origin (`center_lon`, `center_lat`).

# Arguments
- `x::Real`: Planar x-coordinate in kilometers.
- `y::Real`: Planar y-coordinate in kilometers.
- `center_lon`: Tangent plane projection center longitude (default `nothing`).
- `center_lat`: Tangent plane projection center latitude (default `nothing`).
- `crs`: Optional target Coordinate Reference System from `CoordRefSystems`.
- `datum`: Geographic datum (default `WGS84Latest`).

# Returns
- `Tuple{Float64, Float64}`: Geographic coordinates `(lon, lat)` in decimal degrees.
"""
function xy_km_to_lonlat(
    x::Real, y::Real;
    center_lon = nothing,
    center_lat = nothing,
    crs = nothing,
    datum = WGS84Latest
)::Tuple{Float64, Float64}
    if !isnothing(crs) && _HAS_COORDSYS
        try
            abs_x = x * 1.0u"km"
            abs_y = y * 1.0u"km"

            # If a local custom center was used, add it back to get absolute CRS coordinates
            if !isnothing(center_lon) && !isnothing(center_lat)
                center_pt = LatLon{datum}(center_lat, center_lon)
                proj_center = convert(crs, center_pt)
                abs_x += proj_center.x
                abs_y += proj_center.y
            end

            proj_pt = crs(abs_x, abs_y)
            lonlat_pt = convert(LatLon{datum}, proj_pt)

            out_lon = Float64(ustrip(u"°", lonlat_pt.lon))
            out_lat = Float64(ustrip(u"°", lonlat_pt.lat))

            return (out_lon, out_lat)
        catch
        end
    end
    
    # Fallback local tangent plane
    c_lon = isnothing(center_lon) ? 0.0 : Float64(center_lon)
    c_lat = isnothing(center_lat) ? 0.0 : Float64(center_lat)
    
    R  = 6371.0
    ϕ0 = c_lat * (π / 180.0)
    lon = c_lon + (Float64(x) / (R * cos(ϕ0))) * (180.0 / π)
    lat = c_lat + (Float64(y) / R) * (180.0 / π)
    return (lon, lat)
end


"""
    assign_spatial_units(s_x, s_y; area_method=:hexagonal, target_units=10, kwargs...)

Partitions planar or geographic point coordinates into discrete areal units.
"""
function assign_spatial_units(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real};
    area_method::Symbol = :hexagonal, target_units::Union{Nothing, Integer} = 10,
    exact_units::Bool = false, kwargs...
)::NamedTuple
    n_u = !isnothing(target_units) ? Int(target_units) : 10
    dx = maximum(s_x) - minimum(s_x)
    dy = maximum(s_y) - minimum(s_y)
    domain_area = max(1.0, dx * dy)
    unit_area = domain_area / n_u
    r_km = sqrt(unit_area / (3.0 * sqrt(3.0) / 2.0))
    mesh = build_hex_mesh_planar(s_x, s_y; radius_km = max(1.0, r_km))
    return mesh
end

# --- Hexagonal Mesh & Point Mapping ---
function build_hex_mesh_planar(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    radius_km::Real = 5.0,
    crs = nothing,
    datum = WGS84Latest
)::NamedTuple

    center_lon = mean(lon_vec)
    center_lat = mean(lat_vec)

    # Calculate planar coordinates directly
    pts_km = [lonlat_to_xy_km(Float64(lon), Float64(lat);
                  center_lon=center_lon, center_lat=center_lat, 
                  crs=crs, datum=datum)
              for (lon, lat) in zip(lon_vec, lat_vec)]

    r  = Float64(radius_km)
    dx = sqrt(3.0) * r
    dy = 1.5 * r

    # Determine planar domain bounds (expanded by r to ensure complete boundary coverage)
    x_coords = [p[1] for p in pts_km]
    y_coords = [p[2] for p in pts_km]
    x_min, x_max = extrema(x_coords)
    y_min, y_max = extrema(y_coords)

    x_min -= r
    x_max += r
    y_min -= r
    y_max += r

    row_min = floor(Int, y_min / dy)
    row_max = ceil(Int, y_max / dy)

    centroids_km = Tuple{Float64, Float64}[]
    for row in row_min:row_max
        yk   = row * dy
        xoff = isodd(row) ? (dx / 2.0) : 0.0
        col_min = floor(Int, (x_min - xoff) / dx)
        col_max = ceil(Int, (x_max - xoff) / dx)
        for col in col_min:col_max
            xk = col * dx + xoff
            push!(centroids_km, (xk, yk))
        end
    end

    sort!(centroids_km, by = c -> (c[2], c[1]))
    S = length(centroids_km)

    polygons_km      = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    polygons_lonlat  = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    centroids_lonlat = Vector{Tuple{Float64, Float64}}(undef, S)

    # Flat-top hexagon: vertices at 30°, 90°, 150°, 210°, 270°, 330°
    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)

    for (i, (cx, cy)) in enumerate(centroids_km)
        verts_km = [(cx + r * cos(a), cy + r * sin(a)) for a in hex_angles]
        push!(verts_km, verts_km[1]) # Close the polygon
        
        polygons_km[i]      = verts_km
        polygons_lonlat[i]  = [xy_km_to_lonlat(v[1], v[2];
                                    center_lon=center_lon, center_lat=center_lat,
                                    crs=crs, datum=datum)
                                for v in verts_km]
        
        centroids_lonlat[i] = xy_km_to_lonlat(cx, cy;
                                    center_lon=center_lon, center_lat=center_lat,
                                    crs=crs, datum=datum)
    end

    # Efficient matrix allocation for KDTree
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = centroids_km[i][1]
        c_mat[2, i] = centroids_km[i][2]
    end
    
    tree       = KDTree(c_mat)
    adj_thresh = sqrt(3.0) * r * 1.05

    rows_idx = Int[]
    cols_idx = Int[]
    for i in 1:S
        nbrs = inrange(tree, [centroids_km[i][1], centroids_km[i][2]], adj_thresh)
        for j in nbrs
            if j != i
                push!(rows_idx, i)
                push!(cols_idx, j)
            end
        end
    end
    
    W = sparse(rows_idx, cols_idx, ones(Float64, length(rows_idx)), S, S)
    W = max.(W, W')

    area_km2 = (3.0 * sqrt(3.0) / 2.0) * r^2

    return (
        centroids        = centroids_lonlat,
        centroids_km     = centroids_km,
        centroids_lonlat = centroids_lonlat,
        polygons         = polygons_lonlat,
        polygons_km      = polygons_km,
        polygons_lonlat  = polygons_lonlat,
        n_units          = S,
        W                = W,
        radius_km        = r,
        areas_km2        = fill(area_km2, S),
        center_lon       = center_lon,
        center_lat       = center_lat
    )
end

# ── point → unit mapping ───────────────────────────────────────────────────


"""
    map_point_to_units(
        tagging, centroids_km, center_lon, center_lat; crs=nothing, datum=WGS84Latest
    ) -> DataFrame

Assign each point observation to the nearest hexagonal unit via a KDTree on
planar km centroids. Adds `:s_idx` (1-based integer, 1 ≤ s ≤ S).

# Arguments
- `tagging`: Must contain `:lon`, `:lat` (geographic degrees).
- `centroids_km`: Vector of (x, y) km tuples from `build_hex_mesh_planar`.
- `center_lon, center_lat`: Projection origin matching the mesh.
- `crs`: Target Coordinate Reference System (default nothing).
- `datum`: The geographic datum (default: `WGS84Latest`).
- `land_mask`: Optional boolean vector of length `S` (`true` for land units).
- `W`: Optional adjacency matrix (size `S × S`) to identify disconnected units.

# Returns
- Copy of `tagging` with `:s_idx::Int` appended.
"""
function map_point_to_units(
    tagging::DataFrame,
    centroids_km::Vector{Tuple{Float64, Float64}},
    center_lon::Real,
    center_lat::Real;
    target_col::Union{Symbol, AbstractString, Nothing} = nothing,
    crs = nothing,
    datum = WGS84Latest,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    W::Union{Nothing, AbstractMatrix} = nothing
)::DataFrame
    S = length(centroids_km)

    # Filter to active navigable units if masks or topology are provided
    active_mask = trues(S)
    if land_mask !== nothing
        active_mask .&= .!land_mask
    end
    if W !== nothing
        deg = vec(sum(W, dims=2))
        active_mask .&= (deg .> 0.0)
    end
    active_indices = findall(active_mask)
    if isempty(active_indices)
        active_indices = collect(1:S)
    end

    n_act = length(active_indices)
    c_mat = Matrix{Float64}(undef, 2, n_act)
    for (col, idx) in enumerate(active_indices)
        c_mat[1, col] = centroids_km[idx][1]
        c_mat[2, col] = centroids_km[idx][2]
    end

    # Build the KDTree for fast spatial lookups on navigable marine units
    tree = KDTree(c_mat)

    # Pre-allocate and populate the telemetry points matrix
    N = nrow(tagging)
    pts_mat = Matrix{Float64}(undef, 2, N)
    col_x, col_y = _detect_xy_columns(tagging)
    for (i, (lon, lat)) in enumerate(zip(tagging[!, col_x], tagging[!, col_y]))
        x, y = lonlat_to_xy_km(
            Float64(lon), Float64(lat);
            center_lon=center_lon, center_lat=center_lat,
            crs=crs, datum=datum
        )
        pts_mat[1, i] = x
        pts_mat[2, i] = y
    end

    # Perform nearest neighbor search for all points simultaneously
    idxs, _ = knn(tree, pts_mat, 1)

    # Append mapped units to a copy of the DataFrame
    out = copy(tagging)
    mapped_s = [active_indices[first(idx)] for idx in idxs]
    out[!, :s_idx] = mapped_s
    if !isnothing(target_col)
        out[!, Symbol(target_col)] = mapped_s
    end

    return out
end

# --- Bathymetry & Hydrodynamic Dataset Extraction ---
"""
    load_open_bathymetry(;
        source = :synthetic,
        bbox = (-68.0, -57.0, 42.0, 48.0),
        grid_resolution = (60, 50),
        seed = 42,
        crs = nothing
    ) -> NamedTuple

Ingest or synthesize high-resolution open-sourced bathymetry data for coastal
shelf environments (e.g., Scotian Shelf, Cabot Strait, and Gulf of St. Lawrence).

# Mathematical & Physical Foundation
Seafloor elevation ``z_{\\text{bottom}}(\\mathbf{s})`` satisfies:
```math
z_{\\text{bottom}}(\\mathbf{s}) \\le 0, \\quad H(\\mathbf{s}) = -z_{\\text{bottom}}(\\mathbf{s})
```
where ``H(\\mathbf{s})`` is the water-column depth in meters.
Topographic slope is evaluated as:
```math
\\text{Slope}(\\mathbf{s}) = \\arctan\\left( \\sqrt{ \\left(\\frac{\\partial z}{\\partial x}\\right)^2 + \\left(\\frac{\\partial z}{\\partial y}\\right)^2 } \\right)
```

# Arguments
- `source`: Bathymetric data source. Supported options:
  - `:synthetic` (default): Realistic regional shelf synthesis featuring coastal
    shallows (0–40 m), offshore banks (30–70 m), basins (150–250 m), submarine
    canyons (e.g. The Gully), shelf break (200 m), and continental slope (up to 2500 m).
  - `filepath::AbstractString`: Path to a local `.nc`, `.tif`, `.csv`, `.duckdb`, or `.jld2` file.
- `bbox::Tuple{Real, Real, Real, Real}`: Geographic bounding box `(min_lon, max_lon, min_lat, max_lat)`.
- `grid_resolution::Tuple{Int, Int}`: Regular grid dimensions `(nx, ny)`. Default: `(60, 50)`.
- `seed::Int`: Random seed for synthetic bathymetric perturbations.
- `crs`: Optional coordinate reference system identifier.

# Returns
A `NamedTuple` containing:
- `lons::Vector{Float64}`: 1D vector of grid longitudes (length `nx`).
- `lats::Vector{Float64}`: 1D vector of grid latitudes (length `ny`).
- `depth::Matrix{Float64}`: 2D matrix of water column depths in meters (positive down, size `nx × ny`).
- `elevation::Matrix{Float64}`: 2D matrix of seafloor elevation in meters (negative below sea level, size `nx × ny`).
- `slope::Matrix{Float64}`: 2D matrix of topographic seabed slope in degrees.
- `is_land::BitMatrix`: Boolean mask indicating terrestrial units (`elevation >= 0.0`).
- `centroids::Vector{Tuple{Float64, Float64}}`: Centroids for each marine cell.
- `polygons::Vector{Vector{Tuple{Float64, Float64}}}`: Closed bounding polygon vertex rings for each cell.
- `depth_vec::Vector{Float64}`: 1D vector of marine unit depths matching `centroids`.
- `bbox::Tuple{Float64, Float64, Float64, Float64}`: Bounding box.
"""
function load_open_bathymetry(;
    source::Union{Symbol, AbstractString} = :synthetic,
    bbox::Union{Nothing, Tuple{<:Real, <:Real, <:Real, <:Real}} = nothing,
    lon_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    lat_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    grid_resolution::Union{Nothing, Tuple{Int, Int}} = nothing,
    resolution_deg::Union{Nothing, Real} = nothing,
    seed::Int = 42,
    crs = nothing
)::NamedTuple
    actual_bbox = if bbox !== nothing
        Float64.(bbox)
    elseif lon_range !== nothing && lat_range !== nothing
        (Float64(lon_range[1]), Float64(lon_range[2]), Float64(lat_range[1]), Float64(lat_range[2]))
    else
        (-68.0, -57.0, 42.0, 48.0)
    end
    min_lon, max_lon, min_lat, max_lat = actual_bbox

    actual_res = if grid_resolution !== nothing
        grid_resolution
    elseif resolution_deg !== nothing
        nx_calc = max(4, round(Int, (max_lon - min_lon) / Float64(resolution_deg)))
        ny_calc = max(4, round(Int, (max_lat - min_lat) / Float64(resolution_deg)))
        (nx_calc, ny_calc)
    else
        (60, 50)
    end
    nx, ny = actual_res

    @assert nx >= 4 && ny >= 4 "Grid resolution must be at least (4, 4)"
    @assert min_lon < max_lon "Bounding box min_lon must be less than max_lon"
    @assert min_lat < max_lat "Bounding box min_lat must be less than max_lat"

    lons = collect(range(min_lon, max_lon, length=nx))
    lats = collect(range(min_lat, max_lat, length=ny))
    dx = (max_lon - min_lon) / max(1, nx - 1)
    dy = (max_lat - min_lat) / max(1, ny - 1)

    elev = zeros(Float64, nx, ny)
    loaded_from_file = false

    # Check for file-based ingestion
    if source isa AbstractString && isfile(source)
        ext = lowercase(splitext(source)[2])
        try
            if ext == ".csv"
                df = CSV.read(source, DataFrame)
                lon_col = filter(c -> occursin("lon", lowercase(string(c))), names(df))
                lat_col = filter(c -> occursin("lat", lowercase(string(c))), names(df))
                z_col   = filter(c -> occursin("elev", lowercase(string(c))) ||
                                      occursin("depth", lowercase(string(c))) ||
                                      occursin("z", lowercase(string(c))), names(df))
                if !isempty(lon_col) && !isempty(lat_col) && !isempty(z_col)
                    pts = [Float64.([df[i, first(lon_col)], df[i, first(lat_col)]]) for i in 1:nrow(df)]
                    tree = KDTree(hcat(pts...))
                    is_depth = occursin("depth", lowercase(string(first(z_col))))
                    for j in 1:ny, i in 1:nx
                        idx, _ = knn(tree, [lons[i], lats[j]], 1)
                        val = Float64(df[first(idx), first(z_col)])
                        elev[i, j] = is_depth ? -abs(val) : val
                    end
                    loaded_from_file = true
                end
            elseif ext == ".jld2"
                d = JLD2.load(source)
                key = haskey(d, "elevation") ? "elevation" : (haskey(d, "bathymetry") ? "bathymetry" : nothing)
                if key !== nothing
                    raw_elev = d[key]
                    if size(raw_elev) == (nx, ny)
                        elev .= Float64.(raw_elev)
                        loaded_from_file = true
                    end
                end
            end
        catch err
            @warn "Failed to parse open bathymetry file '$(source)': $(err). Falling back to synthetic shelf model."
        end
    end

    if !loaded_from_file
        # Realistic continental shelf synthetic bathymetry
        rng = MersenneTwister(seed)
        for j in 1:ny
            y_norm = (lats[j] - min_lat) / (max_lat - min_lat)
            for i in 1:nx
                x_norm = (lons[i] - min_lon) / (max_lon - min_lon)

                # Distance from shelf-edge line (roughly southwest to northeast)
                # Shelf edge runs from (0.0, 0.35) to (1.0, 0.70)
                shelf_edge_y = 0.35 + 0.35 * x_norm
                dist_to_slope = y_norm - shelf_edge_y

                base_elev = if dist_to_slope < -0.05
                    # Continental Slope and Abyss (deep ocean)
                    slope_t = clamp((-dist_to_slope - 0.05) / 0.35, 0.0, 1.0)
                    -200.0 - 2200.0 * (slope_t ^ 1.8)
                else
                    # Continental Shelf Platform: depth typically 50m to 220m
                    shelf_t = clamp(dist_to_slope / 0.6, 0.0, 1.0)
                    # Outer shelf banks (shallow offshore features)
                    bank_signal = 55.0 * sin(3.0 * π * x_norm) * cos(2.5 * π * y_norm)
                    # Central shelf basins / troughs
                    basin_signal = -80.0 * exp(-((x_norm - 0.5)^2 + (y_norm - 0.55)^2) / 0.04)
                    # Coastal shallowing
                    coastal_rise = 110.0 * (shelf_t ^ 1.2)
                    -170.0 + bank_signal + basin_signal + coastal_rise
                end

                # Submarine Canyon cut (e.g., The Gully at x ≈ 0.65)
                canyon_dist = abs(x_norm - 0.65)
                if canyon_dist < 0.08 && dist_to_slope < 0.15
                    canyon_depth = 450.0 * (1.0 - canyon_dist / 0.08) * max(0.0, 0.15 - dist_to_slope) / 0.15
                    base_elev -= canyon_depth
                end

                # Micro-topographic soundings roughness
                roughness = 4.0 * (rand(rng) - 0.5)
                elev[i, j] = min(5.0, base_elev + roughness)
            end
        end
    end

    # Water column depth: H = max(0.0, -elevation)
    depth = zeros(Float64, nx, ny)
    is_land = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        if elev[i, j] >= 0.0
            is_land[i, j] = true
            depth[i, j] = 0.0
        else
            depth[i, j] = -elev[i, j]
        end
    end

    # Calculate topographic slope: arctan(sqrt(dz_dx^2 + dz_dy^2))
    slope = zeros(Float64, nx, ny)
    for j in 1:ny
        lat_rad = deg2rad(lats[j])
        dx_m = dx * 111320.0 * cos(lat_rad)
        dy_m = dy * 110540.0
        for i in 1:nx
            dz_dx = if i == 1
                (elev[2, j] - elev[1, j]) / dx_m
            elseif i == nx
                (elev[nx, j] - elev[nx - 1, j]) / dx_m
            else
                (elev[i + 1, j] - elev[i - 1, j]) / (2.0 * dx_m)
            end

            dz_dy = if j == 1
                (elev[i, 2] - elev[i, 1]) / dy_m
            elseif j == ny
                (elev[i, ny] - elev[i, ny - 1]) / dy_m
            else
                (elev[i, j + 1] - elev[i, j - 1]) / (2.0 * dy_m)
            end

            grad_mag = sqrt(dz_dx^2 + dz_dy^2)
            slope[i, j] = rad2deg(atan(grad_mag))
        end
    end

    # Build regular grid polygon cells for seamless LibGEOS geometric resharding
    centroids = Tuple{Float64, Float64}[]
    polygons = Vector{Vector{Tuple{Float64, Float64}}}()
    depth_vec = Float64[]

    half_dx = dx / 2.0
    half_dy = dy / 2.0

    for j in 1:ny, i in 1:nx
        cx = lons[i]
        cy = lats[j]
        # Closed counter-clockwise rectangular bounding box
        poly = [
            (cx - half_dx, cy - half_dy),
            (cx + half_dx, cy - half_dy),
            (cx + half_dx, cy + half_dy),
            (cx - half_dx, cy + half_dy),
            (cx - half_dx, cy - half_dy)
        ]
        push!(centroids, (cx, cy))
        push!(polygons, poly)
        push!(depth_vec, depth[i, j])
    end

    au = (
        centroids = centroids,
        centroids_lonlat = centroids,
        polygons = polygons,
        polygons_lonlat = polygons
    )

    return (
        lons = lons,
        lats = lats,
        grid_lon = [c[1] for c in centroids],
        grid_lat = [c[2] for c in centroids],
        depth = depth,
        depths = depth_vec,
        depth_vec = depth_vec,
        elevation = elev,
        slope = slope,
        slopes = vec(slope),
        is_land = is_land,
        centroids = centroids,
        polygons = polygons,
        au = au,
        bbox = actual_bbox,
        crs = crs
    )
end

"""
    extract_hydrodynamic_dataset(
        hydro_input::Any = nothing;
        bathymetry_data::Union{Nothing, NamedTuple} = nothing,
        bbox::Tuple{<:Real, <:Real, <:Real, <:Real} = (-68.0, -57.0, 42.0, 48.0),
        grid_resolution::Tuple{Int, Int} = (60, 50),
        depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0, -150.0, -250.0],
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = nothing,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = nothing
    ) -> NamedTuple

Extract, compute, and format 3D/4D ocean hydrodynamic fields at specific depths and times.

# Physical Formulations
- **Potential Density** (UNESCO Linear Equation of State):
  ```math
  \\rho(T, S) = \\rho_0 \\left[ 1 - \\alpha (T - T_0) + \\beta (S - S_0) \\right]
  ```
  with ``\\rho_0 = 1025.0\\text{ kg/m}^3``, ``\\alpha = 2.0 \\times 10^{-4}\\text{ K}^{-1}``,
  ``\\beta = 7.6 \\times 10^{-4}\\text{ PSU}^{-1}``.
- **Salinity Stratification & Brunt-Väisälä Frequency**:
  ```math
  N^2 = -\\frac{g}{\\rho_0} \\frac{\\partial \\rho}{\\partial z} \\approx g \\left( \\alpha \\frac{\\partial T}{\\partial z} - \\beta \\frac{\\partial S}{\\partial z} \\right)
  ```
- **Turbulent Eddy Diffusivity & Viscosity**:
  ```math
  \\kappa_v(z) = \\kappa_{\\text{surf}} e^{z / h_{\\text{mix}}} + \\frac{\\kappa_{\\text{bkg}}}{1 + 10 Ri} + \\kappa_{\\text{bbl}} e^{-(H + z) / h_{\\text{bbl}}}
  ```
  where ``Ri = N^2 / [(\\partial u / \\partial z)^2 + (\\partial v / \\partial z)^2]``.

# Arguments
- `hydro_input`: Ocean circulation model instance, NamedTuple, Dict, or `nothing`.
- `bathymetry_data`: Optional bathymetric dataset from `load_open_bathymetry`.
- `bbox`: Geographic spatial domain `(min_lon, max_lon, min_lat, max_lat)`.
- `grid_resolution`: Grid dimensions `(nx, ny)`.
- `depths`: Vertical depth coordinate levels (m, negative below surface).
- `depth`: Target continuous depth for 2D slice extraction.
- `depth_level`: Target vertical level index (1 = surface).
- `time_seconds`: Simulation time in seconds.
- `time_index`: Snapshot time index.

# Returns
A `NamedTuple` containing:
- `lons`, `lats`, `depths`: Coordinate axes.
- `temperature`, `salinity`, `density`: 3D scalar fields (`nx × ny × nz`).
- `stratification_N2`, `salinity_gradient`: 3D stratification diagnostics (`nx × ny × nz`).
- `u`, `v`, `w`, `speed`: 3D velocity fields (`nx × ny × nz`).
- `diffusivity_v`, `viscosity_v`: 3D turbulent mixing fields (`nx × ny × nz`).
- `elevation`: 2D sea surface height (m).
- `bathymetry`: 2D seafloor elevation (m).
- `slice_2d`: NamedTuple containing 2D horizontal slices of all fields at the requested `depth`.
- `centroids`, `polygons`: Areal unit representation for LibGEOS resharding.
"""
function extract_hydrodynamic_dataset(
    hydro_input::Any = nothing;
    bathymetry_data::Union{Nothing, NamedTuple} = nothing,
    bbox::Tuple{<:Real, <:Real, <:Real, <:Real} = (-68.0, -57.0, 42.0, 48.0),
    grid_resolution::Tuple{Int, Int} = (60, 50),
    depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0, -150.0, -250.0],
    depth_levels::Union{Nothing, AbstractVector{<:Real}} = nothing,
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = nothing,
    time_seconds::Union{Nothing, Real} = nothing,
    times::Union{Nothing, AbstractVector{<:Real}} = nothing,
    time_index::Union{Nothing, Int} = nothing
)::NamedTuple
    # Auto-detect if bathymetry NamedTuple was supplied as first positional argument
    if hydro_input isa NamedTuple && (hasproperty(hydro_input, :elevation) || hasproperty(hydro_input, :depth)) && !hasproperty(hydro_input, :temperature)
        bathymetry_data = hydro_input
        hydro_input = nothing
    end

    bathy = if bathymetry_data !== nothing
        bathymetry_data
    else
        load_open_bathymetry(source=:synthetic, bbox=bbox, grid_resolution=grid_resolution)
    end

    lons = bathy.lons
    lats = bathy.lats
    nx = length(lons)
    ny = length(lats)

    actual_depths = if depth_levels !== nothing
        collect(Float64, depth_levels)
    else
        collect(Float64, depths)
    end
    nz = length(actual_depths)
    z_levels = [z > 0.0 ? -z : z for z in actual_depths]

    # Resolve target depth index
    active_k = 1
    if depth !== nothing
        target_z = depth > 0.0 ? -Float64(depth) : Float64(depth)
        _, active_k = findmin(abs.(z_levels .- target_z))
    elseif depth_level !== nothing
        active_k = clamp(depth_level, 1, nz)
    end
    resolved_depth_m = z_levels[active_k]

    # Initialize 3D field arrays (nx × ny × nz)
    T_mat = zeros(Float64, nx, ny, nz)
    S_mat = zeros(Float64, nx, ny, nz)
    u_mat = zeros(Float64, nx, ny, nz)
    v_mat = zeros(Float64, nx, ny, nz)
    w_mat = zeros(Float64, nx, ny, nz)
    diff_v = zeros(Float64, nx, ny, nz)
    visc_v = zeros(Float64, nx, ny, nz)

    t_sec = time_seconds !== nothing ? Float64(time_seconds) :
            (time_index !== nothing ? Float64(time_index * 3600.0) : 0.0)

    # 1. Ingest from external model or NamedTuple if provided
    if hydro_input isa NamedTuple || hydro_input isa AbstractDict
        get_f(k_list, def) = begin
            for k in k_list
                if hydro_input isa NamedTuple && hasproperty(hydro_input, k)
                    return getproperty(hydro_input, k)
                elseif hydro_input isa AbstractDict && (haskey(hydro_input, k) || haskey(hydro_input, string(k)))
                    return haskey(hydro_input, k) ? hydro_input[k] : hydro_input[string(k)]
                end
            end
            return def
        end
        raw_u = get_f((:u, :u_velocity), nothing)
        raw_v = get_f((:v, :v_velocity), nothing)
        raw_t = get_f((:temperature, :T, :temp), nothing)
        raw_s = get_f((:salinity, :S, :sal), nothing)

        if raw_u !== nothing && size(raw_u, 1) == nx && size(raw_u, 2) == ny
            u_mat .= ndims(raw_u) == 2 ? repeat(raw_u, 1, 1, nz) : raw_u[:, :, 1:min(nz, size(raw_u, 3))]
        end
        if raw_v !== nothing && size(raw_v, 1) == nx && size(raw_v, 2) == ny
            v_mat .= ndims(raw_v) == 2 ? repeat(raw_v, 1, 1, nz) : raw_v[:, :, 1:min(nz, size(raw_v, 3))]
        end
        if raw_t !== nothing && size(raw_t, 1) == nx && size(raw_t, 2) == ny
            T_mat .= ndims(raw_t) == 2 ? repeat(raw_t, 1, 1, nz) : raw_t[:, :, 1:min(nz, size(raw_t, 3))]
        end
        if raw_s !== nothing && size(raw_s, 1) == nx && size(raw_s, 2) == ny
            S_mat .= ndims(raw_s) == 2 ? repeat(raw_s, 1, 1, nz) : raw_s[:, :, 1:min(nz, size(raw_s, 3))]
        end
    else
        # 2. Physics-based synthetic regional shelf circulation
        min_lon, max_lon, min_lat, max_lat = bathy.bbox
        for j in 1:ny
            y_norm = (lats[j] - min_lat) / (max_lat - min_lat)
            for i in 1:nx
                x_norm = (lons[i] - min_lon) / (max_lon - min_lon)
                h_bed = bathy.depth[i, j]

                # Tidal and seasonal modulation
                t_phase = 2.0 * π * (t_sec / 44714.0)
                u_tide = 0.08 * sin(t_phase + 2.0 * x_norm)
                v_tide = 0.06 * cos(t_phase + 1.5 * y_norm)

                for k in 1:nz
                    z = z_levels[k]

                    # If level is below seabed, mask as NaN
                    if abs(z) > h_bed
                        T_mat[i, j, k] = NaN
                        S_mat[i, j, k] = NaN
                        u_mat[i, j, k] = NaN
                        v_mat[i, j, k] = NaN
                        w_mat[i, j, k] = NaN
                        diff_v[i, j, k] = NaN
                        visc_v[i, j, k] = NaN
                        continue
                    end

                    # Temperature (°C): Surface warm, CIL minimum at -50m, slope warm
                    t_surface = 15.5 - 3.2 * y_norm + 1.5 * x_norm
                    t_cil = 2.0 + 0.9 * sin(π * x_norm)
                    t_slope = 7.8 + 1.2 * (1.0 - y_norm)

                    t_val = if z > -20.0
                        t_surface + (z / 20.0) * (t_surface - 6.0)
                    elseif z > -75.0
                        t_cil + ((z + 50.0) / 35.0)^2 * 2.8
                    else
                        t_cil + ((abs(z) - 75.0) / 100.0) * (t_slope - t_cil)
                    end
                    T_mat[i, j, k] = clamp(t_val, 0.2, 19.0)

                    # Salinity (PSU): Fresher coastal runoff to saline deep slope
                    s_val = 31.2 + 1.9 * (1.0 - y_norm) + 1.3 * x_norm + (abs(z) / 150.0) * 1.4
                    S_mat[i, j, k] = clamp(s_val, 29.8, 35.8)

                    # Advection: Southwestward Nova Scotia Current along coastal shelf
                    z_atten = exp(z / 80.0)
                    u_mean = (-0.18 - 0.12 * y_norm) * z_atten
                    v_mean = (-0.10 - 0.08 * (1.0 - x_norm)) * z_atten

                    u_mat[i, j, k] = u_mean + u_tide * (1.0 + z / 200.0)
                    v_mat[i, j, k] = v_mean + v_tide * (1.0 + z / 200.0)

                    # Vertical velocity w (m/s): Upwelling along shelf break
                    w_up = 0.0004 * sin(2.0 * π * x_norm) * cos(π * y_norm) * (z / max(1.0, h_bed))
                    w_mat[i, j, k] = w_up
                end
            end
        end
    end

    # Physical Density and Stratification Computation
    # UNESCO Linear Equation of State
    rho0 = 1025.0
    alpha_t = 2.0e-4
    beta_s = 7.6e-4
    T0 = 10.0
    S0 = 35.0
    g = 9.80665

    rho_mat = zeros(Float64, nx, ny, nz)
    strat_N2 = zeros(Float64, nx, ny, nz)
    sal_grad = zeros(Float64, nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        t = T_mat[i, j, k]
        s = S_mat[i, j, k]
        if !isnan(t) && !isnan(s)
            rho_mat[i, j, k] = rho0 * (1.0 - alpha_t * (t - T0) + beta_s * (s - S0))
        else
            rho_mat[i, j, k] = NaN
        end
    end

    # Vertical gradients
    for j in 1:ny, i in 1:nx
        for k in 1:nz
            if isnan(rho_mat[i, j, k])
                strat_N2[i, j, k] = NaN
                sal_grad[i, j, k] = NaN
                diff_v[i, j, k] = NaN
                visc_v[i, j, k] = NaN
                continue
            end

            d_rho_dz = if k == 1 && nz > 1
                (rho_mat[i, j, 1] - rho_mat[i, j, 2]) / (z_levels[1] - z_levels[2])
            elseif k == nz && nz > 1
                (rho_mat[i, j, nz - 1] - rho_mat[i, j, nz]) / (z_levels[nz - 1] - z_levels[nz])
            elseif nz >= 3
                (rho_mat[i, j, k - 1] - rho_mat[i, j, k + 1]) / (z_levels[k - 1] - z_levels[k + 1])
            else
                0.0
            end

            ds_dz = if k == 1 && nz > 1
                (S_mat[i, j, 1] - S_mat[i, j, 2]) / (z_levels[1] - z_levels[2])
            elseif k == nz && nz > 1
                (S_mat[i, j, nz - 1] - S_mat[i, j, nz]) / (z_levels[nz - 1] - z_levels[nz])
            elseif nz >= 3
                (S_mat[i, j, k - 1] - S_mat[i, j, k + 1]) / (z_levels[k - 1] - z_levels[k + 1])
            else
                0.0
            end

            sal_grad[i, j, k] = ds_dz
            n2_val = -(g / rho0) * d_rho_dz
            strat_N2[i, j, k] = max(1e-7, n2_val)

            # Turbulent Eddy Diffusivity & Viscosity
            du_dz = (nz > 1 && k < nz) ? (u_mat[i, j, k] - u_mat[i, j, k+1]) / (z_levels[k] - z_levels[k+1]) : 0.005
            dv_dz = (nz > 1 && k < nz) ? (v_mat[i, j, k] - v_mat[i, j, k+1]) / (z_levels[k] - z_levels[k+1]) : 0.005
            shear2 = max(1e-6, du_dz^2 + dv_dz^2)
            ri = clamp(strat_N2[i, j, k] / shear2, 0.05, 50.0)

            z = z_levels[k]
            h_bed = bathy.depth[i, j]
            k_surf = 1.2e-2 * exp(z / 15.0)
            k_pyc  = 1.5e-4 / (1.0 + 8.0 * ri)
            dist_to_bed = max(0.5, h_bed + z)
            k_bbl  = 2.5e-3 * exp(-dist_to_bed / 12.0)

            diff_v[i, j, k] = clamp(k_surf + k_pyc + k_bbl, 1e-5, 0.05)
            visc_v[i, j, k] = diff_v[i, j, k] * (1.0 + 0.5 * ri)
        end
    end

    spd_mat = hypot.(u_mat, v_mat)

    n_cells = nx * ny
    T_2d = reshape(T_mat, n_cells, nz)
    S_2d = reshape(S_mat, n_cells, nz)
    rho_2d = reshape(rho_mat, n_cells, nz)
    strat_N2_2d = reshape(strat_N2, n_cells, nz)
    sal_grad_2d = reshape(sal_grad, n_cells, nz)
    u_2d = reshape(u_mat, n_cells, nz)
    v_2d = reshape(v_mat, n_cells, nz)
    w_2d = reshape(w_mat, n_cells, nz)
    diff_v_2d = reshape(diff_v, n_cells, nz)
    visc_v_2d = reshape(visc_v, n_cells, nz)
    spd_2d = reshape(spd_mat, n_cells, nz)

    # Habitat suitability gradient tied to depth and water column temperature
    d_vec = bathy.depth_vec
    t_surf = T_2d[:, 1]
    hsi_raw = exp.(-((d_vec .- 175.0) ./ 60.0).^2 - ((t_surf .- 3.0) ./ 2.5).^2)
    valid_hsi = filter(!isnan, hsi_raw)
    hsi_min, hsi_max = isempty(valid_hsi) ? (0.0, 1.0) : extrema(valid_hsi)
    hsi_vec = (hsi_raw .- hsi_min) ./ max(1e-6, hsi_max - hsi_min) .* 0.85 .+ 0.1

    # Extract active 2D horizontal slice at resolved_depth_m
    slice_2d = (
        depth_m = resolved_depth_m,
        depth_level = active_k,
        temperature = T_mat[:, :, active_k],
        salinity = S_mat[:, :, active_k],
        density = rho_mat[:, :, active_k],
        stratification = strat_N2[:, :, active_k],
        salinity_gradient = sal_grad[:, :, active_k],
        u = u_mat[:, :, active_k],
        v = v_mat[:, :, active_k],
        w = w_mat[:, :, active_k],
        speed = spd_mat[:, :, active_k],
        diffusivity = diff_v[:, :, active_k],
        viscosity = visc_v[:, :, active_k]
    )

    return (
        lons = lons,
        lats = lats,
        depths = abs.(actual_depths),
        depth_levels = abs.(actual_depths),
        temperature = T_2d,
        temperature_3d = T_mat,
        salinity = S_2d,
        salinity_3d = S_mat,
        density = rho_2d,
        rho = rho_2d,
        density_3d = rho_mat,
        stratification_N2 = strat_N2_2d,
        N2 = strat_N2_2d,
        stratification_3d = strat_N2,
        salinity_gradient = sal_grad_2d,
        u = u_2d,
        advection_u = u_2d,
        v = v_2d,
        advection_v = v_2d,
        w = w_2d,
        speed = spd_2d,
        diffusivity_v = diff_v_2d,
        kappa_v = diff_v_2d,
        viscosity_v = visc_v_2d,
        nu_v = visc_v_2d,
        hsi = hsi_vec,
        elevation = zeros(Float64, n_cells),
        bathymetry = bathy.elevation,
        depth_vec = bathy.depth_vec,
        depths_vec = bathy.depth_vec,
        slice_2d = slice_2d,
        centroids = bathy.centroids,
        polygons = bathy.polygons,
        bbox = bathy.bbox
    )
end

# --- Spatial Field Resharding ---
function reshard_spatial_field(
    P::AbstractMatrix{<:Real},
    values::AbstractArray{<:Real}
)
    nd = ndims(values)
    if nd == 1
        return Vector{Float64}(P * values)
    elseif nd == 2
        return Matrix{Float64}(P * values)
    elseif nd == 3
        s1, s2, s3 = size(values)
        flat = reshape(values, s1, s2 * s3)
        res_flat = Matrix{Float64}(P * flat)
        return reshape(res_flat, size(P, 1), s2, s3)
    else
        error("Unsupported array dimensionality ($nd) for reshard_spatial_field.")
    end
end

function reshard_spatial_field(
    P::AbstractMatrix{<:Real},
    vectors::Tuple{Vararg{AbstractArray{<:Real}}}
)
    return map(arr -> reshard_spatial_field(P, arr), vectors)
end

function reshard_spatial_field(
    P::AbstractMatrix{<:Real},
    nt_or_summary::NamedTuple;
    mode::Symbol = :samples,
    alpha::Real = 0.05
)
    is_summary = hasproperty(nt_or_summary, :mean) && hasproperty(nt_or_summary, :std)

    if is_summary
        if mode in [:samples, :full_mc, :matrix] && hasproperty(nt_or_summary, :samples) &&
           !isnothing(nt_or_summary.samples) && nt_or_summary.samples isa AbstractMatrix
            U_dest = Matrix{Float64}(P * nt_or_summary.samples)
            return summarize_sample_matrix(U_dest; alpha=alpha)
        end

        mean_resharded = Vector{Float64}(P * nt_or_summary.mean)
        sd_resharded = Vector{Float64}(sqrt.(P * (nt_or_summary.std .^ 2)))
        lower_resharded = Vector{Float64}(P * nt_or_summary.lower)
        upper_resharded = Vector{Float64}(P * nt_or_summary.upper)
        median_resharded = hasproperty(nt_or_summary, :median) ?
            Vector{Float64}(P * nt_or_summary.median) : mean_resharded

        return (
            mean = mean_resharded,
            median = median_resharded,
            std = sd_resharded,
            lower = lower_resharded,
            upper = upper_resharded,
            samples = nothing
        )
    else
        res_pairs = Pair{Symbol, Any}[]
        n_src_expected = size(P, 2)
        for (k, v) in pairs(nt_or_summary)
            if v isa AbstractArray{<:Real} && size(v, 1) == n_src_expected
                push!(res_pairs, k => reshard_spatial_field(P, v))
            elseif v isa Tuple && all(x -> x isa AbstractArray{<:Real} && size(x, 1) == n_src_expected, v)
                push!(res_pairs, k => map(x -> reshard_spatial_field(P, x), v))
            else
                push!(res_pairs, k => v)
            end
        end
        return NamedTuple(res_pairs)
    end
end

function reshard_spatial_field(
    values::AbstractArray{<:Real}, au_src::NamedTuple, au_dest::NamedTuple
)
    P = compute_network_transfer_matrix(au_src, au_dest)
    return reshard_spatial_field(P, values)
end

function reshard_spatial_field(
    vectors::Tuple{Vararg{AbstractArray{<:Real}}},
    au_src::NamedTuple,
    au_dest::NamedTuple
)
    P = compute_network_transfer_matrix(au_src, au_dest)
    return reshard_spatial_field(P, vectors)
end

function reshard_spatial_field(
    nt_or_summary::NamedTuple,
    au_src::NamedTuple,
    au_dest::NamedTuple;
    mode::Symbol = :samples,
    alpha::Real = 0.05
)
    P = compute_network_transfer_matrix(au_src, au_dest)
    return reshard_spatial_field(P, nt_or_summary; mode=mode, alpha=alpha)
end


function _safe_mean(vals)
    return isempty(vals) ? NaN : mean(vals)
end

# ── Telemetry aggregation ──────────────────────────────────────────────────────

"""
    aggregate_telemetry_time(tagging; time_interval=:monthly) -> DataFrame

Aggregate high-frequency telemetry pings per individual into regular temporal
bins, with `Missing`-safe numeric aggregation throughout.

# Binning modes
- `:monthly`  — first day of each (year, month).
- `:weekly`   — Monday of each ISO calendar week.
- `:biweekly` — 14-day epochs anchored at 1990-01-01.
- `:daily`    — calendar date.
- `:raw`      — no aggregation; returns a copy.

# Data requirements
`tagging` must contain at minimum: `:tagid`, `:lon`, `:lat`, `:timestamp`, `:tag`.
Optional numeric columns (`:z`, `:cw`, `:chela`, `:wgt`) are aggregated with
`_safe_mean`.  Group columns (`:sex`, `:mat`, `:datasource`) use `first`.
`:is_dead` is propagated via `any(skipmissing)`.

Individuals with < 2 distinct temporal observations after aggregation are
dropped.  `:tag` is re-indexed (0, 1, 2, …) per individual.

# Arguments
- `tagging::DataFrame`: Input telemetry table.
- `time_interval::Symbol`: Binning resolution (default `:monthly`).

# Returns
- `DataFrame`: Aggregated table with `:time` (decimal year) column.
"""
function aggregate_telemetry_time(
    tagging::DataFrame;
    time_interval::Symbol = :monthly
)::DataFrame

    time_interval == :raw && return copy(tagging)

    df = copy(tagging)
    hasproperty(df, :timestamp) || error("DataFrame must contain :timestamp column.")

    if time_interval == :daily
        df[!, :time_bucket] = Date.(df.timestamp)
    elseif time_interval == :weekly
        df[!, :time_bucket] = [Date(t) - Day(dayofweek(Date(t)) - 1)
                                for t in df.timestamp]
    elseif time_interval == :biweekly
        epoch = Date(1990, 1, 1)
        df[!, :time_bucket] = [epoch + Day(fld(Int(Date(t) - epoch), 14) * 14)
                                for t in df.timestamp]
    elseif time_interval == :monthly
        df[!, :time_bucket] = [Date(year(t), month(t), 1) for t in df.timestamp]
    else
        error("Unsupported time_interval: $(time_interval). " *
              "Use :monthly, :weekly, :biweekly, :daily, or :raw.")
    end

    spec = Pair[
        :lon       => _safe_mean => :lon,
        :lat       => _safe_mean => :lat,
        :tag       => minimum    => :tag,
        :timestamp => minimum    => :timestamp,
    ]
    for col in (:z, :cw, :chela, :wgt)
        hasproperty(df, col) && push!(spec, col => _safe_mean => col)
    end
    for col in (:sex, :mat, :datasource)
        hasproperty(df, col) && push!(spec, col => first => col)
    end
    hasproperty(df, :is_dead) &&
        push!(spec, :is_dead => (x -> any(skipmissing(x))) => :is_dead)

    agg = DataFrames.combine(groupby(df, [:tagid, :time_bucket]), spec...)
    agg[!, :time] = [_to_decimal_year(d) for d in agg.timestamp]
    sort!(agg, [:tagid, :time])

    valid = Set{String}()
    for sub in groupby(agg, :tagid)
        nrow(sub) >= 2 && push!(valid, string(first(sub.tagid)))
    end
    filter!(r -> string(r.tagid) in valid, agg)

    parts = DataFrame[]
    for sub in groupby(agg, :tagid)
        sdf     = DataFrame(sub)
        sdf.tag = collect(0:(nrow(sdf) - 1))
        push!(parts, sdf)
    end
    return isempty(parts) ? DataFrame() : vcat(parts...)
end


"""
    match_telemetry_closest_month_hsi(
        telemetry_df, monthly_hsi, month_lookup, years
    ) -> Vector{Float64}

Assign each telemetry observation its HSI value from the nearest (year, month).

# Fallback strategy (applied in order)
1. Clamp year to [y_min, y_max]; look up `(yr_clamped, month)`.
2. Same month in `y_min` (lower boundary).
3. Same month in `y_max` (upper boundary).
4. Assign `NaN` and emit a single summary `@warn` when entries are missing.

# Arguments
- `telemetry_df`: DataFrame with `:timestamp` and `:s_idx`.
- `monthly_hsi`: Matrix (S × 12T).
- `month_lookup`: Dict{(year, month) => column_index}.
- `years`: Valid year range.

# Returns
- `Vector{Float64}` of length `nrow(telemetry_df)`.
"""
function match_telemetry_closest_month_hsi(
    telemetry_df::DataFrame,
    monthly_hsi::AbstractMatrix{<:Real},
    month_lookup::Dict{Tuple{Int, Int}, Int},
    years::AbstractVector{<:Integer}
)::Vector{Float64}

    y_min, y_max = extrema(years)
    n_rows = nrow(telemetry_df)
    
    # Use undef instead of zeros to prevent unnecessary memory writing
    out = Vector{Float64}(undef, n_rows)

    # 1. Extract columns to local variables for type-stable, zero-overhead indexing
    timestamps = telemetry_df.timestamp
    s_idxs     = telemetry_df.s_idx

    missing_count = 0

    @inbounds for i in 1:n_rows
        dt = timestamps[i]
        s  = s_idxs[i]
        
        yr = clamp(Dates.year(dt), y_min, y_max)
        mo = Dates.month(dt)

        # 2. Use 0 as a default instead of `nothing` to keep types strict and fast
        col = get(month_lookup, (yr, mo), 0)
        
        if col == 0
            col = get(month_lookup, (y_min, mo), get(month_lookup, (y_max, mo), 0))
            if col == 0
                out[i] = NaN
                missing_count += 1
                continue
            end
        end
        
        out[i] = monthly_hsi[s, col]
    end
    
    # 3. Emit a single summary warning rather than spamming the console
    if missing_count > 0
        @warn "No month_lookup entry found for $missing_count telemetry observations. Assigned NaN."
    end

    return out
end


"""
    _interpolate_hsi(hsi_2d, years, s_idx, decimal_year; ref_doy=244.0)
    -> Float64

Linear interpolation of the annual posterior-mean HSI for spatial unit `s_idx`
at continuous time `decimal_year`.

The annual predictions are anchored at `ref_doy` (day of year; 244 ≈ Sept 1):

    t_ref = (ref_doy − 1) / 365.25
    u     = decimal_year − t_ref
    y1    = clamp(⌊u⌋, y_min, y_max − 1)
    α     = u − y1                          ∈ [0, 1)
    HSI   = (1 − α) HSI(s, y1) + α HSI(s, y1+1)

# Arguments
- `hsi_2d`: Posterior-mean HSI matrix (S × T).
- `years`: Annual year labels (e.g. 1999:2025).
- `s_idx`: Spatial unit index (1-based).
- `decimal_year`: Continuous time in decimal years.
- `ref_doy`: Reference survey day of year (default 244.0).

# Returns
- `Float64`: Interpolated HSI clamped to [0, 1].
"""
function _interpolate_hsi(
    hsi_2d::AbstractMatrix{Float64},
    years::AbstractVector{<:Integer},
    s_idx::Integer,
    decimal_year::Real;
    ref_doy::Real = 244.0
)::Float64
    y_min = first(years)
    y_max = last(years)
    t_off = (ref_doy - 1.0) / 365.25
    u     = Float64(decimal_year) - t_off
    u_cl  = clamp(u, Float64(y_min), Float64(y_max))
    y1    = clamp(floor(Int, u_cl), y_min, y_max - 1)
    α     = u_cl - Float64(y1)
    t1    = clamp(y1 - y_min + 1, 1, length(years))
    t2    = clamp(t1 + 1, 1, length(years))
    return clamp((1.0 - α) * hsi_2d[s_idx, t1] + α * hsi_2d[s_idx, t2], 0.0, 1.0)
end


"""
    build_monthly_hsi_matrix(
        hsi_mean::AbstractMatrix{Float64},
        years::AbstractVector{<:Integer};
        ref_doy::Real = 244.0
    ) -> Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}

Discretizes multi-year spatial Habitat Suitability Index (HSI) surfaces onto a
regular monthly grid (12 calendar months per survey year) via continuous
piecewise-linear temporal interpolation, evaluating each cell at mid-month (day 15)
with leap-year awareness.

# Mathematical Formulation
Given annual HSI values ``h_s(y)`` for spatial unit ``s`` in year ``y``, the fractional
calendar year at day-of-year ``d`` of year ``y`` is:
```math
t = y + \\frac{d - 1}{D(y)}
```
where ``D(y) \\in \\{365, 366\\}``. Anchored to annual survey reference day
``d_{\\text{ref}}``, the mid-month HSI is linearly interpolated between adjacent survey epochs:
```math
h_s(t) = (1 - \\alpha) h_s(y_1) + \\alpha h_s(y_2)
```
clamped to ``[0, 1]``.

# Arguments
- `hsi_mean`: Spatial HSI posterior mean matrix (size ``S \\times T_{\\text{years}}``).
- `years`: Integer vector of survey years of length ``T_{\\text{years}}``.
- `ref_doy`: Reference survey day of year (default `244.0` = September 1).

# Returns
- `Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}`:
  - `mat`: Monthly discretized HSI matrix (size ``S \\times 12 T_{\\text{years}}``).
  - `lookup`: Dictionary mapping `(year, month)` tuples to matrix column indices.
"""
function build_monthly_hsi_matrix(
    hsi_mean::AbstractMatrix{Float64},
    years::AbstractVector{<:Integer};
    ref_doy::Real = 244.0
)::Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}

    S      = size(hsi_mean, 1)
    n_cols = length(years) * 12
    
    # Use undef since we overwrite every element (avoids zeroing overhead)
    mat    = Matrix{Float64}(undef, S, n_cols)
    
    # Pre-size the dictionary to avoid reallocations
    lookup = Dict{Tuple{Int, Int}, Int}()
    sizehint!(lookup, n_cols)
    
    col = 1
    for yr in years
        days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
        for m in 1:12
            doy_mid = Float64(Dates.dayofyear(Date(yr, m, 15)))
            dec_yr  = Float64(yr) + (doy_mid - 1.0) / days_in_yr
            
            # @inbounds safely removes bounds checking for peak speed
            @inbounds for s in 1:S
                mat[s, col] = _interpolate_hsi(
                    hsi_mean, years, s, dec_yr; ref_doy=ref_doy)
            end
            
            lookup[(yr, m)] = col
            col += 1
        end
    end
    
    return mat, lookup
end


"""
    load_hsi_jld2(path; hsi_key="hsi", years_key="years",
                  auids_key="auids", ref_doy=244.0) -> NamedTuple

Load a JLD2 bundle containing a 3D posterior HSI array and compute derived
summaries needed for movement modelling.

# Expected JLD2 keys
- `hsi`:   Array{Float64, 3} (S × T × N_draws).
- `years`: Vector{Int} of length T.
- `auids`: (optional) spatial unit identifiers.

# Arguments
- `path`: Path to the JLD2 file.
- `hsi_key`, `years_key`, `auids_key`: JLD2 key names.
- `ref_doy`: Reference day of year (default 244.0 = Sept 1).

# Returns
`NamedTuple`:
- `hsi`, `years`, `auids`.
- `hsi_mean` (S × T): posterior mean.
- `hsi_sd` (S × T): posterior standard deviation.
- `hsi_spatial_mean` (length S): multi-year mean per spatial unit.
- `monthly_hsi` (S × 12T): monthly discretization.
- `month_lookup`: Dict{(year, month) => column_index}.
"""
function load_hsi_jld2(
    path::AbstractString;
    hsi_key::AbstractString   = "hsi",
    years_key::AbstractString = "years",
    auids_key::AbstractString = "auids",
    ref_doy::Real             = 244.0
)::NamedTuple
    isfile(path) || error("HSI JLD2 not found: $(path)")
    bundle = JLD2.load(path)
    actual_hsi_key = if haskey(bundle, hsi_key)
        hsi_key
    elseif haskey(bundle, "sims")
        "sims"
    elseif haskey(bundle, "predictions")
        "predictions"
    else
        hsi_key
    end
    hsi    = bundle[actual_hsi_key]
    years  = bundle[years_key]
    auids  = haskey(bundle, auids_key) ? bundle[auids_key] : collect(1:size(hsi, 1))

    hsi_mean          = dropdims(mean(hsi, dims=3), dims=3)
    hsi_sd            = dropdims(std(hsi, dims=3),  dims=3)
    hsi_spatial_mean  = vec(mean(hsi_mean, dims=2))

    monthly_hsi, month_lookup = build_monthly_hsi_matrix(
        Float64.(hsi_mean), years; ref_doy=ref_doy)

    return (
        hsi              = hsi,
        years            = years,
        auids            = auids,
        hsi_mean         = hsi_mean,
        hsi_sd           = hsi_sd,
        hsi_spatial_mean = hsi_spatial_mean,
        monthly_hsi      = monthly_hsi,
        month_lookup     = month_lookup
    )
end
