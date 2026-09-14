"""
    leaflet.jl

Interactive Leaflet-based HTML visualization engine for Bayesian Spatio-Temporal Models (BSTM).
Provides standalone, portable, publication-grade interactive HTML visualizations for:
- Geographic tessellations, areal unit graphs, and boundary hulls
- Scalar field choropleths (HSI, diffusion, residence distribution π, random effects, SVCs)
- Spatiotemporal prediction animations with interactive time sliders
- Advection drift & velocity vector fields (arrowheads, drift speed, and orientation)
- Multi-individual animal movement trajectories, release/recapture points, and telemetry tracks
- Interactive movement dashboards integrating Leaflet maps with interactive summary charts

Version: v1.0.0
"""

# =============================================================================
# Section 1: LeafletMap Type & Serialization Helpers
# =============================================================================

"""
    LeafletMap

Container holding a complete, self-contained interactive Leaflet HTML visualization.

# Fields
- `html::String`: Complete standalone HTML string embedding CSS, JS, GeoJSON, and map logic.
- `title::String`: Human-readable title of the visualization.
- `width::String`: CSS width specification (default: `"100%"`).
- `height::String`: CSS height specification (default: `"650px"`).
- `metadata::Dict{Symbol, Any}`: Additional metadata (bounding box, number of units, layers).
"""
struct LeafletMap
    html::String
    title::String
    width::String
    height::String
    metadata::Dict{Symbol, Any}
end

function LeafletMap(html::String, title::String, width::String, height::String, metadata::AbstractDict)
    return LeafletMap(html, title, width, height, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(metadata)))
end

function LeafletMap(html::String; title::String="BSTM Leaflet Map", width::String="100%",
           height::String="650px", metadata=Dict{Symbol, Any}())
    meta_dict = if metadata isa AbstractDict
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(metadata))
    elseif metadata isa NamedTuple
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(metadata))
    else
        Dict{Symbol, Any}()
    end
    return LeafletMap(html, title, width, height, meta_dict)
end

Base.show(io::IO, ::MIME"text/html", m::LeafletMap) = print(io, m.html)

function Base.show(io::IO, m::LeafletMap)
    print(io, "LeafletMap(\"", m.title, "\", dimensions: ", m.width, " × ", m.height, ")")
end

function Base.getproperty(m::LeafletMap, s::Symbol)
    if s === :html_content
        return getfield(m, :html)
    end
    return getfield(m, s)
end

"""
    save_html(m::LeafletMap, filepath::AbstractString)
    save_html(html_str::AbstractString, filepath::AbstractString)

Saves an interactive `LeafletMap` visualization or raw HTML string to a local `.html` file.

# Arguments
- `m`: `LeafletMap` instance or HTML `AbstractString`.
- `filepath::AbstractString`: Destination path for the HTML file.

# Returns
- `String`: Absolute path of the written HTML file.
"""
function save_html(m::LeafletMap, filepath::AbstractString)::String
    dir = dirname(filepath)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    final_path = endswith(filepath, ".html") ? filepath : (filepath * ".html")
    write(final_path, m.html)
    return abspath(final_path)
end

function save_html(html_str::AbstractString, filepath::AbstractString)::String
    dir = dirname(filepath)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    final_path = endswith(filepath, ".html") ? filepath : (filepath * ".html")
    write(final_path, html_str)
    return abspath(final_path)
end


# =============================================================================
# Section 2: Color Palettes, Interpolation & Colormap Helpers
# =============================================================================

# Curated high-fidelity color ramps (256-stop sampled Hex)
const _LEAFLET_PALETTES = Dict{Symbol, Vector{String}}(
    :viridis => [
        "#440154", "#481567", "#482677", "#453781", "#404688", "#3b528b", "#365d8d",
        "#31688e", "#2c728e", "#287c8e", "#24868e", "#21908d", "#1f9a8a", "#20a486",
        "#27ad81", "#35b779", "#4ac16d", "#63cb5f", "#80d350", "#9fda3a", "#c0df26",
        "#dee318", "#fde725"
    ],
    :plasma => [
        "#0d0887", "#280592", "#41049d", "#5901a5", "#7000a8", "#8707a6", "#9c179e",
        "#b12a90", "#c33d80", "#d45070", "#e3655f", "#f07c4a", "#f99532", "#feb019",
        "#facf27", "#f0f921"
    ],
    :inferno => [
        "#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#ba3655", "#dd513a",
        "#f3761b", "#fca50a", "#f6d543", "#fcffa4"
    ],
    :turbo => [
        "#30123b", "#4145ab", "#4675ed", "#39a2fc", "#1bcfd4", "#24ec9c", "#61fc54",
        "#a4fc3b", "#d1e834", "#f3c63a", "#fe9b2d", "#f36315", "#d93806", "#b11902",
        "#7a0403"
    ],
    :cividis => [
        "#00204d", "#002c69", "#0c3b7a", "#294982", "#415788", "#56668d", "#6b7592",
        "#818597", "#98969b", "#b0a89d", "#c9bc9a", "#e3d191", "#ffea46"
    ],
    :coolwarm => [
        "#3b4cc0", "#5977e3", "#7b9ff9", "#9ebeff", "#c0d4f5", "#dddcdc", "#f2c4b2",
        "#f79b7d", "#e96c51", "#ce3e32", "#b40426"
    ],
    :RdBu => [
        "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0",
        "#92c5de", "#4393c3", "#2166ac", "#053061"
    ],
    :Blues => [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5",
        "#08519c", "#08306b"
    ],
    :YlOrRd => [
        "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c",
        "#bd0026", "#800026"
    ],
    :tab10 => [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf"
    ]
)

"""
    _resolve_palette(cmap::Union{Symbol, AbstractVector{String}})::Vector{String}

Resolves a color palette into a vector of hex color strings.
"""
function _resolve_palette(cmap::Union{Symbol, AbstractVector{String}})::Vector{String}
    if cmap isa Symbol
        return get(_LEAFLET_PALETTES, cmap, _LEAFLET_PALETTES[:viridis])
    elseif cmap isa AbstractVector{String} && !isempty(cmap)
        return collect(cmap)
    else
        return _LEAFLET_PALETTES[:viridis]
    end
end

"""
    _hex_to_rgb(hex::String)::Tuple{Float64, Float64, Float64}

Parses a hex color string (e.g. `"#440154"`) into `(r, g, b)` components in `[0, 1]`.
"""
function _hex_to_rgb(hex::String)::Tuple{Float64, Float64, Float64}
    clean = strip(hex, ['#', ' '])
    if length(clean) == 6
        r = parse(Int, clean[1:2], base=16) / 255.0
        g = parse(Int, clean[3:4], base=16) / 255.0
        b = parse(Int, clean[5:6], base=16) / 255.0
        return (r, g, b)
    elseif length(clean) == 3
        r = parse(Int, clean[1:1] * clean[1:1], base=16) / 255.0
        g = parse(Int, clean[2:2] * clean[2:2], base=16) / 255.0
        b = parse(Int, clean[3:3] * clean[3:3], base=16) / 255.0
        return (r, g, b)
    end
    return (0.5, 0.5, 0.5)
end

"""
    _rgb_to_hex(r::Real, g::Real, b::Real)::String

Converts `(r, g, b)` floats in `[0, 1]` to a 6-digit hex string `"#rrggbb"`.
"""
function _rgb_to_hex(r::Real, g::Real, b::Real)::String
    ir = clamp(round(Int, r * 255.0), 0, 255)
    ig = clamp(round(Int, g * 255.0), 0, 255)
    ib = clamp(round(Int, b * 255.0), 0, 255)
    return @sprintf("#%02x%02x%02x", ir, ig, ib)
end

"""
    _map_val_to_hex(v::Real, vmin::Real, vmax::Real, palette::Vector{String};
                    center_zero::Bool=false)::String

Linearly maps a real value `v` in `[vmin, vmax]` to an interpolated hex color in `palette`.
"""
function _map_val_to_hex(
    v::Real,
    vmin::Real,
    vmax::Real,
    palette::Vector{String};
    center_zero::Bool = false
)::String
    if isnan(v) || isinf(v)
        return "#888888"
    end
    if center_zero
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
    end
    denom = vmax - vmin
    t = abs(denom) > 1e-12 ? (Float64(v) - Float64(vmin)) / denom : 0.5
    t = clamp(t, 0.0, 1.0)

    n_colors = length(palette)
    if n_colors == 1
        return palette[1]
    end
    pos = t * (n_colors - 1) + 1.0
    idx_low = clamp(floor(Int, pos), 1, n_colors - 1)
    idx_high = idx_low + 1
    frac = pos - idx_low

    c1 = _hex_to_rgb(palette[idx_low])
    c2 = _hex_to_rgb(palette[idx_high])
    r = c1[1] + frac * (c2[1] - c1[1])
    g = c1[2] + frac * (c2[2] - c1[2])
    b = c1[3] + frac * (c2[3] - c1[3])
    return _rgb_to_hex(r, g, b)
end

"""
    _is_geographic_coordinates(coords)::Bool

Infers whether spatial coordinates are in geographic `(lon, lat)` space versus abstract
planar cartesian `(x, y)` space.

# Mathematical & Heuristic Criteria:
1. Hard WGS84 bounding box: Longitudes must strictly reside within `[-180, 180]`,
   and Latitudes within `[-90, 90]`.
2. Planar origin check: Non-negative coordinates starting near `(0, 0)` with synthetic
   extents (e.g. `[0, 100]`, `[0, 10]`, `[0, 1]`) are classified as planar.
3. Marine regional domain checks (Scotian Shelf, Atlantic Canada, Pacific, Europe).
"""
function _is_geographic_coordinates(coords)::Bool
    if isempty(coords)
        return false
    end
    xs = Float64[]
    ys = Float64[]
    for pt in coords
        if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
            push!(xs, float(pt[1]))
            push!(ys, float(pt[2]))
        end
    end
    if isempty(xs) || isempty(ys)
        return false
    end
    min_x, max_x = minimum(xs), maximum(xs)
    min_y, max_y = minimum(ys), maximum(ys)

    # 1. Hard limits for WGS84 Geographic degrees
    if min_y < -90.0 || max_y > 90.0 || min_x < -180.0 || max_x > 180.0
        return false
    end

    # 2. Planar Cartesian origin check: non-negative coordinates starting near (0, 0)
    # Synthetic domains like [0, 100], [0, 10], [0, 1], [1, N]
    if min_x >= 0.0 && min_y >= 0.0 && (min_x < 5.0 || min_y < 5.0) && (max_x <= 100.0 && max_y <= 100.0)
        return false
    end

    # 3. Known marine fisheries regions (Atlantic Canada, Scotian Shelf, Pacific, Europe)
    if (min_x <= -20.0 && max_x <= -10.0 && min_y >= 30.0 && max_y <= 85.0) ||
       (min_x >= 100.0 && max_x <= 180.0 && min_y >= -50.0 && max_y <= 70.0) ||
       (min_x >= -180.0 && max_x <= -50.0 && min_y >= -60.0 && max_y <= 75.0)
        return true
    end

    # 4. Standard degrees span check
    if (max_x - min_x > 0.1 || max_y - min_y > 0.1) && (min_x < -5.0 || max_x > 5.0 || min_y > 15.0 || min_y < -15.0)
        return true
    end

    return false
end

"""
    utm_to_lonlat(easting::Real, northing::Real; zone::Int=20, is_km::Bool=true, northern::Bool=true)::Tuple{Float64, Float64}

Converts Universal Transverse Mercator (UTM) coordinates (in km or meters) to WGS84
geographic coordinates `(lon, lat)` in decimal degrees using `CoordRefSystems.jl`.

# Mathematical Formulation
Let (x, y) be the projected UTM Cartesian coordinates in meters. The geodetic
coordinates (lon, lat) on the WGS84 reference ellipsoid are obtained via
the inverse Transverse Mercator mapping:
    (lon, lat) = T_UTM⁻¹(x, y; zone, datum=WGS84)

# Arguments
- `easting::Real`: Projected X coordinate (Easting).
- `northing::Real`: Projected Y coordinate (Northing).
- `zone::Int`: UTM Longitudinal Zone (1 <= zone <= 60, default 20).
- `is_km::Bool`: Whether inputs are in kilometers (`true`, default) or meters (`false`).
- `northern::Bool`: Whether the coordinate is in the Northern Hemisphere (default `true`).

# Returns
- `Tuple{Float64, Float64}`: Geographic `(lon, lat)` in decimal degrees.
"""
function utm_to_lonlat(
    easting::Real,
    northing::Real;
    zone::Int = 20,
    is_km::Bool = true,
    northern::Bool = true
)::Tuple{Float64, Float64}
    scale_m = is_km ? 1000.0 : 1.0
    x_m = Float64(easting) * scale_m
    y_m = Float64(northing) * scale_m
    crs = northern ? utmnorth(zone) : utmsouth(zone)
    pt = crs(x_m * 1.0u"m", y_m * 1.0u"m")
    ll = convert(LatLon, pt)
    lon_deg = Float64(ll.lon.val)
    lat_deg = Float64(ll.lat.val)
    return (lon_deg, lat_deg)
end

const _utm_to_lonlat = utm_to_lonlat

"""
    lonlat_to_utm(lon::Real, lat::Real; zone::Int=20, is_km::Bool=true, northern::Bool=true)::Tuple{Float64, Float64}

Converts WGS84 geographic coordinates `(lon, lat)` in decimal degrees to Universal Transverse
Mercator (UTM) coordinates (in km or meters) using `CoordRefSystems.jl`.

# Mathematical Formulation
Let (lon, lat) be the geodetic longitude and latitude in degrees. The projected
Cartesian coordinates (x, y) on the Transverse Mercator plane with central meridian
λ₀ = 6° * (zone - 1) - 180° + 3° are given by:
    (x, y) = T_UTM(lon, lat; zone, datum=WGS84)

# Arguments
- `lon::Real`: Longitude in decimal degrees ([-180, 180]).
- `lat::Real`: Latitude in decimal degrees ([-90, 90]).
- `zone::Int`: UTM Longitudinal Zone (1 <= zone <= 60, default 20).
- `is_km::Bool`: Whether output should be scaled to kilometers (`true`, default) or meters (`false`).
- `northern::Bool`: Whether the coordinate is in the Northern Hemisphere (default `true`).

# Returns
- `Tuple{Float64, Float64}`: Projected Easting and Northing coordinates.
"""
function lonlat_to_utm(
    lon::Real,
    lat::Real;
    zone::Int = 20,
    is_km::Bool = true,
    northern::Bool = true
)::Tuple{Float64, Float64}
    crs = northern ? utmnorth(zone) : utmsouth(zone)
    ll = LatLon(Float64(lat) * 1.0u"°", Float64(lon) * 1.0u"°")
    pt = convert(crs, ll)
    x_m = Float64(pt.x.val)
    y_m = Float64(pt.y.val)
    scale_out = is_km ? 0.001 : 1.0
    return (x_m * scale_out, y_m * scale_out)
end

const _lonlat_to_utm = lonlat_to_utm

"""
    struct CoordinateTransformer
        mode::Symbol # :geographic, :crs, :utm, :planar
        is_geo::Bool
        crs_type::Any
        utm_zone::Int
        utm_is_km::Bool
        northern::Bool
        scale::Float64
        x_mean::Float64
        y_mean::Float64
        lon_center::Float64
        lat_center::Float64
        wkt::Union{Nothing, String}
    end

Encapsulates coordinate normalization and projection from raw input space (geographic,
CoordRefSystems CRS, UTM, or planar) into Leaflet WGS84 space.
"""
struct CoordinateTransformer
    mode::Symbol
    is_geo::Bool
    crs_type::Any
    utm_zone::Int
    utm_is_km::Bool
    northern::Bool
    scale::Float64
    x_mean::Float64
    y_mean::Float64
    lon_center::Float64
    lat_center::Float64
    wkt::Union{Nothing, String}
end

function _extract_wkt(au)::Union{Nothing, String}
    if isnothing(au)
        return nothing
    end
    for k in (:wkt, :crs, :proj, :projection, :coord_sys)
        if hasproperty(au, k)
            v = getproperty(au, k)
            v isa AbstractString && !isempty(v) && return string(v)
        elseif au isa AbstractDict && haskey(au, k)
            v = au[k]
            v isa AbstractString && !isempty(v) && return string(v)
        end
    end
    return nothing
end

function _is_wgs84_wkt(wkt::Union{Nothing, AbstractString})::Bool
    if isnothing(wkt) || isempty(wkt)
        return false
    end
    w = uppercase(string(wkt))
    return contains(w, "4326") ||
           contains(w, "WGS 84") ||
           contains(w, "WGS_1984") ||
           contains(w, "GCS_WGS_1984") ||
           contains(w, "GEOGCS[\"WGS") ||
           contains(w, "+PROJ=LONGLAT") ||
           contains(w, "+PROJ=LATLONG")
end

function _build_coordinate_transformer(
    all_points;
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    lon_center::Float64 = -160.0,
    lat_center::Float64 = 0.0
)::CoordinateTransformer
    wkt_str = !isnothing(wkt) ? string(wkt) : nothing
    has_wgs84_wkt = _is_wgs84_wkt(wkt_str)

    # 1. Explicit geographic or WGS84 WKT
    if is_geo === true || has_wgs84_wkt
        return CoordinateTransformer(:geographic, true, nothing, 20, false, true, 1.0, 0.0, 0.0, 0.0, 0.0, wkt_str)
    end

    # Extract all valid coordinates to determine spatial envelope
    xs = Float64[]
    ys = Float64[]
    for pt in all_points
        if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
            push!(xs, float(pt[1]))
            push!(ys, float(pt[2]))
        end
    end

    min_x = !isempty(xs) ? minimum(xs) : 0.0
    max_x = !isempty(xs) ? maximum(xs) : 0.0
    min_y = !isempty(ys) ? minimum(ys) : 0.0
    max_y = !isempty(ys) ? maximum(ys) : 0.0

    # 2. Check for UTM signatures or EPSG codes in WKT / metadata
    is_utm_wkt = !isnothing(wkt_str) && contains(lowercase(wkt_str), "utm")
    utm_zone = 20
    is_north = true
    if is_utm_wkt
        m_zone = match(r"zone\s*=?\s*(\d+)", lowercase(wkt_str))
        if m_zone !== nothing
            utm_zone = parse(Int, m_zone.captures[1])
        end
        if contains(lowercase(wkt_str), "south")
            is_north = false
        end
    end

    # Check EPSG code directly if present in WKT / CRS
    if !isnothing(wkt_str)
        m_epsg = match(r"epsg[:\s]+(\d+)"i, wkt_str)
        if m_epsg !== nothing
            code = parse(Int, m_epsg.captures[1])
            if 32601 <= code <= 32660
                utm_zone = code - 32600
                is_north = true
                is_utm_wkt = true
            elseif 32701 <= code <= 32760
                utm_zone = code - 32700
                is_north = false
                is_utm_wkt = true
            end
        end
    end

    is_utm_km = (min_x >= 50.0 && max_x <= 1500.0 && min_y >= 1000.0 && max_y <= 9000.0)
    is_utm_m = (min_x >= 50000.0 && min_y >= 1000000.0)

    if is_utm_wkt || is_utm_km || is_utm_m
        is_km = is_utm_km || (!is_utm_m && !contains(lowercase(something(wkt_str, "")), "units=m"))
        crs = is_north ? utmnorth(utm_zone) : utmsouth(utm_zone)
        return CoordinateTransformer(:utm, true, crs, utm_zone, is_km, is_north, 1.0, 0.0, 0.0, 0.0, 0.0, wkt_str)
    end

    geo_flag = if is_geo !== nothing
        is_geo
    else
        _is_geographic_coordinates(all_points)
    end

    if geo_flag
        return CoordinateTransformer(:geographic, true, nothing, 20, true, true, 1.0, 0.0, 0.0, 0.0, 0.0, wkt_str)
    end

    if isempty(xs) || isempty(ys)
        return CoordinateTransformer(:planar, false, nothing, 20, true, true, 1.0, 0.0, 0.0, lon_center, lat_center, wkt_str)
    end

    dx = max_x - min_x
    dy = max_y - min_y
    x_bar = (min_x + max_x) / 2.0
    y_bar = (min_y + max_y) / 2.0
    s = dy > 1e-9 ? 1.0 / dy : (dx > 1e-9 ? 1.0 / dx : 1.0)

    return CoordinateTransformer(:planar, false, nothing, 20, true, true, s, x_bar, y_bar, lon_center, lat_center, wkt_str)
end

function _transform_point(tf::CoordinateTransformer, pt)::Tuple{Float64, Float64}
    x, y = float(pt[1]), float(pt[2])
    # 1. If coordinate is already in standard geographic WGS84 range
    if -180.0 <= x <= 180.0 && -90.0 <= y <= 90.0
        return (x, y)
    end

    # 2. CoordRefSystems transformation mode
    if tf.mode == :utm || tf.mode == :crs
        if !isnothing(tf.crs_type)
            try
                scale_m = tf.utm_is_km ? 1000.0 : 1.0
                pt_crs = tf.crs_type((x * scale_m) * 1.0u"m", (y * scale_m) * 1.0u"m")
                ll = convert(LatLon, pt_crs)
                lon = Float64(ll.lon.val)
                lat = Float64(ll.lat.val)
                if !isnan(lon) && !isnan(lat) && isfinite(lon) && isfinite(lat)
                    return (lon, lat)
                end
            catch
            end
        end
        return utm_to_lonlat(x, y; zone=tf.utm_zone, is_km=tf.utm_is_km, northern=tf.northern)
    end

    # 3. If explicit geographic mode
    if tf.is_geo
        return (x, y)
    end

    # 4. Fallback planar isometric projection
    lon = (x - tf.x_mean) * tf.scale + tf.lon_center
    lat = (y - tf.y_mean) * tf.scale + tf.lat_center
    return (lon, lat)
end

function _transform_polygon(tf::CoordinateTransformer, poly)
    return [_transform_point(tf, pt) for pt in poly if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])]
end



# =============================================================================
# Section 3: Standalone HTML Template Engine
# =============================================================================

"""
    _generate_leaflet_html_document(;
        title::String,
        map_id::String,
        map_setup_js::String,
        extra_css::String = "",
        extra_html_body::String = "",
        dark_mode::Bool = true,
        width::String = "100%",
        height::String = "650px"
    )::String

Builds a self-contained, publication-grade HTML document embedding Leaflet.js,
modern styling (Outfit/Inter fonts, CSS glassmorphism, responsive controls,
layer toggles, colorbars, planar grid canvas, and tooltip logic).
"""
function _generate_leaflet_html_document(;
    title::String,
    map_id::String,
    map_setup_js::String,
    extra_css::String = "",
    extra_html_body::String = "",
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "650px"
)::String
    bg_color = dark_mode ? "#0f172a" : "#f8fafc"
    panel_bg = dark_mode ? "rgba(30, 41, 59, 0.85)" : "rgba(255, 255, 255, 0.9)"
    text_color = dark_mode ? "#f1f5f9" : "#0f172a"
    text_muted = dark_mode ? "#94a3b8" : "#64748b"
    border_color = dark_mode ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.1)"
    accent_color = "#38bdf8"

    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>$(title) - BSTM Interactive Visualization</title>
  
  <!-- Modern Typography -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
        
  <!-- Chart.js for Interactive Metrics -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

  <style>
    :root {
      --bg-main: $(bg_color);
      --panel-bg: $(panel_bg);
      --text-main: $(text_color);
      --text-muted: $(text_muted);
      --border-main: $(border_color);
      --accent: $(accent_color);
      --font-main: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--font-main);
      background-color: var(--bg-main);
      color: var(--text-main);
      line-height: 1.5;
      padding: 16px;
      overflow-x: hidden;
    }

    .bstm-container {
      max-width: 1440px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .bstm-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 20px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .bstm-title-area h1 {
      font-size: 1.25rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .bstm-badge {
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 8px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .bstm-subtitle {
      font-size: 0.84rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .bstm-map-wrapper {
      position: relative;
      width: $(width);
      height: $(height);
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border-main);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    }

    #$(map_id) {
      width: 100%;
      height: 100%;
      background: $(dark_mode ? "#0b0f19" : "#e2e8f0");
    }

    /* Planar Grid Background (1:1 length scales for non-geographic coordinates) */
    .bstm-planar-bg {
      background-color: $(dark_mode ? "#090d16" : "#f8fafc") !important;
      background-image: 
        linear-gradient($(dark_mode ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.06)") 1px, transparent 1px),
        linear-gradient(90deg, $(dark_mode ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.06)") 1px, transparent 1px),
        linear-gradient($(dark_mode ? "rgba(255, 255, 255, 0.015)" : "rgba(0, 0, 0, 0.02)") 1px, transparent 1px),
        linear-gradient(90deg, $(dark_mode ? "rgba(255, 255, 255, 0.015)" : "rgba(0, 0, 0, 0.02)") 1px, transparent 1px) !important;
      background-size: 50px 50px, 50px 50px, 10px 10px, 10px 10px !important;
    }

    .bstm-coord-indicator {
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      padding: 5px 12px;
      border-radius: 8px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: var(--accent);
      box-shadow: 0 4px 15px rgba(0,0,0,0.25);
      font-weight: 500;
    }

    /* Custom Leaflet Controls & Legends */
    .leaflet-control-layers {
      background: var(--panel-bg) !important;
      color: var(--text-main) !important;
      border: 1px solid var(--border-main) !important;
      border-radius: 10px !important;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
      font-family: var(--font-main) !important;
      font-size: 0.85rem !important;
      backdrop-filter: blur(10px);
    }

    .leaflet-control-layers label {
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 4px 0;
      cursor: pointer;
    }

    .bstm-legend {
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      padding: 10px 14px;
      border-radius: 10px;
      color: var(--text-main);
      font-size: 0.82rem;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
      line-height: 1.3;
      max-width: 260px;
    }

    .bstm-legend-title {
      font-weight: 600;
      margin-bottom: 6px;
      font-size: 0.85rem;
      color: var(--text-main);
    }

    .bstm-colorbar {
      height: 12px;
      border-radius: 4px;
      margin: 6px 0 4px 0;
      border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .bstm-colorbar-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }

    .bstm-popup {
      font-family: var(--font-main);
      font-size: 0.84rem;
      line-height: 1.4;
      color: #0f172a;
    }

    .bstm-popup-title {
      font-weight: 700;
      font-size: 0.92rem;
      margin-bottom: 4px;
      color: #0369a1;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 2px;
    }

    .bstm-popup-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 2px 0;
    }

    .bstm-popup-label {
      color: #64748b;
      font-weight: 500;
    }

    .bstm-popup-val {
      font-family: var(--font-mono);
      font-weight: 600;
      color: #0f172a;
    }

    .bstm-controls-bar {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }

    .bstm-btn {
      background: var(--panel-bg);
      color: var(--text-main);
      border: 1px solid var(--border-main);
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-family: var(--font-main);
      font-size: 0.82rem;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    }

    .bstm-btn:hover {
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--accent);
      color: #fff;
    }

    .bstm-arrow-icon-container {
      background: transparent !important;
      border: none !important;
    }

    .bstm-arrow-head {
      transition: transform 0.15s ease-out;
      filter: drop-shadow(0 1px 2px rgba(0,0,0,0.6));
    }

    $(extra_css)
  </style>
</head>
<body>

  <div class="bstm-container">
    <header class="bstm-header">
      <div class="bstm-title-area">
        <h1>
          <span>$(title)</span>
          <span class="bstm-badge">BSTM Interactive</span>
        </h1>
        <div class="bstm-subtitle">Bayesian Spatio-Temporal Modeling & Movement Engine</div>
      </div>
      <div class="bstm-controls-bar">
        <button class="bstm-btn" onclick="resetMapView()">⟲ Reset View</button>
      </div>
    </header>

    <div class="bstm-map-wrapper">
      <div id="$(map_id)"></div>
    </div>

    $(extra_html_body)
  </div>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>

  <script>
    $(map_setup_js)
  </script>
</body>
</html>
"""
end


# =============================================================================
# Section 4: Geographic Choropleths & Spatial Graph Visualizations
# =============================================================================

"""
    leaflet_choropleth(polygons, values;
                       title="Spatial Choropleth Map", cmap=:viridis,
                       vmin=nothing, vmax=nothing, clims=nothing, center_zero=false,
                       colorbar_label="Value", border_color="#333333", border_width=1.0,
                       fill_opacity=0.82, tooltip_prefix="Unit", extra_props=nothing,
                       width="100%", height="650px", dark_mode=true, kwargs...)

Renders an interactive polygon choropleth map in Leaflet HTML format.

# Mathematical Formulation
Each spatial unit ``i \\in \\{1,\\dots,S\\}`` is mapped to color ``\\mathcal{C}(v_i)``
interpolated across `cmap` over ``[v_{\\min}, v_{\\max}]``.

# Arguments
- `polygons`: Collection of polygon coordinate vectors `Vector{Vector{Tuple{Real, Real}}}`.
- `values`: Scalar value vector across polygons of length ``S``.
- `title::String`: Map super-title. Default: `"Spatial Choropleth Map"`.
- `cmap::Symbol`: Colormap (`:viridis`, `:plasma`, `:inferno`, `:turbo`, `:cividis`, `:coolwarm`, `:RdBu`, `:Blues`, `:YlOrRd`).
- `vmin`, `vmax`: Value domain bounds (defaults to 2nd and 98th empirical percentiles).
- `center_zero::Bool`: Whether to center colormap symmetrically around 0.
- `colorbar_label::String`: Label on interactive colorbar legend.
- `fill_opacity::Real`: Polygon fill transparency (default: `0.82`).

# Returns
- `LeafletMap`: Standalone interactive visualization container.
"""
function leaflet_choropleth(
    polygons,
    values;
    title::String = "Spatial Choropleth Map",
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    clims = nothing,
    center_zero::Bool = false,
    colorbar_label::String = "Value",
    border_color::String = "#475569",
    border_width::Real = 1.0,
    fill_opacity::Real = 0.82,
    tooltip_prefix::String = "Unit",
    extra_props = nothing,
    au = nothing,
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    width::String = "100%",
    height::String = "650px",
    dark_mode::Bool = true,
    kwargs...
)::LeafletMap
    n_poly = length(polygons)
    vals = collect(Float64, values)
    @assert length(vals) == n_poly "length(values) ($(length(vals))) must match length(polygons) ($n_poly)"

    mask = .!isnan.(vals) .& .!isinf.(vals)
    valid_vals = vals[mask]

    palette = _resolve_palette(cmap)
    
    # Calculate robust value bounds
    if clims !== nothing
        vmin = clims[1]
        vmax = clims[2]
    end
    if vmin === nothing
        vmin = !isempty(valid_vals) ? (length(valid_vals) > 1 ? quantile(valid_vals, 0.02) : minimum(valid_vals)) : 0.0
    end
    if vmax === nothing
        vmax = !isempty(valid_vals) ? (length(valid_vals) > 1 ? quantile(valid_vals, 0.98) : maximum(valid_vals)) : 1.0
    end
    if center_zero
        lim = max(abs(vmin), abs(vmax))
        vmin, vmax = -lim, lim
    end
    if abs(vmax - vmin) < 1e-12
        vmax = vmin + 1.0
    end

    # Extract WKT from au if provided
    wkt_str = !isnothing(wkt) ? string(wkt) : _extract_wkt(au)

    # Collect all raw coordinates to configure transformer
    all_raw_pts = Tuple{Float64, Float64}[]
    for poly in polygons
        for pt in poly
            if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
                push!(all_raw_pts, (float(pt[1]), float(pt[2])))
            end
        end
    end

    tf = _build_coordinate_transformer(all_raw_pts; wkt=wkt_str, is_geo=is_geo, lon_center=-160.0, lat_center=0.0)

    # Build GeoJSON Features
    features_json = String[]
    all_lats = Float64[]
    all_lngs = Float64[]

    for i in 1:n_poly
        poly = polygons[i]
        if length(poly) < 3
            continue
        end

        poly_trans = _transform_polygon(tf, poly)
        if length(poly_trans) < 3
            continue
        end

        # Leaflet GeoJSON expects [lng, lat]
        coords_json = String[]
        for pt in poly_trans
            push!(coords_json, "[$(pt[1]), $(pt[2])]")
            push!(all_lngs, pt[1])
            push!(all_lats, pt[2])
        end
        if isempty(coords_json)
            continue
        end
        # Ensure closed polygon ring
        if coords_json[1] != coords_json[end]
            push!(coords_json, coords_json[1])
        end

        raw_xs = [float(p[1]) for p in poly if length(p) >= 2 && !isnan(p[1])]
        raw_ys = [float(p[2]) for p in poly if length(p) >= 2 && !isnan(p[2])]
        orig_cx = !isempty(raw_xs) ? round(mean(raw_xs), digits=2) : 0.0
        orig_cy = !isempty(raw_ys) ? round(mean(raw_ys), digits=2) : 0.0

        val = vals[i]
        hex_col = mask[i] ? _map_val_to_hex(val, vmin, vmax, palette; center_zero=center_zero) : "#64748b"
        val_str = mask[i] ? @sprintf("%.4f", val) : "NaN"

        extra_json = String[]
        if !isnothing(extra_props) && haskey(extra_props, i)
            for (k, v) in pairs(extra_props[i])
                push!(extra_json, "\"$k\": \"$v\"")
            end
        end
        extra_str = !isempty(extra_json) ? ", " * join(extra_json, ", ") : ""

        f = """{
          "type": "Feature",
          "id": $i,
          "properties": {
            "unit_id": $i,
            "value": $val,
            "value_str": "$val_str",
            "orig_x": $orig_cx,
            "orig_y": $orig_cy,
            "fillColor": "$hex_col"$extra_str
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [[$(join(coords_json, ", "))]]
          }
        }"""
        push!(features_json, f)
    end

    geojson_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(features_json, ",\n"))]}"

    # Bounds calculation in projected space
    min_lat = !isempty(all_lats) ? minimum(all_lats) : -1.0
    max_lat = !isempty(all_lats) ? maximum(all_lats) : 1.0
    min_lng = !isempty(all_lngs) ? minimum(all_lngs) : -1.0
    max_lng = !isempty(all_lngs) ? maximum(all_lngs) : 1.0

    gradient_css = "linear-gradient(to right, " * join(palette, ", ") * ")"
    map_id = "bstm_map_" * string(abs(hash(title * string(rand()))), base=16)

    vmin_lbl = @sprintf("%.2f", vmin)
    vmid_lbl = @sprintf("%.2f", (vmin + vmax) / 2.0)
    vmax_lbl = @sprintf("%.2f", vmax)

    orig_row_js = !tf.is_geo ? " + '<div class=\"bstm-popup-row\"><span class=\"bstm-popup-label\">Orig Centroid:</span><span class=\"bstm-popup-val\">(' + props.orig_x + ', ' + props.orig_y + ')</span></div>'" : ""

    map_setup_js = """
    var isGeo = $(tf.is_geo ? "true" : "false");
    var map = L.map('$(map_id)', { attributionControl: false });

    if (!isGeo) {
      var coordControl = L.control({ position: 'bottomleft' });
      coordControl.onAdd = function() {
        var div = L.DomUtil.create('div', 'bstm-coord-indicator');
        div.innerHTML = 'Planar Projection (Equatorial Pacific 1:1)';
        return div;
      };
      coordControl.addTo(map);
      map.on('mousemove', function(e) {
        var el = document.querySelector('.bstm-coord-indicator');
        if (el) {
          var origX = (e.latlng.lng - ($(tf.lon_center))) / ($(tf.scale)) + ($(tf.x_mean));
          var origY = (e.latlng.lat - ($(tf.lat_center))) / ($(tf.scale)) + ($(tf.y_mean));
          el.innerHTML = 'Planar (x: ' + origX.toFixed(2) + ', y: ' + origY.toFixed(2) + ') [1:1]';
        }
      });
    }

    // Base Tile Layers
    var baseLayers = {};
    var cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var cartoLight = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19
    });
    var esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri &copy; GEBCO, NOAA',
      maxZoom: 13
    });

    $(dark_mode ? "cartoDark.addTo(map);" : "cartoLight.addTo(map);")
    baseLayers["CartoDB Dark"] = cartoDark;
    baseLayers["CartoDB Positron"] = cartoLight;
    baseLayers["OpenStreetMap"] = osm;
    baseLayers["Esri Ocean"] = esriOcean;

    var geojsonData = $(geojson_collection);

    function styleFeature(feature) {
      return {
        fillColor: feature.properties.fillColor,
        weight: $(border_width),
        opacity: 0.9,
        color: '$(border_color)',
        fillOpacity: $(fill_opacity)
      };
    }

    function highlightFeature(e) {
      var layer = e.target;
      layer.setStyle({
        weight: $(border_width + 1.5),
        color: '#38bdf8',
        fillOpacity: 0.95
      });
      layer.bringToFront();
    }

    function resetHighlight(e) {
      geojsonLayer.resetStyle(e.target);
    }

    var geojsonLayer = L.geoJSON(geojsonData, {
      style: styleFeature,
      onEachFeature: function(feature, layer) {
        layer.on({
          mouseover: highlightFeature,
          mouseout: resetHighlight
        });
        var props = feature.properties;
        var popupContent = '<div class="bstm-popup">' +
          '<div class="bstm-popup-title">$(tooltip_prefix) #' + props.unit_id + '</div>' +
          '<div class="bstm-popup-row"><span class="bstm-popup-label">$(colorbar_label):</span><span class="bstm-popup-val">' + props.value_str + '</span></div>'$(orig_row_js) +
          '</div>';
        layer.bindPopup(popupContent);
      }
    }).addTo(map);

    // Fit map bounds with 1:1 aspect ratio
    var bounds = [[$(min_lat), $(min_lng)], [$(max_lat), $(max_lng)]];
    map.fitBounds(bounds, { padding: [25, 25] });

    function resetMapView() {
      map.fitBounds(bounds, { padding: [25, 25] });
    }

    // Legend Control
    var legend = L.control({ position: 'bottomright' });
    legend.onAdd = function(map) {
      var div = L.DomUtil.create('div', 'bstm-legend');
      div.innerHTML = '<div class="bstm-legend-title">$(colorbar_label)</div>' +
        '<div class="bstm-colorbar" style="background: $(gradient_css);"></div>' +
        '<div class="bstm-colorbar-labels">' +
        '<span>$(vmin_lbl)</span>' +
        '<span>$(vmid_lbl)</span>' +
        '<span>$(vmax_lbl)</span>' +
        '</div>';
      return div;
    };
    legend.addTo(map);

    L.control.layers(baseLayers, { "Choropleth Layer": geojsonLayer }, { position: 'topright' }).addTo(map);
    """

    doc = _generate_leaflet_html_document(
        title = title,
        map_id = map_id,
        map_setup_js = map_setup_js,
        dark_mode = dark_mode,
        width = width,
        height = height
    )

    return LeafletMap(doc, title=title, width=width, height=height,
                      metadata=Dict(:n_units=>n_poly, :vmin=>vmin, :vmax=>vmax, :is_geo=>tf.is_geo, :wkt=>wkt_str))
end

# Multi-dispatch alias for standard choropleth
const leaflet_spatial_map = leaflet_choropleth


"""
    leaflet_spatial_graph(centroids, g;
                          polygons=nothing, hull_coords=nothing, pts=nothing,
                          au=nothing, wkt=nothing, is_geo=nothing,
                          title="Spatial Partitioning & Graph",
                          node_size=4.5, node_color="#38bdf8",
                          edge_color="#f43f5e", edge_width=1.8,
                          dark_mode=true, width="100%", height="650px", kwargs...)

Generates an interactive Leaflet visualization of the spatial partitioning network,
displaying polygon areal units, graph adjacency edges, centroid nodes, boundary hull,
and raw observation points with toggleable layer controls.
"""
function leaflet_spatial_graph(
    centroids,
    g;
    polygons = nothing,
    hull_coords = nothing,
    pts = nothing,
    au = nothing,
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    title::String = "Spatial Partitioning & Graph",
    node_size::Real = 4.5,
    node_color::String = "#38bdf8",
    edge_color::String = "#f43f5e",
    edge_width::Real = 1.8,
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "650px",
    kwargs...
)::LeafletMap
    S = length(centroids)
    wkt_str = !isnothing(wkt) ? string(wkt) : _extract_wkt(au)

    # Collect all points to configure transformer
    all_raw_pts = Tuple{Float64, Float64}[]
    for c in centroids
        if length(c) >= 2 && !isnan(c[1]) && !isnan(c[2])
            push!(all_raw_pts, (float(c[1]), float(c[2])))
        end
    end
    if !isnothing(polygons)
        for poly in polygons, pt in poly
            if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
                push!(all_raw_pts, (float(pt[1]), float(pt[2])))
            end
        end
    end
    if !isnothing(pts)
        for pt in pts
            if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
                push!(all_raw_pts, (float(pt[1]), float(pt[2])))
            end
        end
    end

    tf = _build_coordinate_transformer(all_raw_pts; wkt=wkt_str, is_geo=is_geo, lon_center=-160.0, lat_center=0.0)

    all_lats = Float64[]
    all_lngs = Float64[]

    # 1. Centroid Nodes GeoJSON
    nodes_json = String[]
    for i in 1:S
        c = centroids[i]
        orig_x, orig_y = round(float(c[1]), digits=2), round(float(c[2]), digits=2)
        c_trans = _transform_point(tf, c)
        push!(all_lngs, c_trans[1])
        push!(all_lats, c_trans[2])
        deg_i = !isnothing(g) && i <= Graphs.nv(g) ? Graphs.degree(g, i) : 0
        f = """{
          "type": "Feature",
          "id": $i,
          "properties": { "unit_id": $i, "degree": $deg_i, "orig_x": $orig_x, "orig_y": $orig_y },
          "geometry": { "type": "Point", "coordinates": [$(c_trans[1]), $(c_trans[2])] }
        }"""
        push!(nodes_json, f)
    end
    nodes_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(nodes_json, ",\n"))]}"

    # 2. Graph Edges GeoJSON
    edges_json = String[]
    if !isnothing(g)
        for e in Graphs.edges(g)
            u, v = Graphs.src(e), Graphs.dst(e)
            if u <= S && v <= S
                c1_trans = _transform_point(tf, centroids[u])
                c2_trans = _transform_point(tf, centroids[v])
                f = """{
                  "type": "Feature",
                  "properties": { "src": $u, "dst": $v },
                  "geometry": {
                    "type": "LineString",
                    "coordinates": [[$(c1_trans[1]), $(c1_trans[2])], [$(c2_trans[1]), $(c2_trans[2])]]
                  }
                }"""
                push!(edges_json, f)
            end
        end
    end
    edges_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(edges_json, ",\n"))]}"

    # 3. Polygons GeoJSON
    polys_json = String[]
    if !isnothing(polygons)
        for (i, p) in enumerate(polygons)
            if length(p) > 2
                p_trans = _transform_polygon(tf, p)
                if length(p_trans) > 2
                    coords = ["[$(pt[1]), $(pt[2])]" for pt in p_trans]
                    if coords[1] != coords[end]
                        push!(coords, coords[1])
                    end
                    f = """{
                      "type": "Feature",
                      "id": $i,
                      "properties": { "unit_id": $i },
                      "geometry": { "type": "Polygon", "coordinates": [[$(join(coords, ", "))]] }
                    }"""
                    push!(polys_json, f)
                end
            end
        end
    end
    polys_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(polys_json, ",\n"))]}"

    # 4. Hull GeoJSON
    hull_json = ""
    if !isnothing(hull_coords) && length(hull_coords) > 2
        h_trans = _transform_polygon(tf, hull_coords)
        if length(h_trans) > 2
            h_coords = ["[$(pt[1]), $(pt[2])]" for pt in h_trans]
            if h_coords[1] != h_coords[end]
                push!(h_coords, h_coords[1])
            end
            hull_json = """{
              "type": "Feature",
              "geometry": { "type": "Polygon", "coordinates": [[$(join(h_coords, ", "))]] }
            }"""
        end
    end

    # 5. Raw Points GeoJSON
    pts_json = String[]
    if !isnothing(pts)
        for (i, pt) in enumerate(pts)
            if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
                pt_trans = _transform_point(tf, pt)
                push!(pts_json, """{
                  "type": "Feature",
                  "properties": { "pt_id": $i },
                  "geometry": { "type": "Point", "coordinates": [$(pt_trans[1]), $(pt_trans[2])] }
                }""")
            end
        end
    end
    pts_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(pts_json, ",\n"))]}"

    # Map Bounds
    min_lat = !isempty(all_lats) ? minimum(all_lats) : -1.0
    max_lat = !isempty(all_lats) ? maximum(all_lats) : 1.0
    min_lng = !isempty(all_lngs) ? minimum(all_lngs) : -1.0
    max_lng = !isempty(all_lngs) ? maximum(all_lngs) : 1.0

    map_id = "bstm_graph_map_" * string(abs(hash(title * string(rand()))), base=16)

    hull_js_block = if !isempty(hull_json)
        """
        var hullData = $(hull_json);
        var hullLayer = L.geoJSON(hullData, {
          style: {
            fillColor: 'transparent',
            color: '#fbbf24',
            weight: 2.2,
            dashArray: '5, 5'
          }
        }).addTo(map);
        overlayLayers["Boundary Hull"] = hullLayer;
        """
    else
        ""
    end

    base_layer_add = dark_mode ? "cartoDark.addTo(map);" : "cartoLight.addTo(map);"
    orig_node_row_js = !tf.is_geo ? " + '<div class=\"bstm-popup-row\"><span class=\"bstm-popup-label\">Orig Coords:</span><span class=\"bstm-popup-val\">(' + p.orig_x + ', ' + p.orig_y + ')</span></div>'" : ""

    map_setup_js = """
    var isGeo = $(tf.is_geo ? "true" : "false");
    var map = L.map('$(map_id)', { attributionControl: false });

    if (!isGeo) {
      var coordControl = L.control({ position: 'bottomleft' });
      coordControl.onAdd = function() {
        var div = L.DomUtil.create('div', 'bstm-coord-indicator');
        div.innerHTML = 'Planar Projection (Equatorial Pacific 1:1)';
        return div;
      };
      coordControl.addTo(map);
      map.on('mousemove', function(e) {
        var el = document.querySelector('.bstm-coord-indicator');
        if (el) {
          var origX = (e.latlng.lng - ($(tf.lon_center))) / ($(tf.scale)) + ($(tf.x_mean));
          var origY = (e.latlng.lat - ($(tf.lat_center))) / ($(tf.scale)) + ($(tf.y_mean));
          el.innerHTML = 'Planar (x: ' + origX.toFixed(2) + ', y: ' + origY.toFixed(2) + ') [1:1]';
        }
      });
    }

    var baseLayers = {};
    var cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var cartoLight = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 });
    var esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', { maxZoom: 13 });

    $(base_layer_add)
    baseLayers["CartoDB Dark"] = cartoDark;
    baseLayers["CartoDB Positron"] = cartoLight;
    baseLayers["OpenStreetMap"] = osm;
    baseLayers["Esri Ocean"] = esriOcean;

    var overlayLayers = {};

    // 1. Polygons
    var polysData = $(polys_collection);
    if (polysData.features.length > 0) {
      var polyLayer = L.geoJSON(polysData, {
        style: {
          fillColor: '#334155',
          fillOpacity: 0.15,
          color: '#64748b',
          weight: 1.0
        },
        onEachFeature: function(feature, layer) {
          layer.bindPopup('<div class="bstm-popup"><div class="bstm-popup-title">Areal Unit #' + feature.properties.unit_id + '</div></div>');
        }
      }).addTo(map);
      overlayLayers["Tessellation Polygons"] = polyLayer;
    }

    // 2. Hull
    $(hull_js_block)

    // 3. Graph Edges
    var edgesData = $(edges_collection);
    if (edgesData.features.length > 0) {
      var edgeLayer = L.geoJSON(edgesData, {
        style: {
          color: '$(edge_color)',
          weight: $(edge_width),
          opacity: 0.75
        }
      }).addTo(map);
      overlayLayers["Adjacency Edges"] = edgeLayer;
    }

    // 4. Centroid Nodes
    var nodesData = $(nodes_collection);
    var nodeLayer = L.geoJSON(nodesData, {
      pointToLayer: function(feature, latlng) {
        return L.circleMarker(latlng, {
          radius: $(node_size),
          fillColor: '$(node_color)',
          color: '#ffffff',
          weight: 1.2,
          opacity: 1.0,
          fillOpacity: 0.95
        });
      },
      onEachFeature: function(feature, layer) {
        var p = feature.properties;
        layer.bindPopup('<div class="bstm-popup"><div class="bstm-popup-title">Centroid Unit #' + p.unit_id + '</div><div class="bstm-popup-row"><span class="bstm-popup-label">Degree (Neighbors):</span><span class="bstm-popup-val">' + p.degree + '</span></div>'$(orig_node_row_js) + '</div>');
      }
    }).addTo(map);
    overlayLayers["Centroids"] = nodeLayer;

    // 5. Raw Points
    var ptsData = $(pts_collection);
    if (ptsData.features.length > 0) {
      var ptLayer = L.geoJSON(ptsData, {
        pointToLayer: function(feature, latlng) {
          return L.circleMarker(latlng, {
            radius: 2.0,
            fillColor: '#94a3b8',
            color: 'transparent',
            fillOpacity: 0.4
          });
        }
      });
      overlayLayers["Raw Points"] = ptLayer;
    }

    var bounds = [[$(min_lat), $(min_lng)], [$(max_lat), $(max_lng)]];
    map.fitBounds(bounds, { padding: [25, 25] });

    function resetMapView() {
      map.fitBounds(bounds, { padding: [25, 25] });
    }

    L.control.layers(baseLayers, overlayLayers, { position: 'topright', collapsed: false }).addTo(map);
    """

    doc = _generate_leaflet_html_document(
        title = title,
        map_id = map_id,
        map_setup_js = map_setup_js,
        dark_mode = dark_mode,
        width = width,
        height = height
    )

    return LeafletMap(doc, title=title, width=width, height=height,
                      metadata=Dict(:n_centroids=>S, :is_geo=>tf.is_geo, :wkt=>wkt_str))
end

function leaflet_spatial_graph(; au=nothing, pts=nothing, title="Spatial Partitioning", kwargs...)
    if isnothing(au)
        error("leaflet_spatial_graph requires either (centroids, g) or `au=(...)`.")
    end
    polygons = hasproperty(au, :polygons) ? au.polygons : (haskey(au, :polygons) ? au[:polygons] : nothing)
    centroids = hasproperty(au, :centroids) ? au.centroids : (haskey(au, :centroids) ? au[:centroids] : nothing)
    g = hasproperty(au, :graph) ? au.graph : (haskey(au, :graph) ? au[:graph] : nothing)
    hull = hasproperty(au, :hull_coords) ? au.hull_coords : (haskey(au, :hull_coords) ? au[:hull_coords] : nothing)
    return leaflet_spatial_graph(centroids, g; polygons=polygons, hull_coords=hull, pts=pts, title=title, kwargs...)
end

leaflet_spatial_graph(au::Union{NamedTuple, AbstractDict}; kwargs...) = leaflet_spatial_graph(; au=au, kwargs...)


# =============================================================================
# Section 5: Movement Maps (HSI, Diffusion, Residence Time, Velocity Vectors)
# =============================================================================

"""
    leaflet_hsi_map(hsi, au; title="Habitat Suitability Index (HSI)",
                    cmap=:viridis, show_centroids=false, show_hull=true,
                    colorbar_label="HSI", kwargs...)

Renders an interactive Leaflet choropleth map of the Habitat Suitability Index (HSI)
over the tessellated spatial domain.
"""
function leaflet_hsi_map(
    hsi::AbstractVector{<:Real},
    au::NamedTuple;
    title::String = "Habitat Suitability Index (HSI)",
    cmap::Symbol = :viridis,
    show_centroids::Bool = false,
    show_hull::Bool = true,
    colorbar_label::String = "HSI",
    kwargs...
)::LeafletMap
    polys = hasproperty(au, :polygons) ? au.polygons : au[:polygons]
    return leaflet_choropleth(
        polys, hsi;
        title = title,
        cmap = cmap,
        colorbar_label = colorbar_label,
        tooltip_prefix = "Habitat Unit",
        kwargs...
    )
end

"""
    leaflet_diffusion_map(diffusion, au; title="Spatial Diffusion Field (D)",
                          cmap=:plasma, colorbar_label="Diffusion (D)", kwargs...)

Renders an interactive Leaflet choropleth map representing the spatial distribution
of dispersal or diffusion rates ``D`` across spatial units.
"""
function leaflet_diffusion_map(
    diffusion::Union{Real, AbstractVector{<:Real}},
    au::NamedTuple;
    title::String = "Spatial Diffusion Field (D)",
    cmap::Symbol = :plasma,
    colorbar_label::String = "Diffusion (D)",
    kwargs...
)::LeafletMap
    polys = hasproperty(au, :polygons) ? au.polygons : au[:polygons]
    S = length(polys)
    diff_vec = diffusion isa Real ? fill(Float64(diffusion), S) : Float64.(diffusion)
    return leaflet_choropleth(
        polys, diff_vec;
        title = title,
        cmap = cmap,
        colorbar_label = colorbar_label,
        tooltip_prefix = "Spatial Unit",
        kwargs...
    )
end

"""
    leaflet_residence_time_map(Gamma, au; title="Stationary Residence Distribution (π)",
                               cmap=:inferno, colorbar_label="Stationary π", kwargs...)

Computes and maps the stationary distribution ``\\pi`` satisfying ``\\pi \\Gamma = \\pi``,
reflecting long-term spatial occupancy and residence probabilities under the fitted movement kernel.
"""
function leaflet_residence_time_map(
    Gamma::AbstractMatrix{<:Real},
    au::NamedTuple;
    title::String = "Stationary Residence Distribution (π)",
    cmap::Symbol = :inferno,
    colorbar_label::String = "Stationary π",
    kwargs...
)::LeafletMap
    S = size(Gamma, 1)
    # Power iteration for robust principal left eigenvector
    pi_vec = fill(1.0 / S, S)
    for _ in 1:250
        pi_next = vec(pi_vec' * Gamma)
        s_sum = sum(pi_next)
        s_sum > 1e-12 && (pi_next ./= s_sum)
        pi_vec = pi_next
    end

    polys = hasproperty(au, :polygons) ? au.polygons : au[:polygons]
    return leaflet_choropleth(
        polys, pi_vec;
        title = title,
        cmap = cmap,
        colorbar_label = colorbar_label,
        tooltip_prefix = "Residence Unit",
        kwargs...
    )
end

"""
    leaflet_advection_arrows(au; hsi=nothing, velocity=1.0, Gamma=nothing,
                            relationship=:exponential, arrow_scale=1.0,
                            arrow_color="#f8fafc", background=:hsi, cmap=:viridis,
                            title="Advection Drift & Velocity Field",
                            min_speed_quantile=0.05, dark_mode=true,
                            width="100%", height="650px", kwargs...)

Generates an interactive Leaflet vector field visualization displaying directional
advection arrows with speed magnitudes, direction angles, and background choropleth.

# Mathematical Formulation
Vectors ``\\mathbf{v}_i = (v_x, v_y)_i`` at unit centroids ``\\mathbf{x}_i`` are computed
from the habitat suitability gradient or the net transition flux:
```math
\\mathbf{v}_i = v \\sum_{j \\in \\mathcal{N}(i)} w_{ij} f(\\text{HSI}_j - \\text{HSI}_i) \\frac{\\mathbf{x}_j - \\mathbf{x}_i}{\\|\\mathbf{x}_j - \\mathbf{x}_i\\|}
```
"""
function leaflet_advection_arrows(
    au::NamedTuple;
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    velocity::Real = 1.0,
    Gamma::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    relationship::Symbol = :exponential,
    arrow_scale::Real = 1.0,
    arrow_color::String = "#f8fafc",
    background::Symbol = :hsi,
    cmap::Symbol = :viridis,
    title::String = "Advection Drift & Velocity Field",
    min_speed_quantile::Real = 0.05,
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "650px",
    kwargs...
)::LeafletMap
    S = length(au.centroids)
    vx = zeros(Float64, S)
    vy = zeros(Float64, S)

    wkt_str = !isnothing(wkt) ? string(wkt) : _extract_wkt(au)
    tf = _build_coordinate_transformer(au.centroids; wkt=wkt_str, is_geo=is_geo, lon_center=-160.0, lat_center=0.0)

    if !isnothing(hsi) && hasproperty(au, :W) && !isnothing(au.W)
        W = au.W
        rows = rowvals(W)
        vals = nonzeros(W)
        hsi_vec = Float64.(hsi)

        for i in 1:S
            ci = au.centroids[i]
            for j_idx in nzrange(W, i)
                j = rows[j_idx]
                i == j && continue
                cj = au.centroids[j]
                dh = hsi_vec[j] - hsi_vec[i]
                if dh > 0.0
                    dx = cj[1] - ci[1]
                    dy = cj[2] - ci[2]
                    dist = sqrt(dx^2 + dy^2) + 1e-9
                    weight = if relationship == :exponential
                        vals[j_idx] * exp(dh)
                    elseif relationship == :logistic
                        vals[j_idx] / (1.0 + exp(-4.0 * dh))
                    else
                        vals[j_idx] * dh
                    end
                    vx[i] += weight * (dx / dist)
                    vy[i] += weight * (dy / dist)
                end
            end
        end
        vx .*= Float64(velocity)
        vy .*= Float64(velocity)
    elseif !isnothing(Gamma)
        for i in 1:S
            ci = au.centroids[i]
            for j in 1:S
                i == j && continue
                cj = au.centroids[j]
                p_ij = Gamma[i, j]
                vx[i] += p_ij * (cj[1] - ci[1])
                vy[i] += p_ij * (cj[2] - ci[2])
            end
        end
    end

    speeds = sqrt.(vx.^2 .+ vy.^2)
    max_sp = maximum(speeds)
    speed_cutoff = max_sp > 1e-12 ? quantile(speeds, clamp(min_speed_quantile, 0.0, 0.5)) : 0.0

    # Determine average nearest centroid spacing
    dists = Float64[]
    for i in 1:min(S, 25)
        ci = au.centroids[i]
        nn_d = minimum([sqrt((cj[1]-ci[1])^2 + (cj[2]-ci[2])^2) for (j, cj) in enumerate(au.centroids) if j != i])
        push!(dists, nn_d)
    end
    avg_spacing = !isempty(dists) ? mean(dists) : 1.0
    scale_fac = max_sp > 1e-12 ? (avg_spacing * 0.7 * arrow_scale) / max_sp : 1.0

    # Build Vectors GeoJSON in transformed coordinates
    arrows_json = String[]
    heads_json = String[]
    bases_json = String[]

    for i in 1:S
        sp = speeds[i]
        if sp >= speed_cutoff && max_sp > 1e-12
            c0_trans = _transform_point(tf, au.centroids[i])
            dx_trans = vx[i] * scale_fac * tf.scale
            dy_trans = vy[i] * scale_fac * tf.scale
            x1, y1 = c0_trans[1] + dx_trans, c0_trans[2] + dy_trans
            ang_deg = rad2deg(atan(vy[i], vx[i]))
            rot_deg = 90.0 - ang_deg
            sp_str = @sprintf("%.4f", sp)
            ang_str = @sprintf("%.1f", ang_deg)
            
            # 1. Shaft Line
            push!(arrows_json, """{
              "type": "Feature",
              "properties": {
                "unit_id": $i,
                "speed": $sp,
                "speed_str": "$sp_str",
                "angle_deg": $ang_str
              },
              "geometry": {
                "type": "LineString",
                "coordinates": [[$(c0_trans[1]), $(c0_trans[2])], [$x1, $y1]]
              }
            }""")

            # 2. Tip Point for Single Intermediate SVG Arrowhead Marker
            push!(heads_json, """{
              "type": "Feature",
              "properties": {
                "unit_id": $i,
                "speed": $sp,
                "speed_str": "$sp_str",
                "angle_deg": $ang_str,
                "rot_deg": $rot_deg
              },
              "geometry": {
                "type": "Point",
                "coordinates": [$x1, $y1]
              }
            }""")

            # 3. Base Point for Origin Dot Marker
            push!(bases_json, """{
              "type": "Feature",
              "properties": { "unit_id": $i },
              "geometry": {
                "type": "Point",
                "coordinates": [$(c0_trans[1]), $(c0_trans[2])]
              }
            }""")
        end
    end
    arrows_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(arrows_json, ",\n"))]}"
    heads_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(heads_json, ",\n"))]}"
    bases_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(bases_json, ",\n"))]}"

    # Background Base Map (HSI or Polygons)
    base_map = if background == :hsi && !isnothing(hsi) && hasproperty(au, :polygons)
        leaflet_hsi_map(hsi, au; title=title, cmap=cmap, wkt=wkt_str, is_geo=is_geo, dark_mode=dark_mode, width=width, height=height)
    elseif hasproperty(au, :polygons)
        polys = au.polygons
        leaflet_choropleth(polys, fill(0.5, length(polys)); au=au, title=title, wkt=wkt_str, is_geo=is_geo, dark_mode=dark_mode, width=width, height=height)
    else
        leaflet_spatial_graph(au.centroids, nothing; au=au, title=title, wkt=wkt_str, is_geo=is_geo, dark_mode=dark_mode, width=width, height=height)
    end

    # Inject vector arrows layer into HTML
    arrow_inject_js = """
    // Advection Vectors Layer (Shafts + Single SVG Arrowheads + Base Nodes)
    var arrowsData = $(arrows_collection);
    var headsData = $(heads_collection);
    var basesData = $(bases_collection);

    var arrowLayer = L.geoJSON(arrowsData, {
      style: {
        color: '$(arrow_color)',
        weight: 2.2,
        opacity: 0.95
      },
      onEachFeature: function(feature, layer) {
        var p = feature.properties;
        layer.bindPopup('<div class="bstm-popup">' +
          '<div class="bstm-popup-title">Advective Drift (Unit #' + p.unit_id + ')</div>' +
          '<div class="bstm-popup-row"><span class="bstm-popup-label">Drift Speed:</span><span class="bstm-popup-val">' + p.speed_str + '</span></div>' +
          '<div class="bstm-popup-row"><span class="bstm-popup-label">Heading:</span><span class="bstm-popup-val">' + p.angle_deg + '°</span></div>' +
          '</div>');
      }
    }).addTo(map);

    var headIconLayer = L.geoJSON(headsData, {
      pointToLayer: function(feature, latlng) {
        var p = feature.properties;
        var iconHtml = '<div class="bstm-arrow-head" style="transform: rotate(' + p.rot_deg + 'deg); width: 14px; height: 14px; display: flex; align-items: center; justify-content: center; pointer-events: auto;">' +
          '<svg width="13" height="13" viewBox="0 0 13 13" style="overflow: visible;">' +
          '<polygon points="6.5,0 12,12 6.5,9 1,12" fill="$(arrow_color)" stroke="#0f172a" stroke-width="0.75" stroke-linejoin="round"/>' +
          '</svg>' +
          '</div>';
        var arrowIcon = L.divIcon({
          className: 'bstm-arrow-icon-container',
          html: iconHtml,
          iconSize: [14, 14],
          iconAnchor: [7, 7]
        });
        return L.marker(latlng, { icon: arrowIcon, interactive: true });
      },
      onEachFeature: function(feature, layer) {
        var p = feature.properties;
        layer.bindPopup('<div class="bstm-popup">' +
          '<div class="bstm-popup-title">Advective Drift (Unit #' + p.unit_id + ')</div>' +
          '<div class="bstm-popup-row"><span class="bstm-popup-label">Drift Speed:</span><span class="bstm-popup-val">' + p.speed_str + '</span></div>' +
          '<div class="bstm-popup-row"><span class="bstm-popup-label">Heading:</span><span class="bstm-popup-val">' + p.angle_deg + '°</span></div>' +
          '</div>');
      }
    }).addTo(map);

    var baseLayer = L.geoJSON(basesData, {
      pointToLayer: function(feature, latlng) {
        return L.circleMarker(latlng, {
          radius: 2.2,
          fillColor: '$(arrow_color)',
          color: '#0f172a',
          weight: 0.8,
          fillOpacity: 1.0
        });
      }
    }).addTo(map);

    if (typeof overlayLayers !== 'undefined') {
      var vectorGroup = L.featureGroup([arrowLayer, headIconLayer, baseLayer]);
      overlayLayers["Advection Vectors"] = vectorGroup;
    }
    """

    html_modified = replace(base_map.html, "</script>\n</body>" => arrow_inject_js * "\n</script>\n</body>")
    return LeafletMap(html_modified, title=title, width=width, height=height,
                      metadata=Dict(:n_arrows=>length(arrows_json), :max_speed=>max_sp, :is_geo=>base_map.metadata[:is_geo], :wkt=>wkt_str))
end

const leaflet_velocity_field = leaflet_advection_arrows


# =============================================================================
# Section 6: Movement Trajectories & Telemetry Tracks
# =============================================================================

"""
    leaflet_tracks_map(paths, au;
                       hsi=nothing, background=:polygons,
                       title="Animal Movement Trajectories",
                       palette=:tab10, show_start_end=true,
                       max_paths=50, animated=true,
                       wkt=nothing, is_geo=nothing,
                       dark_mode=true, width="100%", height="650px", kwargs...)

Renders an interactive Leaflet visualization of individual animal movement tracks
over the spatial domain, featuring distinct individual colorings, release (start)
and recapture (end) pins, step-by-step tooltips, and interactive path playback.
"""
function leaflet_tracks_map(
    paths,
    au::Union{Nothing, NamedTuple} = NamedTuple();
    empirical_paths = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    background::Symbol = :polygons,
    title::String = "Animal Movement Trajectories",
    palette::Symbol = :tab10,
    show_start_end::Bool = true,
    max_paths::Int = 50,
    max_empirical_paths::Int = 500,
    animated::Bool = true,
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "650px",
    kwargs...
)::LeafletMap
    cents = if !isnothing(au)
        hasproperty(au, :centroids) ? au.centroids :
        (hasproperty(au, :centroids_lonlat) ? au.centroids_lonlat : Tuple{Float64, Float64}[])
    else
        Tuple{Float64, Float64}[]
    end
    polys = if !isnothing(au)
        hasproperty(au, :polygons) ? au.polygons :
        (hasproperty(au, :polygons_lonlat) ? au.polygons_lonlat : Vector{Vector{Tuple{Float64, Float64}}}())
    else
        Vector{Vector{Tuple{Float64, Float64}}}()
    end

    n_paths_total = !isnothing(paths) ? size(paths, 1) : 0
    n_to_render = min(n_paths_total, max_paths)
    colors = _resolve_palette(palette)
    wkt_str = !isnothing(wkt) ? string(wkt) : (!isnothing(au) ? _extract_wkt(au) : nothing)

    # Collect all points to configure transformer
    all_raw_pts = Tuple{Float64, Float64}[]
    for c in cents
        if length(c) >= 2 && !isnan(c[1]) && !isnan(c[2])
            push!(all_raw_pts, (float(c[1]), float(c[2])))
        end
    end
    if !isempty(polys)
        for poly in polys, pt in poly
            if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
                push!(all_raw_pts, (float(pt[1]), float(pt[2])))
            end
        end
    end
    if isempty(all_raw_pts) && !isnothing(paths) && isa(paths, AbstractVector)
        for path in paths
            if isa(path, AbstractVector)
                for pt in path
                    if !isa(pt, Number) && length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
                        push!(all_raw_pts, (float(pt[1]), float(pt[2])))
                    end
                end
            end
        end
    end
    if isempty(all_raw_pts) && !isnothing(empirical_paths) && isa(empirical_paths, AbstractVector)
        for path in empirical_paths
            if isa(path, AbstractVector)
                for pt in path
                    if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
                        push!(all_raw_pts, (float(pt[1]), float(pt[2])))
                    end
                end
            end
        end
    end

    tf = _build_coordinate_transformer(all_raw_pts; wkt=wkt_str, is_geo=is_geo, lon_center=-160.0, lat_center=0.0)

    all_lats = Float64[]
    all_lngs = Float64[]

    tracks_json = String[]
    starts_json = String[]
    ends_json = String[]

    if !isnothing(paths)
        for i in 1:n_to_render
            t_col = colors[(i - 1) % length(colors) + 1]

            coords_raw = Tuple{Float64, Float64}[]
            tag_id_str = string(i)
            n_steps_val = 0
            dur_days_val = 0.0
            dist_val_str = ""
            disp_val_str = ""
            tort_val_str = ""
            hsi_val_str = ""
            start_date_str = ""
            end_date_str = ""

            if isa(paths, AbstractMatrix)
                n_steps = size(paths, 2)
                for t in 1:n_steps
                    u_idx = paths[i, t]
                    if 1 <= u_idx <= length(cents)
                        push!(coords_raw, (float(cents[u_idx][1]), float(cents[u_idx][2])))
                    end
                end
                n_steps_val = length(coords_raw) - 1
            elseif isa(paths, AbstractVector) && !isempty(paths) && paths[1] isa NamedTuple
                tr = paths[i]
                coords_raw = tr.coords
                tag_id_str = string(tr.tagid)
                n_steps_val = hasproperty(tr, :n_steps) ?
                    tr.n_steps : length(coords_raw) - 1
                dur_days_val = hasproperty(tr, :duration_days) ?
                    tr.duration_days : 0.0
                dist_val_str = hasproperty(tr, :total_dist_km) ?
                    @sprintf("%.1f", tr.total_dist_km) : ""
                disp_val_str = hasproperty(tr, :displacement_km) ?
                    @sprintf("%.1f", tr.displacement_km) : ""
                start_date_str = hasproperty(tr, :start_date) ?
                    string(tr.start_date) : ""
                end_date_str = hasproperty(tr, :end_date) ?
                    string(tr.end_date) : ""
                if hasproperty(tr, :color)
                    t_col = string(tr.color)
                end
                tort_val_str = hasproperty(tr, :tortuosity) ?
                    @sprintf("%.2f", tr.tortuosity) : ""
                hsi_val_str = hasproperty(tr, :mean_hsi) ?
                    @sprintf("%.3f", tr.mean_hsi) : ""
            elseif isa(paths, AbstractVector) && isa(paths[1], AbstractVector)
                if !isempty(paths[1]) && paths[1][1] isa Integer && !isempty(cents)
                    for u_idx in paths[i]
                        if 1 <= u_idx <= length(cents)
                            push!(coords_raw, (float(cents[u_idx][1]), float(cents[u_idx][2])))
                        end
                    end
                else
                    for pt in paths[i]
                        if !isa(pt, Number) && length(pt) >= 2
                            push!(coords_raw, (float(pt[1]), float(pt[2])))
                        end
                    end
                end
                n_steps_val = length(coords_raw) - 1
            end

            if length(coords_raw) >= 2
                coords_trans = [_transform_point(tf, pt) for pt in coords_raw]
                coord_str = join(["[$(pt[1]), $(pt[2])]" for pt in coords_trans], ", ")
                for pt in coords_trans
                    push!(all_lngs, pt[1])
                    push!(all_lats, pt[2])
                end

                if isempty(dist_val_str)
                    dist_total = 0.0
                    for k in 2:length(coords_trans)
                        dist_total += haversine_distance(
                            coords_trans[k-1][1], coords_trans[k-1][2],
                            coords_trans[k][1], coords_trans[k][2]
                        ) / 1000.0
                    end
                    dist_val_str = @sprintf("%.1f", dist_total)
                end

                push!(tracks_json, """{
                  "type": "Feature",
                  "id": $i,
                  "properties": {
                    "tag_id": "$tag_id_str",
                    "n_steps": $n_steps_val,
                    "duration_days": $(round(dur_days_val, digits=1)),
                    "total_dist": "$dist_val_str",
                    "displacement": "$disp_val_str",
                    "tortuosity": "$tort_val_str",
                    "mean_hsi": "$hsi_val_str",
                    "start_date": "$start_date_str",
                    "end_date": "$end_date_str",
                    "color": "$t_col"
                  },
                  "geometry": { "type": "LineString", "coordinates": [$coord_str] }
                }""")

                p_start = coords_trans[1]
                push!(starts_json, """{
                  "type": "Feature",
                  "properties": { "tag_id": "$tag_id_str", "date": "$start_date_str", "type": "Release (Start)" },
                  "geometry": { "type": "Point", "coordinates": [$(p_start[1]), $(p_start[2])] }
                }""")

                p_end = coords_trans[end]
                push!(ends_json, """{
                  "type": "Feature",
                  "properties": { "tag_id": "$tag_id_str", "date": "$end_date_str", "type": "Recapture (End)" },
                  "geometry": { "type": "Point", "coordinates": [$(p_end[1]), $(p_end[2])] }
                }""")
            end
        end
    end

    tracks_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(tracks_json, ",\n"))]}"
    starts_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(starts_json, ",\n"))]}"
    ends_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(ends_json, ",\n"))]}"

    # Empirical Observation Tracks
    emp_tracks_json = String[]
    if !isnothing(empirical_paths)
        n_emp_total = length(empirical_paths)
        n_emp_render = min(n_emp_total, max_empirical_paths)
        for i in 1:n_emp_render
            pt_seq = empirical_paths[i]
            if length(pt_seq) >= 2
                coords_trans = [_transform_point(tf, pt) for pt in pt_seq]
                coord_str = join(["[$(pt[1]), $(pt[2])]" for pt in coords_trans], ", ")
                for pt in coords_trans
                    push!(all_lngs, pt[1])
                    push!(all_lats, pt[2])
                end
                push!(emp_tracks_json, """{
                  "type": "Feature",
                  "id": $i,
                  "properties": { "tag_id": $i, "n_obs": $(length(pt_seq)) },
                  "geometry": { "type": "LineString", "coordinates": [$coord_str] }
                }""")
            end
        end
    end
    emp_tracks_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(emp_tracks_json, ",\n"))]}"

    # Background Polygons in transformed space with HSI coloring
    polys_json = String[]
    has_hsi = !isnothing(hsi) && !isempty(hsi)
    hsi_vec = has_hsi ? Float64.(hsi) : Float64[]
    pal_hsi = _resolve_palette(:viridis)

    if !isempty(polys)
        for (i, p) in enumerate(polys)
            if length(p) > 2
                p_trans = _transform_polygon(tf, p)
                if length(p_trans) > 2
                    c_str = ["[$(pt[1]), $(pt[2])]" for pt in p_trans]
                    if c_str[1] != c_str[end]
                        push!(c_str, c_str[1])
                    end
                    for pt in p_trans
                        push!(all_lngs, pt[1])
                        push!(all_lats, pt[2])
                    end
                    h_val = (has_hsi && i <= length(hsi_vec)) ? hsi_vec[i] : NaN
                    h_col = !isnan(h_val) ? _map_val_to_hex(h_val, 0.0, 1.0, pal_hsi) : "#1e293b"
                    h_str = !isnan(h_val) ? @sprintf("%.3f", h_val) : "N/A"
                    push!(polys_json, """{
                      "type": "Feature",
                      "id": $i,
                      "properties": { "unit_id": $i, "hsi": "$h_str", "fill_color": "$h_col" },
                      "geometry": { "type": "Polygon", "coordinates": [[$(join(c_str, ", "))]] }
                    }""")
                end
            end
        end
    elseif !isempty(cents)
        for c in cents
            if length(c) >= 2 && !isnan(c[1]) && !isnan(c[2])
                c_trans = _transform_point(tf, (float(c[1]), float(c[2])))
                push!(all_lngs, c_trans[1])
                push!(all_lats, c_trans[2])
            end
        end
    end
    polys_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(polys_json, ",\n"))]}"

    # Bounds with safeguard against zero-height or zero-width collapse
    min_lat = !isempty(all_lats) ? minimum(all_lats) : -1.0
    max_lat = !isempty(all_lats) ? maximum(all_lats) : 1.0
    min_lng = !isempty(all_lngs) ? minimum(all_lngs) : -1.0
    max_lng = !isempty(all_lngs) ? maximum(all_lngs) : 1.0

    if max_lat - min_lat < 0.05
        mid_lat = (max_lat + min_lat) / 2.0
        min_lat = mid_lat - 0.05
        max_lat = mid_lat + 0.05
    end
    if max_lng - min_lng < 0.05
        mid_lng = (max_lng + min_lng) / 2.0
        min_lng = mid_lng - 0.05
        max_lng = mid_lng + 0.05
    end

    map_id = "bstm_tracks_map_" * string(abs(hash(title * string(rand()))), base=16)

    map_setup_js = """
    var isGeo = $(tf.is_geo ? "true" : "false");
    var map = L.map('$(map_id)', { attributionControl: false });

    if (!isGeo) {
      var coordControl = L.control({ position: 'bottomleft' });
      coordControl.onAdd = function() {
        var div = L.DomUtil.create('div', 'bstm-coord-indicator');
        div.innerHTML = 'Planar Projection (Equatorial Pacific 1:1)';
        return div;
      };
      coordControl.addTo(map);
      map.on('mousemove', function(e) {
        var el = document.querySelector('.bstm-coord-indicator');
        if (el) {
          var origX = (e.latlng.lng - ($(tf.lon_center))) / ($(tf.scale)) + ($(tf.x_mean));
          var origY = (e.latlng.lat - ($(tf.lat_center))) / ($(tf.scale)) + ($(tf.y_mean));
          el.innerHTML = 'Planar (x: ' + origX.toFixed(2) + ', y: ' + origY.toFixed(2) + ') [1:1]';
        }
      });
    }

    var baseLayers = {};
    var cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var cartoLight = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19
    });
    var esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri &copy; GEBCO, NOAA',
      maxZoom: 13
    });

    $(dark_mode ? "cartoDark.addTo(map);" : "cartoLight.addTo(map);")
    baseLayers["CartoDB Dark"] = cartoDark;
    baseLayers["CartoDB Positron"] = cartoLight;
    baseLayers["OpenStreetMap"] = osm;
    baseLayers["Esri Ocean"] = esriOcean;

    var overlayLayers = {};

    // 1. Polygons (HSI / Tessellation)
    var polysData = $(polys_collection);
    var hasHsi = $(has_hsi ? "true" : "false");
    if (polysData.features.length > 0) {
      var polyLayer = L.geoJSON(polysData, {
        style: function(feature) {
          return {
            fillColor: feature.properties.fill_color || '#1e293b',
            fillOpacity: hasHsi ? 0.65 : 0.12,
            color: hasHsi ? '#0f172a' : '#475569',
            weight: hasHsi ? 0.8 : 0.6
          };
        },
        onEachFeature: function(feature, layer) {
          var p = feature.properties;
          layer.on('mouseover', function() {
            layer.setStyle({ weight: 2.5, color: '#ffffff' });
          });
          layer.on('mouseout', function() {
            polyLayer.resetStyle(layer);
          });
          layer.bindPopup('<div class="bstm-popup">' +
            '<div class="bstm-popup-title">Areal Unit #' + p.unit_id + '</div>' +
            (hasHsi ? '<div class="bstm-popup-row"><span class="bstm-popup-label">Habitat Suitability (HSI):</span><span class="bstm-popup-val">' + p.hsi + '</span></div>' : '') +
            '</div>');
        }
      }).addTo(map);
      overlayLayers[hasHsi ? "Habitat Suitability (HSI)" : "Spatial Tessellation"] = polyLayer;
    }

    // 2. Empirical Tag Observations Layer
    var empTracksData = $(emp_tracks_collection);
    if (empTracksData.features.length > 0) {
      var empTrackLayer = L.geoJSON(empTracksData, {
        style: {
          color: '#38bdf8',
          weight: 1.5,
          opacity: 0.45
        },
        onEachFeature: function(feature, layer) {
          var p = feature.properties;
          layer.on('mouseover', function() {
            layer.setStyle({ weight: 3.5, opacity: 1.0, color: '#0284c7' });
          });
          layer.on('mouseout', function() {
            empTrackLayer.resetStyle(layer);
          });
          layer.bindPopup('<div class="bstm-popup">' +
            '<div class="bstm-popup-title" style="color:#0284c7">Direct Tag Vector #' + p.tag_id + '</div>' +
            '<div class="bstm-popup-row"><span class="bstm-popup-label">Observations:</span><span class="bstm-popup-val">' + p.n_obs + '</span></div>' +
            '</div>');
        }
      }).addTo(map);
      overlayLayers["Direct Tag Vectors"] = empTrackLayer;
    }

    // 3. Reconstructed State-Space Trajectories Layer
    var tracksData = $(tracks_collection);
    if (tracksData.features.length > 0) {
      var trackLayer = L.geoJSON(tracksData, {
        style: function(feature) {
          return {
            color: feature.properties.color,
            weight: 2.8,
            opacity: 0.85
          };
        },
        onEachFeature: function(feature, layer) {
          var p = feature.properties;
          layer.on('mouseover', function() {
            layer.setStyle({ weight: 5.0, opacity: 1.0 });
          });
          layer.on('mouseout', function() {
            trackLayer.resetStyle(layer);
          });
          var tTitle = (p.tag_id.startsWith('Tag') ||
                        p.tag_id.startsWith('Method') ||
                        p.tag_id.startsWith('Path')) ?
                        p.tag_id : 'Snow Crab Tag #' + p.tag_id;
          var popupHtml = '<div class="bstm-popup">' +
            '<div class="bstm-popup-title" style="color: ' + p.color + '">' +
            tTitle + '</div>' +
            '<div class="bstm-popup-row">' +
            '<span class="bstm-popup-label">State-Space Steps:</span>' +
            '<span class="bstm-popup-val">' + p.n_steps + '</span></div>';
          if (p.duration_days > 0) {
            popupHtml += '<div class="bstm-popup-row">' +
              '<span class="bstm-popup-label">Duration:</span>' +
              '<span class="bstm-popup-val">' + p.duration_days + ' days</span></div>';
          }
          if (p.total_dist && p.total_dist != '') {
            popupHtml += '<div class="bstm-popup-row">' +
              '<span class="bstm-popup-label">Estimated Path Length:</span>' +
              '<span class="bstm-popup-val">' + p.total_dist + ' km</span></div>';
          }
          if (p.displacement && p.displacement != '') {
            popupHtml += '<div class="bstm-popup-row">' +
              '<span class="bstm-popup-label">Net Displacement:</span>' +
              '<span class="bstm-popup-val">' + p.displacement + ' km</span></div>';
          }
          if (p.tortuosity && p.tortuosity != '') {
            popupHtml += '<div class="bstm-popup-row">' +
              '<span class="bstm-popup-label">Tortuosity Index:</span>' +
              '<span class="bstm-popup-val">' + p.tortuosity + '</span></div>';
          }
          if (p.mean_hsi && p.mean_hsi != '') {
            popupHtml += '<div class="bstm-popup-row">' +
              '<span class="bstm-popup-label">Mean Habitat HSI:</span>' +
              '<span class="bstm-popup-val">' + p.mean_hsi + '</span></div>';
          }
          if (p.start_date && p.start_date != '') {
            popupHtml += '<div class="bstm-popup-row">' +
              '<span class="bstm-popup-label">Release:</span>' +
              '<span class="bstm-popup-val">' + p.start_date + '</span></div>';
          }
          if (p.end_date && p.end_date != '') {
            popupHtml += '<div class="bstm-popup-row">' +
              '<span class="bstm-popup-label">Recapture:</span>' +
              '<span class="bstm-popup-val">' + p.end_date + '</span></div>';
          }
          popupHtml += '</div>';
          layer.bindPopup(popupHtml);
        }
      }).addTo(map);
      overlayLayers["Estimated Movement Paths (Markov Bridges)"] = trackLayer;
    }

    // 4. Start Markers (Green Circles)
    var startsData = $(starts_collection);
    if (startsData.features.length > 0) {
      var startLayer = L.geoJSON(startsData, {
        pointToLayer: function(feature, latlng) {
          return L.circleMarker(latlng, {
            radius: 5.0,
            fillColor: '#10b981',
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 0.95
          });
        },
        onEachFeature: function(feature, layer) {
          var p = feature.properties;
          layer.bindPopup('<div class="bstm-popup"><div class="bstm-popup-title" style="color:#10b981;">Release (Tag #' + p.tag_id + ')</div>' +
            (p.date ? '<div class="bstm-popup-row"><span class="bstm-popup-label">Date:</span><span class="bstm-popup-val">' + p.date + '</span></div>' : '') +
            '</div>');
        }
      }).addTo(map);
      overlayLayers["Releases (Starts)"] = startLayer;
    }

    // 5. End Markers (Red Circles)
    var endsData = $(ends_collection);
    if (endsData.features.length > 0) {
      var endLayer = L.geoJSON(endsData, {
        pointToLayer: function(feature, latlng) {
          return L.circleMarker(latlng, {
            radius: 5.5,
            fillColor: '#ef4444',
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 0.95
          });
        },
        onEachFeature: function(feature, layer) {
          var p = feature.properties;
          layer.bindPopup('<div class="bstm-popup"><div class="bstm-popup-title" style="color:#ef4444;">Recapture (Tag #' + p.tag_id + ')</div>' +
            (p.date ? '<div class="bstm-popup-row"><span class="bstm-popup-label">Date:</span><span class="bstm-popup-val">' + p.date + '</span></div>' : '') +
            '</div>');
        }
      }).addTo(map);
      overlayLayers["Recaptures (Ends)"] = endLayer;
    }


    // Layer Control
    L.control.layers(baseLayers, overlayLayers, { collapsed: false }).addTo(map);

    // HSI Colorbar Legend
    if (hasHsi) {
      var legend = L.control({ position: 'bottomright' });
      legend.onAdd = function() {
        var div = L.DomUtil.create('div', 'bstm-legend');
        div.style.backgroundColor = 'rgba(15, 23, 42, 0.88)';
        div.style.padding = '8px 12px';
        div.style.borderRadius = '6px';
        div.style.border = '1px solid #334155';
        div.style.color = '#f8fafc';
        div.style.fontSize = '12px';
        div.innerHTML = '<div style="font-weight:600;margin-bottom:4px;">Habitat Suitability (HSI)</div>' +
          '<div style="display:flex;align-items:center;gap:6px;">' +
          '<span>0.0</span>' +
          '<div style="width:120px;height:12px;border-radius:2px;background:linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725);"></div>' +
          '<span>1.0</span>' +
          '</div>';
        return div;
      };
      legend.addTo(map);
    }

    var bounds = [[$(min_lat), $(min_lng)], [$(max_lat), $(max_lng)]];
    map.fitBounds(bounds, { padding: [25, 25] });

    function resetMapView() {
      map.fitBounds(bounds, { padding: [25, 25] });
    }
    """

    if animated
        anim_js = """
        // Trajectory Playback Animation Controller
        var animControl = L.control({ position: 'bottomleft' });
        animControl.onAdd = function() {
          var c = L.DomUtil.create('div', 'bstm-playback-panel');
          c.style.backgroundColor = 'rgba(15, 23, 42, 0.92)';
          c.style.border = '1px solid #334155';
          c.style.borderRadius = '8px';
          c.style.padding = '10px 14px';
          c.style.color = '#f8fafc';
          c.style.fontFamily = 'Outfit, sans-serif';
          c.style.fontSize = '12px';
          c.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.4)';
          c.style.minWidth = '260px';
          c.style.zIndex = '1000';
          c.innerHTML =
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">' +
              '<span style="font-weight:600;color:#38bdf8;">▶ Trajectory Playback</span>' +
              '<span id="bstm-anim-percent" style="font-family:monospace;color:#94a3b8;">0%</span>' +
            '</div>' +
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">' +
              '<button id="bstm-anim-play" style="background:#0284c7;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-weight:600;">Play</button>' +
              '<button id="bstm-anim-reset" style="background:#334155;color:#f8fafc;border:none;border-radius:4px;padding:4px 8px;cursor:pointer;">Reset</button>' +
              '<select id="bstm-anim-speed" style="background:#1e293b;color:#f8fafc;border:1px solid #475569;border-radius:4px;padding:3px 6px;">' +
                '<option value="1">1x</option>' +
                '<option value="2">2x</option>' +
                '<option value="5">5x</option>' +
              '</select>' +
            '</div>' +
            '<input id="bstm-anim-slider" type="range" min="0" max="100" value="0" style="width:100%;cursor:pointer;">';
          L.DomEvent.disableClickPropagation(c);
          L.DomEvent.disableScrollPropagation(c);
          return c;
        };
        animControl.addTo(map);

        // Initialize Animated Trajectory Entities
        var animEntities = [];
        var animGroup = L.layerGroup().addTo(map);
        overlayLayers["Trajectory Animation"] = animGroup;

        if (tracksData && tracksData.features) {
          tracksData.features.forEach(function(feat) {
            if (!feat.geometry || !feat.geometry.coordinates || feat.geometry.coordinates.length < 2) return;
            var coords = feat.geometry.coordinates;
            var col = feat.properties && feat.properties.color ? feat.properties.color : '#38bdf8';
            var tag = feat.properties && feat.properties.tag_id ? feat.properties.tag_id : '';
            var marker = L.circleMarker([coords[0][1], coords[0][0]], {
              radius: 5.5,
              fillColor: col,
              color: '#ffffff',
              weight: 1.5,
              fillOpacity: 0.95
            });
            marker.bindTooltip('Tag #' + tag, { permanent: false, direction: 'top' });
            animGroup.addLayer(marker);

            var trail = L.polyline([], {
              color: col,
              weight: 2.5,
              opacity: 0.85
            });
            animGroup.addLayer(trail);

            animEntities.push({
              coords: coords,
              marker: marker,
              trail: trail
            });
          });
        }

        var isPlaying = false;
        var animProgress = 0.0;
        var animSpeed = 0.002;
        var animReqId = null;

        function updateAnimation(progress) {
          animProgress = Math.max(0.0, Math.min(1.0, progress));
          var pct = Math.round(animProgress * 100);
          var pctEl = document.getElementById('bstm-anim-percent');
          var sliderEl = document.getElementById('bstm-anim-slider');
          if (pctEl) pctEl.innerText = pct + '%';
          if (sliderEl) sliderEl.value = pct;

          animEntities.forEach(function(ent) {
            var pts = ent.coords;
            var totalSegments = pts.length - 1;
            var curPos = animProgress * totalSegments;
            var i0 = Math.floor(curPos);
            var i1 = Math.min(totalSegments, i0 + 1);
            var frac = curPos - i0;

            var p0 = pts[i0];
            var p1 = pts[i1];
            var curLng = p0[0] + frac * (p1[0] - p0[0]);
            var curLat = p0[1] + frac * (p1[1] - p0[1]);

            ent.marker.setLatLng([curLat, curLng]);

            var trailLatlngs = [];
            for (var k = 0; k <= i0; k++) {
              trailLatlngs.push([pts[k][1], pts[k][0]]);
            }
            trailLatlngs.push([curLat, curLng]);
            ent.trail.setLatLngs(trailLatlngs);
          });
        }

        function animLoop() {
          if (!isPlaying) return;
          var speedEl = document.getElementById('bstm-anim-speed');
          var speedMultiplier = speedEl ? (parseFloat(speedEl.value) || 1.0) : 1.0;
          animProgress += animSpeed * speedMultiplier;
          if (animProgress >= 1.0) {
            animProgress = 1.0;
            isPlaying = false;
            var btn = document.getElementById('bstm-anim-play');
            if (btn) btn.innerText = 'Play';
          }
          updateAnimation(animProgress);
          if (isPlaying) {
            animReqId = requestAnimationFrame(animLoop);
          }
        }

        setTimeout(function() {
          var playBtn = document.getElementById('bstm-anim-play');
          var resetBtn = document.getElementById('bstm-anim-reset');
          var slider = document.getElementById('bstm-anim-slider');

          if (playBtn) {
            playBtn.addEventListener('click', function() {
              if (isPlaying) {
                isPlaying = false;
                playBtn.innerText = 'Play';
                if (animReqId) cancelAnimationFrame(animReqId);
              } else {
                if (animProgress >= 1.0) animProgress = 0.0;
                isPlaying = true;
                playBtn.innerText = 'Pause';
                animLoop();
              }
            });
          }

          if (resetBtn) {
            resetBtn.addEventListener('click', function() {
              isPlaying = false;
              if (animReqId) cancelAnimationFrame(animReqId);
              if (playBtn) playBtn.innerText = 'Play';
              updateAnimation(0.0);
            });
          }

          if (slider) {
            slider.addEventListener('input', function() {
              isPlaying = false;
              if (animReqId) cancelAnimationFrame(animReqId);
              if (playBtn) playBtn.innerText = 'Play';
              updateAnimation(parseFloat(this.value) / 100.0);
            });
          }
          updateAnimation(0.0);
        }, 100);
        """
        map_setup_js = string(map_setup_js, "\n", anim_js)
    end

    doc = _generate_leaflet_html_document(
        title = title,
        map_id = map_id,
        map_setup_js = map_setup_js,
        dark_mode = dark_mode,
        width = width,
        height = height
    )

    return LeafletMap(doc, title=title, width=width, height=height,
                      metadata=Dict(:n_rendered=>n_to_render, :n_total=>n_paths_total, :is_geo=>tf.is_geo, :wkt=>wkt_str))
end

const leaflet_render_paths = leaflet_tracks_map


# =============================================================================
# Section 7: Spatiotemporal Time-Slider & Animation Visualizations
# =============================================================================

"""
    leaflet_spacetime_map(unique_t, st_slice_data, au;
                          title="Spatiotemporal Field Evolution",
                          cmap=:viridis, colorbar_label="Fitted Mean",
                          wkt=nothing, is_geo=nothing,
                          dark_mode=true, width="100%", height="650px", kwargs...)

Renders an interactive spatiotemporal map featuring an interactive timeline slider
and play/pause animation control, allowing the user to scrub across time slices
and observe the dynamic spatial evolution of predictions or risk fields.
"""
function leaflet_spacetime_map(
    unique_t::AbstractVector,
    st_slice_data::Dict,
    au::NamedTuple;
    title::String = "Spatiotemporal Field Evolution",
    cmap::Symbol = :viridis,
    colorbar_label::String = "Fitted Mean",
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "650px",
    kwargs...
)::LeafletMap
    polys = hasproperty(au, :polygons) ? au.polygons : au[:polygons]
    n_poly = length(polys)
    palette = _resolve_palette(cmap)
    wkt_str = !isnothing(wkt) ? string(wkt) : _extract_wkt(au)

    # Collect all polygon coordinates for transformer
    all_raw_pts = Tuple{Float64, Float64}[]
    for p in polys, pt in p
        if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
            push!(all_raw_pts, (float(pt[1]), float(pt[2])))
        end
    end

    tf = _build_coordinate_transformer(all_raw_pts; wkt=wkt_str, is_geo=is_geo, lon_center=-160.0, lat_center=0.0)

    # Compute global min and max across all time slices
    all_vals = Float64[]
    for (t, v) in pairs(st_slice_data)
        for val in v
            if !isnan(val) && !isinf(val)
                push!(all_vals, float(val))
            end
        end
    end
    vmin = !isempty(all_vals) ? quantile(all_vals, 0.02) : 0.0
    vmax = !isempty(all_vals) ? quantile(all_vals, 0.98) : 1.0
    abs(vmax - vmin) < 1e-12 && (vmax = vmin + 1.0)

    # Convert polygons to transformed GeoJSON
    features_json = String[]
    all_lats, all_lngs = Float64[], Float64[]
    for i in 1:n_poly
        p = polys[i]
        if length(p) > 2
            p_trans = _transform_polygon(tf, p)
            if length(p_trans) > 2
                c_str = ["[$(pt[1]), $(pt[2])]" for pt in p_trans]
                for pt in p_trans
                    push!(all_lngs, pt[1])
                    push!(all_lats, pt[2])
                end
                if !isempty(c_str)
                    if c_str[1] != c_str[end]
                        push!(c_str, c_str[1])
                    end
                    push!(features_json, """{
                      "type": "Feature",
                      "id": $i,
                      "properties": { "unit_id": $i },
                      "geometry": { "type": "Polygon", "coordinates": [[$(join(c_str, ", "))]] }
                    }""")
                end
            end
        end
    end
    geojson_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(features_json, ",\n"))]}"

    # Encode time slices matrix as JSON
    time_keys_str = [string(t) for t in unique_t]
    slice_data_json_entries = String[]
    for (idx, t_val) in enumerate(unique_t)
        v_vec = get(st_slice_data, t_val, zeros(Float64, n_poly))
        color_arr = [_map_val_to_hex(v_vec[k], vmin, vmax, palette) for k in 1:n_poly]
        val_arr_str = join([@sprintf("%.4f", v) for v in v_vec], ", ")
        col_arr_str = join(["\"$c\"" for c in color_arr], ", ")
        push!(slice_data_json_entries, "\"$t_val\": { \"vals\": [$val_arr_str], \"colors\": [$col_arr_str] }")
    end
    time_series_json = "{" * join(slice_data_json_entries, ",\n") * "}"

    min_lat = !isempty(all_lats) ? minimum(all_lats) : -1.0
    max_lat = !isempty(all_lats) ? maximum(all_lats) : 1.0
    min_lng = !isempty(all_lngs) ? minimum(all_lngs) : -1.0
    max_lng = !isempty(all_lngs) ? maximum(all_lngs) : 1.0

    gradient_css = "linear-gradient(to right, " * join(palette, ", ") * ")"
    map_id = "bstm_spacetime_map_" * string(abs(hash(title * string(rand()))), base=16)

    extra_css = """
    .bstm-timeline-panel {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 14px 20px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .bstm-slider {
      flex: 1;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .bstm-time-display {
      font-family: var(--font-mono);
      font-weight: 600;
      font-size: 1.05rem;
      min-width: 90px;
      color: var(--accent);
    }
    """

    first_t_str = string(unique_t[1])
    n_t_steps = length(unique_t) - 1

    extra_html = """
    <div class="bstm-timeline-panel">
      <button id="playBtn" class="bstm-btn" onclick="togglePlay()">▶ Play</button>
      <span id="timeDisplay" class="bstm-time-display">T = $(first_t_str)</span>
      <input type="range" id="timeSlider" class="bstm-slider" min="0" max="$(n_t_steps)" value="0" step="1" oninput="onTimeChange(this.value)">
    </div>
    """

    vmin_str = @sprintf("%.2f", vmin)
    vmid_str = @sprintf("%.2f", (vmin + vmax) / 2.0)
    vmax_str = @sprintf("%.2f", vmax)

    map_setup_js = """
    var isGeo = $(tf.is_geo ? "true" : "false");
    var map = L.map('$(map_id)', { attributionControl: false });

    if (!isGeo) {
      var coordControl = L.control({ position: 'bottomleft' });
      coordControl.onAdd = function() {
        var div = L.DomUtil.create('div', 'bstm-coord-indicator');
        div.innerHTML = 'Planar Projection (Equatorial Pacific 1:1)';
        return div;
      };
      coordControl.addTo(map);
      map.on('mousemove', function(e) {
        var el = document.querySelector('.bstm-coord-indicator');
        if (el) {
          var origX = (e.latlng.lng - ($(tf.lon_center))) / ($(tf.scale)) + ($(tf.x_mean));
          var origY = (e.latlng.lat - ($(tf.lat_center))) / ($(tf.scale)) + ($(tf.y_mean));
          el.innerHTML = 'Planar (x: ' + origX.toFixed(2) + ', y: ' + origY.toFixed(2) + ') [1:1]';
        }
      });
    }

    var baseLayers = {};
    var cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var cartoLight = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB &copy; OpenStreetMap',
      maxZoom: 19
    });
    var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19
    });
    var esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri &copy; GEBCO, NOAA',
      maxZoom: 13
    });

    $(dark_mode ? "cartoDark.addTo(map);" : "cartoLight.addTo(map);")
    baseLayers["CartoDB Dark"] = cartoDark;
    baseLayers["CartoDB Positron"] = cartoLight;
    baseLayers["OpenStreetMap"] = osm;
    baseLayers["Esri Ocean"] = esriOcean;

    var timeSteps = [$(join(["\"$t\"" for t in unique_t], ", "))];
    var timeData = $(time_series_json);
    var currentIdx = 0;
    var isPlaying = false;
    var playInterval = null;

    var geojsonData = $(geojson_collection);
    var layerMap = {};

    var geojsonLayer = L.geoJSON(geojsonData, {
      style: {
        weight: 1.0,
        color: '#475569',
        fillOpacity: 0.85,
        fillColor: '#334155'
      },
      onEachFeature: function(feature, layer) {
        layerMap[feature.id] = layer;
        layer.bindPopup('<div class="bstm-popup"><div class="bstm-popup-title">Unit #' + feature.id + '</div><div id="popupVal_' + feature.id + '"></div></div>');
      }
    }).addTo(map);

    function updateTimeSlice(idx) {
      currentIdx = parseInt(idx);
      var tVal = timeSteps[currentIdx];
      document.getElementById('timeDisplay').innerText = 'T = ' + tVal;
      document.getElementById('timeSlider').value = currentIdx;

      var stepObj = timeData[tVal];
      if (!stepObj) return;

      for (var id in layerMap) {
        var uIdx = parseInt(id) - 1;
        if (uIdx >= 0 && uIdx < stepObj.colors.length) {
          layerMap[id].setStyle({ fillColor: stepObj.colors[uIdx] });
        }
      }
    }

    function onTimeChange(val) {
      updateTimeSlice(val);
    }

    function togglePlay() {
      isPlaying = !isPlaying;
      var btn = document.getElementById('playBtn');
      if (isPlaying) {
        btn.innerText = '⏸ Pause';
        playInterval = setInterval(function() {
          currentIdx = (currentIdx + 1) % timeSteps.length;
          updateTimeSlice(currentIdx);
        }, 650);
      } else {
        btn.innerText = '▶ Play';
        clearInterval(playInterval);
      }
    }

    var bounds = [[$(min_lat), $(min_lng)], [$(max_lat), $(max_lng)]];
    map.fitBounds(bounds, { padding: [25, 25] });

    function resetMapView() {
      map.fitBounds(bounds, { padding: [25, 25] });
    }

    var legend = L.control({ position: 'bottomright' });
    legend.onAdd = function(map) {
      var div = L.DomUtil.create('div', 'bstm-legend');
      div.innerHTML = '<div class="bstm-legend-title">$(colorbar_label)</div>' +
        '<div class="bstm-colorbar" style="background: $(gradient_css);"></div>' +
        '<div class="bstm-colorbar-labels">' +
        '<span>$(vmin_str)</span>' +
        '<span>$(vmid_str)</span>' +
        '<span>$(vmax_str)</span>' +
        '</div>';
      return div;
    };
    legend.addTo(map);

    // Initial slice render
    updateTimeSlice(0);
    """

    doc = _generate_leaflet_html_document(
        title = title,
        map_id = map_id,
        map_setup_js = map_setup_js,
        extra_css = extra_css,
        extra_html_body = extra_html,
        dark_mode = dark_mode,
        width = width,
        height = height
    )

    return LeafletMap(doc, title=title, width=width, height=height,
                      metadata=Dict(:n_time_slices=>length(unique_t), :time_points=>unique_t, :is_geo=>tf.is_geo, :wkt=>wkt_str))
end


# =============================================================================
# Section 8: Interactive Movement Dashboard & Analytical Charts
# =============================================================================

"""
    leaflet_movement_dashboard(result, paths;
                               hsi=nothing, strata=nothing,
                               title="BSTM Movement Diagnostics Dashboard",
                               dark_mode=true, width="100%", kwargs...)

Creates a unified, publication-grade interactive HTML web dashboard integrating:
1. **Interactive Multi-Layer Leaflet Map**:
   - Habitat Suitability Index (HSI) choropleth
   - Spatial Diffusion Field (D)
   - Stationary Residence Occupancy Distribution (π)
   - Advection Drift & Velocity Field Vectors (direction and speed)
   - Animal Movement Trajectories (with release/recapture markers)
   - Spatial Partitioning Tessellation & Centroids
2. **Interactive Movement Summary Analytics (Chart.js & SVG)**:
   - **Dispersal Kernel Distance Decay**: Pairwise transition probabilities Γ_ij vs distance with binned empirical mean ± SE and fitted exponential decay curve (λ).
   - **Step Length & Turning Angle Diagnostics**: Histograms of movement steps and directional turning angles.
   - **Macro-Regional Connectivity Matrix**: Interactive heatmap of regional transition transfer probabilities (C_rs).
   - **Advection-to-Diffusion (Péclet-like) Ratio Distribution**: Histogram of local drift vs diffusion ratios across units.
   - **KPI Metric Summary Cards**: Velocity, diffusion, mean step displacement, active tag metrics.
"""
function leaflet_movement_dashboard(
    result::NamedTuple,
    paths::Union{AbstractMatrix, AbstractVector};
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    strata::Union{Nothing, AbstractVector} = nothing,
    title::String = "BSTM Movement Diagnostics Dashboard",
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    dark_mode::Bool = true,
    width::String = "100%",
    kwargs...
)::LeafletMap
    au = result.au
    Gamma = result.transition_matrix
    S = size(Gamma, 1)
    cents = au.centroids
    wkt_str = !isnothing(wkt) ? string(wkt) : _extract_wkt(au)
    hsi_vec = !isnothing(hsi) ? Float64.(hsi) : (hasproperty(result.opts, :hsi) && !isnothing(result.opts.hsi) ? Float64.(result.opts.hsi) : fill(0.5, S))
    d_val = hasproperty(result, :d_val) ? result.d_val : (hasproperty(result, :parameters) && hasproperty(result.parameters, :diffusion_mean) ? result.parameters.diffusion_mean : 0.5)
    v_val = hasproperty(result, :v_val) ? result.v_val : (hasproperty(result, :parameters) && hasproperty(result.parameters, :velocity_mean) ? result.parameters.velocity_mean : 1.0)

    # 1. Stationary Residence Distribution pi
    pi_vec = fill(1.0 / S, S)
    for _ in 1:250
        pi_next = vec(pi_vec' * Gamma)
        s_sum = sum(pi_next)
        s_sum > 1e-12 && (pi_next ./= s_sum)
        pi_vec = pi_next
    end

    # 2. Regional Connectivity Matrix
    cents_x = [c[1] for c in cents]
    mid_x = (minimum(cents_x) + maximum(cents_x)) / 2.0
    strata_vec = !isnothing(strata) ? strata : [x > mid_x ? "East" : "West" for x in cents_x]
    C_reg = calculate_regional_connectivity(Gamma, strata_vec)
    strata_names = unique(strata_vec)

    # 3. Dispersal Decay Pairs & Binned Curve
    distances = Float64[]
    probs = Float64[]
    for i in 1:S
        ci = cents[i]
        for j in 1:S
            i == j && continue
            cj = cents[j]
            d = sqrt((cj[1] - ci[1])^2 + (cj[2] - ci[2])^2)
            push!(distances, d)
            push!(probs, Gamma[i, j])
        end
    end
    n_bins = 15
    d_min, d_max = !isempty(distances) ? (minimum(distances), maximum(distances)) : (0.0, 10.0)
    bin_edges = range(d_min, d_max, length=n_bins+1)
    bin_mids = Float64[]
    bin_means = Float64[]
    for b in 1:n_bins
        idx = findall(d -> bin_edges[b] <= d < bin_edges[b+1], distances)
        if !isempty(idx)
            push!(bin_mids, (bin_edges[b] + bin_edges[b+1]) / 2.0)
            push!(bin_means, mean(probs[idx]))
        end
    end

    # 4. Step Diagnostics
    n_indiv = isa(paths, AbstractMatrix) ? size(paths, 1) : length(paths)
    step_lengths = Float64[]
    turning_angles = Float64[]
    if isa(paths, AbstractMatrix)
        n_steps = size(paths, 2)
        for i in 1:n_indiv
            xs = [cents[paths[i, t]][1] for t in 1:n_steps if 1 <= paths[i, t] <= length(cents)]
            ys = [cents[paths[i, t]][2] for t in 1:n_steps if 1 <= paths[i, t] <= length(cents)]
            for t in 2:length(xs)
                d = sqrt((xs[t] - xs[t-1])^2 + (ys[t] - ys[t-1])^2)
                push!(step_lengths, d)
                if t >= 3
                    dx1, dy1 = xs[t-1] - xs[t-2], ys[t-1] - ys[t-2]
                    dx2, dy2 = xs[t] - xs[t-1], ys[t] - ys[t-1]
                    a1 = atan(dy1, dx1)
                    a2 = atan(dy2, dx2)
                    d_ang = atan(sin(a2 - a1), cos(a2 - a1))
                    push!(turning_angles, d_ang)
                end
            end
        end
    elseif isa(paths, AbstractVector)
        for p in paths
            coords = if p isa NamedTuple && hasproperty(p, :coords)
                p.coords
            elseif p isa AbstractVector
                p
            else
                nothing
            end
            if !isnothing(coords) && length(coords) >= 2
                for t in 2:length(coords)
                    pt1 = coords[t-1]
                    pt2 = coords[t]
                    push!(step_lengths, sqrt((pt2[1] - pt1[1])^2 + (pt2[2] - pt1[2])^2))
                    if t >= 3
                        pt0 = coords[t-2]
                        dx1, dy1 = pt1[1] - pt0[1], pt1[2] - pt0[2]
                        dx2, dy2 = pt2[1] - pt1[1], pt2[2] - pt1[2]
                        a1 = atan(dy1, dx1)
                        a2 = atan(dy2, dx2)
                        push!(turning_angles, atan(sin(a2 - a1), cos(a2 - a1)))
                    end
                end
            end
        end
    end

    # 5. Advection-to-Diffusion Ratios
    vel_x = zeros(Float64, S)
    vel_y = zeros(Float64, S)
    if hasproperty(au, :W) && !isnothing(au.W)
        W = au.W
        rows = rowvals(W)
        vals = nonzeros(W)
        for i in 1:S
            ci = cents[i]
            for j_idx in nzrange(W, i)
                j = rows[j_idx]
                i == j && continue
                cj = au.centroids[j]
                dh = hsi_vec[j] - hsi_vec[i]
                if dh > 0.0
                    dx, dy = cj[1] - ci[1], cj[2] - ci[2]
                    dist = sqrt(dx^2 + dy^2) + 1e-9
                    w = vals[j_idx] * exp(dh)
                    vel_x[i] += w * (dx / dist)
                    vel_y[i] += w * (dy / dist)
                end
            end
        end
        vel_x .*= Float64(v_val)
        vel_y .*= Float64(v_val)
    end
    ad_speeds = sqrt.(vel_x.^2 .+ vel_y.^2)
    ad_ratios = ad_speeds ./ (d_val + 1e-6)

    # Construct GeoJSON Collections for Base Map
    tracks_map = leaflet_tracks_map(paths, au; title=title, wkt=wkt_str, is_geo=is_geo, dark_mode=dark_mode, height="580px")

    # Extra Layout HTML for Dashboard
    dashboard_css = """
    .bstm-dashboard-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 1024px) {
      .bstm-dashboard-grid {
        grid-template-columns: 1fr;
      }
    }
    .bstm-card {
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .bstm-card-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .bstm-kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    @media (max-width: 768px) {
      .bstm-kpi-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }
    .bstm-kpi-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-main);
      border-radius: 10px;
      padding: 12px 14px;
      text-align: left;
    }
    .bstm-kpi-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .bstm-kpi-val {
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--accent);
      font-family: var(--font-mono);
      margin-top: 2px;
    }
    .bstm-chart-wrapper {
      position: relative;
      width: 100%;
      height: 240px;
    }
    """

    mean_step = !isempty(step_lengths) ? mean(step_lengths) : 0.0
    mean_speed = mean(ad_speeds)

    v_val_str = @sprintf("%.3f", v_val)
    d_val_str = @sprintf("%.3f", d_val)
    mean_step_str = @sprintf("%.2f", mean_step)

    dashboard_html = """
    <!-- KPI Summary Row -->
    <div class="bstm-kpi-grid">
      <div class="bstm-kpi-card">
        <div class="bstm-kpi-label">Drift Velocity (v)</div>
        <div class="bstm-kpi-val">$(v_val_str)</div>
      </div>
      <div class="bstm-kpi-card">
        <div class="bstm-kpi-label">Diffusion Rate (D)</div>
        <div class="bstm-kpi-val">$(d_val_str)</div>
      </div>
      <div class="bstm-kpi-card">
        <div class="bstm-kpi-label">Mean Step Length</div>
        <div class="bstm-kpi-val">$(mean_step_str)</div>
      </div>
      <div class="bstm-kpi-card">
        <div class="bstm-kpi-label">Active Individuals</div>
        <div class="bstm-kpi-val">$(n_indiv)</div>
      </div>
    </div>

    <!-- Analytics Charts Grid -->
    <div class="bstm-dashboard-grid">
      <!-- Chart 1: Dispersal Distance Decay -->
      <div class="bstm-card">
        <div class="bstm-card-title">
          <span>(A) Dispersal Kernel Distance Decay</span>
          <span class="bstm-badge">Decay Curve</span>
        </div>
        <div class="bstm-chart-wrapper">
          <canvas id="decayChart"></canvas>
        </div>
      </div>

      <!-- Chart 2: Step Diagnostics -->
      <div class="bstm-card">
        <div class="bstm-card-title">
          <span>(B) Empirical Step Length Distribution</span>
          <span class="bstm-badge">Displacements</span>
        </div>
        <div class="bstm-chart-wrapper">
          <canvas id="stepChart"></canvas>
        </div>
      </div>

      <!-- Chart 3: Regional Connectivity Matrix -->
      <div class="bstm-card">
        <div class="bstm-card-title">
          <span>(C) Macro-Regional Connectivity Matrix</span>
          <span class="bstm-badge">Transfer Probabilities</span>
        </div>
        <div class="bstm-chart-wrapper">
          <canvas id="connChart"></canvas>
        </div>
      </div>

      <!-- Chart 4: Advection-to-Diffusion Ratio -->
      <div class="bstm-card">
        <div class="bstm-card-title">
          <span>(D) Advection-to-Diffusion (Péclet) Distribution</span>
          <span class="bstm-badge">v / D</span>
        </div>
        <div class="bstm-chart-wrapper">
          <canvas id="adRatioChart"></canvas>
        </div>
      </div>
    </div>
    """

    # Build Chart.js Setup Scripts
    decay_x_js = join([@sprintf("%.2f", x) for x in bin_mids], ", ")
    decay_y_js = join([@sprintf("%.4f", y) for y in bin_means], ", ")

    step_hist_bins = 15
    s_min, s_max = !isempty(step_lengths) ? (minimum(step_lengths), maximum(step_lengths)) : (0.0, 10.0)
    s_edges = range(s_min, s_max, length=step_hist_bins+1)
    s_counts = [count(x -> s_edges[b] <= x < s_edges[b+1], step_lengths) for b in 1:step_hist_bins]
    step_mids = [(@sprintf("%.1f", (s_edges[b]+s_edges[b+1])/2.0)) for b in 1:step_hist_bins]
    step_x_js = join(["\"" * m * "\"" for m in step_mids], ", ")
    step_y_js = join(s_counts, ", ")

    # Regional Connectivity matrix
    conn_labels = join(["\"$(name)\"" for name in strata_names], ", ")
    reg_entries = String[]
    for r in 1:length(strata_names)
        for c in 1:length(strata_names)
            val = C_reg[r, c]
            from_n = strata_names[r]
            to_n = strata_names[c]
            p_s = @sprintf("%.3f", val)
            push!(reg_entries, "{ from: \"$(from_n)\", to: \"$(to_n)\", prob: $(p_s) }")
        end
    end
    conn_cells_js = "var connCells = [" * join(reg_entries, ", ") * "];"

    # Advection ratios
    ad_bins = 15
    a_max = !isempty(ad_ratios) ? maximum(ad_ratios) : 5.0
    a_edges = range(0.0, max(a_max, 2.0), length=ad_bins+1)
    a_counts = [count(x -> a_edges[b] <= x < a_edges[b+1], ad_ratios) for b in 1:ad_bins]
    ad_mids = [(@sprintf("%.2f", (a_edges[b]+a_edges[b+1])/2.0)) for b in 1:ad_bins]
    ad_x_js = join(["\"" * m * "\"" for m in ad_mids], ", ")
    ad_y_js = join(a_counts, ", ")

    chart_init_js = """
    // Chart 1: Dispersal Decay
    new Chart(document.getElementById('decayChart'), {
      type: 'line',
      data: {
        labels: [$(decay_x_js)],
        datasets: [{
          label: 'Transition Prob (Γ_ij)',
          data: [$(decay_y_js)],
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.15)',
          fill: true,
          tension: 0.35,
          pointRadius: 4,
          pointBackgroundColor: '#38bdf8'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'Distance (units)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'P(Transition)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });

    // Chart 2: Step Lengths
    new Chart(document.getElementById('stepChart'), {
      type: 'bar',
      data: {
        labels: [$(step_x_js)],
        datasets: [{
          label: 'Frequency',
          data: [$(step_y_js)],
          backgroundColor: '#818cf8',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'Step Displacement', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'Count', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });

    // Chart 3: Regional Connectivity
    var regData = [];
    var rNames = [$(conn_labels)];
    $(conn_cells_js)
    new Chart(document.getElementById('connChart'), {
      type: 'bar',
      data: {
        labels: connCells.map(c => c.from + ' ➔ ' + c.to),
        datasets: [{
          label: 'Transfer Probability',
          data: connCells.map(c => c.prob),
          backgroundColor: '#34d399',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'Region Transfer Path', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { min: 0, max: 1, title: { display: true, text: 'Probability', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });

    // Chart 4: A/D Ratios
    new Chart(document.getElementById('adRatioChart'), {
      type: 'bar',
      data: {
        labels: [$(ad_x_js)],
        datasets: [{
          label: 'Spatial Units',
          data: [$(ad_y_js)],
          backgroundColor: '#fb7185',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'Ratio (v / D)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'Units', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });
    """

    map_html = tracks_map.html
    # Insert dashboard layout before </div>\n  <!-- Leaflet JS -->
    split_tag = "</div>\n\n  <!-- Leaflet JS -->"
    if !contains(map_html, split_tag)
        split_tag = "<!-- Leaflet JS -->"
    end
    map_html = replace(map_html, split_tag => dashboard_html * "\n  </div>\n\n  <!-- Leaflet JS -->")
    # Append chart JS at end of script
    map_html = replace(map_html, "</script>\n</body>" => chart_init_js * "\n</script>\n</body>")

    return LeafletMap(map_html, title=title, width=width, height="600px",
                      metadata=Dict(:n_indiv=>n_indiv, :v=>v_val, :d=>d_val))
end


# =============================================================================
# Section 9: Individual Summary Statistics Chart Generators
# =============================================================================

"""
    leaflet_dispersal_kernel(Gamma, au; title="Dispersal Kernel Distance Decay",
                             dark_mode=true, width="100%", height="400px", kwargs...)

Generates a standalone interactive HTML chart showing isotropic transition probability
decay against pairwise spatial distances with empirical binned means.
"""
function leaflet_dispersal_kernel(
    Gamma::AbstractMatrix{<:Real},
    au::NamedTuple;
    title::String = "Dispersal Kernel Distance Decay",
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "400px",
    kwargs...
)::LeafletMap
    cents = au.centroids
    S = size(Gamma, 1)
    distances = Float64[]
    probs = Float64[]
    for i in 1:S
        ci = cents[i]
        for j in 1:S
            i == j && continue
            cj = cents[j]
            d = sqrt((cj[1] - ci[1])^2 + (cj[2] - ci[2])^2)
            push!(distances, d)
            push!(probs, Gamma[i, j])
        end
    end
    n_bins = 15
    d_min, d_max = !isempty(distances) ? (minimum(distances), maximum(distances)) : (0.0, 10.0)
    bin_edges = range(d_min, d_max, length=n_bins+1)
    bin_mids, bin_means = Float64[], Float64[]
    for b in 1:n_bins
        idx = findall(d -> bin_edges[b] <= d < bin_edges[b+1], distances)
        if !isempty(idx)
            push!(bin_mids, (bin_edges[b] + bin_edges[b+1]) / 2.0)
            push!(bin_means, mean(probs[idx]))
        end
    end

    dx_str = join([@sprintf("%.2f", x) for x in bin_mids], ", ")
    dy_str = join([@sprintf("%.4f", y) for y in bin_means], ", ")

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>\$(title)</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { background: #0f172a; color: #f8fafc; font-family: 'Outfit', sans-serif; padding: 20px; }
    .card { background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; max-width: 900px; margin: auto; }
    h2 { font-size: 1.15rem; margin-bottom: 12px; color: #38bdf8; }
  </style>
</head>
<body>
  <div class="card">
    <h2>\$(title)</h2>
    <div style="height: 320px;"><canvas id="chart"></canvas></div>
  </div>
  <script>
    new Chart(document.getElementById('chart'), {
      type: 'line',
      data: {
        labels: [\$dx_str],
        datasets: [{ label: 'Transition Probability', data: [\$dy_str], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.2)', fill: true, tension: 0.3 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Distance (units)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'P(Transition)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });
  </script>
</body>
</html>"""
    return LeafletMap(html, title=title, width=width, height=height, metadata=Dict(:n_pairs=>length(distances)))
end

"""
    leaflet_step_diagnostics(paths, au; title="Step Length & Turning Angle Diagnostics", kwargs...)

Generates interactive step displacement and turning angle distribution charts.
"""
function leaflet_step_diagnostics(
    paths,
    au::NamedTuple;
    title::String = "Step Length & Turning Angle Diagnostics",
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "400px",
    kwargs...
)::LeafletMap
    cents = au.centroids
    step_lengths = Float64[]
    turning_angles = Float64[]

    if isa(paths, AbstractMatrix)
        n_indiv, n_steps = size(paths)
        for i in 1:n_indiv
            xs = [cents[paths[i, t]][1] for t in 1:n_steps if 1 <= paths[i, t] <= length(cents)]
            ys = [cents[paths[i, t]][2] for t in 1:n_steps if 1 <= paths[i, t] <= length(cents)]
            for t in 2:length(xs)
                push!(step_lengths, sqrt((xs[t] - xs[t-1])^2 + (ys[t] - ys[t-1])^2))
                if t >= 3
                    dx1, dy1 = xs[t-1] - xs[t-2], ys[t-1] - ys[t-2]
                    dx2, dy2 = xs[t] - xs[t-1], ys[t] - ys[t-1]
                    a1 = atan(dy1, dx1)
                    a2 = atan(dy2, dx2)
                    push!(turning_angles, atan(sin(a2 - a1), cos(a2 - a1)))
                end
            end
        end
    elseif isa(paths, AbstractVector)
        for p in paths
            coords = if p isa NamedTuple && hasproperty(p, :coords)
                p.coords
            elseif p isa AbstractVector
                p
            else
                nothing
            end
            if !isnothing(coords) && length(coords) >= 2
                for t in 2:length(coords)
                    pt1 = coords[t-1]
                    pt2 = coords[t]
                    push!(step_lengths, sqrt((pt2[1] - pt1[1])^2 + (pt2[2] - pt1[2])^2))
                    if t >= 3
                        pt0 = coords[t-2]
                        dx1, dy1 = pt1[1] - pt0[1], pt1[2] - pt0[2]
                        dx2, dy2 = pt2[1] - pt1[1], pt2[2] - pt1[2]
                        a1 = atan(dy1, dx1)
                        a2 = atan(dy2, dx2)
                        push!(turning_angles, atan(sin(a2 - a1), cos(a2 - a1)))
                    end
                end
            end
        end
    end

    s_bins = 15
    s_min, s_max = !isempty(step_lengths) ? (minimum(step_lengths), maximum(step_lengths)) : (0.0, 10.0)
    s_edges = range(s_min, s_max, length=s_bins+1)
    s_counts = [count(x -> s_edges[b] <= x < s_edges[b+1], step_lengths) for b in 1:s_bins]
    s_mids = [(@sprintf("%.1f", (s_edges[b]+s_edges[b+1])/2.0)) for b in 1:s_bins]
    sx_str = join(["\"" * m * "\"" for m in s_mids], ", ")
    sy_str = join(s_counts, ", ")

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>$(title)</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { background: #0f172a; color: #f8fafc; font-family: 'Outfit', sans-serif; padding: 20px; }
    .card { background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; max-width: 900px; margin: auto; }
    h2 { font-size: 1.15rem; margin-bottom: 12px; color: #818cf8; }
  </style>
</head>
<body>
  <div class="card">
    <h2>$(title)</h2>
    <div style="height: 320px;"><canvas id="chart"></canvas></div>
  </div>
  <script>
    new Chart(document.getElementById('chart'), {
      type: 'bar',
      data: {
        labels: [$(sx_str)],
        datasets: [{ label: 'Frequency', data: [$(sy_str)], backgroundColor: '#818cf8', borderRadius: 4 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Step Displacement', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'Count', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });
  </script>
</body>
</html>"""
    return LeafletMap(html, title=title, width=width, height=height, metadata=Dict(:n_steps=>length(step_lengths)))
end

"""
    leaflet_regional_connectivity(C_regional; strata_names=nothing, title="Macro-Regional Connectivity Matrix", kwargs...)

Renders an interactive macro-regional transfer probability matrix chart.
"""
function leaflet_regional_connectivity(
    C_regional::AbstractMatrix{<:Real};
    strata_names::Union{Nothing, Vector{String}} = nothing,
    title::String = "Macro-Regional Connectivity Matrix",
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "400px",
    kwargs...
)::LeafletMap
    R = size(C_regional, 1)
    names = !isnothing(strata_names) && length(strata_names) == R ? strata_names : (R == 2 ? ["West", "East"] : ["Region $i" for i in 1:R])
    reg_entries = String[]
    for r in 1:R
        for s in 1:R
            p_val = @sprintf("%.3f", C_regional[r, s])
            push!(reg_entries, "{ from: \"$(names[r])\", to: \"$(names[s])\", prob: $(p_val) }")
        end
    end
    cells_js = "[" * join(reg_entries, ", ") * "]"

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>$(title)</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { background: #0f172a; color: #f8fafc; font-family: 'Outfit', sans-serif; padding: 20px; }
    .card { background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; max-width: 900px; margin: auto; }
    h2 { font-size: 1.15rem; margin-bottom: 12px; color: #34d399; }
  </style>
</head>
<body>
  <div class="card">
    <h2>$(title)</h2>
    <div style="height: 320px;"><canvas id="chart"></canvas></div>
  </div>
  <script>
    var cells = $(cells_js);
    new Chart(document.getElementById('chart'), {
      type: 'bar',
      data: {
        labels: cells.map(c => c.from + ' ➔ ' + c.to),
        datasets: [{ label: 'Transfer Prob', data: cells.map(c => c.prob), backgroundColor: '#34d399', borderRadius: 4 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Region Path', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { min: 0, max: 1, title: { display: true, text: 'Probability', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });
  </script>
</body>
</html>"""
    return LeafletMap(html, title=title, width=width, height=height, metadata=Dict(:n_regions=>R))
end

"""
    leaflet_ad_ratio_distribution(advection_field, diffusion_field; title="Advection-to-Diffusion Ratio Distribution", kwargs...)

Renders an interactive histogram of local Advection-to-Diffusion (Péclet-like) ratios.
"""
function leaflet_ad_ratio_distribution(
    advection_field::AbstractVector{<:Real},
    diffusion_field::AbstractVector{<:Real};
    title::String = "Advection-to-Diffusion Ratio Distribution",
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "400px",
    kwargs...
)::LeafletMap
    ratios = Float64.(advection_field) ./ (mean(diffusion_field) + 1e-6)
    n_bins = 20
    max_r = !isempty(ratios) ? maximum(ratios) : 2.0
    edges = range(0.0, max(max_r, 2.0), length=n_bins+1)
    counts = [count(x -> edges[b] <= x < edges[b+1], ratios) for b in 1:n_bins]
    r_mids = [(@sprintf("%.2f", (edges[b]+edges[b+1])/2.0)) for b in 1:n_bins]
    rx_str = join(["\"" * m * "\"" for m in r_mids], ", ")
    ry_str = join(counts, ", ")

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>$(title)</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { background: #0f172a; color: #f8fafc; font-family: 'Outfit', sans-serif; padding: 20px; }
    .card { background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; max-width: 900px; margin: auto; }
    h2 { font-size: 1.15rem; margin-bottom: 12px; color: #fb7185; }
  </style>
</head>
<body>
  <div class="card">
    <h2>$(title)</h2>
    <div style="height: 320px;"><canvas id="chart"></canvas></div>
  </div>
  <script>
    new Chart(document.getElementById('chart'), {
      type: 'bar',
      data: {
        labels: [\$rx_str],
        datasets: [{ label: 'Spatial Units', data: [\$ry_str], backgroundColor: '#fb7185', borderRadius: 4 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Ratio (v / D)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
          y: { title: { display: true, text: 'Count', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
        }
      }
    });
  </script>
</body>
</html>"""
    return LeafletMap(html, title=title, width=width, height=height, metadata=Dict(:n_units=>length(ratios)))
end


# =============================================================================
# Section: Multi-Depth Hexagonal Hydrodynamic Leaflet Dashboard
# =============================================================================

"""
    leaflet_hydrodynamic_dashboard(
        au::NamedTuple,
        hydro_data::NamedTuple;
        trajectories::Union{Nothing, NamedTuple, AbstractVector} = nothing,
        title::AbstractString = "Hydrodynamic Ocean Circulation & Stratification Dashboard",
        dark_mode::Bool = true,
        width::String = "100%",
        height::String = "750px",
        wkt = nothing,
        is_geo::Union{Nothing, Bool} = nothing,
        quiver_scale::Real = 1.0
    ) -> LeafletMap

Constructs an interactive, publication-grade standalone Leaflet HTML dashboard rendering
multi-depth ocean hydrodynamic model outputs (temperature, salinity stratification,
turbulent diffusion, and advection current velocity vectors) over a spatial mesh
(e.g., fine-resolution hexagonal lattice).

# Features
- **Depth Selector**: Live switching between discrete vertical water-column levels
  (Surface, Subsurface, Cold Intermediate Layer, and Benthic Shelf).
- **Interactive Layer Toggles**:
  - Seawater Potential Temperature (``T``, °C)
  - Practical Salinity (``S``, PSU)
  - Brunt-Väisälä Stratification Frequency Squared (``N^2``, ``\\text{s}^{-2}``)
  - Turbulent Vertical Eddy Diffusivity (``\\kappa_v``, ``\\text{m}^2/\\text{s}``)
  - Current Speed Magnitude (``|\\boldsymbol{u}_h|``, cm/s)
  - Seafloor Bathymetric Depth (``H``, m)
- **Advection Velocity Quivers**: Directional arrowheads originating from cell centroids
  scaled by horizontal flow speed.
- **Dynamic HUD Tooltip**: Real-time display of localized physical diagnostics on mouse hover.
- **Trajectory Integration**: Optional rendering of Lagrangian particle tracks or telemetry paths.

# Arguments
- `au`: Spatial tessellation NamedTuple containing `:centroids` and `:polygons`
  (or `:polygons_lonlat` / `:centroids_lonlat`).
- `hydro_data`: NamedTuple containing multi-depth hydrodynamic variables (`:depths`,
  `:temperature`, `:salinity`, `:stratification_N2`, `:diffusivity_v`, `:u`, `:v`).
- `trajectories`: Optional trajectory records or path sequences.
- `title`: Header title of the interactive map.
- `dark_mode`: Whether to style with dark theme basemaps (default: `true`).
"""
function leaflet_hydrodynamic_dashboard(
    arg1::NamedTuple,
    arg2::NamedTuple;
    trajectories::Union{Nothing, NamedTuple, AbstractVector} = nothing,
    title::AbstractString = "Hydrodynamic Ocean Circulation & Stratification Dashboard",
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "750px",
    wkt = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    quiver_scale::Real = 1.0
)::LeafletMap
    # Auto-detect argument order: either (au, hydro_data) or (hydro_data, au)
    au, hydro_data = if (hasproperty(arg1, :temperature) || hasproperty(arg1, :stratification_N2) || hasproperty(arg1, :N2)) && !(hasproperty(arg1, :W) || hasproperty(arg1, :polygons_lonlat))
        arg2, arg1
    else
        arg1, arg2
    end
    # 1. Resolve geometry coordinates (prefer geographic lon/lat)
    polys = if hasproperty(au, :polygons_lonlat) && !isempty(au.polygons_lonlat)
        au.polygons_lonlat
    elseif hasproperty(au, :polygons)
        au.polygons
    else
        error("Spatial partition `au` must contain `:polygons` or `:polygons_lonlat`.")
    end

    cents = if hasproperty(au, :centroids_lonlat) && !isempty(au.centroids_lonlat)
        au.centroids_lonlat
    elseif hasproperty(au, :centroids)
        au.centroids
    else
        error("Spatial partition `au` must contain `:centroids` or `:centroids_lonlat`.")
    end

    S = length(cents)
    wkt_str = !isnothing(wkt) ? string(wkt) : _extract_wkt(au)
    tf = _build_coordinate_transformer(cents; wkt=wkt_str, is_geo=is_geo, lon_center=-160.0, lat_center=0.0)

    # 2. Extract multi-depth hydrodynamic variables
    depth_levels = hasproperty(hydro_data, :depths) ? Float64.(hydro_data.depths) : [-2.5, -25.0, -50.0, -100.0]
    nz = length(depth_levels)

    to_2d(field, def_val) = begin
        if field === nothing
            return fill(def_val, S, nz)
        elseif ndims(field) == 1
            if length(field) == S
                return repeat(Float64.(field), 1, nz)
            else
                return fill(def_val, S, nz)
            end
        elseif ndims(field) == 2
            if size(field, 1) == S
                f_cols = size(field, 2)
                if f_cols == nz
                    return Float64.(field)
                else
                    res = fill(def_val, S, nz)
                    for k in 1:min(nz, f_cols)
                        res[:, k] .= Float64.(field[:, k])
                    end
                    return res
                end
            else
                return fill(def_val, S, nz)
            end
        else
            return fill(def_val, S, nz)
        end
    end

    T_2d = to_2d(hasproperty(hydro_data, :temperature) ? hydro_data.temperature : nothing, 5.0)
    S_2d = to_2d(hasproperty(hydro_data, :salinity) ? hydro_data.salinity : nothing, 32.5)

    N2_field = hasproperty(hydro_data, :stratification_N2) ? hydro_data.stratification_N2 :
               (hasproperty(hydro_data, :N2) ? hydro_data.N2 :
               (hasproperty(hydro_data, :stratification) ? hydro_data.stratification : nothing))
    N2_2d = to_2d(N2_field, 1e-4)

    diff_field = hasproperty(hydro_data, :diffusivity_v) ? hydro_data.diffusivity_v :
                 (hasproperty(hydro_data, :kappa_v) ? hydro_data.kappa_v :
                 (hasproperty(hydro_data, :diffusivity) ? hydro_data.diffusivity : nothing))
    diff_2d = to_2d(diff_field, 1e-3)

    u_field = hasproperty(hydro_data, :u) ? hydro_data.u :
              (hasproperty(hydro_data, :advection_u) ? hydro_data.advection_u : nothing)
    v_field = hasproperty(hydro_data, :v) ? hydro_data.v :
              (hasproperty(hydro_data, :advection_v) ? hydro_data.advection_v : nothing)
    u_2d = to_2d(u_field, 0.0)
    v_2d = to_2d(v_field, 0.0)
    spd_2d = hypot.(u_2d, v_2d) .* 100.0 # convert m/s to cm/s

    bathy_vec = if hasproperty(hydro_data, :depths_vec) && length(hydro_data.depths_vec) == S
        Float64.(hydro_data.depths_vec)
    elseif hasproperty(hydro_data, :depth_vec) && length(hydro_data.depth_vec) == S
        Float64.(hydro_data.depth_vec)
    elseif hasproperty(hydro_data, :depths) && length(hydro_data.depths) == S
        Float64.(hydro_data.depths)
    elseif hasproperty(hydro_data, :bathymetry) && length(hydro_data.bathymetry) == S
        Float64.(hydro_data.bathymetry)
    elseif hasproperty(au, :depth_vec) && length(au.depth_vec) == S
        Float64.(au.depth_vec)
    elseif hasproperty(au, :depths) && length(au.depths) == S
        Float64.(au.depths)
    else
        fill(120.0, S)
    end

    # 3. Build GeoJSON Polygons with multi-depth attributes
    features_json = String[]
    all_lats = Float64[]
    all_lngs = Float64[]

    for i in 1:S
        p = polys[i]
        length(p) < 3 && continue

        p_trans = _transform_polygon(tf, p)
        length(p_trans) < 3 && continue

        coords_str = String[]
        for pt in p_trans
            push!(coords_str, "[$(pt[1]), $(pt[2])]")
            push!(all_lngs, pt[1])
            push!(all_lats, pt[2])
        end
        if coords_str[1] != coords_str[end]
            push!(coords_str, coords_str[1])
        end

        # Multi-depth attributes for cell i
        t_vals = [isnan(T_2d[i, k]) ? "null" : @sprintf("%.2f", T_2d[i, k]) for k in 1:nz]
        s_vals = [isnan(S_2d[i, k]) ? "null" : @sprintf("%.2f", S_2d[i, k]) for k in 1:nz]
        n2_vals = [isnan(N2_2d[i, k]) ? "null" : @sprintf("%.2e", N2_2d[i, k]) for k in 1:nz]
        diff_vals = [isnan(diff_2d[i, k]) ? "null" : @sprintf("%.2e", diff_2d[i, k]) for k in 1:nz]
        spd_vals = [isnan(spd_2d[i, k]) ? "null" : @sprintf("%.1f", spd_2d[i, k]) for k in 1:nz]
        u_vals = [isnan(u_2d[i, k]) ? "null" : @sprintf("%.3f", u_2d[i, k]) for k in 1:nz]
        v_vals = [isnan(v_2d[i, k]) ? "null" : @sprintf("%.3f", v_2d[i, k]) for k in 1:nz]
        b_val = isnan(bathy_vec[i]) ? "null" : @sprintf("%.1f", abs(bathy_vec[i]))

        feat = """{
          "type": "Feature",
          "id": $i,
          "properties": {
            "unit_id": $i,
            "bathy": $b_val,
            "t": [$(join(t_vals, ", "))],
            "s": [$(join(s_vals, ", "))],
            "n2": [$(join(n2_vals, ", "))],
            "diff": [$(join(diff_vals, ", "))],
            "spd": [$(join(spd_vals, ", "))],
            "u": [$(join(u_vals, ", "))],
            "v": [$(join(v_vals, ", "))]
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [[$(join(coords_str, ", "))]]
          }
        }"""
        push!(features_json, feat)
    end

    geojson_cells = "{\"type\": \"FeatureCollection\", \"features\": [$(join(features_json, ",\n"))]}"

    # 4. Build Multi-Depth Velocity Quiver Features
    quivers_per_depth = String[]
    for k in 1:nz
        arrows_k = String[]
        for i in 1:S
            u_val = u_2d[i, k]
            v_val = v_2d[i, k]
            spd = spd_2d[i, k]
            if !isnan(u_val) && !isnan(v_val) && spd > 0.5
                c_trans = _transform_point(tf, cents[i])
                # Scale arrow tip (quiver_scale)
                arrow_len = min(0.08, 0.003 * spd * quiver_scale)
                angle = atan(v_val, u_val)
                tip_x = c_trans[1] + arrow_len * cos(angle)
                tip_y = c_trans[2] + arrow_len * sin(angle)

                arrow_feat = """{
                  "type": "Feature",
                  "properties": { "speed": $(round(spd, digits=1)), "dir": $(round(rad2deg(angle), digits=1)) },
                  "geometry": {
                    "type": "LineString",
                    "coordinates": [[$(c_trans[1]), $(c_trans[2])], [$(tip_x), $(tip_y)]]
                  }
                }"""
                push!(arrows_k, arrow_feat)
            end
        end
        push!(quivers_per_depth, "{\"type\": \"FeatureCollection\", \"features\": [$(join(arrows_k, ",\n"))]}")
    end
    quivers_json_array = "[" * join(quivers_per_depth, ", ") * "]"

    # 5. Trajectory Paths Overlay if provided
    tracks_json_str = "null"
    if trajectories !== nothing
        t_feats = String[]
        p_list = if trajectories isa NamedTuple && hasproperty(trajectories, :lon) && hasproperty(trajectories, :lat)
            # Flattened coordinate arrays
            [[(trajectories.lon[i], trajectories.lat[i]) for i in 1:length(trajectories.lon)]]
        elseif trajectories isa AbstractVector
            trajectories
        else
            []
        end
        for (tr_idx, track) in enumerate(p_list)
            if track isa AbstractVector && length(track) >= 2
                coords_t = [_transform_point(tf, pt) for pt in track if !isnan(pt[1]) && !isnan(pt[2])]
                if length(coords_t) >= 2
                    c_str = join(["[$(pt[1]), $(pt[2])]" for pt in coords_t], ", ")
                    push!(t_feats, """{
                      "type": "Feature",
                      "properties": { "track_id": $tr_idx },
                      "geometry": { "type": "LineString", "coordinates": [$c_str] }
                    }""")
                end
            end
        end
        if !isempty(t_feats)
            tracks_json_str = "{\"type\": \"FeatureCollection\", \"features\": [$(join(t_feats, ",\n"))]}"
        end
    end

    # 6. HTML Template with Controls and Palette Interpolator
    map_id = "bstm_hydro_" * string(abs(hash(title * string(rand()))), base=16)
    depth_labels_js = "[" * join(["\"$(abs(round(d, digits=1)))m\"" for d in depth_levels], ", ") * "]"

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>$(title)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Outfit', sans-serif; background: #090d16; color: #f1f5f9; overflow: hidden; }
    #$(map_id) { width: $(width); height: $(height); position: relative; }
    .hydro-panel {
      position: absolute; top: 16px; left: 16px; z-index: 1000;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px;
      padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); width: 330px;
    }
    .hydro-title { font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-bottom: 12px; }
    .hydro-group { margin-bottom: 12px; }
    .hydro-group-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 6px; }
    .btn-group { display: flex; flex-wrap: wrap; gap: 4px; }
    .btn-tab {
      background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08);
      color: #cbd5e1; padding: 5px 10px; border-radius: 6px; font-size: 0.75rem;
      cursor: pointer; transition: all 0.15s ease; font-family: 'Outfit', sans-serif;
    }
    .btn-tab:hover { background: rgba(56, 189, 248, 0.2); color: #fff; border-color: rgba(56, 189, 248, 0.4); }
    .btn-tab.active { background: #0284c7; color: #fff; border-color: #38bdf8; font-weight: 600; }
    .legend-card {
      position: absolute; bottom: 24px; right: 24px; z-index: 1000;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 10px;
      padding: 12px 16px; width: 260px; box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .legend-title { font-size: 0.78rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
    .legend-bar { height: 12px; border-radius: 4px; margin-bottom: 6px; }
    .legend-labels { display: flex; justify-content: space-between; font-size: 0.70rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
    .hud-tooltip {
      position: absolute; top: 16px; right: 16px; z-index: 1000;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 10px;
      padding: 12px 16px; min-width: 240px; pointer-events: none; font-size: 0.75rem;
    }
    .hud-title { font-weight: 700; color: #38bdf8; margin-bottom: 6px; font-size: 0.82rem; }
    .hud-row { display: flex; justify-content: space-between; margin-bottom: 3px; font-family: 'JetBrains Mono', monospace; }
    .hud-label { color: #94a3b8; }
    .hud-val { color: #f8fafc; font-weight: 500; }
  </style>
</head>
<body>
  <div id="$(map_id)">
    <div class="hydro-panel">
      <div class="hydro-title">$(title)</div>
      <div class="hydro-group">
        <div class="hydro-group-label">Depth Slices</div>
        <div class="btn-group" id="depth-tabs"></div>
      </div>
      <div class="hydro-group">
        <div class="hydro-group-label">Diagnostic Layers</div>
        <div class="btn-group" id="layer-tabs">
          <button class="btn-tab active" data-layer="t">Temperature</button>
          <button class="btn-tab" data-layer="s">Salinity</button>
          <button class="btn-tab" data-layer="n2">Stratification (N²)</button>
          <button class="btn-tab" data-layer="diff">Diffusivity (κ)</button>
          <button class="btn-tab" data-layer="spd">Current Speed</button>
          <button class="btn-tab" data-layer="bathy">Bathymetry</button>
        </div>
      </div>
      <div class="hydro-group" style="margin-bottom: 0;">
        <label style="font-size: 0.75rem; color: #cbd5e1; cursor: pointer; display: flex; align-items: center; gap: 6px;">
          <input type="checkbox" id="toggle-quivers" checked> Advection Quiver Arrows
        </label>
      </div>
    </div>

    <div class="hud-tooltip" id="hud-box">
      <div class="hud-title">Hydrodynamic Diagnostics</div>
      <div style="color: #64748b;">Hover over a cell to view localized water-column metrics.</div>
    </div>

    <div class="legend-card">
      <div class="legend-title" id="legend-label">Temperature (°C)</div>
      <div class="legend-bar" id="legend-gradient"></div>
      <div class="legend-labels">
        <span id="leg-min">-</span>
        <span id="leg-mid">-</span>
        <span id="leg-max">-</span>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    var cellData = $(geojson_cells);
    var quiversData = $(quivers_json_array);
    var tracksData = $(tracks_json_str);
    var depthLabels = $(depth_labels_js);
    var nz = $(nz);

    var curDepthIdx = 0;
    var curLayer = 't';
    var showQuivers = true;

    // Palettes
    var palettes = {
      t: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027'],
      s: ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
      n2: ['#0d0887', '#5402a3', '#8b0aa5', '#b93289', '#db5c68', '#f48849', '#febd2a', '#f0f921'],
      diff: ['#000004', '#2c105c', '#711f81', '#b63679', '#ee605e', '#fca068', '#fcdea2'],
      spd: ['#30123b', '#4662d8', '#28bbec', '#35eb93', '#a2fc3c', '#e2d312', '#fa8603', '#d61b03'],
      bathy: ['#081d58', '#253494', '#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', '#c7e9b4', '#edf8b1']
    };

    var units = {
      t: '°C', s: 'PSU', n2: 's⁻²', diff: 'm²/s', spd: 'cm/s', bathy: 'm'
    };
    var layerNames = {
      t: 'Temperature', s: 'Salinity', n2: 'Stratification Frequency (N²)',
      diff: 'Turbulent Diffusivity (κ)', spd: 'Current Speed (|u|)', bathy: 'Seafloor Depth'
    };

    // Calculate dynamic layer bounds across cells
    var layerRanges = {};
    ['t', 's', 'n2', 'diff', 'spd'].forEach(function(lKey) {
      layerRanges[lKey] = [];
      for (var k = 0; k < nz; k++) {
        var vals = [];
        cellData.features.forEach(function(f) {
          var v = f.properties[lKey][k];
          if (v !== null && !isNaN(v)) vals.push(v);
        });
        vals.sort(function(a, b) { return a - b; });
        var minVal = vals.length > 0 ? vals[Math.floor(vals.length * 0.02)] : 0;
        var maxVal = vals.length > 0 ? vals[Math.floor(vals.length * 0.98)] : 1;
        if (minVal === maxVal) maxVal += 0.1;
        layerRanges[lKey].push({ min: minVal, max: maxVal });
      }
    });

    // Bathymetry range
    var bVals = [];
    cellData.features.forEach(function(f) {
      if (f.properties.bathy !== null) bVals.push(f.properties.bathy);
    });
    bVals.sort(function(a,b){return a-b;});
    layerRanges['bathy'] = [{ min: bVals[0] || 0, max: bVals[bVals.length - 1] || 2000 }];

    function getColor(val, min, max, pal) {
      if (val === null || isNaN(val)) return 'rgba(30, 41, 59, 0.4)';
      var t = (val - min) / (max - min);
      t = Math.max(0, Math.min(1, t));
      var idx = t * (pal.length - 1);
      var i = Math.floor(idx);
      var f = idx - i;
      if (i >= pal.length - 1) return pal[pal.length - 1];
      return pal[i];
    }

    var map = L.map('$(map_id)', { attributionControl: false, zoomControl: false });
    L.control.zoom({ position: 'topright' }).addTo(map);

    var baseTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 13
    }).addTo(map);

    var cellLayer = L.geoJSON(cellData, {
      style: function(f) {
        var rng = (curLayer === 'bathy') ? layerRanges['bathy'][0] : layerRanges[curLayer][curDepthIdx];
        var val = (curLayer === 'bathy') ? f.properties.bathy : f.properties[curLayer][curDepthIdx];
        var col = getColor(val, rng.min, rng.max, palettes[curLayer]);
        return { fillColor: col, fillOpacity: 0.82, weight: 0.6, color: '#0f172a' };
      },
      onEachFeature: function(f, layer) {
        layer.on({
          mouseover: function(e) {
            e.target.setStyle({ weight: 2, color: '#38bdf8' });
            e.target.bringToFront();
            updateHUD(f.properties);
          },
          mouseout: function(e) {
            cellLayer.resetStyle(e.target);
          }
        });
      }
    }).addTo(map);

    map.fitBounds(cellLayer.getBounds(), { padding: [20, 20] });

    var quiverLayers = [];
    for (var k = 0; k < nz; k++) {
      var ql = L.geoJSON(quiversData[k], {
        style: function(f) {
          return { color: '#ffffff', weight: 1.5, opacity: 0.75 };
        }
      });
      quiverLayers.push(ql);
    }
    if (showQuivers) quiverLayers[curDepthIdx].addTo(map);

    if (tracksData) {
      L.geoJSON(tracksData, {
        style: { color: '#f43f5e', weight: 2.5, dashArray: '4, 4', opacity: 0.9 }
      }).addTo(map);
    }

    function updateHUD(p) {
      var hud = document.getElementById('hud-box');
      var d = depthLabels[curDepthIdx];
      var html = '<div class="hud-title">Hex Unit #' + p.unit_id + ' [' + d + ']</div>';
      html += '<div class="hud-row"><span class="hud-label">Seafloor Depth:</span><span class="hud-val">' + p.bathy + ' m</span></div>';
      html += '<div class="hud-row"><span class="hud-label">Temperature:</span><span class="hud-val">' + p.t[curDepthIdx] + ' °C</span></div>';
      html += '<div class="hud-row"><span class="hud-label">Salinity:</span><span class="hud-val">' + p.s[curDepthIdx] + ' PSU</span></div>';
      html += '<div class="hud-row"><span class="hud-label">Stratification (N²):</span><span class="hud-val">' + p.n2[curDepthIdx] + ' s⁻²</span></div>';
      html += '<div class="hud-row"><span class="hud-label">Diffusivity (κ):</span><span class="hud-val">' + p.diff[curDepthIdx] + ' m²/s</span></div>';
      html += '<div class="hud-row"><span class="hud-label">Velocity (u, v):</span><span class="hud-val">(' + p.u[curDepthIdx] + ', ' + p.v[curDepthIdx] + ') m/s</span></div>';
      html += '<div class="hud-row"><span class="hud-label">Current Speed:</span><span class="hud-val">' + p.spd[curDepthIdx] + ' cm/s</span></div>';
      hud.innerHTML = html;
    }

    function refreshLayers() {
      var rng = (curLayer === 'bathy') ? layerRanges['bathy'][0] : layerRanges[curLayer][curDepthIdx];
      var pal = palettes[curLayer];

      cellLayer.eachLayer(function(l) {
        var f = l.feature;
        var val = (curLayer === 'bathy') ? f.properties.bathy : f.properties[curLayer][curDepthIdx];
        l.setStyle({ fillColor: getColor(val, rng.min, rng.max, pal) });
      });

      for (var k = 0; k < nz; k++) {
        map.removeLayer(quiverLayers[k]);
      }
      if (showQuivers && curLayer !== 'bathy') {
        quiverLayers[curDepthIdx].addTo(map);
      }

      // Update Legend
      document.getElementById('legend-label').innerText = layerNames[curLayer] + ' (' + units[curLayer] + ') [' + depthLabels[curDepthIdx] + ']';
      document.getElementById('legend-gradient').style.background = 'linear-gradient(to right, ' + pal.join(', ') + ')';
      document.getElementById('leg-min').innerText = rng.min.toExponential ? (rng.min < 0.01 ? rng.min.toExponential(2) : rng.min.toFixed(2)) : rng.min;
      document.getElementById('leg-mid').innerText = ((rng.min + rng.max) / 2).toFixed(2);
      document.getElementById('leg-max').innerText = rng.max.toExponential ? (rng.max < 0.01 ? rng.max.toExponential(2) : rng.max.toFixed(2)) : rng.max;
    }

    // Initialize UI Buttons
    var depthContainer = document.getElementById('depth-tabs');
    depthLabels.forEach(function(dLabel, idx) {
      var btn = document.createElement('button');
      btn.className = 'btn-tab' + (idx === 0 ? ' active' : '');
      btn.innerText = dLabel;
      btn.onclick = function() {
        document.querySelectorAll('#depth-tabs .btn-tab').forEach(function(b){b.classList.remove('active');});
        btn.classList.add('active');
        curDepthIdx = idx;
        refreshLayers();
      };
      depthContainer.appendChild(btn);
    });

    document.querySelectorAll('#layer-tabs .btn-tab').forEach(function(btn) {
      btn.onclick = function() {
        document.querySelectorAll('#layer-tabs .btn-tab').forEach(function(b){b.classList.remove('active');});
        btn.classList.add('active');
        curLayer = btn.getAttribute('data-layer');
        refreshLayers();
      };
    });

    document.getElementById('toggle-quivers').onchange = function(e) {
      showQuivers = e.target.checked;
      refreshLayers();
    };

    refreshLayers();
  </script>
</body>
</html>"""

    return LeafletMap(
        html_content,
        title=title,
        width=width,
        height=height,
        metadata=Dict(:n_units=>S, :n_depths=>nz, :depths=>depth_levels)
    )
end


# ==============================================================================
# SECTION: TWO-CLICK DYNAMIC INTERACTIVE CORRIDOR & PATH VISUALIZATION
# ==============================================================================

"""
    leaflet_interactive_corridor_dashboard(
        P::Union{AbstractMatrix{<:Real}, AbstractVector{<:AbstractMatrix{<:Real}}},
        au::NamedTuple;
        hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        empirical_paths = nothing,
        group_labels::Vector{String} = String[],
        title::String = "Interactive Movement Corridor & Path Ensemble Explorer",
        dark_mode::Bool = true,
        width::String = "100%",
        height::String = "750px",
        k_default::Int = 4,
        max_paths_render::Int = 30
    ) -> LeafletMap

Constructs a standalone interactive Leaflet visualization of dynamic movement corridors
and probabilistic path ensembles between user-selected spatial endpoints.

# Interaction Design
- **First Click on Hexagon \$A\$**: Designates Start / Release unit (green pulsing pin).
- **Second Click on Hexagon \$B\$**: Designates Destination / Recapture unit (red pulsing pin).
- **Dynamic Path Ensemble**: Instantly computes and renders all probable paths from \$A\$
  to \$B\$ in \$k\$ steps. More probable paths are rendered with darker, thicker lines
  (high opacity, bold stroke), while less probable paths are drawn with lighter, faint
  lines (low opacity, thin stroke).
- **Interactive Controls**: Features client-side sliders for discrete elapsed steps \$k\$,
  top-path density filters, and biological stratum selectors.

# Mathematical Formulation
For discrete transition kernel \$\\mathbf{P}\$ and endpoints \$A, B\$ over \$k\$ steps,
the posterior probability of a candidate trajectory \$\\pi = (u_0=A, u_1, \\dots, u_k=B)\$ is:
```math
\\mathbb{P}(\\pi \\mid X_0 = A, X_k = B) = \\frac{\\prod_{\\tau=0}^{k-1} P_{u_\\tau, u_{\\tau+1}}}{[\\mathbf{P}^k]_{A, B}}
```
Paths are rendered with stroke opacity and width proportional to relative posterior probability:
```math
w_\\pi = \\frac{\\mathbb{P}(\\pi \\mid A \\to B)}{\\max_{\\pi'} \\mathbb{P}(\\pi' \\mid A \\to B)} \\in (0, 1]
```

# Arguments
- `P`: Row-stochastic transition matrix (or vector of matrices for multiple biological groups).
- `au`: Spatial areal unit NamedTuple containing `:centroids` and `:polygons`.
- `hsi`: Optional spatial habitat suitability index vector of length ``S``.
- `empirical_paths`: Optional collection of observed mark-recapture trajectories.
- `group_labels`: Optional labels matching the group transition kernels.
- `title`: Visualization header title.
- `dark_mode`: Toggle modern dark theme (default `true`).
- `width, height`: CSS dimension specifications.
- `k_default`: Initial transition step duration (default `4`).
- `max_paths_render`: Maximum number of paths to draw simultaneously (default `30`).

# Returns
- `LeafletMap`: Self-contained HTML dashboard with embedded client-side Markov path engine.
"""
function leaflet_interactive_corridor_dashboard(
    P::Union{AbstractMatrix{<:Real}, AbstractVector{<:AbstractMatrix{<:Real}}},
    au::NamedTuple;
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    empirical_paths = nothing,
    group_labels::Vector{String} = String[],
    title::String = "Interactive Movement Corridor & Path Ensemble Explorer",
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "750px",
    k_default::Int = 4,
    max_paths_render::Int = 30
)::LeafletMap

    # Extract centroids and polygons
    cents_raw = hasproperty(au, :centroids_lonlat) ? au.centroids_lonlat :
                (hasproperty(au, :centroids) ? au.centroids : Tuple{Float64, Float64}[])
    polys_raw = hasproperty(au, :polygons_lonlat) ? au.polygons_lonlat :
                (hasproperty(au, :polygons) ? au.polygons : Vector{Vector{Tuple{Float64, Float64}}}())
    S = length(cents_raw)

    # Normalize kernel input into group dictionary
    kernels_dict = Dict{String, Matrix{Float64}}()
    if P isa AbstractVector
        for (idx, mat) in enumerate(P)
            g_lbl = idx <= length(group_labels) ? group_labels[idx] : "Group \$idx"
            kernels_dict[g_lbl] = Matrix{Float64}(mat)
        end
    else
        g_lbl = !isempty(group_labels) ? first(group_labels) : "All"
        kernels_dict[g_lbl] = Matrix{Float64}(P)
    end
    group_names = collect(keys(kernels_dict))

    # Helper to serialize sparse P matrix into compact JSON lookup
    function _matrix_to_sparse_json(mat::Matrix{Float64})
        S_dim = size(mat, 1)
        entries = String[]
        for i in 1:S_dim
            nbrs = String[]
            for j in 1:S_dim
                p_val = mat[i, j]
                if p_val > 1e-5
                    push!(nbrs,
                        string("[", j, ",",
                               round(p_val, digits=5), "]"))
                end
            end
            push!(entries,
                string("\"" , i, "\":[",
                       join(nbrs, ","), "]"))
        end
        return "{" * join(entries, ",") * "}"
    end

    groups_json_parts = String[]
    for g_name in group_names
        p_json = _matrix_to_sparse_json(kernels_dict[g_name])
        push!(groups_json_parts,
            string("\"" , g_name, "\": ", p_json))
    end
    all_kernels_json = "{" * join(groups_json_parts, ",\n") * "}"

    # Centroids JSON: lookup of [lon, lat] per unit ID
    cents_json_parts = String[]
    for (i, c) in enumerate(cents_raw)
        push!(cents_json_parts,
            string("\"" , i, "\": [",
                   round(Float64(c[1]), digits=6), ", ",
                   round(Float64(c[2]), digits=6), "]"))
    end
    cents_json = "{" * join(cents_json_parts, ",") * "}"

    # Background Polygons GeoJSON
    has_hsi = !isnothing(hsi) && length(hsi) == S
    hsi_vals = has_hsi ? Float64.(hsi) : fill(0.5, S)
    pal_hsi = _resolve_palette(:viridis)

    polys_json = String[]
    for (i, poly) in enumerate(polys_raw)
        if length(poly) >= 3
            c_str = ["[$(round(Float64(pt[1]), digits=6)), $(round(Float64(pt[2]), digits=6))]" for pt in poly]
            if c_str[1] != c_str[end]
                push!(c_str, c_str[1])
            end
            h_val = has_hsi ? hsi_vals[i] : NaN
            h_col = has_hsi ? _map_val_to_hex(h_val, 0.0, 1.0, pal_hsi) : "#1e293b"
            h_str = has_hsi ? @sprintf("%.3f", h_val) : "N/A"
            push!(polys_json, """{
              "type": "Feature",
              "id": $i,
              "properties": { "unit_id": $i, "hsi": "$h_str", "fill_color": "$h_col" },
              "geometry": { "type": "Polygon", "coordinates": [[$(join(c_str, ", "))]] }
            }""")
        end
    end
    polys_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(polys_json, ",\n"))]}"

    # Optional Empirical Vectors GeoJSON
    emp_json = String[]
    if !isnothing(empirical_paths)
        for (idx, tr) in enumerate(empirical_paths)
            if length(tr) >= 2
                c_str = join(["[$(round(Float64(pt[1]), digits=6)), $(round(Float64(pt[2]), digits=6))]" for pt in tr], ", ")
                push!(emp_json, """{
                  "type": "Feature",
                  "id": $idx,
                  "properties": { "track_id": $idx },
                  "geometry": { "type": "LineString", "coordinates": [$c_str] }
                }""")
            end
        end
    end
    emp_collection = "{\"type\": \"FeatureCollection\", \"features\": [$(join(emp_json, ",\n"))]}"

    # Compute bounding box center
    all_lons = [Float64(c[1]) for c in cents_raw if !isnan(c[1])]
    all_lats = [Float64(c[2]) for c in cents_raw if !isnan(c[2])]
    mid_lon = !isempty(all_lons) ? (minimum(all_lons) + maximum(all_lons)) / 2.0 : -60.0
    mid_lat = !isempty(all_lats) ? (minimum(all_lats) + maximum(all_lats)) / 2.0 : 45.0

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>$title - BSTM Interactive Dynamic Corridor Explorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
  <style>
    :root {
      --bg-main: #0b1329;
      --panel-bg: rgba(15, 23, 42, 0.88);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-main: rgba(255, 255, 255, 0.12);
      --accent: #38bdf8;
      --start-color: #10b981;
      --end-color: #f43f5e;
      --font-main: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-main);
      background-color: var(--bg-main);
      color: var(--text-main);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .bstm-header {
      background: var(--panel-bg);
      border-bottom: 1px solid var(--border-main);
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 1000;
    }
    .bstm-title {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .bstm-badge {
      font-size: 0.70rem;
      text-transform: uppercase;
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 600;
    }
    .bstm-map-container {
      position: relative;
      flex: 1;
      width: 100%;
    }
    #corridor_map {
      width: 100%;
      height: 100%;
      background: #090e1f;
    }
    .bstm-floating-panel {
      position: absolute;
      top: 16px;
      right: 16px;
      width: 340px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 12px;
      padding: 16px;
      z-index: 1000;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .bstm-panel-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-main);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .bstm-status-box {
      font-size: 0.82rem;
      background: rgba(11, 19, 41, 0.7);
      border: 1px solid var(--border-main);
      border-radius: 8px;
      padding: 10px;
      color: var(--text-muted);
      line-height: 1.4;
    }
    .bstm-endpoints-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .bstm-endpoint-card {
      background: rgba(11, 19, 41, 0.7);
      border: 1px solid var(--border-main);
      border-radius: 8px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
    }
    .bstm-ep-label {
      font-size: 0.70rem;
      text-transform: uppercase;
      font-weight: 600;
    }
    .bstm-ep-val {
      font-size: 1.05rem;
      font-weight: 700;
      font-family: var(--font-mono);
      margin-top: 2px;
    }
    .bstm-control-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .bstm-control-label {
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    .bstm-slider {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .bstm-select {
      background: rgba(11, 19, 41, 0.8);
      border: 1px solid var(--border-main);
      color: var(--text-main);
      padding: 6px 10px;
      border-radius: 6px;
      font-family: var(--font-main);
      font-size: 0.85rem;
    }
    .bstm-btn-group {
      display: flex;
      gap: 8px;
      margin-top: 4px;
    }
    .bstm-btn {
      flex: 1;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: var(--accent);
      padding: 7px 12px;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.18s ease;
    }
    .bstm-btn:hover {
      background: rgba(56, 189, 248, 0.30);
      color: #fff;
    }
    .bstm-btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-main);
      color: var(--text-muted);
    }
    .bstm-btn-secondary:hover {
      background: rgba(255, 255, 255, 0.12);
      color: var(--text-main);
    }
    .bstm-pulse-icon {
      border-radius: 50%;
      box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
      animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
      70% { box-shadow: 0 0 0 14px rgba(56, 189, 248, 0); }
      100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }
  </style>
</head>
<body>
  <header class="bstm-header">
    <div class="bstm-title">
      <span>$title</span>
      <span class="bstm-badge">Two-Click Dynamic Markov Path Explorer</span>
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted);">
      Click any polygon for <b>Release</b>, then a second polygon for <b>Recapture</b>
    </div>
  </header>

  <div class="bstm-map-container">
    <div id="corridor_map"></div>

    <div class="bstm-floating-panel">
      <div class="bstm-panel-title">
        <span>Path Ensemble Controls</span>
        <span id="paths_count_badge" class="bstm-badge" style="display:none;">0 paths</span>
      </div>

      <div class="bstm-endpoints-grid">
        <div class="bstm-endpoint-card" style="border-left: 3px solid var(--start-color);">
          <span class="bstm-ep-label" style="color: var(--start-color);">Release (Start)</span>
          <span class="bstm-ep-val" id="ep_start">None</span>
        </div>
        <div class="bstm-endpoint-card" style="border-left: 3px solid var(--end-color);">
          <span class="bstm-ep-label" style="color: var(--end-color);">Recapture (End)</span>
          <span class="bstm-ep-val" id="ep_end">None</span>
        </div>
      </div>

      <div class="bstm-status-box" id="status_text">
        Click a polygon on the map to select the <b>Release (Start)</b> unit.
      </div>

      <div class="bstm-control-group">
        <div class="bstm-control-label">
          <span>Migration Elapsed Steps (k)</span>
          <b id="k_val_label" style="color: var(--accent); font-family: var(--font-mono);">$k_default</b>
        </div>
        <input type="range" class="bstm-slider" id="k_slider" min="1" max="20" value="$k_default">
      </div>

      <div class="bstm-control-group">
        <div class="bstm-control-label">
          <span>Max Paths to Draw</span>
          <b id="paths_limit_label" style="color: var(--accent); font-family: var(--font-mono);">$max_paths_render</b>
        </div>
        <input type="range" class="bstm-slider" id="paths_slider" min="5" max="50" value="$max_paths_render">
      </div>

      $(length(group_names) > 1 ? """
      <div class="bstm-control-group">
        <div class="bstm-control-label">Biological Group</div>
        <select class="bstm-select" id="group_select">
          $(join(["<option value=\"$g\">$g</option>" for g in group_names], "\n"))
        </select>
      </div>
      """ : "")

      <div class="bstm-btn-group">
        <button class="bstm-btn bstm-btn-secondary" onclick="clearSelection()">⟲ Clear</button>
        <button class="bstm-btn" onclick="resetMapView()">🗺 Reset Map</button>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script>
    // 1. Data Ingestion
    var allKernels = $all_kernels_json;
    var centroids = $cents_json;
    var polysData = $polys_collection;
    var empData = $emp_collection;
    var defaultGroup = "$(first(group_names))";

    // 2. Map Initialization
    var map = L.map('corridor_map', { attributionControl: false }).setView([$mid_lat, $mid_lon], 7);

    var cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19 }).addTo(map);
    var esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', { maxZoom: 13 });

    var baseLayers = { "CartoDB Dark": cartoDark, "Esri Ocean": esriOcean };
    var overlayLayers = {};

    // Base Polygons Layer
    var polyLayer = L.geoJSON(polysData, {
      style: function(feature) {
        return {
          fillColor: feature.properties.fill_color || '#1e293b',
          fillOpacity: 0.55,
          color: '#334155',
          weight: 0.8
        };
      },
      onEachFeature: function(feature, layer) {
        var uid = feature.properties.unit_id;
        layer.on('click', function(e) {
          handleUnitClick(uid, e.latlng);
        });
        layer.on('mouseover', function() {
          if (uid !== state.startUnit && uid !== state.endUnit) {
            layer.setStyle({ weight: 2.0, color: '#38bdf8' });
          }
        });
        layer.on('mouseout', function() {
          if (uid !== state.startUnit && uid !== state.endUnit) {
            polyLayer.resetStyle(layer);
          }
        });
        layer.bindTooltip("Unit #" + uid + (feature.properties.hsi ? " (HSI: " + feature.properties.hsi + ")" : ""));
      }
    }).addTo(map);
    overlayLayers["Spatial Mesh (HSI)"] = polyLayer;

    if (polyLayer && polyLayer.getBounds().isValid()) {
      map.fitBounds(polyLayer.getBounds(), { padding: [25, 25] });
    }

    // Optional Empirical Tracks Layer
    if (empData.features && empData.features.length > 0) {
      var empTrackLayer = L.geoJSON(empData, {
        style: { color: '#0ea5e9', weight: 1.2, opacity: 0.35 }
      });
      overlayLayers["Direct Tag Vectors"] = empTrackLayer;
    }

    var pathsGroup = L.layerGroup().addTo(map);
    overlayLayers["Probabilistic Paths"] = pathsGroup;

    L.control.layers(baseLayers, overlayLayers, { collapsed: false }).addTo(map);

    // 3. State Management
    var state = {
      startUnit: null,
      endUnit: null,
      startMarker: null,
      endMarker: null,
      k: $k_default,
      maxPaths: $max_paths_render,
      group: defaultGroup
    };

    function updateStatus(msg) {
      document.getElementById('status_text').innerHTML = msg;
    }

    function handleUnitClick(uid, latlng) {
      if (state.startUnit === null) {
        setStartUnit(uid, latlng);
      } else if (state.endUnit === null && uid !== state.startUnit) {
        setEndUnit(uid, latlng);
        computeAndRenderPaths();
      } else {
        clearSelection();
        setStartUnit(uid, latlng);
      }
    }

    function setStartUnit(uid, latlng) {
      state.startUnit = uid;
      document.getElementById('ep_start').innerText = "#" + uid;
      var icon = L.divIcon({
        className: 'bstm-pulse-icon',
        html: '<div style="width:14px;height:14px;background:#10b981;border:2px solid #fff;border-radius:50%;"></div>',
        iconSize: [14, 14]
      });
      state.startMarker = L.marker(latlng, { icon: icon }).addTo(map).bindPopup("<b>Release (Start): #" + uid + "</b>").openPopup();
      updateStatus("Selected Release <b>#" + uid + "</b>. Now click a second polygon to set the <b>Recapture (Destination)</b>.");
    }

    function setEndUnit(uid, latlng) {
      state.endUnit = uid;
      document.getElementById('ep_end').innerText = "#" + uid;
      var icon = L.divIcon({
        className: 'bstm-pulse-icon',
        html: '<div style="width:14px;height:14px;background:#f43f5e;border:2px solid #fff;border-radius:50%;"></div>',
        iconSize: [14, 14]
      });
      state.endMarker = L.marker(latlng, { icon: icon }).addTo(map).bindPopup("<b>Recapture (End): #" + uid + "</b>").openPopup();
      updateStatus("Calculating probabilistic path ensemble between <b>#" + state.startUnit + "</b> and <b>#" + uid + "</b>...");
    }

    function clearSelection() {
      state.startUnit = null;
      state.endUnit = null;
      if (state.startMarker) map.removeLayer(state.startMarker);
      if (state.endMarker) map.removeLayer(state.endMarker);
      pathsGroup.clearLayers();
      document.getElementById('ep_start').innerText = "None";
      document.getElementById('ep_end').innerText = "None";
      document.getElementById('paths_count_badge').style.display = 'none';
      updateStatus("Click a polygon on the map to select the <b>Release (Start)</b> unit.");
    }

    function resetMapView() {
      map.setView([$mid_lat, $mid_lon], 7);
    }

    // 4. Client-Side Dynamic Path Ensemble Search
    function computeAndRenderPaths() {
      if (state.startUnit === null || state.endUnit === null) return;

      pathsGroup.clearLayers();
      var P = allKernels[state.group] || allKernels[defaultGroup];
      if (!P) return;

      var start = String(state.startUnit);
      var end = Number(state.endUnit);
      var k = state.k;
      var maxRender = state.maxPaths;

      // Beam search for top candidate paths from start to end in k steps
      // Path representation: { seq: [start, ...], prob: 1.0 }
      var beam = [{ seq: [Number(start)], prob: 1.0 }];
      var BEAM_WIDTH = 150;

      for (var step = 1; step <= k; step++) {
        var nextBeam = [];
        for (var b = 0; b < beam.length; b++) {
          var pItem = beam[b];
          var lastNode = String(pItem.seq[pItem.seq.length - 1]);
          var nbrs = P[lastNode];
          if (!nbrs || nbrs.length === 0) continue;

          for (var n = 0; n < nbrs.length; n++) {
            var target = nbrs[n][0];
            var pVal = nbrs[n][1];
            nextBeam.push({
              seq: pItem.seq.concat([target]),
              prob: pItem.prob * pVal
            });
          }
        }
        nextBeam.sort(function(a, b) { return b.prob - a.prob; });
        beam = nextBeam.slice(0, BEAM_WIDTH);
      }

      // Filter paths that ended at destination
      var validPaths = beam.filter(function(item) {
        return item.seq[item.seq.length - 1] === end;
      });

      // If exact step reach is empty, fallback to paths that reached destination at intermediate steps
      if (validPaths.length === 0) {
        validPaths = beam.filter(function(item) {
          return item.seq.indexOf(end) !== -1;
        });
      }

      if (validPaths.length === 0) {
        updateStatus("<span style='color:#f87171;'>No viable path found</span> from #" + start + " to #" + end + " in " + k + " steps. Try increasing the <b>k</b> slider.");
        document.getElementById('paths_count_badge').style.display = 'none';
        return;
      }

      validPaths.sort(function(a, b) { return b.prob - a.prob; });
      var renderPaths = validPaths.slice(0, maxRender);
      var maxProb = renderPaths[0].prob;

      document.getElementById('paths_count_badge').style.display = 'inline';
      document.getElementById('paths_count_badge').innerText = renderPaths.length + " paths";
      updateStatus("Rendered <b>" + renderPaths.length + " paths</b> between #" + start + " and #" + end + " (" + k + " steps). <i>Darker = higher posterior probability</i>.");

      // Render Polylines with probability-weighted opacity, thickness, and color
      for (var pIdx = 0; pIdx < renderPaths.length; pIdx++) {
        var pathObj = renderPaths[pIdx];
        var relWeight = (maxProb > 0) ? (pathObj.prob / maxProb) : 1.0;

        var coords = [];
        for (var s = 0; s < pathObj.seq.length; s++) {
          var c = centroids[String(pathObj.seq[s])];
          if (c) coords.push([c[1], c[0]]); // Leaflet [lat, lon]
        }

        // More probable = darker, thicker, higher opacity
        var opacity = 0.20 + 0.75 * relWeight;
        var weight = 1.4 + 3.6 * relWeight;
        
        // Color mapping: dark navy / blue for high, soft slate/cyan for low
        var col = (relWeight > 0.6) ? '#0284c7' : (relWeight > 0.25 ? '#38bdf8' : '#94a3b8');

        var line = L.polyline(coords, {
          color: col,
          weight: weight,
          opacity: opacity
        }).addTo(pathsGroup);

        var probPct = (relWeight * 100).toFixed(1);
        line.bindPopup(
          "<div style='font-family:var(--font-main);font-size:0.85rem;'>" +
          "<b>Probabilistic Path #" + (pIdx + 1) + "</b><br>" +
          "Relative Probability: <b>" + probPct + "%</b><br>" +
          "Steps: " + (pathObj.seq.length - 1) + " hops<br>" +
          "<span style='font-size:0.75rem;color:#94a3b8;font-family:var(--font-mono);'>Route: " + pathObj.seq.join(" → ") + "</span>" +
          "</div>"
        );
      }
    }

    // 5. Control Listeners
    document.getElementById('k_slider').oninput = function(e) {
      state.k = Number(e.target.value);
      document.getElementById('k_val_label').innerText = state.k;
      computeAndRenderPaths();
    };

    document.getElementById('paths_slider').oninput = function(e) {
      state.maxPaths = Number(e.target.value);
      document.getElementById('paths_limit_label').innerText = state.maxPaths;
      computeAndRenderPaths();
    };

    var grpSelect = document.getElementById('group_select');
    if (grpSelect) {
      grpSelect.onchange = function(e) {
        state.group = e.target.value;
        computeAndRenderPaths();
      };
    }
  </script>
</body>
</html>"""

    return LeafletMap(
        html_content,
        title=title,
        width=width,
        height=height,
        metadata=Dict(:n_units=>S, :groups=>group_names, :k_default=>k_default)
    )
end


# ==============================================================================
# SECTION: MIGRATORY CURRENT DENSITY & ECOLOGICAL PINCH-POINTS
# ==============================================================================

"""
    leaflet_current_density_map(
        mesh::NamedTuple,
        current_density::AbstractVector{<:Real};
        pinch_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        pinch_score::Union{Nothing, AbstractVector{<:Real}} = nothing,
        centroids::Union{Nothing, AbstractVector} = nothing,
        output_html::Union{Nothing, AbstractString} = nothing,
        title::String = "BSTM Migratory Current Density & Ecological Pinch-Points",
        colormap::Symbol = :inferno,
        dark_mode::Bool = true,
        width::String = "100%",
        height::String = "750px",
        legend_title::String = "Current Density (J)"
    ) -> LeafletMap

Constructs a standalone interactive Leaflet visualization displaying the population-level
migratory current density field and critical ecological pinch-points across the mesh.

# Visual Design
- Polygons are filled with a continuous color ramp (`:inferno`, `:plasma`, `:turbo`)
  reflecting migratory current density flux ``J_i``.
- Critical bottleneck cells (`pinch_mask .== true`) are emphasized with glowing borders.
- Interactive popups display polygon ID, current density, bottleneck status, and coordinates.
- Includes a floating stats HUD and toggleable pinch-point filter.
"""

"""
    leaflet_current_density_map(
        mesh::NamedTuple,
        res::PosteriorCircuitResult;
        prob_threshold::Real = 0.80,
        title::String = "BSTM Posterior Migratory Current Density & Pinch-Points",
        legend_title::String = "Posterior Mean Flux (J)",
        kwargs...
    ) -> LeafletMap

Overload of `leaflet_current_density_map` for `PosteriorCircuitResult`, plotting the
posterior mean current density field and highlighting robust pinch-points that exceed
`prob_threshold` certainty.
"""
function leaflet_current_density_map(
    mesh::NamedTuple,
    res::PosteriorCircuitResult;
    prob_threshold::Real = 0.80,
    title::String = "BSTM Posterior Migratory Current Density & Pinch-Points",
    legend_title::String = "Posterior Mean Flux (J)",
    kwargs...
)
    robust_mask, _, _ = identify_stochastic_pinchpoints(
        res; prob_threshold = prob_threshold
    )
    return leaflet_current_density_map(
        mesh,
        res.mean_density;
        pinch_mask = robust_mask,
        pinch_score = res.pinchpoint_prob,
        title = title,
        legend_title = legend_title,
        kwargs...
    )
end

function leaflet_current_density_map(
    mesh::NamedTuple,
    current_density::AbstractVector{<:Real};
    pinch_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    pinch_score::Union{Nothing, AbstractVector{<:Real}} = nothing,
    centroids::Union{Nothing, AbstractVector} = nothing,
    output_html::Union{Nothing, AbstractString} = nothing,
    title::String = "BSTM Migratory Current Density & Ecological Pinch-Points",
    colormap::Symbol = :inferno,
    dark_mode::Bool = true,
    width::String = "100%",
    height::String = "750px",
    legend_title::String = "Current Density (J)"
)
    polys_raw = if hasproperty(mesh, :polygons)
        mesh.polygons
    elseif hasproperty(mesh, :poly)
        mesh.poly
    elseif hasproperty(mesh, :polygons_lonlat)
        mesh.polygons_lonlat
    elseif hasproperty(mesh, :polygons_km)
        mesh.polygons_km
    else
        error("mesh must provide :polygons, :poly, or :polygons_lonlat")
    end

    S = length(polys_raw)
    if length(current_density) != S
        throw(DimensionMismatch(
            "current_density length $(length(current_density)) must match mesh units $S"
        ))
    end

    cents_raw = if !isnothing(centroids) && length(centroids) == S
        centroids
    elseif hasproperty(mesh, :centroids) && length(mesh.centroids) == S
        mesh.centroids
    elseif hasproperty(mesh, :centroids_lonlat) && length(mesh.centroids_lonlat) == S
        mesh.centroids_lonlat
    elseif hasproperty(mesh, :centroids_km) && length(mesh.centroids_km) == S
        mesh.centroids_km
    else
        fill([NaN, NaN], S)
    end

    pal = _resolve_palette(colormap)
    dens_vals = Float64.(current_density)
    max_d = maximum(dens_vals)
    min_d = minimum(dens_vals)

    has_pinch = !isnothing(pinch_mask) && length(pinch_mask) == S
    pinch_bools = has_pinch ? pinch_mask : fill(false, S)
    n_pinch = sum(pinch_bools)

    # Convert polygons to GeoJSON FeatureCollection
    features_json = String[]
    for i in 1:S
        poly = polys_raw[i]
        length(poly) < 3 && continue
        c_str = [
            "[$(round(Float64(pt[1]), digits=6)), $(round(Float64(pt[2]), digits=6))]"
            for pt in poly
        ]
        if c_str[1] != c_str[end]
            push!(c_str, c_str[1])
        end

        d_val = dens_vals[i]
        col = _map_val_to_hex(d_val, min_d, max_d, pal)
        is_p = pinch_bools[i]
        p_str = is_p ? "CRITICAL PINCH-POINT" : "Normal Corridor"
        ps_str = if !isnothing(pinch_score) && length(pinch_score) == S
            @sprintf("%.4f", Float64(pinch_score[i]))
        else
            "N/A"
        end

        push!(features_json, """{
          "type": "Feature",
          "id": $i,
          "properties": {
            "unit_id": $i,
            "density": $(round(d_val, digits=5)),
            "is_pinch": $(is_p ? "true" : "false"),
            "pinch_status": "$p_str",
            "pinch_score": "$ps_str",
            "fill_color": "$col"
          },
          "geometry": { "type": "Polygon", "coordinates": [[$(join(c_str, ", "))]] }
        }""")
    end
    geojson_str = "{\"type\": \"FeatureCollection\", \"features\": [\n" *
                  join(features_json, ",\n") * "]}"

    all_lons = [Float64(c[1]) for c in cents_raw if !isnan(c[1])]
    all_lats = [Float64(c[2]) for c in cents_raw if !isnan(c[2])]
    mid_lon = !isempty(all_lons) ? (minimum(all_lons) + maximum(all_lons)) / 2.0 : -60.0
    mid_lat = !isempty(all_lats) ? (minimum(all_lats) + maximum(all_lats)) / 2.0 : 45.0

    grad_css = "linear-gradient(to right, " * join(pal, ", ") * ")"
    min_str = @sprintf("%.3e", min_d)
    max_str = @sprintf("%.3e", max_d)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>$title</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet"
    href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap">
  <style>
    :root {
      --bg-main: #090d16;
      --panel-bg: rgba(15, 23, 42, 0.90);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-main: rgba(255, 255, 255, 0.12);
      --accent: #38bdf8;
      --pinch-glow: #06b6d4;
      --font-main: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-main);
      background-color: var(--bg-main);
      color: var(--text-main);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .bstm-header {
      background: var(--panel-bg);
      border-bottom: 1px solid var(--border-main);
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 1000;
    }
    .bstm-title {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .bstm-badge {
      font-size: 0.70rem;
      text-transform: uppercase;
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 600;
    }
    .bstm-map-container {
      position: relative;
      flex: 1;
      width: 100%;
    }
    #density_map {
      width: 100%;
      height: 100%;
      background: #0b1120;
    }
    .bstm-floating-panel {
      position: absolute;
      top: 15px;
      right: 15px;
      width: 320px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 12px;
      padding: 16px;
      z-index: 1000;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .bstm-stat-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 0;
      font-size: 0.85rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .bstm-stat-val {
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--accent);
    }
    .bstm-legend {
      position: absolute;
      bottom: 25px;
      left: 20px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 10px;
      padding: 12px 16px;
      z-index: 1000;
      width: 280px;
    }
    .bstm-legend-bar {
      height: 12px;
      border-radius: 6px;
      margin: 8px 0 4px 0;
      background: $grad_css;
    }
    .bstm-legend-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }
    .bstm-toggle-btn {
      width: 100%;
      margin-top: 12px;
      padding: 8px;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid var(--accent);
      color: var(--accent);
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .bstm-toggle-btn:hover {
      background: rgba(56, 189, 248, 0.30);
      color: #fff;
    }
  </style>
</head>
<body>
  <header class="bstm-header">
    <div class="bstm-title">
      <span>$title</span>
      <span class="bstm-badge">Circuit Theory / Ohm's Law</span>
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted);">
      Population Migratory Flux & Bottleneck Corridors
    </div>
  </header>

  <div class="bstm-map-container">
    <div id="density_map"></div>

    <div class="bstm-floating-panel">
      <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 10px;">
        Connectivity Statistics
      </div>
      <div class="bstm-stat-row">
        <span style="color: var(--text-muted);">Active Spatial Units:</span>
        <span class="bstm-stat-val">$S</span>
      </div>
      <div class="bstm-stat-row">
        <span style="color: var(--text-muted);">Max Current Density:</span>
        <span class="bstm-stat-val">$max_str</span>
      </div>
      <div class="bstm-stat-row">
        <span style="color: var(--text-muted);">Critical Pinch-Points:</span>
        <span class="bstm-stat-val" style="color: #38bdf8;">$n_pinch</span>
      </div>
      <button id="toggle_pinch_btn" class="bstm-toggle-btn">
        Toggle Pinch-Point Highlight
      </button>
    </div>

    <div class="bstm-legend">
      <div style="font-size: 0.82rem; font-weight: 600;">$legend_title</div>
      <div class="bstm-legend-bar"></div>
      <div class="bstm-legend-labels">
        <span>$min_str</span>
        <span>$max_str</span>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    var map = L.map('density_map', {
      center: [$mid_lat, $mid_lon],
      zoom: 7,
      preferCanvas: true
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap, &copy; CARTO',
      maxZoom: 19
    }).addTo(map);

    var rawGeojson = $geojson_str;
    var showPinchHighlight = true;

    function getFeatureStyle(feature) {
      var isP = feature.properties.is_pinch;
      var strokeColor = (isP && showPinchHighlight) ? '#06b6d4' : '#1e293b';
      var strokeWeight = (isP && showPinchHighlight) ? 2.5 : 0.6;
      var strokeOpacity = (isP && showPinchHighlight) ? 0.95 : 0.4;

      return {
        fillColor: feature.properties.fill_color,
        fillOpacity: 0.82,
        color: strokeColor,
        weight: strokeWeight,
        opacity: strokeOpacity
      };
    }

    var geojsonLayer = L.geoJSON(rawGeojson, {
      style: getFeatureStyle,
      onEachFeature: function(feature, layer) {
        var p = feature.properties;
        var badgeHtml = p.is_pinch
          ? "<span style='background:rgba(6,182,212,0.2);color:#06b6d4;" +
            "padding:2px 6px;border-radius:4px;font-size:0.75rem;font-weight:700;'>" +
            "CRITICAL BOTTLENECK</span>"
          : "<span style='color:#94a3b8;font-size:0.75rem;'>Standard Marine Corridor</span>";

        var scoreRow = (p.pinch_score !== 'N/A')
          ? "<div>Pinch Score: <b style='font-family:var(--font-mono);'>" +
            p.pinch_score + "</b></div>"
          : "";

        layer.bindPopup(
          "<div style='font-family:var(--font-main);font-size:0.85rem;line-height:1.4;'>" +
          "<div style='font-weight:700;font-size:0.95rem;margin-bottom:4px;'>" +
          "Polygon #" + p.unit_id + "</div>" +
          "<div>Status: " + badgeHtml + "</div>" +
          "<div>Current Density: <b style='font-family:var(--font-mono);'>" +
          p.density.toExponential(4) + "</b></div>" +
          scoreRow +
          "</div>"
        );

        layer.on('mouseover', function() {
          this.setStyle({ weight: 3.5, color: '#f8fafc', fillOpacity: 0.95 });
        });
        layer.on('mouseout', function() {
          geojsonLayer.resetStyle(this);
        });
      }
    }).addTo(map);

    try {
      map.fitBounds(geojsonLayer.getBounds(), { padding: [20, 20] });
    } catch(e) {}

    document.getElementById('toggle_pinch_btn').onclick = function() {
      showPinchHighlight = !showPinchHighlight;
      geojsonLayer.setStyle(getFeatureStyle);
      this.innerText = showPinchHighlight ? "Hide Pinch Highlights" : "Highlight Pinch-Points";
    };
  </script>
</body>
</html>"""

    if !isnothing(output_html)
        write(output_html, html_content)
    end

    return LeafletMap(
        html_content,
        title=title,
        width=width,
        height=height,
        metadata=Dict(:n_units=>S, :n_pinch=>n_pinch, :max_density=>max_d)
    )
end

"""
    leaflet_graph_wavelet_dashboard(
        mesh,
        sgwt_result::SpectralGraphWaveletResult;
        reconstruction::Union{Nothing, AbstractVector{<:Real}} = nothing,
        centroids::Union{Nothing, AbstractVector} = nothing,
        output_html::Union{Nothing, AbstractString} = nothing,
        title::String = "BSTM Spectral Graph Wavelet Multi-Scale Dashboard",
        signal_name::String = "Spatial Field",
        width::String = "100%",
        height::String = "750px"
    ) -> LeafletMap

Generates a standalone, interactive Leaflet HTML dashboard for visual inspection of
multiresolution Spectral Graph Wavelet decompositions across irregular spatial meshes.

# Features
- **Dynamic Layer Toggling**: Seamlessly switch between Original Signal ``\\mathbf{f}``,
  Macro Baseline (Low-pass Approximation ``\\mathbf{a}``), Detail Scales
  ``\\mathbf{d}_1, \\dots, \\mathbf{d}_J`` (fine to coarse), and Denoised Reconstruction
  ``\\hat{\\mathbf{f}}`` without re-rendering the DOM.
- **Adaptive Colormaps**: Uses sequential palettes for baseline and reconstructed fields,
  and zero-centered diverging palettes for oscillating wavelet detail coefficients.
- **Energy Spectrum Analytics**: Real-time progress bars showing the percentage of
  spatial variance and energy captured across each resolution scale.
- **Cell Inspection Popups**: Displays exact numerical values across all resolution
  scales simultaneously upon clicking or hovering any spatial unit.

# Arguments
- `mesh`: Spatial partitioning mesh (named tuple or object with `:polygons` and `:centroids`).
- `sgwt_result::SpectralGraphWaveletResult`: Decomposed wavelet result from
  `spectral_graph_wavelet_transform`.
- `reconstruction`: Optional reconstructed or denoised signal vector (length ``S``).
- `centroids`: Optional explicit centroid coordinates (length ``S``).
- `output_html`: Optional file path to write standalone HTML.
- `title::String`: Dashboard title.
- `signal_name::String`: Name of the analyzed ecological field (default: `"Spatial Field"`).
- `width::String`: CSS width (default: `"100%"`).
- `height::String`: CSS height (default: `"750px"`).

# Returns
- `map::LeafletMap`: Standalone LeafletMap object containing complete HTML and metadata.
"""
function leaflet_graph_wavelet_dashboard(
    mesh,
    sgwt_result::SpectralGraphWaveletResult;
    reconstruction::Union{Nothing, AbstractVector{<:Real}} = nothing,
    centroids::Union{Nothing, AbstractVector} = nothing,
    output_html::Union{Nothing, AbstractString} = nothing,
    title::String = "BSTM Spectral Graph Wavelet Multi-Scale Dashboard",
    signal_name::String = "Spatial Field",
    width::String = "100%",
    height::String = "750px"
)
    polys_raw = if hasproperty(mesh, :polygons)
        mesh.polygons
    elseif hasproperty(mesh, :poly)
        mesh.poly
    elseif hasproperty(mesh, :polygons_lonlat)
        mesh.polygons_lonlat
    elseif hasproperty(mesh, :polygons_km)
        mesh.polygons_km
    else
        error("mesh must provide :polygons, :poly, or :polygons_lonlat")
    end

    S = length(polys_raw)
    if length(sgwt_result.original) != S
        throw(DimensionMismatch(
            "Wavelet signal length $(length(sgwt_result.original)) != mesh units $S"
        ))
    end

    cents_raw = if !isnothing(centroids) && length(centroids) == S
        centroids
    elseif hasproperty(mesh, :centroids) && length(mesh.centroids) == S
        mesh.centroids
    elseif hasproperty(mesh, :centroids_lonlat) && length(mesh.centroids_lonlat) == S
        mesh.centroids_lonlat
    elseif hasproperty(mesh, :centroids_km) && length(mesh.centroids_km) == S
        mesh.centroids_km
    else
        fill([NaN, NaN], S)
    end

    J = length(sgwt_result.scales)
    has_recon = !isnothing(reconstruction) && length(reconstruction) == S
    recon_vec = has_recon ? Float64.(reconstruction) : zeros(Float64, S)

    # Compute min/max bounds for each layer to allow client-side color normalization
    orig_min, orig_max = extrema(sgwt_result.original)
    app_min, app_max = extrema(sgwt_result.approximation)
    detail_max_abs = [maximum(abs, sgwt_result.details[:, j]) for j in 1:J]
    rec_min, rec_max = has_recon ? extrema(recon_vec) : (0.0, 1.0)

    # Build GeoJSON feature collection
    features_json = String[]
    for i in 1:S
        poly = polys_raw[i]
        length(poly) < 3 && continue
        c_str = [
            "[$(round(Float64(pt[1]), digits=6)), $(round(Float64(pt[2]), digits=6))]"
            for pt in poly
        ]
        if c_str[1] != c_str[end]
            push!(c_str, c_str[1])
        end

        det_vals = [round(sgwt_result.details[i, j], digits=5) for j in 1:J]
        rec_val = has_recon ? round(recon_vec[i], digits=5) : 0.0

        push!(features_json, """{
          "type": "Feature",
          "id": $i,
          "properties": {
            "unit_id": $i,
            "orig": $(round(sgwt_result.original[i], digits=5)),
            "approx": $(round(sgwt_result.approximation[i], digits=5)),
            "details": [$(join(det_vals, ", "))],
            "recon": $rec_val
          },
          "geometry": { "type": "Polygon", "coordinates": [[$(join(c_str, ", "))]] }
        }""")
    end
    geojson_str = "{\"type\": \"FeatureCollection\", \"features\": [\n" *
                  join(features_json, ",\n") * "]}"

    all_lons = [Float64(c[1]) for c in cents_raw if !isnan(c[1])]
    all_lats = [Float64(c[2]) for c in cents_raw if !isnan(c[2])]
    mid_lon = !isempty(all_lons) ? (minimum(all_lons) + maximum(all_lons)) / 2.0 : -60.0
    mid_lat = !isempty(all_lats) ? (minimum(all_lats) + maximum(all_lats)) / 2.0 : 45.0

    # Build energy bars HTML
    energy_bars_html = String[]
    # Macro Approx bar
    p0 = sgwt_result.relative_energies[1] * 100.0
    push!(energy_bars_html, """
      <div class="bstm-energy-row">
        <div class="bstm-energy-label">
          <span>Macro Baseline</span>
          <span>$(round(p0, digits=1))%</span>
        </div>
        <div class="bstm-bar-track">
          <div class="bstm-bar-fill fill-approx" style="width: $(clamp(p0, 1.0, 100.0))%;"></div>
        </div>
      </div>
    """)
    for j in 1:J
        pj = sgwt_result.relative_energies[j + 1] * 100.0
        sj = sgwt_result.scales[j]
        push!(energy_bars_html, """
          <div class="bstm-energy-row">
            <div class="bstm-energy-label">
              <span>Scale $j (s=$(round(sj, digits=2)))</span>
              <span>$(round(pj, digits=1))%</span>
            </div>
            <div class="bstm-bar-track">
              <div class="bstm-bar-fill fill-scale-$j"
                   style="width: $(clamp(pj, 1.0, 100.0))%;"></div>
            </div>
          </div>
        """)
    end
    energy_section_html = join(energy_bars_html, "\n")

    # Layer buttons HTML
    layer_btns_html = [
        """<button class="bstm-layer-btn active" data-layer="orig">Original Field</button>""",
        """<button class="bstm-layer-btn" data-layer="approx">Macro Baseline</button>"""
    ]
    for j in 1:J
        sj = round(sgwt_result.scales[j], digits=2)
        push!(layer_btns_html,
            """<button class="bstm-layer-btn" data-layer="detail_$j">Scale $j (s=$sj)</button>""")
    end
    if has_recon
        push!(layer_btns_html,
            """<button class="bstm-layer-btn" data-layer="recon">Denoised / Recon</button>""")
    end
    buttons_html = join(layer_btns_html, "\n")

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>$title</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="stylesheet"
    href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap">
  <style>
    :root {
      --bg-main: #0b1120;
      --panel-bg: rgba(15, 23, 42, 0.90);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-main: rgba(255, 255, 255, 0.12);
      --accent: #38bdf8;
      --font-main: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-main);
      background-color: var(--bg-main);
      color: var(--text-main);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .bstm-header {
      background: var(--panel-bg);
      border-bottom: 1px solid var(--border-main);
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 1000;
    }
    .bstm-title {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .bstm-badge {
      font-size: 0.70rem;
      text-transform: uppercase;
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 600;
    }
    .bstm-map-container {
      position: relative;
      flex: 1;
      width: 100%;
    }
    #wavelet_map {
      width: 100%;
      height: 100%;
      background: #0b1120;
    }
    .bstm-floating-panel {
      position: absolute;
      top: 15px;
      right: 15px;
      width: 320px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 12px;
      padding: 16px;
      z-index: 1000;
      max-height: calc(100vh - 120px);
      overflow-y: auto;
    }
    .bstm-panel-title {
      font-size: 0.95rem;
      font-weight: 700;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border-main);
      padding-bottom: 6px;
      color: var(--text-main);
    }
    .bstm-layer-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-bottom: 16px;
    }
    .bstm-layer-btn {
      padding: 6px 10px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-main);
      color: var(--text-muted);
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.78rem;
      cursor: pointer;
      transition: all 0.15s ease;
      text-align: center;
    }
    .bstm-layer-btn:hover {
      background: rgba(56, 189, 248, 0.15);
      color: #fff;
    }
    .bstm-layer-btn.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #0b1120;
      font-weight: 700;
    }
    .bstm-stat-row {
      display: flex;
      justify-content: space-between;
      padding: 5px 0;
      font-size: 0.82rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .bstm-stat-val {
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--accent);
    }
    .bstm-energy-row {
      margin-bottom: 8px;
    }
    .bstm-energy-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.76rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
      margin-bottom: 2px;
    }
    .bstm-bar-track {
      height: 6px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 3px;
      overflow: hidden;
    }
    .bstm-bar-fill {
      height: 100%;
      border-radius: 3px;
    }
    .fill-approx { background: #38bdf8; }
    .fill-scale-1 { background: #ec4899; }
    .fill-scale-2 { background: #f59e0b; }
    .fill-scale-3 { background: #10b981; }
    .fill-scale-4 { background: #8b5cf6; }
    .bstm-legend {
      position: absolute;
      bottom: 25px;
      left: 20px;
      background: var(--panel-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-main);
      border-radius: 10px;
      padding: 12px 16px;
      z-index: 1000;
      width: 280px;
    }
    .bstm-legend-bar {
      height: 12px;
      border-radius: 6px;
      margin: 8px 0 4px 0;
    }
    .bstm-legend-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <header class="bstm-header">
    <div class="bstm-title">
      <span>$title</span>
      <span class="bstm-badge">Spectral Graph Wavelets</span>
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted);">
      Irregular Mesh Multiresolution Decomposition
    </div>
  </header>

  <div class="bstm-map-container">
    <div id="wavelet_map"></div>

    <div class="bstm-floating-panel">
      <div class="bstm-panel-title">Spatial Resolution Layer</div>
      <div class="bstm-layer-grid" id="layer_button_container">
        $buttons_html
      </div>

      <div class="bstm-panel-title" style="margin-top: 10px;">Spatial Energy Distribution</div>
      <div style="margin-bottom: 12px;">
        $energy_section_html
      </div>

      <div class="bstm-panel-title">Spectral Metrics</div>
      <div class="bstm-stat-row">
        <span>Mesh Units (S):</span>
        <span class="bstm-stat-val">$S</span>
      </div>
      <div class="bstm-stat-row">
        <span>Chebyshev Degree (M):</span>
        <span class="bstm-stat-val">$(sgwt_result.order)</span>
      </div>
      <div class="bstm-stat-row">
        <span>Laplacian Bound (λ_max):</span>
        <span class="bstm-stat-val">$(round(sgwt_result.lambda_max, digits=4))</span>
      </div>
      <div class="bstm-stat-row">
        <span>Total Energy:</span>
        <span class="bstm-stat-val">$(@sprintf("%.4e", sgwt_result.total_energy))</span>
      </div>
    </div>

    <div class="bstm-legend">
      <div id="legend_title"
           style="font-size: 0.82rem; font-weight: 700; color: var(--text-main);">
        $signal_name (Original)
      </div>
      <div class="bstm-legend-bar" id="legend_gradient"></div>
      <div class="bstm-legend-labels">
        <span id="legend_min"></span>
        <span id="legend_max"></span>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    var map = L.map('wavelet_map', {
      center: [$mid_lat, $mid_lon],
      zoom: 7,
      preferCanvas: true
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap, &copy; CARTO',
      maxZoom: 19
    }).addTo(map);

    var rawGeojson = $geojson_str;
    var currentLayer = 'orig';
    var numDetailScales = $J;
    var hasRecon = $(has_recon ? "true" : "false");

    var bounds = {
      orig: { min: $orig_min, max: $orig_max, name: '$signal_name (Original)', type: 'seq' },
      approx: { min: $app_min, max: $app_max, name: 'Macro Baseline Approx', type: 'seq' },
      recon: { min: $rec_min, max: $rec_max, name: 'Reconstructed / Denoised', type: 'seq' }
    };
    var detailMaxAbs = [$(join(detail_max_abs, ", "))];
    for (var j = 1; j <= numDetailScales; j++) {
      var dMax = detailMaxAbs[j - 1] > 1e-12 ? detailMaxAbs[j - 1] : 1.0;
      bounds['detail_' + j] = {
        min: -dMax,
        max: dMax,
        name: 'Scale ' + j + ' Detail (Band-pass)',
        type: 'div'
      };
    }

    var seqPalette = [
      [48, 18, 59], [70, 98, 216], [53, 171, 248], [26, 228, 182],
      [114, 254, 94], [199, 241, 53], [250, 186, 57], [245, 91, 17],
      [217, 40, 11], [122, 4, 3]
    ];

    var divPalette = [
      [2, 132, 199], [56, 189, 248], [147, 197, 253], [226, 232, 240],
      [252, 165, 165], [248, 113, 113], [239, 68, 68]
    ];

    function interpolatePalette(t, palette) {
      t = Math.max(0.0, Math.min(1.0, t));
      var n = palette.length - 1;
      var idx = Math.floor(t * n);
      if (idx >= n) return 'rgb(' + palette[n].join(',') + ')';
      var frac = (t * n) - idx;
      var c1 = palette[idx];
      var c2 = palette[idx + 1];
      var r = Math.round(c1[0] + frac * (c2[0] - c1[0]));
      var g = Math.round(c1[1] + frac * (c2[1] - c1[1]));
      var b = Math.round(c1[2] + frac * (c2[2] - c1[2]));
      return 'rgb(' + r + ',' + g + ',' + b + ')';
    }

    function getFeatureVal(props, layer) {
      if (layer === 'orig') return props.orig;
      if (layer === 'approx') return props.approx;
      if (layer === 'recon') return props.recon;
      if (layer.startsWith('detail_')) {
        var idx = parseInt(layer.split('_')[1], 10) - 1;
        return props.details[idx];
      }
      return props.orig;
    }

    function getFeatureFillColor(props, layer) {
      var b = bounds[layer] || bounds.orig;
      var val = getFeatureVal(props, layer);
      if (b.type === 'div') {
        var maxAbs = Math.max(Math.abs(b.min), Math.abs(b.max));
        var t = maxAbs > 1e-12 ? 0.5 + 0.5 * (val / maxAbs) : 0.5;
        return interpolatePalette(t, divPalette);
      } else {
        var range = b.max - b.min;
        var t = range > 1e-12 ? (val - b.min) / range : 0.5;
        return interpolatePalette(t, seqPalette);
      }
    }

    function getFeatureStyle(feature) {
      return {
        fillColor: getFeatureFillColor(feature.properties, currentLayer),
        fillOpacity: 0.85,
        color: '#1e293b',
        weight: 0.8,
        opacity: 0.6
      };
    }

    var geojsonLayer = L.geoJSON(rawGeojson, {
      style: getFeatureStyle,
      onEachFeature: function(feature, layer) {
        layer.on('click', function() {
          var p = feature.properties;
          var detailsRows = '';
          for (var j = 0; j < p.details.length; j++) {
            var signSym = p.details[j] >= 0 ? '+' : '';
            detailsRows += '<div style="display:flex;justify-content:space-between;' +
              'padding:2px 0;">' +
              '<span style="color:#94a3b8;">Scale ' + (j + 1) + ':</span>' +
              '<span style="font-family:var(--font-mono);font-weight:600;">' +
              signSym + p.details[j].toExponential(3) + '</span></div>';
          }
          var reconRow = hasRecon ?
            '<div style="display:flex;justify-content:space-between;padding:2px 0;' +
            'border-top:1px solid rgba(255,255,255,0.1);margin-top:4px;">' +
            '<span style="color:#38bdf8;">Denoised:</span>' +
            '<span style="font-family:var(--font-mono);font-weight:700;color:#38bdf8;">' +
            p.recon.toFixed(4) + '</span></div>' : '';

          var popupContent =
            '<div style="font-family:var(--font-main);font-size:0.85rem;min-width:180px;">' +
            '<div style="font-weight:700;font-size:0.95rem;margin-bottom:6px;' +
            'border-bottom:1px solid rgba(255,255,255,0.15);padding-bottom:3px;">' +
            'Polygon Unit #' + p.unit_id + '</div>' +
            '<div style="display:flex;justify-content:space-between;padding:2px 0;">' +
            '<span style="color:#94a3b8;">Original:</span>' +
            '<span style="font-family:var(--font-mono);font-weight:600;">' +
            p.orig.toFixed(4) + '</span></div>' +
            '<div style="display:flex;justify-content:space-between;padding:2px 0;">' +
            '<span style="color:#94a3b8;">Macro Baseline:</span>' +
            '<span style="font-family:var(--font-mono);font-weight:600;">' +
            p.approx.toFixed(4) + '</span></div>' +
            '<div style="font-size:0.75rem;text-transform:uppercase;color:#38bdf8;' +
            'font-weight:700;margin-top:6px;margin-bottom:2px;">Detail Scales</div>' +
            detailsRows +
            reconRow +
            '</div>';

          layer.bindPopup(popupContent).openPopup();
        });

        layer.on('mouseover', function() {
          this.setStyle({ weight: 2.5, color: '#f8fafc', fillOpacity: 0.95 });
        });
        layer.on('mouseout', function() {
          geojsonLayer.resetStyle(this);
        });
      }
    }).addTo(map);

    try {
      map.fitBounds(geojsonLayer.getBounds(), { padding: [20, 20] });
    } catch(e) {}

    function updateLegend() {
      var b = bounds[currentLayer] || bounds.orig;
      document.getElementById('legend_title').innerText = b.name;
      document.getElementById('legend_min').innerText = b.min.toExponential(2);
      document.getElementById('legend_max').innerText = b.max.toExponential(2);

      var gradEl = document.getElementById('legend_gradient');
      if (b.type === 'div') {
        var cols = divPalette.map(function(c) { return 'rgb(' + c.join(',') + ')'; });
        gradEl.style.background = 'linear-gradient(to right, ' + cols.join(', ') + ')';
      } else {
        var cols = seqPalette.map(function(c) { return 'rgb(' + c.join(',') + ')'; });
        gradEl.style.background = 'linear-gradient(to right, ' + cols.join(', ') + ')';
      }
    }
    updateLegend();

    var buttons = document.querySelectorAll('.bstm-layer-btn');
    buttons.forEach(function(btn) {
      btn.onclick = function() {
        buttons.forEach(function(b) { b.classList.remove('active'); });
        this.classList.add('active');
        currentLayer = this.getAttribute('data-layer');
        geojsonLayer.setStyle(getFeatureStyle);
        updateLegend();
      };
    });
  </script>
</body>
</html>"""

    if !isnothing(output_html)
        write(output_html, html_content)
    end

    return LeafletMap(
        html_content,
        title = title,
        width = width,
        height = height,
        metadata = Dict(
            :n_units => S,
            :num_scales => J,
            :lambda_max => sgwt_result.lambda_max,
            :total_energy => sgwt_result.total_energy
        )
    )
end
