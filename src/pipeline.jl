const _DepthRangeArg = Union{
    Nothing,
    Tuple{<:Real, <:Real},
    AbstractVector{<:Real},
    AbstractString,
}


using DataFrames

const PooledArrays = DataFrames.PooledArrays

"""
    _stable_string_hash(s::AbstractString) -> UInt64

Deterministic 64-bit FNV-1a hash of `s`.

Julia's built-in `hash` for strings is salted with a per-process seed, so the
same input yields a different value in every session. Any seed that has to be
reproducible from a configured `seed` must be derived with this function
instead, otherwise stochastic perturbations are not reproducible even when the
configured seed is fixed.
"""
function _stable_string_hash(s::AbstractString)::UInt64
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    return h
end

function _load_snowcrab_sppoly(sppoly_file::AbstractString)
    isfile(sppoly_file) || return (geometries=LibGEOS.AbstractGeometry[], bounds=nothing)
    table = JLD2.load(sppoly_file, "sppoly")
    hasproperty(table, :wkt_geo) ||
        throw(ArgumentError("Snow crab sppoly file must contain a :wkt_geo column."))

    geometries = LibGEOS.AbstractGeometry[]
    for wkt in table.wkt_geo
        ismissing(wkt) && continue
        geom = LibGEOS.readgeom(String(wkt))
        if !LibGEOS.isValid(geom)
            geom = LibGEOS.buffer(geom, 0.0)
        end
        LibGEOS.isEmpty(geom) || push!(geometries, geom)
    end
    isempty(geometries) &&
        throw(ArgumentError("Snow crab sppoly file contains no valid geographic polygons."))

    bounds = (
        west = minimum(LibGEOS.getXMin.(geometries)),
        south = minimum(LibGEOS.getYMin.(geometries)),
        east = maximum(LibGEOS.getXMax.(geometries)),
        north = maximum(LibGEOS.getYMax.(geometries))
    )
    return (geometries=geometries, bounds=bounds)
end

function _snowcrab_mesh_keep_mask(
    mesh,
    geometries::AbstractVector,
    anchor_points::AbstractVector;
    depths = nothing,
    depth_range = nothing,
    bounds = nothing,
    crs = nothing,
    datum = WGS84Latest
)
    keep = falses(mesh.n_units)
    polygon_tree = isempty(geometries) ? nothing : LibGEOS.STRtree(collect(geometries))
    if !isempty(anchor_points)
        anchor_xy = [lonlat_to_xy_km(
            point[1], point[2];
            center_lon=mesh.center_lon, center_lat=mesh.center_lat,
            crs=crs, datum=datum
        ) for point in anchor_points]
        tree = KDTree(hcat([[point[1], point[2]] for point in mesh.centroids_km]...))
        for (index, point) in enumerate(anchor_xy)
            nearest = knn(tree, [point[1], point[2]], 1)[1][1]
            lon, lat = anchor_points[index]
            point_geom = LibGEOS.Point(lon, lat)
            in_sppoly = polygon_tree !== nothing && any(
                polygon -> LibGEOS.intersects(polygon, point_geom),
                LibGEOS.query(polygon_tree, point_geom)
            )
            in_extension = bounds !== nothing &&
                lon >= bounds.west && lat >= bounds.north
            (in_sppoly || in_extension) && (keep[nearest] = true)
        end
    end

    depth_limits = depth_range === nothing ? nothing : _parse_depth_range(depth_range)
    if depths !== nothing && length(depths) != mesh.n_units
        throw(DimensionMismatch("Depth vector must match the snow crab mesh units."))
    end

    for i in 1:mesh.n_units
        lon, lat = mesh.centroids_lonlat[i]
        if polygon_tree !== nothing
            hex_coords = [[Float64(point[1]), Float64(point[2])]
                          for point in mesh.polygons_lonlat[i]]
            hexagon = LibGEOS.Polygon([hex_coords])
            for polygon in LibGEOS.query(polygon_tree, hexagon)
                if LibGEOS.intersects(polygon, hexagon) &&
                   LibGEOS.area(LibGEOS.intersection(polygon, hexagon)) > 1e-12
                    keep[i] = true
                    break
                end
            end
        end

        if !keep[i] && depths !== nothing && depth_limits !== nothing && bounds !== nothing
            depth = Float64(depths[i])
            keep[i] = isfinite(depth) &&
                depth_limits[1] <= depth <= depth_limits[2] &&
                lon >= bounds.west && lat >= bounds.north
        end
    end
    return keep
end

function _clip_snowcrab_mesh_to_sppoly(
    mesh,
    geometries::AbstractVector;
    crs = nothing,
    datum = WGS84Latest
)
    isempty(geometries) && return mesh
    tree = LibGEOS.STRtree(collect(geometries))
    centroids_lonlat = copy(mesh.centroids_lonlat)
    centroids_km = copy(mesh.centroids_km)
    polygons_lonlat = copy(mesh.polygons_lonlat)
    polygons_km = copy(mesh.polygons_km)

    for i in 1:mesh.n_units
        hex_coords = [[Float64(point[1]), Float64(point[2])] for point in polygons_lonlat[i]]
        hexagon = LibGEOS.Polygon([hex_coords])
        clipped = nothing
        for polygon in LibGEOS.query(tree, hexagon)
            LibGEOS.intersects(hexagon, polygon) || continue
            overlap = LibGEOS.intersection(hexagon, polygon)
            LibGEOS.isEmpty(overlap) && continue
            LibGEOS.area(overlap) <= 1e-12 && continue
            clipped = clipped === nothing ? overlap : LibGEOS.union(clipped, overlap)
        end
        clipped === nothing && continue

        pieces = filter(
            piece -> piece isa LibGEOS.Polygon && LibGEOS.area(piece) > 1e-12,
            LibGEOS.getGeometries(clipped)
        )
        isempty(pieces) && continue
        piece = pieces[argmax(LibGEOS.area.(pieces))]
        ring = LibGEOS.getCoordinates(LibGEOS.getCoordSeq(LibGEOS.exteriorRing(piece)))
        polygons_lonlat[i] = [(point[1], point[2]) for point in ring]
        interior = first(LibGEOS.getCoordinates(
            LibGEOS.getCoordSeq(LibGEOS.pointOnSurface(piece))
        ))
        centroids_lonlat[i] = (interior[1], interior[2])
        centroids_km[i] = lonlat_to_xy_km(
            interior[1], interior[2];
            center_lon=mesh.center_lon, center_lat=mesh.center_lat,
            crs=crs, datum=datum
        )
        polygons_km[i] = [lonlat_to_xy_km(
            point[1], point[2];
            center_lon=mesh.center_lon, center_lat=mesh.center_lat,
            crs=crs, datum=datum
        ) for point in polygons_lonlat[i]]
    end

    return merge(mesh, (
        centroids=centroids_lonlat,
        centroids_lonlat=centroids_lonlat,
        centroids_km=centroids_km,
        polygons=polygons_lonlat,
        polygons_lonlat=polygons_lonlat,
        polygons_km=polygons_km
    ))
end

function _subset_snowcrab_mesh(mesh, keep::AbstractVector{Bool})
    length(keep) == mesh.n_units ||
        throw(DimensionMismatch("Keep mask must match mesh units."))
    indices = findall(keep)
    isempty(indices) && throw(ArgumentError("Snow crab domain mask removed every mesh unit."))

    subset = merge(mesh, (
        centroids = mesh.centroids[indices],
        centroids_km = mesh.centroids_km[indices],
        centroids_lonlat = mesh.centroids_lonlat[indices],
        polygons = hasproperty(mesh, :polygons) ? mesh.polygons[indices] : nothing,
        polygons_km = hasproperty(mesh, :polygons_km) ? mesh.polygons_km[indices] : nothing,
        polygons_lonlat = hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat[indices] : nothing,
        n_units = length(indices),
        W = mesh.W[indices, indices],
        radius_km = mesh.radius_km,
        areas_km2 = hasproperty(mesh, :areas_km2) ? mesh.areas_km2[indices] : nothing
    ))
    if hasproperty(mesh, :W_raw)
        subset = merge(subset, (W_raw=mesh.W_raw[indices, indices],))
    end
    if hasproperty(mesh, :land_mask)
        subset = merge(subset, (land_mask=mesh.land_mask[indices],))
    end
    return subset, indices
end

function _construct_snowcrab_domain(
    tagging::DataFrame,
    sppoly_file::AbstractString;
    radius_km::Real,
    land_polygons,
    crs,
    datum
)
    sppoly = _load_snowcrab_sppoly(sppoly_file)
    anchors = Tuple{Float64, Float64}[]
    for row in eachrow(tagging)
        if !ismissing(row.lon) && !ismissing(row.lat) &&
           isfinite(Float64(row.lon)) && isfinite(Float64(row.lat))
            push!(anchors, (Float64(row.lon), Float64(row.lat)))
        end
    end
    unique!(anchors)

    bounds = sppoly.bounds
    extent_points = copy(anchors)
    if bounds !== nothing
        append!(extent_points, [
            (bounds.west, bounds.south), (bounds.west, bounds.north),
            (bounds.east, bounds.south), (bounds.east, bounds.north)
        ])
    end
    isempty(extent_points) && throw(ArgumentError("No valid snow crab input locations found."))

    candidate = construct_full_movement_domain(
        first.(extent_points), last.(extent_points);
        radius_km=radius_km, land_polygons=land_polygons, crs=crs, datum=datum
    )
    keep = _snowcrab_mesh_keep_mask(
        candidate, sppoly.geometries, anchors; bounds=bounds, crs=crs, datum=datum
    )
    mesh, _ = _subset_snowcrab_mesh(candidate, keep)
    mesh = _clip_snowcrab_mesh_to_sppoly(
        mesh, sppoly.geometries; crs=crs, datum=datum
    )
    return merge(mesh, (
        sppoly_geometries = sppoly.geometries,
        sppoly_bounds = bounds,
        anchor_points_lonlat = anchors
    ))
end

"""
    r_to_ipc(filepath::AbstractString)

Converts R serialized tabular data using the optional `RCall` extension.
Load `RCall` before importing R data files.
"""
function r_to_ipc(filepath::AbstractString)
    extension = Base.get_extension(@__MODULE__, :MovementAnalysisRCallExt)
    extension === nothing && throw(ArgumentError(
        "Reading R serialized files requires the optional RCall dependency. " *
        "Load RCall before importing R data files."
    ))
    return extension.convert_r_data(filepath)
end




# Ingest empirical telemetry data, construct 20 km hexagonal mesh, sever
# terrestrial barriers, and infill unobserved marine HSI values


"""
    snowcrab_movement_data(;
        radius_km = 15.0,
        time_interval = :daily,
        crs = nothing,
        datum = WGS84Latest,
        ref_doy = 182,
        verbose = true,
        pre_mapped = nothing,
        data_dir = nothing
    ) -> NamedTuple

High-level convenience pipeline for empirical snow crab (*Chionoecetes opilio*)
movement, telemetry, and environmental suitability data across Atlantic Canada.
Loads empirical mark-recapture encounters from JLD2 storage, constructs a unified
planar hexagonal domain tessellation covering both the Scotian Shelf and Gulf of
St. Lawrence, classifies and severs terrestrial land barriers, infills unobserved
marine Habitat Suitability Index (HSI) values via screened graph-Laplacian
Dirichlet diffusion, snaps telemetry observations to navigable marine units,
and stratifies event pairs into 4 biological demographic categories.

# Arguments
- `radius_km::Real`: Hexagon circumradius in kilometers (default `15.0`).
- `time_interval::Symbol`: Time discretization step (`:daily`, `:monthly`, `:weekly`).
- `crs`: Target coordinate reference system (defaults to local tangent projection).
- `datum`: Reference ellipsoid datum (defaults to `WGS84Latest`).
- `ref_doy::Int`: Annual survey reference day-of-year (default `182`).
- `verbose::Bool`: Enable progress and summary console output (default `true`).
- `pre_mapped`: Optional pre-computed mesh NamedTuple to bypass regeneration.
- `data_dir`: Directory containing `tagging.jld2`, `hsi.jld2`, and `sppoly.jld2`.
  Defaults to the repository directory `docs/movement/data`.

# Returns
- `NamedTuple` containing:
  - `tagging::DataFrame`: Filtered telemetry records snapped to marine units.
  - `mesh::NamedTuple`: Full-domain tessellation with centroids and polygons.
  - `W::SparseMatrixCSC{Float64, Int}`: Adjacency matrix with land severed.
  - `hsi_vec::Vector{Float64}`: Infilled full-domain spatial HSI vector.
  - `monthly_hsi::Matrix{Float64}`: Monthly discretized HSI matrix.
  - `month_lookup::Dict`: Mapping of `(year, month)` to column indices.
  - `years::Vector{Int}`: Survey year span.
  - `obs::DataFrame`: Extracted release-recapture event pairs.
  - `land_mask::Vector{Bool}`: Terrestrial barrier indicator vector.
"""
function snowcrab_movement_data(;
    radius_km     :: Real    = 15.0,
    time_interval :: Symbol  = :daily,
    crs                      = nothing,
    datum                    = WGS84Latest,
    ref_doy       :: Int     = 182,
    verbose       :: Bool    = true,
    pre_mapped               = nothing,
    data_dir      :: Union{Nothing, AbstractString} = nothing,
    tagging_file  :: Union{Nothing, AbstractString} = nothing,
    hsi_file      :: Union{Nothing, AbstractString} = nothing,
    sppoly_file   :: Union{Nothing, AbstractString} = nothing
)::NamedTuple

    dir = if data_dir !== nothing
        data_dir
    elseif isdir(joinpath(@__DIR__, "..", "data"))
        normpath(joinpath(@__DIR__, "..", "data"))
    elseif isdir(joinpath(@__DIR__, "data"))
        joinpath(@__DIR__, "data")
    else
        normpath(joinpath(@__DIR__, "..", "..", "docs", "movement", "data"))
    end

    actual_tagging = if tagging_file !== nothing
        tagging_file
    else
        # Fallback to defaults
        t_jld = joinpath(dir, "tagging.jld2")
        t_rdz = joinpath(dir, "tagging.rdz")
        t_rds = joinpath(dir, "tagging.rds")
        if isfile(t_rdz)
            t_rdz
        elseif isfile(t_rds)
            t_rds
        else
            t_jld
        end
    end

    isfile(actual_tagging) || error("Snow crab tagging file not found at: $actual_tagging")

    ext = lowercase(splitext(actual_tagging)[2])
    tagging = if ext == ".jld2"
        loaded_tag = JLD2.load(actual_tagging)
        loaded_tag isa AbstractDict ? (haskey(loaded_tag, "tagging") ? loaded_tag["tagging"] : first(values(loaded_tag))) : loaded_tag
    elseif ext in (".rdz", ".rds", ".rda", ".rdata", ".qs")
        r_to_ipc(actual_tagging)
    else
        error("Unsupported tagging file format: $ext")
    end

    actual_hsi = if hsi_file !== nothing
        hsi_file
    else
        joinpath(dir, "hsi.jld2")
    end

    actual_sppoly = if sppoly_file !== nothing
        sppoly_file
    else
        joinpath(dir, "sppoly.jld2")
    end

    snowcrab_mesh = if pre_mapped !== nothing
        pre_mapped
    elseif isfile(actual_sppoly)
        _construct_snowcrab_domain(
            tagging, actual_sppoly;
            radius_km=radius_km, land_polygons=:maritimes, crs=crs, datum=datum
        )
    else
        nothing
    end

    return prepare_movement_data(
        tagging;
        hsi_file      = isfile(actual_hsi) ? actual_hsi : nothing,
        sppoly_file   = isfile(actual_sppoly) ? actual_sppoly : nothing,
        radius_km     = radius_km,
        time_interval = time_interval,
        land_polygons = :maritimes,
        crs           = crs,
        datum         = datum,
        ref_doy       = ref_doy,
        verbose       = verbose,
        pre_mapped    = snowcrab_mesh
    )
end

# sc_data = snowcrab_movement_data(radius_km = 20.0, verbose = true)

# println("Active mark-recapture records: ", nrow(sc_data.obs))
# println("Domain units: ", sc_data.mesh.n_units, 
#         " (marine: ", count(!, sc_data.land_mask), 
#         ", land: ", sum(sc_data.land_mask), ")")


# =============================================================================
# Parameter Functions
# =============================================================================

function _bundled_config_path(name::AbstractString)
    return normpath(joinpath(@__DIR__, "..", "configs", name))
end

function _load_movement_config(config_path::AbstractString)
    path = abspath(config_path)
    raw = TOML.parsefile(path)
    valid_keys = Set(keys(TOML.parsefile(_bundled_config_path("default.toml"))))
    unknown_keys = sort!(collect(setdiff(Set(keys(raw)), valid_keys)))
    isempty(unknown_keys) || throw(ArgumentError(
        "Unknown parameter names in $path: $(join(unknown_keys, ", ")). " *
        "Use names defined in configs/default.toml."
    ))
    values = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in raw)

    if haskey(values, :depth_range)
        range = values[:depth_range]
        values[:depth_range] = isempty(range) ? nothing :
            (Float64(range[1]), Float64(range[2]))
    end
    for key in (:data_source, :model_mode, :time_interval, :path_method,
                :depth_barrier_mode, :cmap, :bathymetry_source, :rest_advantage_form)
        haskey(values, key) && (values[key] = Symbol(values[key]))
    end
    for key in (:region_labels,)
        haskey(values, key) && (values[key] = String.(values[key]))
    end
    for key in (:region_labels, :region_map, :data_dir, :tagging_file, :hsi_file,
                :sppoly_file)
        if haskey(values, key) && isempty(values[key])
            values[key] = nothing
        end
    end
    for key in (:output_dir, :data_dir, :tagging_file, :hsi_file, :sppoly_file)
        if haskey(values, key) && values[key] !== nothing
            value = String(values[key])
            values[key] = normpath(isabspath(value) ? value : joinpath(dirname(path), value))
        end
    end

    return (; values...)
end

"""
    movement_parameters_snowcrab() -> NamedTuple

Loads the generic parameters from `configs/default.toml` and overlays the
species-specific settings in `configs/snowcrab.toml`.
"""
function movement_parameters_snowcrab()
    return merge(
        movement_parameters_default(),
        _load_movement_config(_bundled_config_path("snowcrab.toml"))
    )
end

"""
    movement_parameters_default() -> NamedTuple

Returns the default analysis configuration for a generic species.
All fields can be individually overridden by merging with a
species-specific parameter function:
    params = merge(movement_parameters_default(), (max_paths = 50,))

# Returns
`NamedTuple` with:
- `data_source::Symbol`: `:simulate` or `:snowcrab`.
- `model_mode::String`: `"telemetry"`, `"telemetry_and_survey"`, or `"both"`.
- `reshard_hex::Bool`: Reshard domain to finer hexagonal lattice.
- `hex_radius_km::Real`: Cell radius for fine hexagonal lattice (km).
- `use_hydrodynamics::Bool`: Ingest 3D hydrodynamic diagnostics.
- `depth_range`: `(min_d, max_d)` traversal barrier in m, or `nothing`.
- `max_paths::Int`: Maximum individual trajectories to reconstruct.
- `path_method::Symbol`: `:astar` (default) or `:viterbi`.
- `smooth_paths::Bool`: Apply marine line-of-sight raycast smoothing.
- `compute_circuit::Bool`: Compute electrical circuit current density.
- `compute_stochastic::Bool`: Compute stochastic least-cost path ensembles.
- `compute_bottlenecks::Bool`: Compute domain-wide bottleneck index B(u).
- `compute_wavelets::Bool`: Compute Chebyshev spectral graph wavelets.
- `n_stochastic_draws::Int`: Monte Carlo draws per stochastic path.
- `hsi_se::Real`: Observation standard error on HSI (sigma).
- `n_samples::Int`: MCMC posterior draw count.
- `n_warmup::Int`: Initial draws discarded as burn-in. Metropolis-Hastings does not
  adapt, so this is a burn-in discard, not a tuning phase. It was previously
  declared here and in `configs/default.toml` but never applied to the sampler.
- `seed::Int`: Random seed for reproducibility.
- `render_html::Bool`: Export interactive HTML dashboards.
- `output_dir::String`: Output directory for all artifacts.
- `verbose::Bool`: Enable progress logging.
- `advection::Real`: Fallback pooled advection fraction when no posterior is fitted.
- `residence::Real`: Fallback pooled residence probability when no posterior is fitted.
- `gamma::Real`: Fallback pooled habitat sensitivity when no posterior is fitted.
- `species_name::String`: Display name used in dashboard titles.
- `adaptive_mesh::Bool`: Use adaptive multiresolution hexagonal mesh.
- `coarse_radius_km::Real`: Cell radius for offshore/coarse units (km).
- `fine_radius_km::Real`: Cell radius for coastal/refined units (km).
- `dynamic_kernels::Bool`: Use time-varying dynamic transition kernels.
- `hmm_smoothing::Bool`: Use global multi-segment HMM Viterbi smoothing.
- `compute_validation::Bool`: Run validation path uncertainty & connectivity.
- `run_bayesian_ensemble::Bool`: Reconstruct Bayesian MCMC path ensembles.
- `resume_from_checkpoint::Bool`: Load intermediate states if checkpoint file exists.
- `region_labels`: Human-readable labels for stock connectivity regions.
- `region_map`: Spatial mapping assigning mesh units to regions.
"""
function movement_parameters_default()
    return _load_movement_config(_bundled_config_path("default.toml"))
end
 

# =============================================================================
# Private Helpers
# =============================================================================

"""
    _parse_depth_range(depth_range) -> Union{Tuple{Float64,Float64}, Nothing}

Parses a depth range specification into a `(min, max)` float tuple, or
`nothing` when no depth constraint is desired. Accepts `Tuple`,
`AbstractVector`, or a comma/colon-separated `AbstractString` (e.g.
`"50,500"` or `"50:500"`).
"""
function _parse_depth_range(
    depth_range::_DepthRangeArg
)::Union{Tuple{Float64,Float64}, Nothing}
    depth_range === nothing && return nothing
    if depth_range isa AbstractString
        sep   = occursin(":", depth_range) ? ":" : ","
        parts = Base.split(depth_range, sep)
        length(parts) == 2 || throw(ArgumentError(
            "Invalid depth_range \"$depth_range\". Expected \"min,max\"."
        ))
        return (parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2])))
    elseif length(depth_range) >= 2
        return (Float64(depth_range[1]), Float64(depth_range[2]))
    end
    return nothing
end

"""
    _snowcrab_land_polygons(data) -> Union{Nothing, Vector}

Returns the survey-derived terrestrial barrier geometries carried on the loaded
snow crab dataset (`data.mesh.sppoly_geometries`), or `nothing` when the dataset
carries no survey geometry.

Mesh constructors that accept `land_polygons` expect geometry collections and use
the sentinels `:none`/`:maritimes`. Passing a Boolean land mask there is not
merely redundant, it is reinterpreted as a polygon collection and discarded.
"""
function _snowcrab_land_polygons(data)
    m = get(data, :mesh, nothing)
    m === nothing && return nothing
    hasproperty(m, :sppoly_geometries) || return nothing
    polys = m.sppoly_geometries
    (polys === nothing || isempty(polys)) && return nothing
    return collect(polys)
end

"""
    _extract_domain_depths(mesh, data, resharded_depths) -> Vector{Float64}

Extracts or synthesizes bathymetric depth values (m) for all mesh units.
Prioritizes: resharded fine hexagonal depths > loaded dataset depths >
mesh depth vector > synthetic bathymetric contours.
"""
function _extract_domain_depths(mesh, data, resharded_depths)::Vector{Float64}
    n = mesh.n_units
    if !isnothing(resharded_depths) && length(resharded_depths) == n
        return Float64.(resharded_depths)
    elseif hasproperty(data, :depth_vec) &&
           !isnothing(data.depth_vec) &&
           length(data.depth_vec) == n
        return Float64.(data.depth_vec)
    elseif hasproperty(mesh, :depth_vec) &&
           !isnothing(mesh.depth_vec) &&
           length(mesh.depth_vec) == n
        return Float64.(mesh.depth_vec)
    elseif hasproperty(mesh, :centroids_km) &&
           !isnothing(mesh.centroids_km) &&
           length(mesh.centroids_km) == n
        return [
            150.0 + 60.0 * sin(mesh.centroids_km[s][1] / 40.0) +
            40.0 * cos(mesh.centroids_km[s][2] / 40.0)
            for s in 1:n
        ]
    elseif hasproperty(mesh, :centroids_lonlat) &&
           !isnothing(mesh.centroids_lonlat) &&
           length(mesh.centroids_lonlat) == n
        return [
            150.0 + 60.0 * sin(
                (mesh.centroids_lonlat[s][1] + 63.0) * 5.0
            ) + 40.0 * cos(
                (mesh.centroids_lonlat[s][2] - 44.0) * 5.0
            )
            for s in 1:n
        ]
    else
        return fill(150.0, n)
    end
end
"""
    _resolve_hsi_for_time(loaded, t_decimal) -> Vector{Float64}

Returns the spatial HSI vector appropriate for decimal year `t_decimal`.

If `loaded.monthly_hsi` is non-empty and `loaded.month_lookup` contains an
entry for the (year, month) inferred from `t_decimal`, the corresponding
column of `monthly_hsi` is returned.  Otherwise the climatological mean
`loaded.hsi_vec` is returned as a fallback.

Decimal year `t_decimal` is decomposed as:
```
year  = floor(Int, t_decimal)
month = clamp(ceil(Int, (t_decimal - year) * 12), 1, 12)
```

# Arguments
- `loaded`: NamedTuple from `load_movement_data` (must contain `hsi_vec`,
  `monthly_hsi`, `month_lookup`).
- `t_decimal`: Release time in decimal years (e.g. 2018.583 ≈ August 2018).

# Returns
`Vector{Float64}` of length `n_spatial`.
"""
function _resolve_hsi_for_time(loaded, t_decimal::Real)::Vector{Float64}
    mhsi = loaded.monthly_hsi
    mlup = loaded.month_lookup
    if !isempty(mhsi) && !isempty(mlup)
        yr  = floor(Int, t_decimal)
        mo  = clamp(ceil(Int, (t_decimal - yr) * 12), 1, 12)
        col = get(mlup, (yr, mo), nothing)
        if col !== nothing && 1 <= col <= size(mhsi, 2)
            return mhsi[:, col]
        end
    end
    return loaded.hsi_vec
end

"""
    _bridge_depth_disconnected_basins(
        W_depth    :: SparseMatrixCSC,
        W_marine   :: SparseMatrixCSC,
        land_mask  :: BitVector,
        depth_mask :: BitVector
    ) -> SparseMatrixCSC

Augments the depth-constrained adjacency `W_depth` with minimal local bridge
edges that allow movement between depth-valid basins separated only by
out-of-depth marine transit zones.

## Algorithm

For each out-of-depth marine node `t` (i.e. `!land_mask[t] && depth_mask[t]`),
collect the set of its in-depth marine neighbours `N_t` in `W_marine`. Add a
symmetric bridge edge between every pair `(u, v) ∈ N_t × N_t` that is not
already adjacent in `W_depth`. Bridge weight equals the minimum non-zero entry
along column `t` of `W_marine`, preserving the adjacency scale.

This local one-hop strategy adds at most `deg(t)*(deg(t)-1)/2` edges per
transit node and is O(nnz(W_marine)) overall — no global BFS needed, no
O(n²) component-pair enumeration.
"""
function _bridge_depth_disconnected_basins(
    W_depth   ::SparseMatrixCSC{Float64, Int},
    W_marine  ::SparseMatrixCSC{Float64, Int},
    land_mask ::BitVector,
    depth_mask::BitVector
)::SparseMatrixCSC{Float64, Int}
    S           = size(W_depth, 1)
    depth_valid = BitVector(.!land_mask .& .!depth_mask)   # in-depth marine
    transit     = BitVector(.!land_mask .&  depth_mask)     # out-of-depth marine

    extra_I = Int[]
    extra_J = Int[]
    extra_V = Float64[]

    # Iterate over each transit node t; find its in-depth marine neighbours
    for t in findall(transit)
        # Collect in-depth neighbours of t in W_marine
        indepth_nbrs = Int[]
        w_min = Inf
        for ptr in W_marine.colptr[t]:(W_marine.colptr[t+1]-1)
            v = W_marine.rowval[ptr]
            if depth_valid[v]
                push!(indepth_nbrs, v)
            end
            w = W_marine.nzval[ptr]
            w > 0.0 && (w_min = min(w_min, w))
        end
        length(indepth_nbrs) < 2 && continue   # no bridge needed
        bridge_w = isinf(w_min) ? 1.0 : w_min

        # Add bridge between every pair of in-depth neighbours not yet adjacent
        for ii in 1:length(indepth_nbrs)
            u = indepth_nbrs[ii]
            for jj in (ii+1):length(indepth_nbrs)
                v = indepth_nbrs[jj]
                # Check whether u-v already adjacent in W_depth
                already = false
                for ptr in W_depth.colptr[u]:(W_depth.colptr[u+1]-1)
                    W_depth.rowval[ptr] == v && (already = true; break)
                end
                already && continue
                push!(extra_I, u); push!(extra_J, v); push!(extra_V, bridge_w)
                push!(extra_I, v); push!(extra_J, u); push!(extra_V, bridge_w)
            end
        end
    end

    isempty(extra_I) && return W_depth
    W_aug = W_depth + sparse(extra_I, extra_J, extra_V, S, S)
    dropzeros!(W_aug)
    return W_aug
end

"""
    _is_graph_reachable(W, start_node, end_node, max_k) -> Bool

Breadth-first search (BFS) on adjacency matrix `W` verifying that
`end_node` is reachable from `start_node` within `max_k` hops.
Returns `false` if either endpoint is isolated by a barrier.
"""
function _is_graph_reachable(
    W          ::AbstractMatrix{<:Real},
    start_node ::Int,
    end_node   ::Int,
    max_k      ::Int
)::Bool
    start_node == end_node && return true
    max_k <= 0             && return false
    S = size(W, 1)
    (!(1 <= start_node <= S) || !(1 <= end_node <= S)) && return false

    W_sp     = W isa SparseMatrixCSC ? W : sparse(W)
    visited  = falses(S)
    frontier = Int[start_node]
    visited[start_node] = true

    for _ in 1:max_k
        next_frontier = Int[]
        for u in frontier
            for ptr in W_sp.colptr[u]:(W_sp.colptr[u + 1] - 1)
                v = W_sp.rowval[ptr]
                v == end_node && return true
                if !visited[v]
                    visited[v] = true
                    push!(next_frontier, v)
                end
            end
        end
        frontier = next_frontier
        isempty(frontier) && break
    end
    return visited[end_node]
end

"""
    _resolve_centroids(mesh, n_spatial)
        -> (cents_planar, cents_lonlat, cents_mesh)

Resolves planar (km) and geographic (lon/lat) centroid vectors from a
mesh object, returning `(planar, lonlat, preferred_for_mapping)`.
"""
function _resolve_centroids(mesh, n_spatial)
    cents_km = hasproperty(mesh, :centroids_km) &&
               !isnothing(mesh.centroids_km) &&
               length(mesh.centroids_km) == n_spatial
    cents_ll = hasproperty(mesh, :centroids_lonlat) &&
               !isnothing(mesh.centroids_lonlat) &&
               length(mesh.centroids_lonlat) == n_spatial
    cents_c  = hasproperty(mesh, :centroids) && !isnothing(mesh.centroids)

    cents_planar = cents_km ? mesh.centroids_km :
                   cents_ll ? mesh.centroids_lonlat :
                   cents_c  ? mesh.centroids : nothing
    cents_lonlat = cents_ll ? mesh.centroids_lonlat :
                   cents_c  ? mesh.centroids : nothing
    cents_mesh   = cents_lonlat !== nothing ? cents_lonlat : cents_planar
    return (cents_planar, cents_lonlat, cents_mesh)
end

# =============================================================================
# Phase 1: Data Ingestion
# =============================================================================

"""
    load_movement_data(params) -> NamedTuple

Phase 1 of the pipeline. Loads or simulates a spatial telemetry dataset,
optionally reshards the domain to a finer regular hexagonal lattice using
LibGEOS, ingests 3D hydrodynamic bathymetry, and enforces depth-range
traversal barriers.

# Arguments
- `params`: Configuration NamedTuple from `movement_parameters_*`.
  Relevant keys: `data_source`, `reshard_hex`, `hex_radius_km`,
  `use_hydrodynamics`, `depth_range`, `seed`, `verbose`.

# Returns
`NamedTuple` with fields: `data`, `mesh`, `W`, `hsi_vec`, `monthly_hsi`,
`month_lookup`, `years`, `obs_df`, `survey_df`, `land_mask`, `n_spatial`,
`resharded_hydro`, `resharded_depths`, `parsed_depth_range`.
"""
function load_movement_data(params)::NamedTuple
    verbose = params.verbose

    verbose && println("=" ^ 72)
    verbose && println("  MovementAnalysis Pipeline")
    verbose && println("=" ^ 72)

    # -- 1a. Load or simulate dataset ----------------------------------------
    verbose && println(
        "\n[Phase 1] Ingesting dataset (source: :$(params.data_source))..."
    )
    data = if Symbol(params.data_source) == :snowcrab
        sc_rad = params.hex_radius_km != 10.0 ? params.hex_radius_km : 15.0
        try
            kw = Dict{Symbol, Any}(:radius_km => sc_rad, :verbose => verbose)
            kw[:time_interval] = get(params, :time_interval, :daily)
            kw[:ref_doy] = get(params, :ref_doy, 182)
            if hasproperty(params, :data_dir) && !isnothing(params.data_dir)
                kw[:data_dir] = params.data_dir
            end
            if hasproperty(params, :tagging_file) && !isnothing(params.tagging_file)
                kw[:tagging_file] = params.tagging_file
            end
            if hasproperty(params, :hsi_file) && !isnothing(params.hsi_file)
                kw[:hsi_file] = params.hsi_file
            end
            if hasproperty(params, :sppoly_file) && !isnothing(params.sppoly_file)
                kw[:sppoly_file] = params.sppoly_file
            end
            snowcrab_movement_data(; kw...)
        catch err
            @warn "Could not load snow crab data: $err -- using simulate."
            generate_movement_data()
        end
    elseif Symbol(params.data_source) == :simulate
        generate_movement_data()
    else
        generate_movement_data()
    end

    mesh      = data.mesh
    W         = data.W
    hsi_vec   = data.hsi_vec
    obs_df    = data.obs
    survey_df = hasproperty(data, :survey_df) ? data.survey_df : nothing
    land_mask = hasproperty(data, :land_mask) ? data.land_mask : nothing
    n_spatial = mesh.n_units

    if verbose
        println("  Spatial mesh units : $n_spatial")
        if land_mask !== nothing
            println("    Marine    : $(count(!, land_mask))")
            println("    Land      : $(sum(land_mask))")
        end
        println("  Mark-recapture obs : $(nrow(obs_df))")
    end

    # -- 1b. LibGEOS Hexagonal Resharding & Hydrodynamics --------------------
    resharded_hydro  = nothing
    resharded_depths = nothing

    # Composed unit-index mappings, one entry per mesh stage. Each stage records
    # the complete map from the immediately preceding mesh's index space onto the
    # next one, so an index is never looked up in a mesh it does not belong to.
    # `to_final` is the composition of every stage, mapping each source unit
    # directly onto the final mesh.
    src_cents_0 = collect(mesh.centroids_lonlat)
    unit_mapping = (
        source_centroids = src_cents_0,
        to_final = collect(1:length(src_cents_0)),
        stages = [(
            name     = :source,
            n_units  = length(src_cents_0),
            to_next  = collect(1:length(src_cents_0)),
        )],
    )

    # Bathymetry provenance is tracked explicitly so that HSI derived from a
    # synthetic shelf model is never reported as if it were empirical snowcrab
    # input. The empirical observation HSI is always carried across mesh changes
    # by transfer; only bathymetry-derived quantities come from the hydro dataset.
    bathymetry_provenance = (
        requested_source = :none,
        resolved_source  = :none,
        file_backed      = false,
        hsi_origin       = :none,
    )

    if params.reshard_hex || params.use_hydrodynamics
        verbose && println(
            "\n[Phase 1b] Resharding to fine hexagons via LibGEOS..."
        )
        cents_lon = [Float64(c[1]) for c in mesh.centroids_lonlat]
        cents_lat = [Float64(c[2]) for c in mesh.centroids_lonlat]
        min_lon, max_lon = extrema(cents_lon)
        min_lat, max_lat = extrema(cents_lat)

        bathy_req = get(params, :bathymetry_source, :synthetic)
        bathy_req = bathy_req === nothing ? :synthetic : Symbol(bathy_req)
        bathy = load_open_bathymetry(;
            source           = bathy_req,
            lon_range        = (min_lon - 0.2, max_lon + 0.2),
            lat_range        = (min_lat - 0.2, max_lat + 0.2),
            resolution_deg   = 0.08
        )
        bathymetry_provenance = (
            requested_source = bathy_req,
            resolved_source  = Symbol(get(bathy, :source, bathy_req)),
            file_backed      = Bool(get(bathy, :loaded_from_file, false)),
            hsi_origin       = :none,
        )
        bathy_is_empirical = bathymetry_provenance.resolved_source in
                            (:empirical, :synthetic) &&
                            bathymetry_provenance.file_backed
        if bathymetry_provenance.file_backed
            verbose && println(
                "  Bathymetry source: $(bathymetry_provenance.resolved_source) " *
                "(file-backed)."
            )
        else
            @warn "Bathymetry is not file-backed (source = " *
                  "$(bathymetry_provenance.requested_source)); HSI and depth " *
                  "derived from it are synthetic and must not be reported as " *
                  "empirical input."
            verbose && println(
                "  Bathymetry source: $(bathymetry_provenance.requested_source) " *
                "(synthetic)."
            )
        end
        hydro = extract_hydrodynamic_dataset(bathy;
            depth_levels = [0.0, 25.0, 50.0, 100.0, 175.0],
            times        = [1.0]
        )
        fine_mesh = build_hex_mesh_planar(
            bathy.grid_lon, bathy.grid_lat;
            radius_km = Float64(params.hex_radius_km)
        )
        verbose && println(
            "  Fine hexagonal units: $(fine_mesh.n_units) " *
            "(radius = $(params.hex_radius_km) km)"
        )

        P_transfer       = compute_network_transfer_matrix(
            bathy.au, fine_mesh; method = :area_weighted
        )
        fine_depths = reshard_spatial_field(P_transfer, bathy.depths)
        if Symbol(params.data_source) == :snowcrab &&
           hasproperty(data.mesh, :sppoly_geometries) &&
           !isempty(data.mesh.sppoly_geometries) &&
           !isnothing(data.mesh.sppoly_bounds)
            keep_fine = _snowcrab_mesh_keep_mask(
                fine_mesh, data.mesh.sppoly_geometries,
                data.mesh.anchor_points_lonlat;
                depths=fine_depths,
                depth_range=params.depth_range,
                bounds=data.mesh.sppoly_bounds
            )
            fine_mesh, fine_indices = _subset_snowcrab_mesh(fine_mesh, keep_fine)
            fine_mesh = _clip_snowcrab_mesh_to_sppoly(
                fine_mesh, data.mesh.sppoly_geometries
            )
            P_transfer = P_transfer[fine_indices, :]
            fine_depths = fine_depths[fine_indices]
        end
        resharded_hydro  = reshard_spatial_field(P_transfer, hydro)
        resharded_depths = fine_depths

        mesh      = fine_mesh
        W         = fine_mesh.W
        land_mask = identify_land_units(
            fine_mesh.centroids_lonlat; depth = resharded_depths
        )

        # Carry the loaded empirical HSI onto the refined mesh instead of
        # replacing it with bathymetry-derived HSI. `obs_hsi_source` is the HSI
        # that is actually attached to the observation records, and it has the
        # same length as the source mesh, so the transfer is one hop.
        hsi_vec = resharded_hydro.hsi
        hsi_origin = :bathymetry_derived
        if bathy_is_empirical && length(data.hsi_vec) == length(data.mesh.centroids_lonlat)
            hsi_vec = reshard_spatial_field(
                data.hsi_vec, data.mesh, fine_mesh
            )
            hsi_origin = :empirical_transferred
        end
        bathymetry_provenance = merge(
            bathymetry_provenance, (hsi_origin = hsi_origin,)
        )
        verbose && println("  HSI origin: $hsi_origin")

        W, hsi_vec = apply_land_barrier(fine_mesh.W, hsi_vec, land_mask)
        sever_land_crossing_edges!(
            W, fine_mesh.centroids_lonlat;
            land_polygons = _snowcrab_land_polygons(data)
        )
        n_spatial  = fine_mesh.n_units

        # Remap mark-recapture observations to active marine units. Both the
        # endpoints and the survey rows are remapped from the immediately
        # preceding mesh (the source mesh), and the unit mapping is retained.
        obs_df       = copy(obs_df)
        fine_cents   = fine_mesh.centroids_lonlat
        marine_mask  = .!land_mask .& (vec(sum(W; dims = 2)) .> 0)
        marine_units = let mu = findall(marine_mask)
            isempty(mu) ? collect(1:n_spatial) : mu
        end
        if isempty(marine_units)
            throw(ArgumentError(
                "Resharding left no navigable marine unit; observation endpoints " *
                "cannot be placed. Check bathymetry_source, hex_radius_km, and " *
                "depth_range."
            ))
        end
        marine_cents = fine_cents[marine_units]

        # Endpoints are still indexed against the source mesh here, so they are
        # remapped from the source centroids onto the fine mesh.
        rel_sub = map_to_units(
            [Float64(c[1]) for c in src_cents_0[obs_df.release]],
            [Float64(c[2]) for c in src_cents_0[obs_df.release]],
            marine_cents
        )
        rec_sub = map_to_units(
            [Float64(c[1]) for c in src_cents_0[obs_df.recapture]],
            [Float64(c[2]) for c in src_cents_0[obs_df.recapture]],
            marine_cents
        )
        obs_df.release   = marine_units[rel_sub]
        obs_df.recapture = marine_units[rec_sub]

        # Complete source -> fine mapping over every source unit. The destination
        # must be the marine subset, so the result is a valid fine index.
        src_to_marine_sub = map_to_units(
            [Float64(c[1]) for c in src_cents_0],
            [Float64(c[2]) for c in src_cents_0],
            marine_cents
        )
        src_to_marine = [marine_units[src_to_marine_sub[i]]
                         for i in 1:length(src_cents_0)]
        unit_mapping = merge(
            unit_mapping,
            (
                to_final = [src_to_marine[unit_mapping.to_final[i]]
                            for i in 1:length(src_cents_0)],
                stages = vcat(
                    unit_mapping.stages,
                    [(
                        name     = :fine,
                        n_units  = fine_mesh.n_units,
                        to_next  = src_to_marine,
                    )],
                ),
            )
        )

        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            survey_df       = copy(survey_df)
            surf_sub        = map_to_units(
                [Float64(c[1]) for c in src_cents_0[survey_df.s_idx]],
                [Float64(c[2]) for c in src_cents_0[survey_df.s_idx]],
                marine_cents
            )
            survey_df.s_idx = marine_units[surf_sub]
            survey_df.depth = resharded_depths[survey_df.s_idx]
        end
        verbose && println("  Resharding complete.")
    end

    # -- 1b-ii. Adaptive Multiresolution Hexagonal Mesh ----------------------
    if get(params, :adaptive_mesh, false)
        verbose && println(
            "\n[Phase 1b-ii] Constructing adaptive multiresolution domain..."
        )
        # The multiresolution constructor expects barrier POLYGONS. Passing the
        # Boolean land mask here was silently interpreted as a polygon collection
        # and discarded the snowcrab survey geometry.
        mesh_prev = mesh
        adaptive_polygons = _snowcrab_land_polygons(data)
        mesh = construct_adaptive_multiresolution_domain(
            [Float64(c[1]) for c in mesh.centroids_lonlat],
            [Float64(c[2]) for c in mesh.centroids_lonlat];
            coarse_radius_km = Float64(get(params, :coarse_radius_km, 25.0)),
            fine_radius_km   = Float64(get(params, :fine_radius_km, 8.0)),
            land_polygons    = adaptive_polygons
        )
        W = mesh.W
        land_mask = mesh.land_mask
        n_spatial = mesh.n_units

        # Carry HSI across the fine -> adaptive mesh hop. The adaptive mesh has a
        # different unit count and its own centroids, so a transfer is required;
        # replacing HSI with a constant would discard the habitat signal entirely.
        hsi_vec = reshard_spatial_field(
            collect(Float64, hsi_vec), mesh_prev, mesh
        )
        W, hsi_vec = apply_land_barrier(W, hsi_vec, land_mask)

        # Remap from the immediately preceding mesh (the fine mesh), not the
        # original source mesh: obs_df already holds fine-mesh indices here.
        mult_cents = mesh.centroids_lonlat
        mult_marine = findall(
            i -> !land_mask[i] && vec(sum(W; dims = 2))[i] > 0, 1:n_spatial
        )
        if isempty(mult_marine)
            throw(ArgumentError(
                "Adaptive multiresolution mesh left no navigable marine unit; " *
                "observation endpoints cannot be placed. Check coarse_radius_km, " *
                "fine_radius_km, and the land polygons."
            ))
        end
        mult_marine_cents = mult_cents[mult_marine]
        prev_cents = mesh_prev.centroids_lonlat
        rel_sub = map_to_units(
            [Float64(c[1]) for c in prev_cents[obs_df.release]],
            [Float64(c[2]) for c in prev_cents[obs_df.release]],
            mult_marine_cents
        )
        rec_sub = map_to_units(
            [Float64(c[1]) for c in prev_cents[obs_df.recapture]],
            [Float64(c[2]) for c in prev_cents[obs_df.recapture]],
            mult_marine_cents
        )
        obs_df.release   = mult_marine[rel_sub]
        obs_df.recapture = mult_marine[rec_sub]

        # Complete fine -> adaptive mapping over every unit of the previous mesh.
        prev_cents_all = collect(mesh_prev.centroids_lonlat)
        prev_to_mult_sub = map_to_units(
            [Float64(c[1]) for c in prev_cents_all],
            [Float64(c[2]) for c in prev_cents_all],
            mult_marine_cents
        )
        prev_to_mult = [mult_marine[prev_to_mult_sub[i]]
                        for i in 1:length(prev_cents_all)]
        unit_mapping = merge(
            unit_mapping,
            (
                to_final = [prev_to_mult[unit_mapping.to_final[i]]
                            for i in 1:length(src_cents_0)],
                stages = vcat(
                    unit_mapping.stages,
                    [(
                        name     = :adaptive,
                        n_units  = n_spatial,
                        to_next  = prev_to_mult,
                    )],
                ),
            )
        )
        verbose && println(
            "  Multiresolution mesh: $n_spatial units " *
            "(Fine: $(count(mesh.is_fine)))"
        )
    end

    # -- 1c. Depth Range Traversal Barrier -----------------------------------
    parsed_depth_range = _parse_depth_range(params.depth_range)

    if parsed_depth_range !== nothing
        min_d, max_d = parsed_depth_range
        verbose && println(
            "\n[Phase 1c] Enforcing depth range: [$min_d, $max_d] m..."
        )
        depths_vec   = _extract_domain_depths(mesh, data, resharded_depths)
        out_of_depth = BitVector([d < min_d || d > max_d for d in depths_vec])

        if verbose
            n_allowed = count(!, out_of_depth)
            println(
                "  Units in range : $n_allowed / $(length(depths_vec))"
            )
        end

        # depth_barrier_mode controls how out-of-depth marine nodes are treated:
        #
        #   :hsi_only (default)
        #       W uses land-only barrier (full marine connectivity preserved).
        #       Out-of-depth nodes have HSI clamped to a small floor value so
        #       the habitat-gradient term strongly discourages transit through
        #       them without creating structural disconnections.
        #
        #   :hard
        #       W uses depth+land combined barrier (hard structural constraint).
        #       Observations spanning disconnected depth-corridor basins are
        #       dropped. Use when depth imposes a true physical barrier.
        #
        #   :bridge
        #       W uses depth+land combined barrier, then adds sparse bridge
        #       edges wherever two in-depth basins share a marine transit
        #       corridor (shallow/deep). Avoids drops but can generate many
        #       extra edges when basins are numerous.
        depth_mode = Symbol(get(params, :depth_barrier_mode, :hsi_only))

        # Land-only W: always needed for reachability checks and :hsi_only mode
        land_only_bv = land_mask !== nothing ?
            BitVector(land_mask) : falses(n_spatial)
        W_marine_only, _ = apply_land_barrier(W, hsi_vec, land_only_bv)

        if depth_mode == :hsi_only
            # Structural W: land barrier only
            W, hsi_vec = apply_land_barrier(W, hsi_vec, land_only_bv)
            # Encode depth preference via HSI floor for out-of-depth nodes
            hsi_floor  = Float64(get(params, :hsi_ood_floor, 0.01))
            hsi_vec[out_of_depth .& .!land_only_bv] .= min.(
                hsi_vec[out_of_depth .& .!land_only_bv], hsi_floor
            )
            land_mask = land_only_bv   # structural mask = land only
            verbose && println(
                "  Mode :hsi_only -- depth preference via HSI floor ($hsi_floor);"*
                " W connectivity uses land-only barrier."
            )
        else
            combined_barrier = land_mask !== nothing ?
                               BitVector(land_mask .| out_of_depth) :
                               out_of_depth
            W, hsi_vec = apply_land_barrier(W, hsi_vec, combined_barrier)
            land_mask  = combined_barrier

            if depth_mode == :bridge
                W_pre = W
                land_only_snap = land_only_bv
                W = _bridge_depth_disconnected_basins(
                    W, W_marine_only, land_only_snap, out_of_depth
                )
                n_bridge_edges = div(nnz(W) - nnz(W_pre), 2)
                verbose && n_bridge_edges > 0 && println(
                    "  Added $n_bridge_edges bridge edges through out-of-depth " *
                    "marine transit zones."
                )
            end
        end

        # Identify in-depth (or in-mode) valid units for endpoint remapping
        # In :hsi_only mode valid = marine (land_mask = land-only)
        # In :hard/:bridge mode valid = in-depth marine
        valid_units = if depth_mode == :hsi_only
            findall(.!out_of_depth .& .!land_only_bv)   # prefer in-depth
        else
            findall(.!land_mask)
        end
        n_obs_orig = nrow(obs_df)

        if !isempty(valid_units)
            cents_raw = hasproperty(mesh, :centroids_lonlat) ?
                mesh.centroids_lonlat :
                hasproperty(mesh, :centroids) ? mesh.centroids : nothing

            obs_df = copy(obs_df)
            obs_df[!, :release_orig]   = copy(obs_df.release)
            obs_df[!, :recapture_orig] = copy(obs_df.recapture)

            if cents_raw !== nothing
                valid_cents = cents_raw[valid_units]

                # Combined barrier for endpoint-out-of-range check
                out_bv = depth_mode == :hsi_only ? out_of_depth : land_mask

                bad_rel_mask = [out_bv[r] for r in obs_df.release]
                bad_rec_mask = [out_bv[r] for r in obs_df.recapture]
                n_remapped_rel = count(bad_rel_mask)
                n_remapped_rec = count(bad_rec_mask)

                if n_remapped_rel > 0
                    bad_rel_idx   = findall(bad_rel_mask)
                    bad_rel_units = obs_df.release[bad_rel_idx]
                    lons = [Float64(cents_raw[u][1]) for u in bad_rel_units]
                    lats = [Float64(cents_raw[u][2]) for u in bad_rel_units]
                    local_idx = map_to_units(lons, lats, valid_cents)
                    obs_df.release[bad_rel_idx] = valid_units[local_idx]
                end

                if n_remapped_rec > 0
                    bad_rec_idx   = findall(bad_rec_mask)
                    bad_rec_units = obs_df.recapture[bad_rec_idx]
                    lons = [Float64(cents_raw[u][1]) for u in bad_rec_units]
                    lats = [Float64(cents_raw[u][2]) for u in bad_rec_units]
                    local_idx = map_to_units(lons, lats, valid_cents)
                    obs_df.recapture[bad_rec_idx] = valid_units[local_idx]
                end

                if verbose && (n_remapped_rel + n_remapped_rec) > 0
                    println(
                        "  Remapped $n_remapped_rel release / " *
                        "$n_remapped_rec recapture endpoints to nearest " *
                        "in-depth marine unit."
                    )
                end
            end
        end

        # Reachability check: use land-only W so depth-mode differences don't
        # matter here; if truly unreachable through any marine path the
        # observation is a coordinate error or genuine isolation.
        W_reach = W_marine_only   # land-only W for checking
        n_k0    = count(r -> r.k <= 0 && r.release != r.recapture, eachrow(obs_df))
        reachable_mask = [
            _is_graph_reachable(W_reach, r.release, r.recapture, r.k)
            for r in eachrow(obs_df)
        ]
        n_dropped = count(.!reachable_mask)

        if n_dropped > 0
            dropped_df = obs_df[.!reachable_mask, :]
            if verbose
                println(
                    "  Dropped $n_dropped obs unreachable via any marine route " *
                    "(likely coordinate/datum errors):"
                )
                cents_raw = hasproperty(mesh, :centroids_lonlat) ?
                    mesh.centroids_lonlat : nothing
                for r in eachrow(dropped_df)
                    rel_coord = cents_raw !== nothing ?
                        "($(round(cents_raw[r.release_orig][1]; digits=4)), " *
                        "$(round(cents_raw[r.release_orig][2]; digits=4)))" :
                        "node $(r.release_orig)"
                    rec_coord = cents_raw !== nothing ?
                        "($(round(cents_raw[r.recapture_orig][1]; digits=4)), " *
                        "$(round(cents_raw[r.recapture_orig][2]; digits=4)))" :
                        "node $(r.recapture_orig)"
                    println(
                        "    tagid=$(r.tagid)  k=$(r.k)" *
                        "  rel=$rel_coord  rec=$rec_coord"
                    )
                end
            end
        end
        obs_df = obs_df[reachable_mask, :]

        nrow(obs_df) == 0 && error(
            "Depth constraint [$min_d, $max_d] m excluded all " *
            "mark-recapture events. Widen depth_range."
        )

        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            # In :hsi_only mode survey units don't need depth filtering
            if depth_mode != :hsi_only
                survey_df = survey_df[
                    [!land_mask[s] for s in survey_df.s_idx], :
                ]
            end
        end
    end

    # -- 1d. Observation Endpoint Validity (runs after every mesh stage) ------
    # Each refinement, adaptive, and depth stage can leave an endpoint on a land
    # or isolated unit. This check is unconditional so a defect introduced in an
    # earlier stage is caught even when no depth filter was requested.
    n_spatial == length(land_mask) == size(W, 1) || error(
        "Inconsistent spatial state: n_spatial=$n_spatial, " *
        "land_mask=$(length(land_mask)), W=$(size(W, 1))."
    )
    length(hsi_vec) == n_spatial || error(
        "HSI length $(length(hsi_vec)) does not match n_spatial=$n_spatial; " *
        "the habitat field was not carried across the mesh transformation."
    )

    if nrow(obs_df) > 0
        node_degree = vec(sum(W; dims = 2))
        n_ev = nrow(obs_df)
        in_range = falses(n_ev)
        on_marine = falses(n_ev)
        for i in 1:n_ev
            r = obs_df.release[i]
            c = obs_df.recapture[i]
            ok_range = 1 <= r <= n_spatial && 1 <= c <= n_spatial
            in_range[i] = ok_range
            ok_range || continue
            on_marine[i] = !land_mask[r] && !land_mask[c] &&
                           node_degree[r] > 0 && node_degree[c] > 0
        end
        usable = in_range .& on_marine
        n_invalid = count(.!usable)
        if n_invalid > 0
            @warn "Dropping $n_invalid of $n_ev mark-recapture events whose " *
                  "endpoints are not on navigable marine units after mesh " *
                  "transformation."
            verbose && for i in findall(.!usable)
                reason = in_range[i] ? "non-marine or isolated" : "out-of-range"
                println("    tagid=$(obs_df.tagid[i])  k=$(obs_df.k[i])  " *
                        "rel=$(obs_df.release[i])  rec=$(obs_df.recapture[i])  " *
                        "$reason")
            end
            obs_df = obs_df[usable, :]
        end
        nrow(obs_df) > 0 || error(
            "No mark-recapture event has both endpoints on a navigable marine " *
            "unit after mesh transformation. Check bathymetry_source, " *
            "hex_radius_km, depth_range, and the land polygons."
        )
    end

    # Forward monthly HSI and lookup from data for time-varying kernel support
    monthly_hsi  = hasproperty(data, :monthly_hsi)  ? data.monthly_hsi  : Matrix{Float64}(undef, 0, 0)
    month_lookup = hasproperty(data, :month_lookup)  ? data.month_lookup  : Dict{Tuple{Int, Int}, Int}()
    years_vec    = hasproperty(data, :years)         ? data.years         : Int[]

    return (
        data               = data,
        mesh               = mesh,
        W                  = W,
        hsi_vec            = hsi_vec,
        monthly_hsi        = monthly_hsi,
        month_lookup       = month_lookup,
        years              = years_vec,
        obs_df             = obs_df,
        survey_df          = survey_df,
        land_mask          = land_mask,
        n_spatial          = n_spatial,
        resharded_hydro    = resharded_hydro,
        resharded_depths   = resharded_depths,
        parsed_depth_range = parsed_depth_range,
        bathymetry_provenance = bathymetry_provenance,
        unit_mapping       = unit_mapping,
    )
end

# =============================================================================
# Phase 2: Model Fitting
# =============================================================================

"""
    _sample_pooled_chain(rng, model, params, verbose)

Draw the posterior for a pooled model, honouring `n_warmup`.

`n_warmup` was previously declared in `configs/default.toml`, documented, and
parsed by the CLI, but never passed to the sampler: every reported chain was the
raw output of `sample(rng, model, MH(), n_samples)`, which includes the burn-in
trajectory. For a three-parameter posterior that is an arbitrary amount of
unburned chain feeding every downstream corridor, path, and circuit result.

Metropolis-Hastings has no adaptation, so this discards `n_warmup` initial draws
as a burn-in rather than tuning anything. That is stated rather than implied,
because a burn-in discard and an adaptation-based warmup are not the same thing
and only the latter would make the chain's convergence meaningful on its own.
"""
function _sample_pooled_chain(rng, model, params, verbose::Bool)
    n_samples = Int(get(params, :n_samples, 200))
    n_warmup = max(0, Int(get(params, :n_warmup, 0)))
    if verbose
        println("  Sampling $n_samples draws (MH, burn-in discarded: $n_warmup).")
    end
    n_warmup == 0 && return sample(rng, model, MH(), n_samples; progress = false)
    return sample(
        rng, model, MH(), n_samples + n_warmup;
        discard_initial = n_warmup, progress = false
    )
end

"""
    fit_movement_models(loaded, params) -> NamedTuple

Phase 2 of the pipeline. Fits Bayesian movement models via Turing MCMC.

Supported model modes (set via `params.model_mode`):

- `"telemetry"`: Pure categorical mark-recapture transition likelihood.
    recapture ~ Categorical(P_g^k[release, :])
- `"telemetry_and_survey"`: Joint NegBin survey density + telemetry.
    density ~ NegBin(exp(eta_s), r), where eta_s drives advection A_g(eta).
- `"both"`: Fits both models.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `params`: Configuration NamedTuple. Relevant keys: `model_mode`,
  `n_samples`, `seed`, `verbose`.

# Returns
"""
`NamedTuple` with `models::Dict` and `chains::Dict`.
"""

function fit_movement_models(loaded, params)::NamedTuple
    verbose   = params.verbose
    mode_str  = lowercase(string(params.model_mode))
    fit_tel   = mode_str in ("telemetry", "both")
    fit_joint = mode_str in ("telemetry_and_survey", "both")
    rng       = MersenneTwister(params.seed)
    models    = Dict{Symbol, Any}()
    chains    = Dict{Symbol, Any}()

    # -- Pure Telemetry Model ------------------------------------------------
    if fit_tel
        verbose && println("\n[Phase 2] Fitting Pure Telemetry model...")
        verbose && println(
            "  recapture ~ Categorical(P_g^k[release, :])"
        )
        obs_df     = loaded.obs_df
        releases   = Int.(obs_df.release)
        recaptures = Int.(obs_df.recapture)
        ks         = Int.(round.(obs_df.k))

        m_tel = pure_telemetry_turing_model(
            releases, recaptures, ks,
            loaded.W, loaded.hsi_vec, loaded.land_mask;
            rest_coupling = Float64(get(params, :rest_coupling, 0.0)),
            rest_advantage_form = get(params, :rest_advantage_form, :difference),
        )
        models[:telemetry] = m_tel
        chn = _sample_pooled_chain(rng, m_tel, params, verbose)
        chains[:telemetry] = chn
        verbose && println("  Pure telemetry model complete.")
    end

    # -- Joint Survey + Telemetry Model --------------------------------------
    if fit_joint && !isnothing(loaded.survey_df)
        verbose && println(
            "\n[Phase 3] Fitting Joint Survey + Telemetry model..."
        )
        verbose && println("  density ~ NegBin(exp(eta_s), r)")

        obs_df     = loaded.obs_df
        releases   = Int.(obs_df.release)
        recaptures = Int.(obs_df.recapture)
        ks         = Int.(round.(obs_df.k))

        survey_df  = loaded.survey_df
        counts     = Int.(round.(survey_df.density))
        depths     = hasproperty(survey_df, :depth) ?
                     Float64.(survey_df.depth) : zeros(Float64, length(counts))

        m_joint = joint_survey_telemetry_turing_model(
            counts, depths,
            releases, recaptures, ks,
            loaded.W, loaded.hsi_vec, loaded.land_mask;
            rest_coupling = Float64(get(params, :rest_coupling, 0.0)),
            rest_advantage_form = get(params, :rest_advantage_form, :difference),
        )
        models[:telemetry_and_survey] = m_joint
        chn_j = _sample_pooled_chain(rng, m_joint, params, verbose)
        chains[:telemetry_and_survey] = chn_j
        verbose && println("  Joint model complete.")
    end

    return (models = models, chains = chains)
end

# =============================================================================
# Phase 3: Kernel Construction
# =============================================================================

"""
    extract_transition_kernels(loaded, fitted, params) -> NamedTuple

Phase 3 of the pipeline. Extracts pooled posterior parameters and constructs
one transition kernel for all observations:

    P = (1 - rho)[(1 - alpha) T_diff + alpha A(eta)] + rho I

# Arguments
- `loaded`: Output of `load_movement_data`.
- `fitted`: Output of `fit_movement_models`.
- `params`: Configuration NamedTuple with fallback `advection`, `residence`,
  and `gamma` values.

# Returns
`NamedTuple` with one `P_kernel` and scalar parameter estimates/samples.
"""
function extract_transition_kernels(loaded, fitted, params)::NamedTuple
    verbose = params.verbose
    chains  = fitted.chains

    active_chain = if haskey(chains, :telemetry)
        chains[:telemetry]
    elseif haskey(chains, :telemetry_and_survey)
        chains[:telemetry_and_survey]
    elseif !isempty(chains)
        first(values(chains))
    else
        nothing
    end

    if active_chain === nothing
        alpha_samples = [clamp(Float64(get(params, :advection, 0.4)), 0.0, 1.0)]
        rho_samples = [clamp(Float64(get(params, :residence, 0.25)), 0.01, 0.95)]
        gamma_samples = [Float64(get(params, :gamma, 1.0))]
    else
        function posterior_draws(parameter::Symbol, fallback::Float64)
            parameter in keys(active_chain) || return [fallback]
            values = vec(Float64.(Array(active_chain[parameter])))
            return isempty(values) ? [fallback] : values
        end
        velocity_samples = posterior_draws(:velocity, 0.3)
        diffusion_samples = posterior_draws(:diffusion, 0.1)
        gamma_samples = posterior_draws(:gamma, Float64(get(params, :gamma, 1.0)))
        n_draws = minimum((length(velocity_samples), length(diffusion_samples), length(gamma_samples)))
        velocity_samples = velocity_samples[1:n_draws]
        diffusion_samples = diffusion_samples[1:n_draws]
        gamma_samples = gamma_samples[1:n_draws]
        total_samples = velocity_samples .+ diffusion_samples .+ 1e-6
        alpha_samples = clamp.(velocity_samples ./ total_samples, 0.0, 1.0)
        rho_samples = clamp.(1.0 ./ (1.0 .+ total_samples), 0.01, 0.95)
    end
    alpha_hat = mean(alpha_samples)
    rho_hat = mean(rho_samples)
    gamma_hat = mean(gamma_samples)

    if verbose
        println("\n[Phase 3] Pooled posterior parameters:")
        println("  alpha=$(round(alpha_hat; digits=4)), rho=$(round(rho_hat; digits=4)), gamma=$(round(gamma_hat; digits=4))")
    end

    P_kernel = construct_stochastic_transition_kernel(
        loaded.W, loaded.hsi_vec;
        gamma     = gamma_hat,
        residence = rho_hat,
        advection = alpha_hat,
        rest_coupling = Float64(get(params, :rest_coupling, 0.0)),
        rest_advantage_form = get(params, :rest_advantage_form, :difference),
        land_mask = loaded.land_mask
    )

    return (
        P_kernel        = P_kernel,
        alpha_hat       = alpha_hat,
        rho_hat         = rho_hat,
        gamma_hat       = gamma_hat,
        alpha_samples   = alpha_samples,
        rho_samples     = rho_samples,
        gamma_samples   = gamma_samples,
        active_chain    = active_chain,
    )
end

# =============================================================================
# Phase 4: Path Reconstruction & Bottleneck Detection
# =============================================================================

"""
    reconstruct_paths_and_diagnostics(loaded, kernels, params) -> NamedTuple

Phase 4 of the pipeline. Reconstructs individual movement trajectories and
Markov bridge corridor heatmaps, computes optional stochastic least-cost
path ensembles, and assembles domain-wide Bayesian posterior averages for
bottleneck detection:

    F_domain(u, v) = sum_i w_i E_i(u, v)     [directed migration flux]
    C_domain(u)    = sum_i w_i rho_i(u)       [nodal transit density]
    B(u) = C_domain(u) / max(1, deg_marine(u))  [bottleneck index]

# Arguments
- `loaded`: Output of `load_movement_data`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Relevant keys: `max_paths`, `path_method`, `smooth_paths`,
  `compute_stochastic`, `compute_bottlenecks`, `n_stochastic_draws`,
  `hsi_se`, `seed`, `verbose`.

# Returns
`NamedTuple` with: `paths`, `corridors`, `stochastic_paths`,
`domain_bottlenecks`, `cents_planar`, `cents_lonlat`, `cents_mesh`.
"""
function reconstruct_paths_and_diagnostics(
    loaded, kernels, params
)::NamedTuple
    verbose     = params.verbose
    obs_df      = loaded.obs_df
    W           = loaded.W
    hsi_vec     = loaded.hsi_vec
    land_mask   = loaded.land_mask
    n_spatial   = loaded.n_spatial
    P_kernel    = kernels.P_kernel

    cents_planar, cents_lonlat, cents_mesh =
        _resolve_centroids(loaded.mesh, n_spatial)

    all_tags    = unique(obs_df.tagid)
    n_sample    = min(params.max_paths, length(all_tags))
    sample_tags = all_tags[1:n_sample]

    verbose && println(
        "\n[Phase 4] Reconstructing $(params.path_method) trajectories " *
        "for $n_sample / $(length(all_tags)) individuals..."
    )

    max_k_dyn = 0
    if get(params, :dynamic_kernels, false)
        for tid in sample_tags
            sub_obs = filter(:tagid => ==(tid), obs_df)
            isempty(sub_obs) && continue
            max_k_dyn = max(max_k_dyn, first(sub_obs).k)
        end
    end

    P_dyn_seq_cache = nothing
    if max_k_dyn > 0
        hsi_dyn_all = [
            clamp.(hsi_vec .+ 0.05 * sin(t * π / 2), 0.01, 1.0)
            for t in 1:max_k_dyn
        ]
        P_dyn_seq_cache = construct_dynamic_transition_kernels(
            W, hsi_dyn_all; land_mask = land_mask
        )
    end

    P_seg_cache = Dict{Float64, Any}()

    reconstructed_paths     = Dict{String, Vector{Int}}()
    reconstructed_corridors = Dict{String, Matrix{Float64}}()
    stochastic_paths        = Dict{String, Any}()
    forward_ibm_paths       = Dict{String, Vector{Int}}()

    for tid in sample_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue

                P_k = P_kernel

        # Concatenate multi-segment trajectories for this individual.
        # Segment-wise routing is the default reconstruction. HMM smoothing, when
        # enabled, replaces it only if a global decode actually succeeded; an
        # inadmissible decode falls back here rather than storing a fabricated path.
        fpath = Int[sub_obs.release[1]]
        for row in eachrow(sub_obs)
            # Select the monthly HSI slice for this segment's release time
            t_rel  = hasproperty(row, :rel_time) ? row.rel_time : NaN
            hsi_row = isnan(t_rel) ? hsi_vec :
                      _resolve_hsi_for_time(loaded, t_rel)

            # Build a segment-specific kernel if HSI differs from the
            # pre-computed P_k (time-varying case)
            P_seg = if hsi_row === hsi_vec
                P_k
            else
                get!(P_seg_cache, Float64(t_rel)) do
            construct_stochastic_transition_kernel(
                W, hsi_row;
                gamma     = kernels.gamma_hat,
                residence = kernels.rho_hat,
                advection = kernels.alpha_hat,
                rest_coupling = Float64(get(params, :rest_coupling, 0.0)),
                rest_advantage_form = get(params, :rest_advantage_form, :difference),
                land_mask = land_mask
            )
                end
            end

            seg = if hasproperty(loaded.mesh, :is_fine)
                astar_multiresolution_path(
                    loaded.mesh, row.release, row.recapture;
                    hsi = hsi_row, land_mask = land_mask
                )
            else
                predict_path(
                    P_seg, row.release, row.recapture, row.k;
                    centroids = cents_mesh,
                    method    = params.path_method,
                    land_mask = land_mask
                )
            end
            if isempty(seg)
                @warn "Path reconstruction: no valid route for tag $tid " *
                      "($(row.release) -> $(row.recapture), k=$(row.k)); " *
                      "the segment is omitted rather than padded."
                continue
            end
            append!(fpath, seg[2:end])
        end

        if get(params, :hmm_smoothing, false) && nrow(sub_obs) > 1
            times = Int[1]
            cum_t = 1
            locs = Any[sub_obs.release[1]]
            for r in eachrow(sub_obs)
                cum_t += max(1, r.k)
                push!(times, cum_t)
                push!(locs, r.recapture)
            end
            res_hmm = viterbi_hmm_path_smoothing(
                times, locs, P_k;
                mesh = loaded.mesh, land_mask = land_mask
            )
            if get(res_hmm, :valid, true) && !isempty(res_hmm.path)
                fpath = res_hmm.path
            else
                @warn "Path reconstruction: HMM smoothing could not decode tag $tid; " *
                      "the segment-wise route is used instead."
            end
        end

        full_path = fpath
        if params.smooth_paths && cents_mesh !== nothing
            full_path = smooth_marine_path(full_path, cents_mesh)
        end
        reconstructed_paths[string(tid)] = full_path

        # Generate unconditioned forward IBM path
        fwd_ibm = simulate_forward_ibm(P_k, first(sub_obs).release, sum(sub_obs.k); land_mask=land_mask)
        forward_ibm_paths[string(tid)] = fwd_ibm

        # Markov bridge corridor heatmap for the first segment
        first_row = first(sub_obs)
        t_rel_first = hasproperty(first_row, :rel_time) ? first_row.rel_time : NaN
        hsi_first   = isnan(t_rel_first) ? hsi_vec :
                      _resolve_hsi_for_time(loaded, t_rel_first)
        prop_hsi  = get(params, :propagate_hsi_error, true) && params.hsi_se > 0.0
        n_hsi_m   = prop_hsi ? min(params.n_stochastic_draws, 5) : 1

        corr_accum = nothing
        n_valid_corr = 0
        for d in 1:n_hsi_m
            P_eval = if prop_hsi && d > 1
                rng_d = MersenneTwister(
                    params.seed + Int(_stable_string_hash(string(tid)) % 10_000) + d
                )
                hsi_d = clamp.(
                    hsi_first .+ randn(rng_d, n_spatial) .* params.hsi_se,
                    0.001, 1.0
                )
                construct_stochastic_transition_kernel(
                    W, hsi_d;
                    gamma     = kernels.gamma_hat,
                    residence = kernels.rho_hat,
                    advection = kernels.alpha_hat,
                    land_mask = land_mask
                )
            else
                P_k
            end
            corr_d = if get(params, :dynamic_kernels, false)
                k_val = max(1, first_row.k)
                P_dyn_seq = P_dyn_seq_cache[1:k_val]
                predict_dynamic_corridor(
                    P_dyn_seq, first_row.release, first_row.recapture;
                    land_mask = land_mask
                )
            else
                predict_corridor(
                    P_eval, first_row.release, first_row.recapture,
                    first_row.k; land_mask = land_mask
                )
            end
            if !any(isnan, corr_d)
                if corr_accum === nothing
                    corr_accum = zeros(Float64, size(corr_d)...)
                end
                corr_accum .+= corr_d
                n_valid_corr += 1
            end
        end
        reconstructed_corridors[string(tid)] = n_valid_corr > 0 ?
            corr_accum ./ n_valid_corr :
            (isnothing(corr_accum) ? zeros(Float64, n_spatial, max(1, first_row.k) + 1) : fill(NaN, size(corr_accum)))

        # Optional stochastic least-cost path ensemble
        if params.compute_stochastic && cents_planar !== nothing
            try
                stoch_res = astar_stochastic_least_cost_path(
                    cents_planar, W,
                    first_row.release, first_row.recapture;
                    hsi_mean         = hsi_vec,
                    hsi_se           = fill(params.hsi_se, n_spatial),
                    n_draws          = params.n_stochastic_draws,
                    friction_power   = 2.0,
                    land_mask        = land_mask,
                    centroids_lonlat = cents_lonlat,
                    smooth           = params.smooth_paths,
                    seed             = Int(
                        params.seed + Int(_stable_string_hash(string(tid)) % 10_000)
                    )
                )
                stochastic_paths[string(tid)] = stoch_res
                d = stoch_res.mean_distance
                dist_km = d > 10_000.0 ? d / 1_000.0 : d
                verbose && println(
                    "  Tag $tid stochastic: " *
                    "$(length(stoch_res.medoid_path)) hops, " *
                    "$(round(dist_km; digits=1)) km"
                )
            catch e
                verbose && println("  (Stochastic A* note [$tid]: $e)")
            end
        end

        verbose && println(
            "  Tag $tid: " *
            "$(length(full_path)) units " *
            "($(first(full_path)) -> $(last(full_path)))"
        )
    end

    # -- 4b. Domain-Wide Posterior Averaging & Bottleneck Detection ----------
    domain_bottlenecks = nothing

    if params.compute_bottlenecks && cents_planar !== nothing
        verbose && println("\n[Phase 4b] Domain bottleneck detection...")
        n_obs              = nrow(obs_df)
        tag_sample_indices = Int[]

        cand = findall(
            i -> obs_df.release[i] != obs_df.recapture[i] &&
                 (land_mask === nothing ||
                  (!land_mask[obs_df.release[i]] &&
                   !land_mask[obs_df.recapture[i]])),
            1:n_obs
        )
        n_take = min(25, length(cand))
        n_take > 0 && append!(tag_sample_indices, cand[1:n_take])
        isempty(tag_sample_indices) &&
            append!(tag_sample_indices, collect(1:min(10, n_obs)))

        verbose && println(
            "  Aggregating $(length(tag_sample_indices)) events..."
        )

        C_domain = zeros(Float64, n_spatial)
        C_events = zeros(Float64, length(tag_sample_indices), n_spatial)
        F_domain = spzeros(Float64, n_spatial, n_spatial)
        hsi_se_v = fill(params.hsi_se, n_spatial)

        for (e_idx, idx) in enumerate(tag_sample_indices)
            u_rel = obs_df.release[idx]
            u_rec = obs_df.recapture[idx]
            res_i = astar_stochastic_least_cost_path(
                cents_planar, W, u_rel, u_rec;
                hsi_mean         = hsi_vec,
                hsi_se           = hsi_se_v,
                n_draws          = params.n_stochastic_draws,
                friction_power   = 2.0,
                land_mask        = land_mask,
                centroids_lonlat = cents_lonlat,
                smooth           = params.smooth_paths,
                seed             = params.seed + idx
            )
            C_domain .+= res_i.corridor_prob
            C_events[e_idx, :] .= res_i.corridor_prob
            F_domain .+= res_i.edge_prob
        end

        # Structural bottleneck index: B(u) = C(u) / deg_marine(u)
        deg_marine = [
            count(
                j -> W[i, j] > 0 &&
                     (land_mask === nothing || !land_mask[j]),
                1:n_spatial
            )
            for i in 1:n_spatial
        ]
        bottleneck_score = zeros(Float64, n_spatial)
        bottleneck_se    = zeros(Float64, n_spatial)
        n_ev = length(tag_sample_indices)

        for u in 1:n_spatial
            is_marine = land_mask === nothing ? true : !land_mask[u]
            if is_marine && deg_marine[u] > 0
                bottleneck_score[u] = C_domain[u] / deg_marine[u]
                if n_ev > 1
                    scores_u = [C_events[e, u] / deg_marine[u] for e in 1:n_ev]
                    bottleneck_se[u] = std(scores_u) / sqrt(n_ev)
                end
            end
        end

        marine_idx = land_mask === nothing ?
                     collect(1:n_spatial) : findall(!, land_mask)
        pos_scores = filter(>(0.0), bottleneck_score[marine_idx])
        b_thresh   = isempty(pos_scores) ? 0.0 : quantile(pos_scores, 0.90)
        bottleneck_mask = (bottleneck_score .>= b_thresh) .&
                          (bottleneck_score .> 0.0) .&
                          (land_mask === nothing ?
                           trues(n_spatial) : .!land_mask)

        if verbose
            println(
                "  Peak transit density: " *
                "$(round(maximum(C_domain); digits=2)) tag equiv."
            )
            println("  Active directed edges: $(nnz(F_domain))")
            println(
                "  Top-10% bottleneck threshold: " *
                "$(round(b_thresh; digits=4))"
            )
            println(
                "  Critical bottleneck units: " *
                "$(sum(bottleneck_mask)) / $n_spatial"
            )
        end

        domain_bottlenecks = (
            transit_density  = C_domain,
            edge_flux        = F_domain,
            bottleneck_score = bottleneck_score,
            bottleneck_se    = bottleneck_se,
            bottleneck_mask  = bottleneck_mask,
            threshold        = b_thresh,
        )
    end

    return (
        paths              = reconstructed_paths,
        corridors          = reconstructed_corridors,
        stochastic_paths   = stochastic_paths,
        forward_ibm_paths  = forward_ibm_paths,
        domain_bottlenecks = domain_bottlenecks,
        cents_planar       = cents_planar,
        cents_lonlat       = cents_lonlat,
        cents_mesh         = cents_mesh,
    )
end

# =============================================================================
# Phase 5: Advanced Diagnostics (Circuit Theory & Wavelets)
# =============================================================================

"""
    compute_advanced_diagnostics(loaded, path_results, params) -> NamedTuple

Phase 5 of the pipeline. Optionally computes:

**Circuit Theory** (`params.compute_circuit = true`):
- Multi-pair electrical current density I = C nabla V across all
  mark-recapture source-sink pairs.
- Identifies ecological pinch-points (top 10% current density).
- Posterior circuit inference via Monte Carlo HSI sampling:
    HSI_draw ~ N(hsi_mean, hsi_se^2)
  over `n_stochastic_draws * 2` replicates.

**Spectral Graph Wavelets** (`params.compute_wavelets = true`):
- Multi-scale Chebyshev SGWT decomposition on HSI (3 scales, order 25).
- BayesShrink adaptive soft-threshold spatial denoising.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `path_results`: Output of `reconstruct_paths_and_diagnostics`.
- `params`: Relevant keys: `compute_circuit`, `compute_wavelets`,
  `n_stochastic_draws`, `hsi_se`, `seed`, `verbose`.

# Returns
`NamedTuple` with `circuit` and `wavelets` (each `nothing` if not computed).
"""
function compute_advanced_diagnostics(
    loaded, path_results, params
)::NamedTuple
    verbose   = params.verbose
    W         = loaded.W
    hsi_vec   = loaded.hsi_vec
    land_mask = loaded.land_mask
    n_spatial = loaded.n_spatial
    obs_df    = loaded.obs_df
    sources_v = Int.(obs_df.release)
    sinks_v   = Int.(obs_df.recapture)
    hsi_se_v  = fill(params.hsi_se, n_spatial)

    circuit_res::Union{NamedTuple, Nothing} = nothing
    wavelet_res::Union{NamedTuple, Nothing} = nothing

    # -- Circuit Theory ------------------------------------------------------
    if params.compute_circuit
        verbose && println(
            "\n[Phase 5a] Computing Circuit Theory Current Density..."
        )
        try
            cur_dens, _, _, _ = current_density_map(
                W, sources_v, sinks_v;
                hsi       = hsi_vec,
                land_mask = land_mask
            )
            p_mask, p_score, p_thresh = identify_ecological_pinchpoints(
                cur_dens; top_quantile = 0.90
            )
            verbose && println(
                "  Current density range: " *
                "[$(round(minimum(cur_dens); digits=4)), " *
                "$(round(maximum(cur_dens); digits=4))]"
            )
            verbose && println(
                "  Critical pinch-points (top 10%): " *
                "$(sum(p_mask)) / $n_spatial"
            )

            stoch_circuit = posterior_circuit_inference(
                W, sources_v, sinks_v;
                hsi_mean     = hsi_vec,
                hsi_se       = hsi_se_v,
                n_draws      = params.n_stochastic_draws * 2,
                land_mask    = land_mask,
                top_quantile = 0.90,
                seed         = params.seed
            )
            robust_mask, _, n_robust = identify_stochastic_pinchpoints(
                stoch_circuit; prob_threshold = 0.80
            )
            verbose && println(
                "  Robust pinch-points (P >= 80%): $n_robust / $n_spatial"
            )

            circuit_res = (
                current_density = cur_dens,
                pinch_mask      = p_mask,
                pinch_score     = p_score,
                threshold       = p_thresh,
                stochastic      = stoch_circuit,
                robust_mask     = robust_mask,
                n_robust        = n_robust,
            )
        catch e
            verbose && println("  (Circuit computation note: $e)")
        end
    end

    # -- Chebyshev Spectral Graph Wavelets -----------------------------------
    if params.compute_wavelets
        verbose && println(
            "\n[Phase 5b] Multi-Scale Chebyshev Spectral Graph Wavelets..."
        )
        try
            res_hsi = spectral_graph_wavelet_transform(
                W, hsi_vec; num_scales = 3, order = 25
            )
            verbose && println(
                "  SGWT decomposed across 3 scales (Chebyshev order 25)."
            )

            noisy_hsi = hsi_vec .+ 0.10 .* randn(
                MersenneTwister(params.seed + 101), n_spatial
            )
            hsi_clean, _, sigma_est = denoise_spatial_signal_wavelet(
                W, noisy_hsi; threshold_rule = :bayesshrink
            )
            verbose && println(
                "  BayesShrink noise sigma: $(round(sigma_est; digits=4))"
            )

            wavelet_res = (
                sgwt            = res_hsi,
                denoised_hsi    = hsi_clean,
                estimated_noise = sigma_est,
            )
        catch e
            verbose && println("  (Wavelet decomposition note: $e)")
        end
    end

    return (circuit = circuit_res, wavelets = wavelet_res)
end

# =============================================================================
# Phase 6: Dashboard Export
# =============================================================================

# General movement ecology metrics (compute_movement_statistics,
# analyze_seasonal_movement_phenology, model_trait_movement_associations,
# export_movement_summary_csv) and standalone SVG/HTML dashboard exporters
# (export_movement_posterior_dashboard, export_movement_flow_dashboard,
# export_movement_summary_dashboard) have been consolidated into src/movement.jl
# and are exported by the core MovementAnalysis module.


function _agent_trajectories_to_paths(agent_trajectories, mesh, hsi_vec)
    agent_trajectories === nothing && return NamedTuple[]
    centroids = hasproperty(mesh, :centroids_lonlat) ?
        mesh.centroids_lonlat : mesh.centroids
    paths = NamedTuple[]

    for agent_rows in groupby(agent_trajectories, :tagid)
        rows = sort(agent_rows, :step)
        units = Int.(rows.mesh_unit)
        length(units) < 2 && continue
        coords = Tuple{Float64, Float64}[centroids[u] for u in units if 1 <= u <= length(centroids)]
        length(coords) < 2 && continue
        distance_km = sum(
            haversine_distance(coords[i - 1][1], coords[i - 1][2], coords[i][1], coords[i][2]) / 1000.0
            for i in 2:length(coords)
        )
        displacement_km = haversine_distance(
            first(coords)[1], first(coords)[2], last(coords)[1], last(coords)[2]
        ) / 1000.0
        valid_units = filter(u -> 1 <= u <= length(hsi_vec), units)
        push!(paths, (
            tagid = "Agent $(first(rows.tagid))",
            path = units,
            coords = coords,
            n_steps = length(coords) - 1,
            total_dist_km = distance_km,
            displacement_km = displacement_km,
            tortuosity = displacement_km > 0.01 ? distance_km / displacement_km : 1.0,
            mean_hsi = isempty(valid_units) ? NaN : mean(hsi_vec[valid_units]),
            color = "#f97316",
            trajectory_kind = "Agent simulation"
        ))
    end
    return paths
end

function _write_agent_simulation_summary(paths, statistics, output_dir::AbstractString)
    isempty(paths) && return nothing
    mkpath(output_dir)
    summary_file = joinpath(output_dir, "agent_simulation_summary.csv")
    open(summary_file, "w") do io
        write(io, "agent_id,steps,total_dist_km,displacement_km,efficiency,tortuosity,bearing_deg\n")
        for (i, path) in enumerate(paths)
            write(io, string(
                path.tagid, ",", path.n_steps, ",",
                round(path.total_dist_km; digits=3), ",",
                round(path.displacement_km; digits=3), ",",
                round(statistics.path_efficiency[i]; digits=4), ",",
                round(statistics.tortuosity[i]; digits=4), ",",
                round(statistics.bearings_deg[i]; digits=1), "\n"
            ))
        end
    end
    return summary_file
end

"""
    export_dashboards(loaded, kernels, path_results, diagnostics, params)

Phase 6 of the pipeline. Exports interactive Leaflet HTML dashboards to
`params.output_dir`. Individual dashboards are silently skipped on
rendering errors so the pipeline is never aborted. Exports:

- Movement path trajectories and empirical displacement vectors.
- Interactive two-click migration corridor explorer.
- Hydrodynamic stratification dashboard (when domain was resharded).
- Circuit current density and posterior pinch-point maps.
- Domain-wide bottleneck conduit heatmap.
- Multi-scale SGWT wavelet decomposition dashboard.
"""
function export_dashboards(
    loaded, kernels, path_results, diagnostics, params;
    agent_paths::AbstractVector{<:NamedTuple} = NamedTuple[]
)::Union{NamedTuple, Nothing}
    params.render_html || return nothing
    verbose = params.verbose

    verbose && println("\n[Phase 6] Generating Leaflet dashboards...")
    out_dir = params.output_dir
    mkpath(out_dir)

    mesh        = loaded.mesh
    hsi_vec     = loaded.hsi_vec
    obs_df      = loaded.obs_df
    P_kernel    = kernels.P_kernel
    spp         = params.species_name

    # Use the resolved centroids from path reconstruction so node
    # indices are consistent with the (possibly resharded) mesh.
    cents_ll = path_results.cents_lonlat !== nothing ?
               path_results.cents_lonlat :
               (hasproperty(mesh, :centroids_lonlat) ?
                mesh.centroids_lonlat : mesh.centroids)
    n_units = length(cents_ll)

    polys_ll = hasproperty(mesh, :polygons_lonlat) ?
               mesh.polygons_lonlat :
               (hasproperty(mesh, :polygons) ?
                mesh.polygons : nothing)
    au_mesh  = (
        centroids        = cents_ll,
        centroids_lonlat = cents_ll,
        polygons         = polys_ll,
        polygons_lonlat  = polys_ll,
        W                = loaded.W,
        n_units          = n_units,
    )

    # Empirical release-recapture displacement vectors
    emp_tracks = [
        [
            (Float64(cents_ll[r.release][1]),
             Float64(cents_ll[r.release][2])),
            (Float64(cents_ll[r.recapture][1]),
             Float64(cents_ll[r.recapture][2])),
        ]
        for r in eachrow(obs_df)
        if 1 <= r.release   <= n_units &&
           1 <= r.recapture <= n_units
    ]

    depth_lbl = if !isnothing(loaded.parsed_depth_range)
        min_d, max_d = loaded.parsed_depth_range
        " [Depth: $(round(Int, min_d))-$(round(Int, max_d)) m]"
    else
        ""
    end
    reshard_lbl = (params.reshard_hex || params.use_hydrodynamics) ?
                  " (Fine Hexagons)" : ""

    # -- Build rich path NamedTuples for the tracks dashboard ------
    #
    # Convert raw Dict{String, Vector{Int}} into NamedTuples with
    # centroid coordinates, per-path distance, displacement,
    # tortuosity and mean HSI.
    palette_colors = [
        "#38bdf8", "#f43f5e", "#10b981", "#fbbf24",
        "#a78bfa", "#fb923c", "#22d3ee", "#e879f9",
    ]
    all_paths_rich = NamedTuple[]
    path_dists_km  = Float64[]
    path_vels      = Float64[]
    path_bearings  = Float64[]

    for (path_index, tid) in enumerate(keys(path_results.paths))
        node_vec = path_results.paths[tid]
        length(node_vec) < 2 && continue

        # Coordinate series via resolved centroids
        coords = Tuple{Float64, Float64}[
            (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
            for u in node_vec
            if 1 <= u <= n_units
        ]
        length(coords) < 2 && continue

        # Cumulative Haversine distance along the path (km)
        total_dist = 0.0
        for h in 2:length(coords)
            total_dist += haversine_distance(
                coords[h-1][1], coords[h-1][2],
                coords[h][1],   coords[h][2]
            ) / 1_000.0
        end

        # Net displacement (great-circle, km)
        displacement = haversine_distance(
            coords[1][1], coords[1][2],
            coords[end][1], coords[end][2]
        ) / 1_000.0

        # Tortuosity (path length / displacement); clamp for
        # coincident release-recapture
        tort = displacement > 0.01 ? total_dist / displacement : 1.0

        # Mean habitat suitability along the trajectory
        mean_h = mean([
            (1 <= u <= length(hsi_vec)) ? hsi_vec[u] : 0.5
            for u in node_vec
        ])

        sub_obs = filter(:tagid => ==(tid), obs_df)
        color = palette_colors[
            (path_index - 1) % length(palette_colors) + 1
        ]

        # Duration in time steps (sum of k across segments)
        k_total = !isempty(sub_obs) ? sum(sub_obs.k) : 1

        push!(all_paths_rich, (
            tagid           = string(tid),
            path            = node_vec,
            coords          = coords,
            n_steps         = length(coords) - 1,
            total_dist_km   = total_dist,
            displacement_km = displacement,
            tortuosity      = tort,
            mean_hsi        = mean_h,
            color           = color,
            duration_days   = Float64(k_total),
        ))

        push!(path_dists_km, total_dist)
        if k_total > 0
            push!(path_vels, total_dist / k_total)
        end

        # Bearing from release to recapture (degrees from north)
        Δlon = coords[end][1] - coords[1][1]
        Δlat = coords[end][2] - coords[1][2]
        bearing = atand(Δlon, Δlat)
        bearing < 0.0 && (bearing += 360.0)
        push!(path_bearings, bearing)
    end

    verbose && println(
        "  Rich path NamedTuples built: $(length(all_paths_rich))"
    )

    # -- Append IBM stochastic path realizations -------------------------
    # Each StochasticAStarResult stores a full ensemble of individual IBM
    # paths (all_paths::Vector{Vector{Int}}). These are added as distinct
    # lighter-coloured entries so they are visible alongside the MAP track.
    ibm_palette = [
        "#7dd3fc", "#fda4af", "#6ee7b7", "#fde68a",
        "#c4b5fd", "#fdba74", "#67e8f9", "#f0abfc",
    ]
    n_ibm_added = 0
    for (path_index, tid) in enumerate(keys(path_results.stochastic_paths))
        stoch_res = path_results.stochastic_paths[tid]
        stoch_res isa StochasticAStarResult || continue
        sub_obs = filter(:tagid => ==(tid), obs_df)
        base_color = ibm_palette[(path_index - 1) % length(ibm_palette) + 1]

        for (r_idx, node_vec) in enumerate(stoch_res.all_paths)
            length(node_vec) < 2 && continue
            coords = Tuple{Float64, Float64}[
                (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
                for u in node_vec
                if 1 <= u <= n_units
            ]
            length(coords) < 2 && continue

            total_dist = sum(
                haversine_distance(
                    coords[h-1][1], coords[h-1][2],
                    coords[h][1],   coords[h][2]
                ) / 1_000.0
                for h in 2:length(coords)
            )
            push!(all_paths_rich, (
                tagid           = string(tid) * "_ibm$r_idx",
                path            = node_vec,
                coords          = coords,
                n_steps         = length(coords) - 1,
                total_dist_km   = total_dist,
                displacement_km = haversine_distance(
                    coords[1][1], coords[1][2],
                    coords[end][1], coords[end][2]
                ) / 1_000.0,
                tortuosity      = 1.0,
                mean_hsi        = mean([
                    (1 <= u <= length(hsi_vec)) ? hsi_vec[u] : 0.5
                    for u in node_vec
                ]),
                color           = base_color,
                duration_days   = Float64(
                    !isempty(sub_obs) ? sum(sub_obs.k) : 1
                ),
            ))
            n_ibm_added += 1
        end
    end
    verbose && n_ibm_added > 0 && println(
        "  IBM stochastic realizations added: $n_ibm_added"
    )

    # -- Append Unconditioned Forward IBM paths --------------------------
    if haskey(path_results, :forward_ibm_paths)
        for (tid, node_vec) in path_results.forward_ibm_paths
            length(node_vec) < 2 && continue
            coords = Tuple{Float64, Float64}[
                (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
                for u in node_vec
                if 1 <= u <= n_units
            ]
            length(coords) < 2 && continue

            total_dist = sum(
                haversine_distance(
                    coords[h-1][1], coords[h-1][2],
                    coords[h][1],   coords[h][2]
                ) / 1_000.0
                for h in 2:length(coords)
            )

            sub_obs = filter(:tagid => ==(tid), obs_df)

            push!(all_paths_rich, (
                tagid           = string(tid) * "_forward_ibm",
                path            = node_vec,
                coords          = coords,
                n_steps         = length(coords) - 1,
                total_dist_km   = total_dist,
                displacement_km = haversine_distance(
                    coords[1][1], coords[1][2],
                    coords[end][1], coords[end][2]
                ) / 1_000.0,
                tortuosity      = 1.0,
                mean_hsi        = mean([
                    (1 <= u <= length(hsi_vec)) ? hsi_vec[u] : 0.5
                    for u in node_vec
                ]),
                color           = "#f97316", # orange
                duration_days   = Float64(
                    !isempty(sub_obs) ? sum(sub_obs.k) : 1
                ),
            ))
        end
    end

    # -- Movement paths dashboard ---------------------------------
    try
        html_file = joinpath(out_dir, "movement_paths_dashboard.html")
        map_obj = leaflet_tracks_map(
            vcat(all_paths_rich, agent_paths), au_mesh;
            empirical_paths     = emp_tracks,
            max_paths           = max(100, length(all_paths_rich)),
            max_empirical_paths = max(500, length(emp_tracks)),
            hsi                 = hsi_vec,
            title               = "$spp Movement Trajectories" *
                                  reshard_lbl * depth_lbl
        )
        save_html(map_obj, html_file)
        verbose && println("  Paths dashboard: $html_file")
    catch e
        verbose && println("  (Leaflet paths note: $e)")
    end

    # -- Forward projection explorer (agent mode) ------------------
    if get(params, :model_mode, "telemetry") == "agent"
        try
            html_file = joinpath(out_dir, "forward_projection_map.html")
            map_obj = leaflet_forward_projection_map(
                kernels.P_kernel, au_mesh;
                hsi      = hsi_vec,
                n_paths  = 40,
                n_steps  = 20,
                seed     = params.seed,
                title    = "$spp Forward Movement Projection",
                subtitle = "Click a unit to project synthetic trajectories forward " *
                           "from it and accumulate expected visit frequency",
            )
            save_html(map_obj, html_file)
            verbose && println("  Forward projection map: $html_file")
        catch e
            verbose && println("  (Forward projection note: $e)")
        end
    end

    # -- Movement ecology statistics & phenology ----------------------
    mov_stats = compute_movement_statistics(
        all_paths_rich, path_results, loaded; params = params
    )
    pheno_res = analyze_seasonal_movement_phenology(
        obs_df, mov_stats, loaded
    )
    trait_res = model_trait_movement_associations(
        obs_df, mov_stats, loaded
    )

    # -- Export movement summary CSV ----------------------------------
    csv_file = joinpath(out_dir, "movement_path_metrics.csv")
    try
        export_movement_summary_csv(
            csv_file, all_paths_rich, mov_stats, obs_df
        )
        verbose && println("  Summary CSV table: $csv_file")
    catch e
        verbose && println("  (Summary CSV note: $e)")
    end

    # -- Movement summary diagnostics dashboard -----------------------
    if !isempty(path_dists_km)
        try
            summ_file = joinpath(
                out_dir, "movement_summary_diagnostics.html"
            )
            export_movement_summary_dashboard(
                summ_file, spp, path_dists_km, path_vels,
                path_bearings, all_paths_rich, mov_stats
            )
            verbose && println(
                "  Summary diagnostics: $summ_file"
            )
        catch e
            verbose && println(
                "  (Summary diagnostics note: $e)"
            )
        end
    end

    # -- Posterior parameter uncertainty dashboard --------------------
    try
        post_file = joinpath(
            out_dir, "movement_posterior_uncertainty.html"
        )
        export_movement_posterior_dashboard(
            post_file, kernels; species = spp
        )
        verbose && println("  Posterior uncertainty: $post_file")
    catch e
        verbose && println("  (Posterior uncertainty note: $e)")
    end

    # -- Directed flow network dashboard ------------------------------
    try
        net_file = joinpath(
            out_dir, "movement_network_flow.html"
        )
        export_movement_flow_dashboard(
            net_file, path_results, loaded; species = spp
        )
        verbose && println("  Network flow diagram: $net_file")
    catch e
        verbose && println("  (Network flow note: $e)")
    end

    # -- Interactive two-click corridor explorer -----------------------------
    try
        corr_file  = joinpath(out_dir, "movement_interactive_corridor.html")
        corr_map   = leaflet_interactive_corridor_dashboard(
            P_kernel, au_mesh;
            hsi             = hsi_vec,
            empirical_paths = emp_tracks,
            title           = "$spp Dynamic Migration Corridor" *
                              reshard_lbl * depth_lbl
        )
        save_html(corr_map, corr_file)
        verbose && println("  Corridor dashboard: $corr_file")
    catch e
        verbose && println("  (Corridor dashboard note: $e)")
    end

    # -- Hydrodynamic dashboard (only when resharded) ------------------------
    if !isnothing(loaded.resharded_hydro)
        try
            hydro_file = joinpath(out_dir, "hydrodynamic_hex_dashboard.html")
            dash = leaflet_hydrodynamic_dashboard(
                loaded.resharded_hydro, au_mesh;
                title = "Hydrodynamics & Stratification (Fine Hexagons)"
            )
            save_html(dash, hydro_file)
            verbose && println("  Hydrodynamic dashboard: $hydro_file")
        catch e
            verbose && println("  (Hydrodynamic dashboard note: $e)")
        end
    end

    # -- Circuit current density & stochastic pinch-point dashboards ---------
    if !isnothing(diagnostics.circuit)
        circ = diagnostics.circuit
        try
            circ_file = joinpath(out_dir, "movement_current_density.html")
            leaflet_current_density_map(
                mesh, circ.current_density;
                pinch_mask   = circ.pinch_mask,
                pinch_score  = circ.pinch_score,
                centroids    = path_results.cents_lonlat,
                output_html  = circ_file,
                title        = "$spp Migratory Current Density & Pinch-Points"
            )
            verbose && println("  Current density dashboard: $circ_file")
        catch e
            verbose && println("  (Circuit density note: $e)")
        end

        try
            stoch_file = joinpath(out_dir, "movement_stochastic_circuit.html")
            leaflet_current_density_map(
                mesh, circ.stochastic;
                prob_threshold = 0.80,
                output_html    = stoch_file,
                title          = "$spp Posterior Migratory Flux & Pinch-Points"
            )
            verbose && println("  Stochastic circuit dashboard: $stoch_file")
        catch e
            verbose && println("  (Stochastic circuit note: $e)")
        end
    end

    # -- Domain-wide bottleneck dashboard ------------------------------------
    if !isnothing(path_results.domain_bottlenecks)
        bn = path_results.domain_bottlenecks
        try
            bn_file = joinpath(out_dir, "movement_domain_bottlenecks.html")
            leaflet_current_density_map(
                mesh, bn.transit_density;
                pinch_mask   = bn.bottleneck_mask,
                pinch_score  = bn.bottleneck_score,
                centroids    = path_results.cents_lonlat,
                output_html  = bn_file,
                title        = "$spp Domain-Wide Pathways & Bottlenecks",
                legend_title = "Transit Density (C)"
            )
            verbose && println("  Bottleneck dashboard: $bn_file")
        catch e
            verbose && println("  (Bottleneck rendering note: $e)")
        end
    end

    # -- Multi-scale wavelet dashboard ---------------------------------------
    if !isnothing(diagnostics.wavelets)
        wv = diagnostics.wavelets
        try
            wv_file = joinpath(out_dir, "movement_wavelet_dashboard.html")
            leaflet_graph_wavelet_dashboard(
                mesh, wv.sgwt;
                reconstruction = wv.denoised_hsi,
                signal_name    = "Habitat Suitability (HSI)",
                title          = "$spp Multi-Scale Habitat (HSI) Wavelets",
                output_html    = wv_file
            )
            verbose && println("  Wavelet dashboard: $wv_file")
        catch e
            verbose && println("  (Wavelet dashboard note: $e)")
        end
    end

    # -- New Missing Visualizations: Speeds, Directions, Home Range, Corridors --
    
    # 1. Step Diagnostics (Speeds, Turning Angles)
    if !isempty(path_results.paths)
        try
            step_file = joinpath(out_dir, "movement_step_diagnostics.html")
            map_obj = leaflet_step_diagnostics(
                path_results.paths, loaded.au_mesh;
                title = "$spp Speeds and Directions Distributions"
            )
            save_html(map_obj, step_file)
            verbose && println("  Step diagnostics dashboard: $step_file")
        catch e
            verbose && println("  (Step diagnostics note: $e)")
        end
    end
    
    # 2. Regional Connectivity & Home Range Estimates
    try
        conn_file = joinpath(out_dir, "movement_regional_connectivity.html")
        map_obj = leaflet_regional_connectivity(
            loaded.au_mesh, path_results.paths;
            title = "$spp Regional Connectivity and Home Range Estimates"
        )
        save_html(map_obj, conn_file)
        verbose && println("  Regional connectivity dashboard: $conn_file")
    catch e
        verbose && println("  (Regional connectivity note: $e)")
    end

    # 3. Advection Velocity Field
    if !isnothing(loaded.advection_x) && !isnothing(loaded.advection_y)
        try
            adv_file = joinpath(out_dir, "movement_advection_velocity.html")
            map_obj = leaflet_velocity_field(
                loaded.advection_x, loaded.advection_y, loaded.au_mesh;
                title = "$spp Advection Drift & Velocity Field Vectors"
            )
            save_html(map_obj, adv_file)
            verbose && println("  Advection velocity dashboard: $adv_file")
        catch e
            verbose && println("  (Advection velocity note: $e)")
        end
        
        try
            ad_file = joinpath(out_dir, "movement_ad_ratio_distribution.html")
            map_obj = leaflet_ad_ratio_distribution(
                loaded.advection_x, loaded.advection_y, 0.1;
                title = "$spp Advection/Diffusion Ratio"
            )
            save_html(map_obj, ad_file)
            verbose && println("  Advection ratio dashboard: $ad_file")
        catch e
            verbose && println("  (Advection ratio note: $e)")
        end
    end

    # 4. Residence Time & Diffusion Field
    if !isnothing(loaded.residence_time)
        try
            res_file = joinpath(out_dir, "movement_residence_time.html")
            map_obj = leaflet_residence_time_map(
                loaded.residence_time, loaded.au_mesh;
                title = "$spp Residence Time Map"
            )
            save_html(map_obj, res_file)
            verbose && println("  Residence time dashboard: $res_file")
        catch e
            verbose && println("  (Residence time note: $e)")
        end
    end

    if !isnothing(loaded.diffusion)
        try
            diff_file = joinpath(out_dir, "movement_diffusion_field.html")
            map_obj = leaflet_diffusion_map(
                loaded.diffusion, loaded.au_mesh;
                title = "$spp Diffusion Field"
            )
            save_html(map_obj, diff_file)
            verbose && println("  Diffusion dashboard: $diff_file")
        catch e
            verbose && println("  (Diffusion note: $e)")
        end
    end
    
    # 5. HSI Map
    if !isnothing(loaded.hsi)
        try
            hsi_file = joinpath(out_dir, "movement_hsi_map.html")
            map_obj = leaflet_hsi_map(
                loaded.hsi, loaded.au_mesh;
                title = "$spp Habitat Suitability Index (HSI)"
            )
            save_html(map_obj, hsi_file)
            verbose && println("  HSI dashboard: $hsi_file")
        catch e
            verbose && println("  (HSI map note: $e)")
        end
    end

    # 6. Dispersal Kernel
    if !isnothing(kernels.P_kernel)
        try
            disp_file = joinpath(out_dir, "movement_dispersal_kernel.html")
            map_obj = leaflet_dispersal_kernel(
                kernels.P_kernel, loaded.au_mesh;
                title = "$spp Empirical Dispersal Kernel"
            )
            save_html(map_obj, disp_file)
            verbose && println("  Dispersal kernel dashboard: $disp_file")
        catch e
            verbose && println("  (Dispersal kernel note: $e)")
        end
    end

    return (
        movement_stats = mov_stats,
        phenology      = pheno_res,
        trait_models   = trait_res,
    )
end


# =============================================================================
# Phase 5c: Validation Mark-Recapture Analyses
# =============================================================================

"""
    execute_validation_analyses(loaded, fitted, kernels, params) ->
        Union{NamedTuple, Nothing}

Executes validation post-processing analyses for mark-recapture telemetry:
1. Path credible intervals and node visitation distributions
   (`path_credible_intervals`)
2. Regional stock connectivity matrix and credible intervals
   (`compute_stock_connectivity_matrix`)
3. Posterior predictive checks (`posterior_predictive_check`)
4. Optional full Bayesian ensemble trajectory and corridor propagation
   (`reconstruct_paths_bayesian_ensemble`)
"""
function execute_validation_analyses(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params
)::Union{NamedTuple, Nothing}
    get(params, :compute_validation, true) || return nothing
    verbose = get(params, :verbose, true)
    out_dir = get(params, :output_dir,
                  normpath(joinpath(@__DIR__, "..", "..", "output")))
    mkpath(out_dir)

    verbose && println(
        "\n[Phase 5c] Running Validation Mark-Recapture Analyses..."
    )

    pa = run_validation_analyses(loaded, fitted, kernels, params, out_dir)

    ensemble_res = if get(params, :run_bayesian_ensemble, false)
        verbose && println(
            "  Evaluating Bayesian ensemble path & corridor propagation..."
        )
        reconstruct_paths_bayesian_ensemble(loaded, fitted, params)
    else
        nothing
    end

    return (
        path_uncertainty         = pa.path_uncertainty,
        connectivity_matrix      = pa.connectivity_matrix,
        connectivity_uncertainty = pa.connectivity_uncertainty,
        posterior_predictive     = pa.posterior_predictive,
        summary_file             = pa.summary_file,
        bayesian_ensemble        = ensemble_res,
    )
end

# =============================================================================
# Orchestrator
# =============================================================================

"""
    run_movement_analysis(params = movement_parameters_default()) -> NamedTuple

Executes the complete MovementAnalysis movement analysis pipeline. All six phases
are called in sequence:

1. `load_movement_data`                 -- data ingestion & depth barriers
2. `fit_movement_models`                -- Bayesian MCMC model fitting
3. `extract_transition_kernels`         -- posterior kernel construction
4. `reconstruct_paths_and_diagnostics` -- paths, corridors, bottlenecks
5. `compute_advanced_diagnostics`      -- circuit theory & wavelets
6. `export_dashboards`                 -- interactive Leaflet HTML maps

# Arguments
- `params`: Configuration NamedTuple. Start from a parameter preset and
  merge individual overrides:

      # Generic simulation
      run_movement_analysis()

      # Snow crab with custom depth range
      p = merge(movement_parameters_snowcrab(),
                (depth_range = (80.0, 400.0), n_samples = 500))
      run_movement_analysis(p)

# Returns
`NamedTuple` with fields: `data`, `models`, `chains`, `P_kernel`,
`paths`, `corridors`, `stochastic_paths`, `domain_bottlenecks`,
`circuit`, `wavelets`, `parameters`, `depth_range`.
"""
function run_movement_analysis(
    params = movement_parameters_default()
)::NamedTuple

    out_dir = get(params, :output_dir, normpath(joinpath(@__DIR__, "..", "..", "output")))
    mkpath(out_dir)
    checkpoint_file = joinpath(out_dir, "movement_checkpoint.jld2")
    resume_from_checkpoint = get(params, :resume_from_checkpoint, false)

    loaded, fitted, kernels = if resume_from_checkpoint && isfile(checkpoint_file)
        if params.verbose
            println("\n[Checkpoint] Resuming from existing checkpoint: $checkpoint_file")
        end
        data = JLD2.load(checkpoint_file)
        (data["loaded"], data["fitted"], data["kernels"])
    else
        loaded_  = load_movement_data(params)
        fitted_  = fit_movement_models(loaded_, params)
        kernels_ = extract_transition_kernels(loaded_, fitted_, params)
        
        if params.verbose
            println("\n[Checkpoint] Saving intermediate states to: $checkpoint_file")
        end
        JLD2.save(checkpoint_file, "loaded", loaded_, "fitted", fitted_, "kernels", kernels_)
        
        (loaded_, fitted_, kernels_)
    end
    
    # -- Agent-Based Forward Projection -------------------------------------
    # In agent mode the telemetry model is fitted as usual, so the projection runs
    # on an estimated kernel rather than on configured constants. Forward
    # projection conditions on the release unit only, unlike the backward
    # reconstruction, which is conditioned on both observed endpoints.
    agent_trajectories = nothing
    agent_paths = NamedTuple[]
    agent_movement_stats = nothing
    agent_space_use = nothing
    agent_summary_file = nothing
    if params.model_mode == "agent"
        if params.verbose
            println(
                "\n[Phase 2b] Forward-projecting synthetic agents from the fitted kernel..."
            )
        end
        n_sim_agents = Int(get(params, :n_agent_projections, 200))
        release_pool = Int.(loaded.obs_df.release)
        duration_pool = max.(1, Int.(round.(loaded.obs_df.k)))
        agent_trajectories = forward_project_agents(
            release_pool, duration_pool;
            n_agents = n_sim_agents,
            transition_kernel = sparse(kernels.P_kernel),
            seed = params.seed,
            centroids = _resolve_centroids(loaded.mesh),
            persistence = Float64(get(params, :persistence, 0.0))
        )
        agent_space_use = forward_space_use(agent_trajectories, loaded.n_spatial)
        if params.verbose
            n_visited = count(>(0), agent_space_use.visits)
            println(
                "  Projected $(n_sim_agents) agents over empirical durations; " *
                "$n_visited of $(loaded.n_spatial) units visited."
            )
        end
        agent_paths = _agent_trajectories_to_paths(
            agent_trajectories, loaded.mesh, loaded.hsi_vec
        )
        agent_path_lookup = Dict(path.tagid => path.path for path in agent_paths)
        agent_movement_stats = compute_movement_statistics(
            agent_paths, (paths=agent_path_lookup,), loaded; params=params
        )
        agent_summary_file = _write_agent_simulation_summary(
            agent_paths, agent_movement_stats, params.output_dir
        )
        params.verbose && !isnothing(agent_summary_file) &&
            println("  Agent summary CSV: $agent_summary_file")
    end
    
    path_res    = reconstruct_paths_and_diagnostics(loaded, kernels, params)
    diagnostics = compute_advanced_diagnostics(loaded, path_res, params)
    validation  = execute_validation_analyses(loaded, fitted, kernels, params)
    dashboards  = export_dashboards(
        loaded, kernels, path_res, diagnostics, params; agent_paths=agent_paths
    )

    if params.verbose
        println("\n" * "=" ^ 72)
        println("  Pipeline Completed Successfully!")
        println("=" ^ 72)
    end

    return (
        data               = loaded.data,
        models             = fitted.models,
        chains             = fitted.chains,
        P_kernel           = kernels.P_kernel,
        paths              = path_res.paths,
        corridors          = path_res.corridors,
        stochastic_paths   = path_res.stochastic_paths,
        domain_bottlenecks = path_res.domain_bottlenecks,
        circuit            = diagnostics.circuit,
        wavelets           = diagnostics.wavelets,
        validation_analyses = validation,
        agent_trajectories = agent_trajectories,
        agent_movement_stats = agent_movement_stats,
        agent_space_use    = agent_space_use,
        agent_summary_file = agent_summary_file,
        movement_stats     = !isnothing(dashboards) && hasproperty(dashboards, :movement_stats) ?
                             dashboards.movement_stats : nothing,
        phenology          = !isnothing(dashboards) && hasproperty(dashboards, :phenology) ?
                             dashboards.phenology : nothing,
        trait_models       = !isnothing(dashboards) && hasproperty(dashboards, :trait_models) ?
                             dashboards.trait_models : nothing,
        parameters         = (
            alpha     = kernels.alpha_hat,
            residence = kernels.rho_hat,
            gamma     = kernels.gamma_hat,
        ),
        depth_range        = loaded.parsed_depth_range,
    )
end
