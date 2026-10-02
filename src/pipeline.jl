const _DepthRangeArg = Union{
    Nothing,
    Tuple{<:Real, <:Real},
    AbstractVector{<:Real},
    AbstractString,
}


using Arrow
using DataFrames

"""
    _require_r_backend(what) -> Nothing

Raise a clear, actionable error when an R-backed code path is reached without
`RCall` installed.

R is an optional dependency. `MovementAnalysisRCallExt` is loaded automatically
when `RCall` and a working R are both present; this stub only fires when they are
not, and it names the specific input that needed them rather than surfacing an
`UndefVarError` from deep inside a loader.
"""
function _require_r_backend(what::AbstractString)
    Base.get_extension(@__MODULE__, :MovementAnalysisRCallExt) === nothing && throw(
        ArgumentError(
            "$what requires the optional `RCall` backend, which is not loaded. " *
            "Install it with `import Pkg; Pkg.add(\"RCall\")` and ensure an R " *
            "installation with the `qs` and `arrow` R packages is configured. " *
            "Alternatively supply the data in Arrow, CSV, GeoJSON, or shapefile " *
            "form, which need no R."
        ),
    )
    return nothing
end

"""
    _r_ipc_convert(filepath, temp_ipc)

Convert a tabular R file to an Arrow IPC file, in R.

Implemented by `MovementAnalysisRCallExt`. This stub exists so the package loads
without `RCall`.
"""
function _r_ipc_convert(filepath::AbstractString, temp_ipc::AbstractString)
    _require_r_backend("Reading R data ($filepath)")
    return nothing
end

"""
    _r_load_object(path)

Load an R object (`.rds` / `.RData`) for region-polygon extraction.

Implemented by `MovementAnalysisRCallExt`. This stub exists so the package loads
without `RCall`.
"""
function _r_load_object(path::AbstractString)
    _require_r_backend("Reading R region polygons ($path)")
    return nothing
end

"""
    r_to_ipc(filepath::AbstractString)

Read tabular R data by utilizing an R session to convert it to a temporary 
Arrow IPC (Feather v2) file, which is then parsed into Julia.

This method achieves extremely high performance by writing the exact in-memory 
representation to disk. Julia reads the bytes into memory and constructs the 
DataFrame pointing directly at those bytes (zero-copy parsing), avoiding 
both memory duplication and deserialization CPU overhead.

# Analytical Assumptions
- **Tabular Data Requirement**: IPC exclusively supports tabular data. The R object 
  stored in the file must be a `data.frame`, `tibble`, or `data.table`.
- **Single Object extraction**: For `.rda` files, this extracts only the first 
  variable loaded alphabetically.
- **Garbage Collection**: The raw bytes of the file are held in a Julia byte array. 
  These bytes will remain in memory until the returned DataFrame is garbage collected.

# Arguments
- `filepath::AbstractString`: The path to the R data file.

# Returns
- A `DataFrame` containing the deserialized tabular data.

# Dependencies
- Julia packages: `RCall`, `Arrow`, `DataFrames`.
- R packages: `qs` and `arrow` must be installed.
"""
function r_to_ipc(filepath::AbstractString)
    if !isfile(filepath)
        error("File not found: ", filepath)
    end
    
    # Generate a temporary file path for the IPC intermediate
    temp_ipc = tempname() * ".arrow"
    
    try
        # The R conversion lives in MovementAnalysisRCallExt so that the package
        # loads without an R installation.
        _r_ipc_convert(filepath, temp_ipc)

        # Read the raw bytes from disk into memory. This avoids Windows file-locking 
        # issues that occur if we were to memory-map the file directly from disk.
        ipc_bytes = read(temp_ipc)
        
        # Construct the Arrow Table from the in-memory byte array, and wrap it in a 
        # DataFrame without copying the underlying columns (zero-copy parsing).
        return DataFrame(Arrow.Table(ipc_bytes), copycols=false)
        
    finally
        # Safely delete the temporary file
        if isfile(temp_ipc)
            rm(temp_ipc)
        end
    end
end




# Ingest empirical telemetry data, construct 20 km hexagonal mesh, sever
# terrestrial barriers, and infill unobserved marine HSI values


"""
    load_movement_dataset(;
        radius_km = 15.0,
        time_interval = :daily,
        crs = nothing,
        datum = WGS84Latest,
        ref_doy = 182,
        verbose = true,
        pre_mapped = nothing,
        data_dir = nothing
    ) -> NamedTuple

High-level convenience pipeline for empirical mark-recapture datasets
movement, telemetry, and environmental suitability data across Atlantic Canada.
Loads empirical mark-recapture encounters from JLD2 storage, constructs a unified
planar hexagonal domain tessellation covering the full extent of the
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
- `tagging_file`, `hsi_file`, `sppoly_file`, `surveydata_file`: dataset inputs.
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
  - `group_lookup::Dict`: Stratum string to integer ID mapping.
  - `land_mask::Vector{Bool}`: Terrestrial barrier indicator vector.
"""
function load_movement_dataset(;
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

    isfile(actual_tagging) || error("Telemetry file not found: $actual_tagging")

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

    return prepare_movement_data(
        tagging;
        hsi_file      = isfile(actual_hsi) ? actual_hsi : nothing,
        sppoly_file   = isfile(actual_sppoly) ? actual_sppoly : nothing,
        radius_km     = radius_km,
        time_interval = time_interval,
        land_polygons = :none,
        crs           = crs,
        datum         = datum,
        ref_doy       = ref_doy,
        verbose       = verbose,
        pre_mapped    = pre_mapped
    )
end

# sc_data = load_movement_dataset(radius_km = 20.0, verbose = true)

# println("Active mark-recapture records: ", nrow(sc_data.obs))
# println("Domain units: ", sc_data.mesh.n_units, 
#         " (marine: ", count(!, sc_data.land_mask), 
#         ", land: ", sum(sc_data.land_mask), ")")


# =============================================================================
# Parameter Functions
# =============================================================================

"""
    movement_parameters_default() -> MovementAnalysisConfig

The single canonical entry point for all analysis parameters.

Every code path -- library call, CLI invocation, or TOML file -- resolves to this
struct. A TOML file is applied as an overlay on top of these defaults (see
`load_config`), so a parameter absent from a config file simply keeps the default
declared in `src/config.jl`. A dataset is selected by its input files, not by a
separate code path: pass `tagging_file` (and the HSI and spatial-unit files that
accompany it) to analyse real data, and leave them unset for synthetic data.
"""
movement_parameters_default() = MovementAnalysisConfig()

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
`NamedTuple` with fields: `data`, `mesh`, `W`, `hsi_vec`, `obs_df`,
`survey_df`, `group_map`, `land_mask`, `n_spatial`, `resharded_hydro`,
`resharded_depths`, `parsed_depth_range`.
"""
function load_movement_data(params)::NamedTuple
    verbose = params.verbose

    verbose && println("=" ^ 72)
    verbose && println("  MovementAnalysis Pipeline")
    verbose && println("=" ^ 72)

    # -- 1a. Load or generate the dataset -----------------------------------
    # The dataset is identified by the input files themselves: when a tagging
# file is configured, that file -- together with whichever HSI and
# spatial-unit files accompany it -- *is* the dataset. With no tagging file
# there is nothing to read, so a synthetic dataset is generated instead.
# There is no dataset name to match on, and no silent fallback: a configured
# file that cannot be read is an error, not a reason to swap in simulated
# data and report a successful run of the wrong analysis.
data = if isnothing(params.tagging_file)
    verbose && println(
        "\n[Phase 1] No tagging file configured; generating a synthetic dataset"
    )
    generate_movement_data()
else
    verbose && println("\n[Phase 1] Ingesting dataset from $(params.tagging_file)")
    isfile(params.tagging_file) || error(
        "Configured tagging_file does not exist: $(params.tagging_file)"
    )
    load_movement_dataset(
        radius_km = params.hex_radius_km,
        verbose = verbose,
        tagging_file = params.tagging_file,
        hsi_file = params.hsi_file,
        sppoly_file = params.sppoly_file,
    )
end
    mesh      = data.mesh
    W         = data.W
    hsi_vec   = data.hsi_vec
    obs_df    = data.obs

    # Delimit domain extent using sppoly bounds when available
    sppoly_path = if !isnothing(params.sppoly_file) && isfile(params.sppoly_file)
        params.sppoly_file
    else
        nothing
    end
    sppoly_bounds = extract_sppoly_bounds(sppoly_path)
    if sppoly_bounds !== nothing && verbose
        println(
            "  Delimiting southern/south-western domain with sppoly bounds: " *
            "lon >= $(round(sppoly_bounds[1]; digits=4)), lat >= $(round(sppoly_bounds[2]; digits=4))"
        )
    end

    # The analysis domain is an explicit bounding box when configured, and
    # otherwise the extent of the input data plus a small padding so the mesh
    # is not clipped flush against the outermost detections.
    domain_bbox = resolve_bbox(
        params.bbox, params.bbox_padding_deg,
        hasproperty(obs_df, :lon) ? Float64[obs_df.lon...] : Float64[],
        hasproperty(obs_df, :lat) ? Float64[obs_df.lat...] : Float64[];
        sppoly_bounds = sppoly_bounds
    )
    verbose && println(
        "  Domain bounding box (W, S, E, N) = " *
        "($(domain_bbox[1]), $(domain_bbox[2]), $(domain_bbox[3]), $(domain_bbox[4]))"
    )

    # Land identification is configuration-driven: a global land/sea raster,
    # user polygons, bathymetry alone, or nothing.
    land_polys = if params.land_source === :polygons
        isempty(params.land_polygon_files) && error(
            "land_source = :polygons requires land_polygon_files"
        )
        reduce(vcat, read_polygon_file.(params.land_polygon_files))
    else
        nothing
    end
    survey_df = if !isnothing(params.surveydata_file)
    load_survey_data(params.surveydata_file; verbose = verbose)
elseif hasproperty(data, :survey_df)
    data.survey_df
else
    nothing
end
    group_map = hasproperty(data, :group_lookup) ?
                data.group_lookup : Dict(1 => "All")

    # Regions of interest come from polygon files paired with `region_labels`.
    # Deriving the unit-to-region map from the polygons is preferred over
    # hand-building it, so an explicit `region_map` only fills the gaps.
    region_map = if !isempty(params.region_polygon_files)
        labels = isempty(params.region_labels) ?
            ["Region $i" for i in eachindex(params.region_polygon_files)] :
            params.region_labels
        polys = load_region_polygons(params.region_polygon_files, labels)
        region_map_from_polygons(mesh.centroids_lonlat, polys)
    else
        nothing
    end
    conn_region_labels = nothing
    if region_map !== nothing
        explicit = isnothing(params.region_map) ? Dict{Int,Int}() : params.region_map
        for (unit, r) in explicit
            1 <= unit <= length(region_map) && (region_map[unit] = r)
        end
        n_r = maximum(region_map)
        n_r >= 1 || error(
            "No mesh unit falls inside any region polygon; check region_polygon_files."
        )
        conn_region_labels = isempty(params.region_labels) ?
            ["Region $i" for i in 1:n_r] : params.region_labels
        verbose && println(
            "  Regions: $(length(conn_region_labels)) from polygon files; " *
            "$(count(>(0), region_map)) / $(length(region_map)) units assigned"
        )
    end
    land_mask = hasproperty(data, :land_mask) ? data.land_mask : nothing
    n_spatial = mesh.n_units

    # Time-varying HSI, taken now rather than just before the return, so the
    # resharding block below can rescale it onto the fine mesh alongside
    # `hsi_vec` and `W`.
    monthly_hsi  = hasproperty(data, :monthly_hsi) ? data.monthly_hsi : Matrix{Float64}(undef, 0, 0)
    month_lookup = hasproperty(data, :month_lookup) ? data.month_lookup : Dict{Tuple{Int,Int}, Int}()
    years_vec    = hasproperty(data, :years) ? data.years : Int[]

    if verbose
        println("  Spatial mesh units : $n_spatial")
        if land_mask !== nothing
            println("    Marine    : $(count(!, land_mask))")
            println("    Land      : $(sum(land_mask))")
        end
        println("  Mark-recapture obs : $(nrow(obs_df))")
        println("  Biological groups  : $(length(group_map))")
    end

    # -- 1b. LibGEOS Hexagonal Resharding & Hydrodynamics --------------------
    resharded_hydro  = nothing
    resharded_depths = nothing

    if params.reshard_hex || params.use_hydrodynamics
        verbose && println(
            "\n[Phase 1b] Resharding to fine hexagons via LibGEOS..."
        )
        cents_lon = [Float64(c[1]) for c in mesh.centroids_lonlat]
        cents_lat = [Float64(c[2]) for c in mesh.centroids_lonlat]
        min_lon, max_lon = extrema(cents_lon)
        min_lat, max_lat = extrema(cents_lat)

        if sppoly_bounds !== nothing
            min_lon = max(min_lon, sppoly_bounds[1])
            min_lat = max(min_lat, sppoly_bounds[2])
        end

        bathy = load_open_bathymetry(;
            lon_range      = (min_lon - 0.2, max_lon + 0.2),
            lat_range      = (min_lat - 0.2, max_lat + 0.2),
            resolution_deg = 0.08
        )
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
        resharded_hydro  = reshard_spatial_field(P_transfer, hydro)
        resharded_depths = reshard_spatial_field(P_transfer, bathy.depths)

        mesh      = fine_mesh
        W         = fine_mesh.W
        land_mask = identify_land_units(
            fine_mesh.centroids_lonlat;
            depth         = resharded_depths,
            land_polygons = land_polys,
        )
        if params.land_source === :landmask
            # A unit can be shallow enough to read as marine on bathymetry yet
            # still sit on land; the global raster settles those disagreements.
            land_mask .|= land_mask_from_global_mask(
                fine_mesh.centroids_lonlat;
                resolution   = params.land_mask_resolution,
                grid_minutes = params.land_mask_grid_minutes,
            )
            verbose && println(
                "  Land units: $(count(land_mask)) / $(fine_mesh.n_units) " *
                "(global land/sea mask)"
            )
        end

        P_orig_to_fine = compute_network_transfer_matrix(
            data.mesh, fine_mesh; method = :area_weighted
        )
        hsi_fine = if hasproperty(data, :hsi_vec) && !isnothing(data.hsi_vec) &&
                      length(data.hsi_vec) == data.mesh.n_units
            reshard_spatial_field(P_orig_to_fine, data.hsi_vec)
        elseif resharded_hydro !== nothing && hasproperty(resharded_hydro, :hsi)
            resharded_hydro.hsi
        else
            fill(0.5, fine_mesh.n_units)
        end

        W, hsi_vec = apply_land_barrier(
            fine_mesh.W, hsi_fine, land_mask
        )

        # `monthly_hsi` was built on the original mesh, so its rows index a
        # different set of units than everything resharded above. It needs a
        # transfer matrix from the *original* mesh -- not from the bathymetry
        # grid, which is what P_transfer maps -- or every time-varying kernel
        # lookup would index the wrong units.
        if !isempty(monthly_hsi) && size(monthly_hsi, 1) != fine_mesh.n_units
            monthly_hsi = reshard_spatial_field(P_orig_to_fine, monthly_hsi)
            for col in axes(monthly_hsi, 2)
                monthly_hsi[land_mask, col] .= 0.0
            end
            verbose && println(
                "  Rescaled monthly HSI to $(size(monthly_hsi, 1)) x " *
                "$(size(monthly_hsi, 2)) fine-mesh units"
            )
        end
        sever_land_crossing_edges!(W, fine_mesh.centroids_lonlat)
        n_spatial  = fine_mesh.n_units

        # Remap mark-recapture observations to active marine units
        obs_df       = copy(obs_df)
        fine_cents   = fine_mesh.centroids_lonlat
        marine_mask  = .!land_mask .& (vec(sum(W; dims = 2)) .> 0)
        marine_units = let mu = findall(marine_mask)
            isempty(mu) ? collect(1:n_spatial) : mu
        end
        marine_cents = fine_cents[marine_units]

        orig_rel = [data.mesh.centroids_lonlat[r] for r in obs_df.release]
        orig_rec = [data.mesh.centroids_lonlat[r] for r in obs_df.recapture]
        rel_sub  = map_to_units(
            [c[1] for c in orig_rel], [c[2] for c in orig_rel], marine_cents
        )
        rec_sub  = map_to_units(
            [c[1] for c in orig_rec], [c[2] for c in orig_rec], marine_cents
        )
        obs_df.release   = marine_units[rel_sub]
        obs_df.recapture = marine_units[rec_sub]

        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            survey_df       = copy(survey_df)
            orig_surv       = [data.mesh.centroids_lonlat[s]
                                for s in survey_df.s_idx]
            surv_sub        = map_to_units(
                [c[1] for c in orig_surv],
                [c[2] for c in orig_surv],
                marine_cents
            )
            survey_df.s_idx = marine_units[surv_sub]
            survey_df.depth = resharded_depths[survey_df.s_idx]
        end
        verbose && println("  Resharding complete.")
    end

    # -- 1b-ii. Adaptive Multiresolution Hexagonal Mesh ----------------------
    if params.adaptive_mesh
        verbose && println(
            "\n[Phase 1b-ii] Constructing adaptive multiresolution domain..."
        )
        cents_lon = [Float64(c[1]) for c in mesh.centroids_lonlat]
        cents_lat = [Float64(c[2]) for c in mesh.centroids_lonlat]
        rc = params.coarse_radius_km
        rf = params.fine_radius_km
        mesh = construct_adaptive_multiresolution_domain(
            cents_lon, cents_lat;
            coarse_radius_km = rc,
            fine_radius_km   = rf,
            land_polygons    = land_mask !== nothing ? land_mask : :none
        )
        W = mesh.W
        land_mask = mesh.land_mask
        n_spatial = mesh.n_units
        hsi_vec = ones(Float64, n_spatial)

        # Remap release/recapture locations to nearest multiresolution units
        cents_xy = [[c[1], c[2]] for c in mesh.centroids_km]
        tree_m = KDTree(hcat(cents_xy...))
        orig_rel_c = [data.mesh.centroids_km[r] for r in obs_df.release]
        orig_rec_c = [data.mesh.centroids_km[r] for r in obs_df.recapture]
        obs_df.release = [
            knn(tree_m, [c[1], c[2]], 1)[1][1] for c in orig_rel_c
        ]
        obs_df.recapture = [
            knn(tree_m, [c[1], c[2]], 1)[1][1] for c in orig_rec_c
        ]
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
        depth_mode = params.depth_barrier_mode

        # Land-only W: always needed for reachability checks and :hsi_only mode
        land_only_bv = land_mask !== nothing ?
            BitVector(land_mask) : falses(n_spatial)
        W_marine_only, _ = apply_land_barrier(W, hsi_vec, land_only_bv)

        if depth_mode == :hsi_only
            # Structural W: land barrier only
            W, hsi_vec = apply_land_barrier(W, hsi_vec, land_only_bv)
            # Encode depth preference via HSI floor for out-of-depth nodes
            hsi_floor  = params.hsi_ood_floor
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
            findall(.!land_only_bv)
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
                out_bv = depth_mode == :hsi_only ? land_only_bv : land_mask

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

        # Prune out-of-depth and unreachable/boundary-exceeded units from the final mesh
        # so that downstream network operations and Leaflet dashboards do not retain
        # an oversized, empty rectangular domain.
        keep_mesh_mask = .!out_of_depth
        if sppoly_bounds !== nothing
            cents_tmp = hasproperty(mesh, :centroids_lonlat) ?
                mesh.centroids_lonlat : mesh.centroids
            for i in eachindex(cents_tmp)
                lon_i = Float64(cents_tmp[i][1])
                lat_i = Float64(cents_tmp[i][2])
                if lon_i < (sppoly_bounds[1] - 0.05) || lat_i < (sppoly_bounds[2] - 0.05)
                    keep_mesh_mask[i] = false
                end
            end
        end

        # Ensure active endpoints in obs_df are preserved
        for r in eachrow(obs_df)
            if 1 <= r.release <= length(keep_mesh_mask)
                keep_mesh_mask[r.release] = true
            end
            if 1 <= r.recapture <= length(keep_mesh_mask)
                keep_mesh_mask[r.recapture] = true
            end
        end

        n_pruned = count(!, keep_mesh_mask)
        if n_pruned > 0 && count(keep_mesh_mask) > 0
            verbose && println(
                "  Pruning $n_pruned out-of-depth/out-of-bounds units from domain " *
                "($(count(keep_mesh_mask)) units retained)..."
            )
            old_to_new = zeros(Int, n_spatial)
            new_idx = 0
            for i in 1:n_spatial
                if keep_mesh_mask[i]
                    new_idx += 1
                    old_to_new[i] = new_idx
                end
            end

            mesh = prune_mesh(mesh, keep_mesh_mask)
            W = W[keep_mesh_mask, keep_mesh_mask]
            hsi_vec = hsi_vec[keep_mesh_mask]
            if !isempty(monthly_hsi) && size(monthly_hsi, 1) == n_spatial
                monthly_hsi = monthly_hsi[keep_mesh_mask, :]
            end
            if land_mask !== nothing && length(land_mask) == n_spatial
                land_mask = land_mask[keep_mesh_mask]
            end
            if resharded_depths !== nothing && length(resharded_depths) == n_spatial
                resharded_depths = resharded_depths[keep_mesh_mask]
            end
            if region_map !== nothing && length(region_map) == n_spatial
                region_map = region_map[keep_mesh_mask]
            end
            if resharded_hydro !== nothing
                resharded_hydro = reshard_spatial_field(
                    spdiagm(0 => ones(Float64, count(keep_mesh_mask))),
                    resharded_hydro
                )
            end

            # Re-index observations to pruned mesh units
            obs_df = copy(obs_df)
            obs_df.release = [old_to_new[r] for r in obs_df.release]
            obs_df.recapture = [old_to_new[r] for r in obs_df.recapture]

            if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
                survey_df = copy(survey_df)
                keep_surv = [keep_mesh_mask[s] for s in survey_df.s_idx]
                survey_df = survey_df[keep_surv, :]
                survey_df.s_idx = [old_to_new[s] for s in survey_df.s_idx]
            end

            n_spatial = mesh.n_units
        end
    end

    # Render and display/save map of resulting polygons after tessellation and pruning
    if params.render_html
        try
            tess_dir = params.output_dir
            mkpath(tess_dir)
            tess_path = joinpath(tess_dir, "tessellation_polygons.html")
            tess_map = leaflet_tessellation_map(
                mesh;
                title = "$(params.species_name) Tessellated Spatial Domain",
                depth = resharded_depths,
                hsi = hsi_vec,
                dark_mode = params.dark_mode,
                output_file = tess_path
            )
            show_map(tess_map; output_file = tess_path)
            verbose && println("  Tessellation polygon map: $tess_path")
        catch err_map
            _record_panel_skip("Tessellation map", err_map);             verbose && println("  (Tessellation map note: $(_error_note(err_map)))")
        end
    end

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
        group_map          = group_map,
        region_labels      = conn_region_labels,
        region_map         = region_map,
        land_mask          = land_mask,
        n_spatial          = n_spatial,
        resharded_hydro    = resharded_hydro,
        resharded_depths   = resharded_depths,
        parsed_depth_range = parsed_depth_range,
    )
end

# =============================================================================
# Phase 2: Model Fitting
# =============================================================================

"""
    movement_sampler(prior_scales, params)

The MCMC sampler for the movement models, as a random walk in linked space with
diagonal covariance `prior_scales .^ 2 .* params.mh_proposal_scale`.

# Why not plain `MH()`
`MH()` in Turing 0.49 is documented as drawing proposals **from the model prior**.
That is unusable here: with 5,294 observations the posterior is far sharper than
the prior, so once the chain has any likelihood at all, a fresh prior draw almost
never beats it. Measured on the snow crab data, `MH()` accepts **0.0%** of
proposals and returns a single distinct value across 200 draws -- the reported
"posterior" is the initialisation, not a fit. Two different models frozen at the
same seeded starting point report *identical* parameters, which is how the bug
surfaced.

# Why a random walk and not NUTS
`NUTS` works (acc = 1.0, 200/200 distinct draws) and adapts its metric
automatically, which is the better long-term answer. On this problem it is ~15x
slower per draw than the random walk, because each of ~2^10 leapfrog steps
rebuilds the transition kernel. That cost is a tuning question, not a correctness
one; the proposal scale is exposed as `mh_proposal_scale` so it can be tuned
without editing code, and convergence diagnostics are tracked in `todo.md`.

`prior_scales` are the priors' standard deviations in linked space, so the proposal
is a fraction of the prior width rather than an arbitrary number.
"""
function movement_sampler(prior_scales::Vector{Float64}, params)
    s = params.mh_proposal_scale
    s > 0 || throw(ArgumentError("mh_proposal_scale must be positive, got $s"))
    return MH(Matrix(Diagonal((s .* prior_scales) .^ 2)))
end

# The three population parameters and their prior scales, in declaration order:
# mu_velocity ~ N(0.3, 0.2) truncated, mu_diffusion ~ N(0.1, 0.2) truncated,
# mu_gamma ~ N(1.0, 1.0). Shared by the discrete and continuous-time models, which
# differ only in their likelihood.
const POPULATION_PRIOR_SCALES = [0.2, 0.2, 1.0]

"""
    report_chain_health(label, chain, params) -> Bool

Print the two numbers that distinguish a chain that moved from one that did not:
the acceptance rate, and how many distinct values the retained draws actually took.

This exists because the alternative is invisible. `MH()` accepted 0.0% of proposals
and returned a single distinct value across 200 draws, and the run reported a
posterior mean as if it meant something. A distinct-value count near 1 is the
signature of a frozen chain and cannot be mistaken for a tight posterior once
stated.

Returns `false` when the chain looks frozen, so callers can escalate.
"""
function report_chain_health(label::AbstractString, chain, params)::Bool
    verbose = params.verbose

    n = 0
    acc = Float64[]
    for k in keys(chain)
        nm = string(k)
        vals = try
            Array(chain[k])
        catch
            continue
        end
        occursin("accept", nm) || (n = max(n, length(vals)); continue)
        m = skipmissing(vec(vals))
        isempty(m) || push!(acc, mean(m))
    end
    n == 0 && return true
    a = isempty(acc) ? NaN : mean(acc)

    # How many distinct values one sampled parameter actually took. This is the
    # signal that catches a frozen chain even when the reported acceptance is
    # misleading, so it is worth computing from whichever parameter is available
    # rather than assuming a name.
    n_distinct = 0
    for k in keys(chain)
        occursin("Parameter", string(k)) || continue
        vals = try
            vec(Array(chain[k]))
        catch
            continue
        end
        all(isfinite, vals) || continue
        n_distinct = length(unique(round.(vals; digits = 9)))
        break
    end

    healthy = (isnan(a) || a > 0.01) && n_distinct > 1
    if healthy
        verbose && println(
            "  $label: acceptance=$(isnan(a) ? "n/a" : string(round(a; digits = 3))), " *
            "distinct values=$n_distinct/$n")
    else
        @warn "$label chain looks frozen" acceptance = a n_distinct = n_distinct n_draws = n
    end
    return healthy
end

"""
    fit_movement_models(loaded, params) -> NamedTuple

Phase 2 of the pipeline. Fits Bayesian movement models via Turing MCMC.

Approaches to fit (set via `params.model_modes`):

- `"telemetry"`: Pure categorical mark-recapture transition likelihood, fitted at
    the population level. recapture ~ Categorical(P^k[release, :]) for free
    parameters `mu_velocity`, `mu_diffusion`, `mu_gamma`; `alpha` and `rho` are
    derived from the first two by `movement_alpha_rho`.
- `"telemetry_and_survey"`: **the survey density likelihood is not implemented** —
    `counts` and `depths` are accepted and ignored, so this reduces to the
    telemetry model and emits a warning.
- `"agent"`: Agent-based movement simulation; no MCMC.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `params`: Configuration struct. Relevant keys: `model_modes`,
  `n_samples`, `seed`, `verbose`.

# Returns
`NamedTuple` with `models::Dict` and `chains::Dict`.

# On burn-in
Every `sample` call passes `num_warmup = params.n_warmup`, which AbstractMCMC
turns into `discard_initial` -- the first `n_warmup` states of the walk are
thrown away and only the following `n_samples` are kept.

This was missing, and it mattered more than it looked. Without it the returned
"posterior" was the *opening segment* of the random walk, dominated by the
initial draw and therefore by the prior. Two different models, seeded the same
way, then reported near-identical parameters because both were mostly reporting
their priors. `n_warmup` was a documented, configurable field that nothing read.

# On reporting whether the chain moved
A frozen chain produces a posterior mean indistinguishable from a converged one, and
that is how the `MH()` problem stayed hidden for so long: two different models
reported identical parameters, and nothing in the output said why. `report_chain_health`
prints acceptance and the number of distinct posterior values, and escalates to a
warning when the chain barely moved, so the failure is visible in the run log
rather than only inferable from the results.
"""
function fit_movement_models(loaded, params)::NamedTuple
    verbose   = params.verbose
    params.n_warmup >= 0 || throw(ArgumentError(
        "n_warmup must be non-negative, got $(params.n_warmup)"))
    params.n_samples >= 1 || throw(ArgumentError(
        "n_samples must be at least 1, got $(params.n_samples)"))
    # `model_modes` is the request; `effective_model_modes` is what survives
    # contact with the data. Survey-dependent approaches are dropped when no
    # readable survey file is configured, so asking for one is not an error --
    # it simply is not run.
    mode_set = Set(effective_model_modes(params))
    for m in skipped_modes(params)
        verbose && println(
            "  Note: :$(m) not run -- no readable surveydata_file configured."
        )
    end
      fit_tel       = :telemetry in mode_set
      fit_joint     = :telemetry_and_survey in mode_set
      rng       = MersenneTwister(params.seed)
      models    = Dict{Symbol, Any}()
      chains    = Dict{Symbol, Any}()

      # -- Pure Telemetry Model ------------------------------------------------
    if fit_tel
        verbose && println("\n[Phase 2] Fitting Pure Telemetry model...")
        verbose && println(
            "  population-level: mu_velocity, mu_diffusion, mu_gamma"
        )
        verbose && println("  recapture ~ Categorical(P^k[release, :])")
        obs_df     = loaded.obs_df
        releases   = Int.(obs_df.release)
        recaptures = Int.(obs_df.recapture)
        ks         = Int.(round.(obs_df.k))
        tagids = obs_df.tagid
        m_tel = pure_telemetry_turing_model(
            releases, recaptures, ks,
            loaded.W, loaded.hsi_vec, loaded.land_mask
        )
        models[:telemetry] = m_tel
        spl = movement_sampler(POPULATION_PRIOR_SCALES, params)
        verbose && println("  Sampling $(params.n_samples) draws " *
                          " (discarding $(params.n_warmup) warmup)...")
        chn = sample(rng, m_tel, spl, params.n_samples; num_warmup = params.n_warmup, progress = false)
                chains[:telemetry] = chn
        report_chain_health("telemetry", chn, params)
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
        tagids = obs_df.tagid
        survey_df  = loaded.survey_df
        counts     = Int.(round.(survey_df.density))
        depths     = hasproperty(survey_df, :depth) ?
                     Float64.(survey_df.depth) : zeros(Float64, length(counts))

        m_joint = joint_survey_telemetry_turing_model(
            counts, depths,
            releases, recaptures, ks,
                        loaded.W, loaded.hsi_vec, loaded.land_mask
        )
        models[:telemetry_and_survey] = m_joint
        # The joint models declare three survey coefficients in addition to the
        # population parameters (see todo.md 1.6). They are prior-only, but the
        # proposal still has to span them or those coordinates cannot move.
        spl = movement_sampler(
            vcat([5.0, 2.0, 1.0], POPULATION_PRIOR_SCALES), params)
        verbose && println("  Sampling $(params.n_samples) draws " *
                          " (discarding $(params.n_warmup) warmup)...")
        chn_j = sample(rng, m_joint, spl, params.n_samples; num_warmup = params.n_warmup, progress = false)
                chains[:telemetry_and_survey] = chn_j
        report_chain_health("telemetry_and_survey", chn_j, params)
        verbose && println("  Joint model complete.")
    end

    return (models = models, chains = chains)
end

"""
    _error_note(e, limit::Int = 200) -> String

A one-line, length-capped rendering of an exception, for use in the `catch` notes
that let an optional dashboard panel fail without ending the run.

Interpolating an exception directly is unsafe here. A `FieldError` or
`MethodError` embeds the full type of every argument, and those types carry the
entire loaded dataset -- so a single missing field on a 7517-unit mesh produced
about 200 KB of type signature on one line, burying the actual message and every
other line of output. Truncating keeps the useful part, which is the exception
type and the reason it failed.
"""
_error_note(e, limit::Int = 200) =
    (s = first(sprint(showerror, e), limit); length(sprint(showerror, e)) > limit ? s * " …" : s)

"""
    PANEL_SKIPS::Vector{String}

Every optional dashboard panel that failed during the last run, as `"name: reason"`.

The export phase wraps roughly twenty-five panels in `try`/`catch` so that one
bad panel cannot end a run that has already spent minutes sampling. That policy
is right, but it was also *silent*: each catch only printed a note, and gated
that print on `verbose`, so under the default quiet settings a panel could fail
every time and nothing in the output said so.

That is not hypothetical. Three real defects hid in exactly these blocks -- an
unbound `hydro`, a `_sample_column` call with a group argument the pooled model
no longer has, and a read of the `group` column after it had been removed. Each
one made a whole panel vanish permanently while the run reported success. The
notes only surfaced because a verbose end-to-end run was printed by hand.

Recording the failures here makes a missing panel an observable fact. It is
module-level rather than threaded through the panel helpers so that panels in
different functions are all captured without changing any of their signatures;
it is diagnostic state only, reset at the start of each run, and never read by
the model.
"""
const PANEL_SKIPS = String[]

function _record_panel_skip(name::AbstractString, e)::Nothing
    push!(PANEL_SKIPS, "$name: $(_error_note(e))")
    return nothing
end

"""
    report_panel_skips(verbose::Bool) -> Int

Print a consolidated summary of `PANEL_SKIPS` and return how many were recorded.
Called once at the end of a run so that silent panel loss is visible by default.
"""
function report_panel_skips(verbose::Bool = true)::Int
    n = length(PANEL_SKIPS)
    n == 0 && return 0
    println("\n  $(n) optional panel(s) did not render:")
    for s in PANEL_SKIPS
        println("    - $s")
    end
    return n
end

# =============================================================================
# Phase 3: Kernel Construction
# =============================================================================

"""
    posterior_kernel_draws(chain) -> (velocity, diffusion, gamma)

Reduce a fitted posterior chain to the three per-draw series the transition
kernel is parameterised by.

The models are population-level, so the kernel parameters are the fitted `mu_*`
draws with the models' own bounds applied:

    velocity_i  = clamp(mu_velocity[i],  0, 0.95)
    diffusion_i = max(mu_diffusion[i],   0)
    gamma_i     = mu_gamma[i]

`sigma_*` and `z_*` are read **if present** and folded in, so a chain from an
older hierarchical model still reduces correctly, but their absence is an
ordinary outcome rather than a warning. There is no `alpha`/`rho` derivation
here: it lives in `movement_alpha_rho`, shared with the models, because a model
and the code reporting its posterior must not each re-derive the same transform.

Two details of the chain layout are handled here rather than at the call site,
because both fail silently if missed:

  * Keys are wrapped (`Parameter(mu_velocity)`, `Extra(:accepted)`), so a plain
    string comparison matches nothing and the bookkeeping columns as well.
  * A vector-valued parameter reads back as one vector per draw nested in a
    trailing singleton axis, which needs unwrapping before it is numeric.
"""
function posterior_kernel_draws(chain)
    n_draws = max(1, size(Array(chain), 1))

    # Chain keys are not plain names: the sampler wraps each one as
    # `Parameter(name)` or `Extra(name)`, so matching on the raw key would both
    # fail and pick up the internal bookkeeping columns.
    function parname(key)
        s = string(key)
        open = findfirst('(', s)
        open === nothing && return s
        close = findlast(')', s)
        close === nothing && return s
        return s[(open + 1):(close - 1)]
    end
    lookup = Dict{String,Any}(parname(k) => k for k in keys(chain))

    "Posterior draws of a scalar parameter, one per draw, or `nothing` if absent."
    function scalar_draws(name::String)
        key = get(lookup, name, nothing)
        key === nothing && return nothing
        d = vec(Array(chain[key]))
        length(d) == n_draws || return nothing
        return Float64.(d)
    end

    "Per-individual effects as an n_draws x n_individuals matrix, or nothing."
    function effect_draws(name::String)
        key = get(lookup, name, nothing)
        key === nothing && return nothing
        a = Array(chain[key])
        # A vector-valued parameter comes back as one vector per draw inside a
        # trailing singleton axis; unwrap the individual axis first.
        if eltype(a) <: AbstractVector
            cols = [Float64.(collect(v)) for v in a]
            isempty(cols) && return nothing
            length(unique(length.(cols))) == 1 || return nothing
            # One row per draw, one column per individual. `hcat` puts the draws
            # in columns, so the result is transposed to the layout used below.
            return permutedims(reduce(hcat, cols))
        end
        ndims(a) == 2 || return nothing
        return Float64.(a)
    end

    # A missing `mu_*` is a real problem: the kernel cannot be built without it.
    mu_v = scalar_draws("mu_velocity")
    mu_d = scalar_draws("mu_diffusion")
    mu_g = scalar_draws("mu_gamma")
    missing_mus = String[]
    mu_v === nothing && push!(missing_mus, "mu_velocity")
    mu_d === nothing && push!(missing_mus, "mu_diffusion")
    mu_g === nothing && push!(missing_mus, "mu_gamma")
    isempty(missing_mus) || error(
        "posterior_kernel_draws: chain is missing " * join(missing_mus, ", ") *
        ", so no transition kernel can be built from it. Chain keys were: " *
        join(sort!(collect(keys(lookup))), ", ")
    )

    # `sigma_*` and `z_*` are optional: present in a chain from an older
    # hierarchical model, absent from the current population-level one.
    sg_v = scalar_draws("sigma_velocity")
    sg_d = scalar_draws("sigma_diffusion")
    sg_g = scalar_draws("sigma_gamma")
    z_v  = effect_draws("z_velocity")
    z_d  = effect_draws("z_diffusion")
    z_g  = effect_draws("z_gamma")

    "Fold optional dispersion and individual effects in, then apply the model's clamp."
    function combine(mu, sigma, z, lo::Float64, hi::Float64)
        (sigma === nothing || z === nothing || size(z, 1) != length(mu)) && return mu
        col_m = reshape(mu, :, 1)
        col_s = reshape(sigma, :, 1)
        return vec(mean(clamp.(col_m .+ col_s .* z, lo, hi); dims = 2))
    end

    velocity  = combine(mu_v, sg_v, z_v, 0.0, 0.95)
    diffusion = combine(mu_d, sg_d, z_d, 0.0, Inf)
    # Gamma is unbounded in the model, so it is averaged without a clamp.
    gamma = (sg_g === nothing || z_g === nothing || size(z_g, 1) != length(mu_g)) ?
        copy(mu_g) :
        vec(mean(reshape(mu_g, :, 1) .+ reshape(sg_g, :, 1) .* z_g; dims = 2))

    return velocity, diffusion, gamma
end

"""
    extract_transition_kernels(loaded, fitted, params) -> NamedTuple

Phase 3 of the pipeline. Extracts the posterior transition kernel and parameter
summaries from the fitted MCMC chains.

Single-group model: all individuals share one parameter set. To fit group-specific
movement, call `run_movement_analysis` once per group with that group's data and
combine the results externally; nothing here borrows parameters across groups.

# Returns
\$NamedTuple\$ with:
- \$P_kernel\$: the posterior-mean stochastic transition kernel
- \$grp_name_lookup\$: Dict mapping group index -> label (empty for single group)
- \$alpha_hat\$, \$rho_hat\$, \$gamma_hat\$: posterior mean advection, residence, HSI
- \$alpha_samples\$, \$rho_samples\$, \$gamma_samples\$: posterior draws, one per draw
- \$active_chain\$: the chain used
"""
function extract_transition_kernels(loaded, fitted, params)::NamedTuple
    verbose = params.verbose
    chains  = fitted.chains

    # Choose the first available chain (prefer joint > telemetry)
    active_chain = if haskey(chains, :telemetry)
        chains[:telemetry]
    elseif haskey(chains, :telemetry_and_survey)
        chains[:telemetry_and_survey]
    elseif !isempty(chains)
        first(values(chains))
    else
        nothing
    end

    # Every downstream phase indexes the kernel and reads alpha/rho/gamma as
    # scalars, so there is no meaningful "empty" result to hand back. Naming the
    # modes that were requested makes the cause obvious: a survey-dependent mode
    # with no survey file is skipped, and could be the only one asked for.
    active_chain === nothing && error(
        "no fitted model produced a posterior chain; requested modes were " *
        join(string.(keys(chains)), ", ", " and ") *
        ". Check that a mode with a readable input was requested -- the " *
        "survey-dependent approaches need `surveydata_file`."
    )

    verbose && println("\n[Phase 3] Extracting transition kernels from posterior...")

    verbose && println("  Extracting posterior means and draws from the active chain...")

    active_chain = _get_active_chain(chains)
    if active_chain === nothing
        # Every downstream phase indexes the kernel and reads alpha/rho/gamma as
        # scalars, so there is no meaningful "empty" result to hand back. Naming
        # the modes that were asked for makes the cause obvious: a survey-dependent
        # mode with no survey file is skipped, and could be the only one requested.
        error("no fitted model produced a posterior chain; requested modes were " *
              join(string.(keys(chains)), ", ", " and ") *
              " (asked for: " * join(string.(get(loaded, :model_modes, Symbol[])), ", ") * ")")
    end

    v_draws, d_draws, g_draws = posterior_kernel_draws(active_chain)

    # `alpha`/`rho` come from the same helper the models use. This line used to
    # re-derive them and got `rho` wrong -- `1/(v+d)` instead of `1/(1+v+d)` --
    # so the reported and propagated residence parameter was a different quantity
    # from the one the likelihood was fitted with. See todo.md 1.1.
    alpha_samples, rho_samples = movement_alpha_rho(v_draws, d_draws)
    gamma_samples = g_draws

    g_mean    = mean(g_draws)
    alpha_hat = mean(alpha_samples)
    rho_hat   = mean(rho_samples)

    # No label-seeded or config-seeded overrides.
    # Parameters are reported exactly as fitted.
    if verbose
        println("\n[Phase 3] Posterior parameters:")
        println("  alpha=$(round(alpha_hat; digits=4)), " *
                "rho=$(round(rho_hat; digits=4)), " *
                "gamma=$(round(g_mean; digits=4))")
    end

# Habitat-coupled residency. `gamma` biases *which* neighbour is chosen;
      # residence decides whether the animal moves at all. With a zero coupling
      # the fitted scalar rho applies everywhere, which is what the posterior
      # actually estimated. A positive coupling makes residency unit-specific
      # from local habitat advantage, which is an extra assumption the telemetry
      # did not fit, so it is opt-in and reported below when active.
      beta = Float64(params.rest_advantage)
      residency = if beta == 0.0
          rho_hat
      else
          adv = local_hsi_advantage(
              loaded.hsi_vec, loaded.W; form = params.rest_advantage_form)
          residency_from_advantage(rho_hat, adv, beta, params.rest_advantage_form)
      end

      # Build the single transition kernel from posterior means
      P_kernel = construct_stochastic_transition_kernel(
          loaded.W, loaded.hsi_vec;
          gamma     = g_mean,
          residence = residency,
          advection = alpha_hat,
          land_mask = loaded.land_mask,
          coupled_residency = beta != 0.0,
      )

      if verbose && beta != 0.0
          println("  Habitat-coupled residency active: beta=$(beta), " *
                  "form=:$(params.rest_advantage_form), " *
                  "rho range $(round(minimum(residency); digits = 3))..$(round(maximum(residency); digits = 3))")
      end

    return (
        P_kernel        = P_kernel,
        grp_name_lookup = Dict{Int, String}(),
        alpha_hat       = alpha_hat,
        rho_hat         = rho_hat,
        gamma_hat       = g_mean,
        G               = 1,
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
- `params`: Relevant keys: `max_paths`, `path_methods`, `smooth_paths`,
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
      G           = 1
      grp_nlookup = Dict{Int, String}()

    cents_planar, cents_lonlat, cents_mesh =
        _resolve_centroids(loaded.mesh, n_spatial)

    all_tags    = unique(obs_df.tagid)
    n_sample    = min(params.max_paths, length(all_tags))
    sample_tags = all_tags[1:n_sample]

    verbose && println(
        "\n[Phase 4] Reconstructing $(join(string.(params.path_methods), ", ")) trajectories " *
        "for $n_sample / $(length(all_tags)) individuals..."
    )

    max_k_dyn = 0
if params.dynamic_kernels
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

    P_seg_cache = Dict{Tuple{Float64, Int}, Any}()

    reconstructed_paths     = Dict{String, Vector{Int}}()
    reconstructed_corridors = Dict{String, Matrix{Float64}}()
    stochastic_paths        = Dict{String, Any}()
    forward_ibm_paths       = Dict{String, Vector{Int}}()

    for tid in sample_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue

        grp = hasproperty(sub_obs, :group) ? first(sub_obs.group) : 1
        P_k = P_kernel isa AbstractVector ?
              P_kernel[clamp(grp, 1, length(P_kernel))] : P_kernel

        # Concatenate multi-segment trajectories for this individual
        full_path = if params.hmm_smoothing && nrow(sub_obs) > 1
            # Global multi-segment Hidden Markov Model Viterbi smoothing
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
            res_hmm.path
        else
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
                    get!(P_seg_cache, (Float64(t_rel), grp)) do
                        a_hat = kernels.alpha_hat isa AbstractVector ?
                            kernels.alpha_hat[clamp(grp, 1, length(kernels.alpha_hat))] :
                            kernels.alpha_hat
                        r_hat = kernels.rho_hat isa AbstractVector ?
                            kernels.rho_hat[clamp(grp, 1, length(kernels.rho_hat))] :
                            kernels.rho_hat
                        g_hat = kernels.gamma_hat isa AbstractVector ?
                            kernels.gamma_hat[clamp(grp, 1, length(kernels.gamma_hat))] :
                            kernels.gamma_hat
                        construct_stochastic_transition_kernel(
                            W, hsi_row;
                            gamma     = g_hat,
                            residence = r_hat,
                            advection = a_hat,
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
                        method    = first(params.path_methods),
                        land_mask = land_mask
                    )
                end
                # The router returns an empty vector when no exact-k route
                  # exists, rather than padding one. Drop the event and say so:
                  # appending a fabricated segment would misattribute movement.
                  isempty(seg) && (verbose && println(
                      "  $(tid) segment $(n)/$(nrow(sub_obs)): no exact-$(row.k)-step " *
                      "route from $(row.release) to $(row.recapture); dropped."))
                  append!(fpath, seg[2:end])
            end
            fpath
        end
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
        prop_hsi  = params.propagate_hsi_error && params.hsi_se > 0.0
        n_hsi_m   = prop_hsi ? min(params.n_stochastic_draws, 5) : 1

        corr_accum = nothing
        for d in 1:n_hsi_m
            P_eval = if prop_hsi && d > 1
                rng_d = MersenneTwister(
                    params.seed + abs(hash(string(tid))) % 10_000 + d
                )
                hsi_d = clamp.(
                    hsi_first .+ randn(rng_d, n_spatial) .* params.hsi_se,
                    0.001, 1.0
                )
                a_hat = kernels.alpha_hat isa AbstractVector ?
                        kernels.alpha_hat[clamp(grp, 1, length(kernels.alpha_hat))] :
                        kernels.alpha_hat
                r_hat = kernels.rho_hat isa AbstractVector ?
                        kernels.rho_hat[clamp(grp, 1, length(kernels.rho_hat))] :
                        kernels.rho_hat
                g_hat = kernels.gamma_hat isa AbstractVector ?
                        kernels.gamma_hat[clamp(grp, 1, length(kernels.gamma_hat))] :
                        kernels.gamma_hat
                construct_stochastic_transition_kernel(
                    W, hsi_d;
                    gamma     = g_hat,
                    residence = r_hat,
                    advection = a_hat,
                    land_mask = land_mask
                )
            else
                P_k
            end
            corr_d = if params.dynamic_kernels
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
            if corr_accum === nothing
                corr_accum = zeros(Float64, size(corr_d)...)
            end
            corr_accum .+= corr_d
        end
        reconstructed_corridors[string(tid)] = corr_accum ./ n_hsi_m

        # Optional stochastic least-cost path ensemble
        if wants_diagnostic(params, :stochastic) && cents_planar !== nothing
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
                    structural_uncertainty = params.add_structural_uncertainty,
                    structural_uncertainty_scale = params.structural_uncertainty_scale,
                    seed             = Int(
                        params.seed + abs(hash(string(tid))) % 10_000
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
                _record_panel_skip("Stochastic A* [$tid]", e); verbose && println("  (Stochastic A* note [$tid]: $e)")
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

    if wants_diagnostic(params, :bottlenecks) && cents_planar !== nothing
        verbose && println("\n[Phase 4b] Domain bottleneck detection...")
        n_obs              = nrow(obs_df)
        tag_sample_indices = Int[]

        if G > 1 && hasproperty(obs_df, :group)
            for g in 1:G
                cand = findall(
                    i -> obs_df.group[i] == g &&
                         obs_df.release[i] != obs_df.recapture[i] &&
                         (land_mask === nothing ||
                          (!land_mask[obs_df.release[i]] &&
                           !land_mask[obs_df.recapture[i]])),
                    1:n_obs
                )
                n_take = min(10, length(cand))
                n_take > 0 && append!(tag_sample_indices, cand[1:n_take])
            end
        else
            cand = findall(
                i -> obs_df.release[i] != obs_df.recapture[i] &&
                     (land_mask === nothing ||
                      (!land_mask[obs_df.release[i]] &&
                       !land_mask[obs_df.recapture[i]])),
                1:n_obs
            )
            n_take = min(25, length(cand))
            n_take > 0 && append!(tag_sample_indices, cand[1:n_take])
        end
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
                structural_uncertainty = params.add_structural_uncertainty,
                structural_uncertainty_scale = params.structural_uncertainty_scale,
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
# Phase 5: Advanced Diagnostics (Circuit Theory)
# =============================================================================

"""
    compute_advanced_diagnostics(loaded, path_results, params) -> NamedTuple

Phase 5 of the pipeline. Optionally computes:

**Circuit Theory** (`:circuit` in `params.diagnostics`):
- Multi-pair electrical current density I = C nabla V across all
  mark-recapture source-sink pairs.
- Identifies ecological pinch-points (top 10% current density).
- Posterior circuit inference via Monte Carlo HSI sampling:
    HSI_draw ~ N(hsi_mean, hsi_se^2)
  over `n_stochastic_draws * 2` replicates.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `path_results`: Output of `reconstruct_paths_and_diagnostics`.
- `params`: Relevant keys: `diagnostics`,
  `n_stochastic_draws`, `hsi_se`, `seed`, `verbose`.

# Returns
`NamedTuple` with `circuit` (or `nothing` if not computed).
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

    # -- Circuit Theory ------------------------------------------------------
    if wants_diagnostic(params, :circuit)
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
            _record_panel_skip("Circuit computation", e);             verbose && println("  (Circuit computation note: $(_error_note(e)))")
        end
    end

    # The trailing comma is required: a one-element parenthesised named tuple
    # without it collapses to the value itself, so `(circuit = nothing)` would
    # return a bare `nothing` and `diagnostics.circuit` would be unreachable.
    return (circuit = circuit_res,)
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
"""
function export_dashboards(
    loaded, kernels, path_results, diagnostics, params,
validation = nothing,
    agent_trajectories = nothing,
    agent_space_use = nothing
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
    G           = 1
    grp_nlookup = Dict{Int, String}()
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
    # tortuosity, mean HSI, and a per-group colour assignment.
    palette_colors = [
        "#38bdf8", "#f43f5e", "#10b981", "#fbbf24",
        "#a78bfa", "#fb923c", "#22d3ee", "#e879f9",
    ]
    all_paths_rich = NamedTuple[]
    path_dists_km  = Float64[]
    path_vels      = Float64[]
    path_bearings  = Float64[]

    for (tid, node_vec) in path_results.paths
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

        # Look up the group-specific colour for this tag
        sub_obs = filter(:tagid => ==(tid), obs_df)
          grp = !isempty(sub_obs) && hasproperty(sub_obs, :group) ?
                first(sub_obs.group) : 1
          color = palette_colors[
              (grp - 1) % length(palette_colors) + 1
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
    for (tid, stoch_res) in path_results.stochastic_paths
        stoch_res isa StochasticAStarResult || continue
        sub_obs = filter(:tagid => ==(tid), obs_df)
        grp = !isempty(sub_obs) && hasproperty(sub_obs, :group) ?
              first(sub_obs.group) : 1
        base_color = ibm_palette[(grp - 1) % length(ibm_palette) + 1]

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
                group           = grp,
                group_label     = "IBM Realization",
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
            grp = !isempty(sub_obs) && hasproperty(sub_obs, :group) ?
                  first(sub_obs.group) : 1

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
                group           = grp,
                group_label     = "IBM Forward Walk",
            ))
        end
    end

    # -- Movement paths dashboard ---------------------------------
    try
        html_file = joinpath(out_dir, "movement_paths_dashboard.html")
# Projected agents in the same shape the map expects for observed tags, so every
      # path source lands on one interactive map. They get their own toggleable
      # layer: reconstructed paths are conditioned on both observed endpoints,
      # projected ones on neither, and conflating them would be misleading.
      agent_segs = nothing
      if !isnothing(agent_trajectories) && nrow(agent_trajectories) > 0
          segs = Vector{Vector{Tuple{Float64, Float64}}}()
          for gdf in groupby(agent_trajectories, :tagid)
              srt = sort(gdf, :step)
              pts = [(Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
                     for u in srt.mesh_unit if 1 <= u <= n_units]
              length(pts) >= 2 && push!(segs, pts)
          end
          isempty(segs) || (agent_segs = segs)
      end

      map_obj = leaflet_tracks_map(
                all_paths_rich, au_mesh;
                empirical_paths     = emp_tracks,
                agent_paths         = agent_segs,
                max_agent_paths     = 200,
            max_paths           = max(100, length(all_paths_rich)),
            max_empirical_paths = length(emp_tracks),
            hsi                 = nothing,
            overlay_hsi         = false,
            dark_mode           = params.dark_mode,
            title               = "$spp Movement Trajectories" *
                                  reshard_lbl * depth_lbl
        )
        save_html(map_obj, html_file)
        verbose && println("  Paths dashboard: $html_file")
    catch e
        _record_panel_skip("Leaflet paths", e);         verbose && println("  (Leaflet paths note: $(_error_note(e)))")
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
        _record_panel_skip("Summary CSV", e);         verbose && println("  (Summary CSV note: $(_error_note(e)))")
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
_record_panel_skip("Summary diagnostics", e); verbose && println(
"  (Summary diagnostics note: $(_error_note(e)))"
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
        _record_panel_skip("Posterior uncertainty", e);         verbose && println("  (Posterior uncertainty note: $(_error_note(e)))")
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
        _record_panel_skip("Network flow", e);         verbose && println("  (Network flow note: $(_error_note(e)))")
    end

    # -- Interactive two-click corridor explorer -----------------------------
    try
        corr_file  = joinpath(out_dir, "movement_interactive_corridor.html")
        corr_file_pl = joinpath(out_dir, "movement_interactive_corridors.html")
# Projected agents, in the same shape the explorer expects for observed tags.
            # Rendered on their own toggleable layer so the two kinds of path stay
            # distinguishable: reconstructed paths are conditioned on both
            # endpoints, projected ones on neither.
corr_map   = leaflet_interactive_corridor_dashboard(
                  P_kernel, au_mesh;
                  hsi             = params.overlay_hsi ? hsi_vec : nothing,
                  overlay_hsi     = params.overlay_hsi,
                  empirical_paths = emp_tracks,
                  dark_mode       = params.dark_mode,
                title           = "$spp Dynamic Migration Corridor" *
                                  reshard_lbl * depth_lbl
            )
        save_html(corr_map, corr_file)
        save_html(corr_map, corr_file_pl)
        verbose && println("  Corridor dashboard: $corr_file")
    catch e
        _record_panel_skip("Corridor dashboard", e);         verbose && println("  (Corridor dashboard note: $(_error_note(e)))")
    end

    # -- Posterior path ensemble --------------------------------------------
    # Only produced when the :bayesian_ensemble diagnostic was requested, and
    # only worth drawing when it actually produced paths for at least one
    # individual. `validation` is `nothing` when neither it nor :validation ran.
    ensemble = isnothing(validation) ? nothing :
               get(validation, :bayesian_ensemble, nothing)
    if !isnothing(ensemble) && !isempty(ensemble.ensemble_paths)
        try
            ens_file = joinpath(out_dir, "movement_posterior_path_ensemble.html")
            ens_map  = leaflet_posterior_path_ensemble(
                ensemble, au_mesh;
                hsi         = params.overlay_hsi ? hsi_vec : nothing,
                overlay_hsi = params.overlay_hsi,
                dark_mode   = params.dark_mode,
                title       = "$spp Posterior Path Ensemble" * reshard_lbl * depth_lbl
            )
            save_html(ens_map, ens_file)
            verbose && println("  Posterior ensemble dashboard: $ens_file")
        catch e
            _record_panel_skip("Posterior ensemble", e);             verbose && println("  (Posterior ensemble note: $(_error_note(e)))")
        end
    end

    # -- Hydrodynamic dashboard (only when resharded) ------------------------
    if !isnothing(loaded.resharded_hydro)
        try
            hydro_file = joinpath(out_dir, "hydrodynamic_hex_dashboard.html")
            dash = leaflet_hydrodynamic_dashboard(
                loaded.resharded_hydro, au_mesh;
                title = "Hydrodynamics & Stratification (Fine Hexagons)",
                dark_mode = params.dark_mode
            )
            save_html(dash, hydro_file)
            verbose && println("  Hydrodynamic dashboard: $hydro_file")
        catch e
            _record_panel_skip("Hydrodynamic dashboard", e);             verbose && println("  (Hydrodynamic dashboard note: $(_error_note(e)))")
        end
    end

    # -- Circuit current density & stochastic pinch-point dashboards ---------
    if !isnothing(diagnostics.circuit)
        circ = diagnostics.circuit
        try
            circ_file = joinpath(out_dir, "movement_current_density.html")
            leaflet_current_density_map(
                mesh, circ.current_density;
                pinch_mask        = circ.pinch_mask,
                pinch_score       = circ.pinch_score,
                centroids         = path_results.cents_lonlat,
                output_html       = circ_file,
                dark_mode         = params.dark_mode,
                title             = "$spp Migratory Current Density & Pinch-Points",
                transparent_zeros = true,
                badge             = nothing
            )
            verbose && println("  Current density dashboard: $circ_file")
        catch e
            _record_panel_skip("Circuit density", e);             verbose && println("  (Circuit density note: $(_error_note(e)))")
        end

        try
            stoch_file = joinpath(out_dir, "movement_stochastic_circuit.html")
            leaflet_current_density_map(
                mesh, circ.stochastic;
                prob_threshold    = 0.80,
                output_html       = stoch_file,
                dark_mode         = params.dark_mode,
                title             = "$spp Posterior Migratory Flux & Pinch-Points",
                transparent_zeros = true,
                badge             = nothing
            )
            verbose && println("  Stochastic circuit dashboard: $stoch_file")
        catch e
            _record_panel_skip("Stochastic circuit", e);             verbose && println("  (Stochastic circuit note: $(_error_note(e)))")
        end
    end

    # -- Domain-wide bottleneck dashboard ------------------------------------
    if !isnothing(path_results.domain_bottlenecks)
        bn = path_results.domain_bottlenecks
        try
            bn_file = joinpath(out_dir, "movement_domain_bottlenecks.html")
            leaflet_current_density_map(
                mesh, bn.transit_density;
                pinch_mask        = bn.bottleneck_mask,
                pinch_score       = bn.bottleneck_score,
                centroids         = path_results.cents_lonlat,
                output_html       = bn_file,
                title             = "$spp Domain-Wide Pathways & Bottlenecks",
                dark_mode         = params.dark_mode,
                legend_title      = "Transit Density (C)",
                transparent_zeros = true,
                badge             = nothing
            )
            verbose && println("  Bottleneck dashboard: $bn_file")
        catch e
            _record_panel_skip("Bottleneck rendering", e);             verbose && println("  (Bottleneck rendering note: $(_error_note(e)))")
        end
    end

    # -- Speeds, Directions, Home Range, Corridors --------------------------
    # `loaded` exposes the mesh as `mesh`, suitability as `hsi_vec`, and the
    # per-unit current fields under `resharded_hydro`. The guards below use
    # `hasproperty` rather than `isnothing` on a named field, so a dataset
    # without hydrodynamics simply skips these panels instead of erroring.

    # 1. Step Diagnostics (Speeds, Turning Angles)
    if !isempty(path_results.paths)
        try
            step_file = joinpath(out_dir, "movement_step_diagnostics.html")
            map_obj = leaflet_step_diagnostics(
                path_results.paths, au_mesh;
                dark_mode = params.dark_mode,
                title = "$spp Speeds and Directions Distributions"
            )
            save_html(map_obj, step_file)
            verbose && println("  Step diagnostics dashboard: $step_file")
        catch e
            _record_panel_skip("Step diagnostics", e);             verbose && println("  (Step diagnostics note: $(_error_note(e)))")
        end
    end

    # 2. Regional Connectivity & Home Range Estimates
    # The panel is a heatmap of the region-to-region matrix, so it is only
    # meaningful once management units are configured. Without them the whole
    # domain collapses to a single region, whose 1x1 matrix describes the mesh
    # rather than connectivity, so the panel is reported as skipped instead of
    # being rendered from a placeholder.
    has_regions = !isnothing(loaded.region_labels) && !isempty(loaded.region_labels)
    if has_regions
        try
            conn_file = joinpath(out_dir, "movement_regional_connectivity.html")
            map_obj = leaflet_regional_connectivity(
                kernels.P_kernel;
                dark_mode = params.dark_mode,
                title = "$spp Regional Connectivity and Home Range Estimates"
            )
            save_html(map_obj, conn_file)
            verbose && println("  Regional connectivity dashboard: $conn_file")
        catch e
            _record_panel_skip("Regional connectivity", e);             verbose && println("  (Regional connectivity note: $(_error_note(e)))")
        end
    elseif verbose
        println(
            "  (Regional connectivity skipped: set `region_labels` and " *
            "`region_polygon_files` in the config file to render this panel)"
        )
    end

    # The hydro dataset reaches this scope only as loaded.resharded_hydro. It was
    # read here as a bare `hydro`, which is local to load_movement_data and so was
    # never in scope at all. Bound once, here, before either panel that reads it.
    hydro = loaded.resharded_hydro

    # 3. Advection Velocity Field
    # When advection or transition kernels are available, render directional drift arrows.
    # Drift vectors are derived from the model-fitted transition kernel (P_kernel)
    # or hydrodynamic current fields, strictly masked against land units.
    if !isnothing(kernels.P_kernel) || (hydro !== nothing && hasproperty(hydro, :advection_u))
        try
            adv_file = joinpath(out_dir, "movement_advection_velocity.html")
            u_vel = (hydro !== nothing && hasproperty(hydro, :u) && !isempty(hydro.u)) ?
                    vec(hydro.u[:, 1]) : nothing
            v_vel = (hydro !== nothing && hasproperty(hydro, :v) && !isempty(hydro.v)) ?
                    vec(hydro.v[:, 1]) : nothing
            map_obj = leaflet_advection_arrows(
                au_mesh;
                hsi        = params.overlay_hsi ? loaded.hsi_vec : nothing,
                background = params.overlay_hsi ? :hsi : :mesh,
                Gamma      = kernels.P_kernel,
                u_velocity = u_vel,
                v_velocity = v_vel,
                land_mask  = loaded.land_mask,
                cmap       = params.cmap,
                title      = "$spp Advection Drift & Velocity Field Vectors",
                dark_mode  = params.dark_mode
            )
            save_html(map_obj, adv_file)
            verbose && println("  Advection velocity dashboard: $adv_file")
        catch e
            _record_panel_skip("Advection velocity", e);             verbose && println("  (Advection velocity note: $(_error_note(e)))")
        end
    end

    # The hydro dataset reaches this scope only as loaded.resharded_hydro; see the
    # binding above, which also serves the advection panel.
    if hydro !== nothing &&
       all(k -> hasproperty(hydro, k), (:advection_u, :kappa_v))
        A = vec(hydro.advection_u)
        K = vec(hydro.kappa_v)
        # The advection/diffusion ratio is only defined where both fields are
        # actually populated. A resharded mesh can leave diffusivity as NaN off
        # the original grid, and a ratio against NaN is not a number to plot.
        if all(isfinite, A) && all(isfinite, K) && any(!iszero, K)
            try
                ad_file = joinpath(out_dir, "movement_ad_ratio_distribution.html")
                map_obj = leaflet_ad_ratio_distribution(
                    A, K;
                    dark_mode = params.dark_mode,
                    title = "$spp Advection/Diffusion Ratio"
                )
                save_html(map_obj, ad_file)
                verbose && println("  Advection ratio dashboard: $ad_file")
            catch e
                _record_panel_skip("Advection ratio", e);                 verbose && println("  (Advection ratio note: $(_error_note(e)))")
            end
        elseif verbose
            println(
                "  (Advection ratio skipped: diffusivity is not populated on the " *
                "resharded mesh, so the ratio is undefined)"
            )
        end
    end

    # 4. Residence Time & Diffusion Field
    # Both are derived per unit from the fitted kernel, so they are reported only
    # where the pipeline actually produced them rather than read off `loaded`.
    if hasproperty(kernels, :residence_by_unit) && !isnothing(kernels.residence_by_unit)
        try
            res_file = joinpath(out_dir, "movement_residence_time.html")
            map_obj = leaflet_residence_time_map(
                kernels.residence_by_unit, au_mesh;
                cmap    = params.cmap,
                    title = "$spp Residence Time Map"
            )
            save_html(map_obj, res_file)
            verbose && println("  Residence time dashboard: $res_file")
        catch e
            _record_panel_skip("Residence time", e);             verbose && println("  (Residence time note: $(_error_note(e)))")
        end
    end

    # `leaflet_diffusion_map` takes one value per spatial unit, so the
    # (depth x month) diffusivity field is collapsed to a long-run mean here.
    #
    # Diffusivity is not carried onto the resharded mesh (todo.md 1.2), so the
    # collapse yields NaN for most units. The panel is skipped in that case rather
    # than rendered: a map that draws successfully from NaN input is worse than no
    # map, because it reads as a result.
    if hydro !== nothing && hasproperty(hydro, :diffusivity_v)
        D_by_unit = vec(mean(Float64.(hydro.diffusivity_v); dims = 2))
        n_finite = count(isfinite, D_by_unit)
        if n_finite == 0
            verbose && println(
                "  (Diffusion dashboard skipped: no finite diffusivity on the " *
                "resharded mesh -- see todo.md 1.2)"
            )
        elseif n_finite < length(D_by_unit)
            verbose && println(
                "  (Diffusion dashboard skipped: only $n_finite of " *
                "$(length(D_by_unit)) units carry finite diffusivity)"
            )
        else
            try
                diff_file = joinpath(out_dir, "movement_diffusion_field.html")
                map_obj = leaflet_diffusion_map(
                    D_by_unit, au_mesh;
                    cmap      = params.cmap,
                    title = "$spp Diffusion Field"
                )
                save_html(map_obj, diff_file)
                verbose && println("  Diffusion dashboard: $diff_file")
            catch e
                _record_panel_skip("Diffusion", e);                 verbose && println("  (Diffusion note: $(_error_note(e)))")
            end
        end
    end

    # 5. HSI Map
    if !isnothing(loaded.hsi_vec)
        try
            hsi_file = joinpath(out_dir, "movement_hsi_map.html")
            map_obj = leaflet_hsi_map(
                loaded.hsi_vec, au_mesh;
                cmap    = params.cmap,
                    title = "$spp Habitat Suitability Index (HSI)"
            )
            save_html(map_obj, hsi_file)
            verbose && println("  HSI dashboard: $hsi_file")
        catch e
            _record_panel_skip("HSI map", e);             verbose && println("  (HSI map note: $(_error_note(e)))")
        end
    end

    # 6. Dispersal Kernel
    if !isnothing(kernels.P_kernel)
        try
            disp_file = joinpath(out_dir, "movement_dispersal_kernel.html")
            map_obj = leaflet_dispersal_kernel(
                kernels.P_kernel,                 au_mesh;
                title = "$spp Empirical Dispersal Kernel",
                dark_mode = params.dark_mode
            )
            save_html(map_obj, disp_file)
            verbose && println("  Dispersal kernel dashboard: $disp_file")
        catch e
            _record_panel_skip("Dispersal kernel", e);             verbose && println("  (Dispersal kernel note: $(_error_note(e)))")
        end
    end

    # 7. Tessellation Polygons Map
    try
        tess_file = joinpath(out_dir, "tessellation_polygons.html")
        tess_map = leaflet_tessellation_map(
            mesh;
            title = "$spp Tessellated Spatial Domain" * reshard_lbl * depth_lbl,
            depth = loaded.resharded_depths,
            hsi = loaded.hsi_vec,
            dark_mode = params.dark_mode,
            output_file = tess_file
        )
        show_map(tess_map; output_file = tess_file)
        verbose && println("  Tessellation polygon map: $tess_file")
    catch e
        _record_panel_skip("Tessellation polygon map", e);         verbose && println("  (Tessellation polygon map note: $(_error_note(e)))")
    end

    # -- Agent-Based Model trajectory dashboard ----------------------------
    # agent_trajectories is a DataFrame with columns: tagid, step, mesh_unit.
    # Convert to lightweight NamedTuples compatible with leaflet_tracks_map
    # and export, then write a companion CSV.
    if !isnothing(agent_trajectories) && nrow(agent_trajectories) > 0
        try
            # Build per-agent coordinate sequences grouped by tagid
            agent_rich = NamedTuple[]
            agent_idx = 0
            for gdf in groupby(agent_trajectories, :tagid)
                sorted = sort(gdf, :step)
                nodes  = sorted.mesh_unit
                agent_idx += 1
                coords = Tuple{Float64, Float64}[
                    (Float64(cents_ll[u][1]), Float64(cents_ll[u][2]))
                    for u in nodes
                    if 1 <= u <= n_units
                ]
                length(coords) < 2 && continue
                path_dist = sum(
                    haversine_distance(
                        coords[h-1][1], coords[h-1][2],
                        coords[h][1],   coords[h][2]
                    ) / 1_000.0
                    for h in 2:length(coords)
                )
                end_to_end = haversine_distance(
                    coords[1][1], coords[1][2],
                    coords[end][1], coords[end][2]
                ) / 1_000.0
                # The kernel is pooled, so every agent shares one transition law and
                # colouring by group no longer distinguishes anything. Cycle the
                # palette by agent instead, so overlapping tracks stay separable.
                color = palette_colors[(agent_idx - 1) % length(palette_colors) + 1]
                push!(agent_rich, (
                    tagid           = string("agent", first(sorted.tagid)),
                    path            = nodes,
                    coords          = coords,
                    n_steps         = length(coords) - 1,
                    total_dist_km   = path_dist,
                    displacement_km = end_to_end,
                    # Straight-line over walked distance. Reported as 1.0 when the
                    # animal returned to its release cell, where the ratio is
                    # genuinely undefined rather than perfectly untangled.
                    tortuosity      = end_to_end > 1e-9 ? path_dist / end_to_end : 1.0,
                    mean_hsi        = mean(
                        (1 <= u <= length(hsi_vec)) ? hsi_vec[u] : 0.5
                        for u in nodes
                    ),
                    color           = color,
                    duration_days   = Float64(length(coords) - 1),
                ))
            end
            if !isempty(agent_rich)
                agent_file = joinpath(out_dir, "movement_agent_trajectories.html")
                agent_map  = leaflet_tracks_map(
                    agent_rich, au_mesh;
                    max_paths   = length(agent_rich),
                    hsi         = params.overlay_hsi ? hsi_vec : nothing,
                    overlay_hsi = params.overlay_hsi,
                    dark_mode   = params.dark_mode,
                    title       = "$spp Agent-Based Model Trajectories"
                )
                save_html(agent_map, agent_file)
                verbose && println("  Agent trajectory dashboard: $agent_file")

                # Companion CSV: mean visit frequency per mesh unit
                visit_freq = zeros(Float64, n_units)
                for row in eachrow(agent_trajectories)
                    u = row.mesh_unit
                    1 <= u <= n_units && (visit_freq[u] += 1)
                end
                n_agents = length(unique(agent_trajectories.tagid))
                visit_freq ./= max(1, n_agents)
                csv_agent = joinpath(out_dir, "movement_agent_visit_frequency.csv")
                open(csv_agent, "w") do io
                    write(io, "mesh_unit,mean_visit_frequency\n")
                    for (u, f) in enumerate(visit_freq)
                        write(io, "$u,$(round(f; digits=6))\n")
                    end
                end
                verbose && println("  Agent visit-frequency CSV: $csv_agent")
            end

            # Projected space use. The track map shows individual samples; this
            # shows what the projection implies for the population, which is the
            # quantity a space-use question is actually about. Without it the
            # forward projection produced a frame that nothing summarised.
            if agent_space_use !== nothing && !isempty(agent_space_use.visit_probability)
                su_file = joinpath(out_dir, "movement_agent_space_use.html")
                su_map = leaflet_choropleth(
                    au_mesh.polygons_lonlat, agent_space_use.visit_probability;
                    title       = "$spp Projected Space Use (untagged animals)",
                    cmap        = params.cmap,
                    vmin        = 0.0,
                    vmax        = 1.0,
                    colorbar_label = "P(unit reached)",
                    dark_mode   = params.dark_mode,
                )
                save_html(su_map, su_file)
                verbose && println("  Projected space-use map: $su_file")

                # Table: one row per unit, ordered by how likely it is to be used.
                su_csv = joinpath(out_dir, "movement_agent_space_use.csv")
                order = sortperm(agent_space_use.visit_probability; rev = true)
                open(su_csv, "w") do io
                    write(io, "mesh_unit,visit_probability,visits,mean_dwell_steps,hsi\n")
                    for u in order
                        write(io, string(u, ',',
                            round(agent_space_use.visit_probability[u]; digits = 6), ',',
                            agent_space_use.visits[u], ',',
                            round(agent_space_use.mean_dwell_steps[u]; digits = 4), ',',
                            (1 <= u <= length(hsi_vec)) ? round(hsi_vec[u]; digits = 6) : "NA",
                            '\n'))
                    end
                end
                verbose && println("  Projected space-use table: $su_csv")

                # Console summary: the units the projection concentrates on.
                top_n = min(5, length(order))
                if top_n > 0
                    println("  Top projected-use units:")
                    for k in 1:top_n
                        u = order[k]
                        println("    unit $(rpad(u, 5)) p=$(round(agent_space_use.visit_probability[u]; digits = 3))  " *
                                "dwell=$(round(agent_space_use.mean_dwell_steps[u]; digits = 1)) steps  " *
                                "hsi=$(round(hsi_vec[u]; digits = 3))")
                    end
                end
            end
        catch e
            _record_panel_skip("Agent trajectory", e);             verbose && println("  (Agent trajectory note: $(_error_note(e)))")
        end

        # Does directional persistence actually buy anything over the memoryless
        # kernel? The agent projection reweights by heading rather than using a
        # (position, heading) kernel, which is a choice that ought to be defended
        # rather than assumed. This scores both against the observed transitions.
        if agent_space_use !== nothing || !isnothing(loaded.obs_df)
            try
                max_units = 400
                n_units <= max_units || throw(ArgumentError(
                    "persistence_gain_report forms T^k explicitly and is limited " *
                    "to $max_units units; this mesh has $n_units."
                ))
                # Bearing needs planar coordinates, so prefer the km centroids over lon/lat.
                gain_cents = hasproperty(au_mesh, :centroids_km) &&
                             !isnothing(au_mesh.centroids_km) ?
                             au_mesh.centroids_km : au_mesh.centroids
                kappas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
                gains = persistence_gain_report(
                    loaded.W, gain_cents, hsi_vec;
                    releases  = Int.(loaded.obs_df.release),
                    recaptures = Int.(loaded.obs_df.recapture),
                    ks         = max.(1, round.(Int, collect(loaded.obs_df.k))),
                    gamma      = kernels.gamma_hat,
                    residence  = kernels.rho_hat,
                    advection  = kernels.alpha_hat,
                    kappas     = kappas,
                    n_headings = params.n_headings,
                )

                csv_gain = joinpath(out_dir, "movement_persistence_gain.csv")
                open(csv_gain, "w") do io
                    write(io, "kappa,mean_loglik,gain_vs_first_order\n")
                    for p in gains.by_persistence
                        write(io, string(p.persistence, ',',
                            round(p.mean_loglik; digits = 6), ',',
                            round(p.gain_vs_first_order; digits = 6), '\n'))
                    end
                end

                # A "best" that sits on the edge of the grid has not been located.
                at_boundary = gains.best_persistence == maximum(kappas) &&
                              gains.improves_on_first_order

                if verbose
                    println("\n  Directional persistence vs the memoryless kernel:")
                    println("  | kappa | mean log-lik | gain vs first-order |")
                    println("  |------:|-------------:|--------------------:|")
                    for p in gains.by_persistence
                        println("  | $(p.persistence) | $(round(p.mean_loglik; digits = 4)) | $(round(p.gain_vs_first_order; digits = 4)) |")
                    end
                    println("  Best kappa = $(gains.best_persistence) " *
                            "(improves: $(gains.improves_on_first_order))")
                    if at_boundary
                        println("  NOTE: the best value is at the edge of the kappa grid " *
                                "($(maximum(kappas))), and the curve is still rising. " *
                                "The optimum has not been located; widen the grid before " *
                                "treating $(gains.best_persistence) as the estimate.")
                    end
                    println("  Table: $csv_gain")
                end
            catch e
                _record_panel_skip("Persistence gain", e); verbose && println("  (Persistence gain note: $(_error_note(e)))")
            end
        end
    end

    # -- Corridor aggregate heatmap -----------------------------------------
    # Sum Markov-bridge corridor matrices across all individuals, normalise to
    # [0, 1] visitation probability, and render as a choropleth.
    if !isempty(path_results.corridors)
        try
            n_sp = loaded.n_spatial
            corr_agg = zeros(Float64, n_sp)
            for (_, C) in path_results.corridors
                if C isa AbstractMatrix
                    corr_agg .+= vec(sum(C; dims = 2))
                elseif C isa AbstractVector
                    corr_agg .+= Float64.(C)
                end
            end
            cmax = maximum(corr_agg)
            cmax > 0 && (corr_agg ./= cmax)
            corr_file = joinpath(out_dir, "movement_corridors_heatmap.html")
            corr_map  = leaflet_choropleth(
                polys_ll, corr_agg;
                title             = "$spp Markov-Bridge Corridor Visitation (Aggregate)",
                cmap              = params.cmap,
                dark_mode         = params.dark_mode,
                transparent_zeros = true
            )
            save_html(corr_map, corr_file)
            verbose && println("  Corridor heatmap: $corr_file")
        catch e
            _record_panel_skip("Corridor heatmap", e);             verbose && println("  (Corridor heatmap note: $(_error_note(e)))")
        end
    end

    # -- Bottleneck uncertainty heatmap -------------------------------------
    # The bottleneck SE field quantifies cross-individual variability in the
    # structural bottleneck index. Compute SE across individual Markov-bridge
    # corridors when available, or fall back to stochastic domain bottlenecks.
    bse_vec = nothing
    if !isempty(path_results.corridors)
        n_sp = loaded.n_spatial
        W_mat = hasproperty(loaded, :W) ? loaded.W : nothing
        l_mask = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing
        deg_marine = ones(Float64, n_sp)
        if W_mat !== nothing
            deg_marine = [
                count(j -> W_mat[i, j] > 0 && (l_mask === nothing || !l_mask[j]), 1:n_sp)
                for i in 1:n_sp
            ]
            replace!(deg_marine, 0 => 1)
        end
        n_ind = length(path_results.corridors)
        if n_ind > 1
            scores_matrix = zeros(Float64, n_ind, n_sp)
            for (idx_k, (_, C_k)) in enumerate(path_results.corridors)
                c_vec = C_k isa AbstractMatrix ? vec(sum(C_k; dims = 2)) : Float64.(C_k)
                scores_matrix[idx_k, :] .= c_vec ./ deg_marine
            end
            bse_vec = [std(scores_matrix[:, u]) / sqrt(n_ind) for u in 1:n_sp]
        end
    end
    if (bse_vec === nothing || !any(>(0.0), bse_vec)) && !isnothing(path_results.domain_bottlenecks)
        bn = path_results.domain_bottlenecks
        if hasproperty(bn, :bottleneck_se) && any(>(0.0), bn.bottleneck_se)
            bse_vec = bn.bottleneck_se
        end
    end
    if bse_vec !== nothing && any(>(0.0), bse_vec)
        try
            bse_file = joinpath(out_dir, "movement_bottleneck_uncertainty.html")
            bse_map  = leaflet_choropleth(
                polys_ll, bse_vec;
                title             = "$spp Bottleneck Index Uncertainty (SE)",
                cmap              = params.cmap,
                dark_mode         = params.dark_mode,
                transparent_zeros = true
            )
            save_html(bse_map, bse_file)
            verbose && println("  Bottleneck SE dashboard: $bse_file")
        catch e
            _record_panel_skip("Bottleneck SE", e);             verbose && println("  (Bottleneck SE note: $(_error_note(e)))")
        end
    end

    # -- Stock connectivity HTML dashboard ----------------------------------
    # The validation connectivity matrix is a small region-to-region table;
    # render it as an interactive heatmap when regions are configured.
    if !isnothing(validation) &&
       !isnothing(get(validation, :connectivity_matrix, nothing))
        conn = get(validation, :connectivity_matrix, nothing)
        if !isnothing(conn) && hasproperty(conn, :connectivity_matrix) &&
           size(conn.connectivity_matrix, 1) > 1
            try
                conn_html = joinpath(out_dir, "movement_stock_connectivity.html")
                conn_map  = leaflet_regional_connectivity(
                    conn.connectivity_matrix;
                    dark_mode = params.dark_mode,
                    title     = "$spp Stock Connectivity Matrix"
                )
                save_html(conn_map, conn_html)
                verbose && println("  Stock connectivity HTML: $conn_html")
            catch e
                _record_panel_skip("Stock connectivity HTML", e);                 verbose && println("  (Stock connectivity HTML note: $(_error_note(e)))")
            end
        end
    end

    # -- Posterior Predictive Check HTML ------------------------------------
    # The existing export writes a plain-text summary; render a richer
    # SVG/HTML panel matching the style of the posterior parameter dashboard.
    if !isnothing(validation) &&
       !isnothing(get(validation, :posterior_predictive, nothing))
        ppc = get(validation, :posterior_predictive, nothing)
        if !isnothing(ppc) && hasproperty(ppc, :brier_scores) &&
           !isempty(ppc.brier_scores)
            try
                ppc_file = joinpath(out_dir, "movement_ppc_summary.html")
                _export_ppc_html(ppc_file, ppc; species = spp)
                verbose && println("  PPC summary dashboard: $ppc_file")
            catch e
                _record_panel_skip("PPC dashboard", e);                 verbose && println("  (PPC dashboard note: $(_error_note(e)))")
            end
        end
    end

    return (
        movement_stats = mov_stats,
        phenology      = pheno_res,
        trait_models   = trait_res,
    )
end


# =============================================================================
# Helper: Posterior Predictive Check HTML
# =============================================================================

"""
    _export_ppc_html(filepath, ppc; species = "generic") -> String

Renders a self-contained SVG/HTML posterior predictive check dashboard.
Includes:
- Brier score trace and KL divergence trace across MCMC draws
- Observed vs predicted recapture probability distribution
- Summary statistics (mean, 95% CI)
"""
function _export_ppc_html(filepath::String, ppc::NamedTuple;
                          species::String = "generic")::String
    mkpath(dirname(filepath))

    # Build SVG trace for a vector of per-draw values.
    function _trace_svg(
        vals::Vector{Float64},
        label::String,
        color::String;
        width::Int = 360, height::Int = 160
    )::String
        N = length(vals)
        N < 2 && return "<p>Insufficient draws</p>"
        vmin, vmax = minimum(vals), maximum(vals)
        abs(vmax - vmin) < 1e-14 && (vmax = vmin + 1.0)
        pad_l, pad_r, pad_t, pad_b = 48, 12, 18, 28
        pw = width  - pad_l - pad_r
        ph = height - pad_t - pad_b
        y_ax = height - pad_b

        io = IOBuffer()
        write(io, "<svg width=\"$width\" height=\"$height\" " *
                  "xmlns=\"http://www.w3.org/2000/svg\">")
        write(io, "<line x1=\"$pad_l\" y1=\"$y_ax\" x2=\"$(width - pad_r)\" " *
                  "y2=\"$y_ax\" stroke=\"#475569\" stroke-width=\"1\"/>")
        write(io, "<line x1=\"$pad_l\" y1=\"$pad_t\" x2=\"$pad_l\" " *
                  "y2=\"$y_ax\" stroke=\"#475569\" stroke-width=\"1\"/>")
        pts = String[]
        for i in 1:N
            px = pad_l + ((i - 1) / (N - 1)) * pw
            py = y_ax  - ((vals[i] - vmin) / (vmax - vmin)) * ph
            push!(pts, (i == 1 ? "M" : "L") *
                       " $(round(px; digits=1)) $(round(py; digits=1))")
        end
        write(io, "<path d=\"$(join(pts, " "))\" fill=\"none\" stroke=\"$color\" " *
                  "stroke-width=\"1.8\"/>")
        mid_x = pad_l + pw ÷ 2
        write(io, "<text x=\"$mid_x\" y=\"$(height - 6)\" " *
                  "text-anchor=\"middle\" fill=\"#94a3b8\" font-size=\"11\" " *
                  "font-family=\"Outfit, sans-serif\">$label</text>")
        # y-axis tick labels
        for (frac, lab) in ((0.0, round(vmin; digits = 4)),
                            (1.0, round(vmax; digits = 4)))
            py_t = round(Int, y_ax - frac * ph)
            write(io, "<text x=\"$(pad_l - 4)\" y=\"$py_t\" " *
                      "text-anchor=\"end\" fill=\"#64748b\" font-size=\"9\" " *
                      "font-family=\"JetBrains Mono, monospace\">$lab</text>")
        end
        write(io, "</svg>")
        String(take!(io))
    end

    # Observed vs mean predicted distribution bar chart.
    function _dist_svg(
        obs::Vector{Float64},
        pred::Vector{Float64};
        width::Int = 720, height::Int = 160
    )::String
        N = length(obs)
        N < 1 && return "<p>No distribution</p>"
        active_units = findall(i -> obs[i] > 0.0 || (i <= length(pred) && pred[i] > 1e-6), 1:N)
        if isempty(active_units)
            active_units = collect(1:min(N, 60))
        end
        sort!(active_units, by = u -> (obs[u], u <= length(pred) ? pred[u] : 0.0), rev = true)
        n_show = min(length(active_units), 60)
        u_sel  = active_units[1:n_show]
        obs_s  = obs[u_sel]
        pred_s = [u <= length(pred) ? pred[u] : 0.0 for u in u_sel]
        vmax   = max(maximum(obs_s), maximum(pred_s), 1e-14)
        pad_l, pad_r, pad_t, pad_b = 8, 8, 14, 22
        bw = max(1, (width - pad_l - pad_r) ÷ n_show)
        ph = height - pad_t - pad_b
        y_ax = height - pad_b
        io = IOBuffer()
        write(io, "<svg width=\"$width\" height=\"$height\" " *
                  "xmlns=\"http://www.w3.org/2000/svg\">")
        for i in 1:n_show
            x0  = pad_l + (i - 1) * bw
            h_o = round(Int, (obs_s[i]  / vmax) * ph)
            h_p = round(Int, (pred_s[i] / vmax) * ph)
            write(io, "<rect x=\"$x0\" y=\"$(y_ax - h_o)\" " *
                      "width=\"$(max(1, bw - 1))\" height=\"$h_o\" " *
                      "fill=\"#38bdf8\" opacity=\"0.75\"/>")
            write(io, "<rect x=\"$x0\" y=\"$(y_ax - h_p)\" " *
                      "width=\"$(max(1, bw - 1))\" height=\"$h_p\" " *
                      "fill=\"#f43f5e\" opacity=\"0.45\"/>")
        end
        mid_x = width ÷ 2
        write(io, "<text x=\"$mid_x\" y=\"$(height - 6)\" " *
                  "text-anchor=\"middle\" fill=\"#94a3b8\" font-size=\"11\" " *
                  "font-family=\"Outfit, sans-serif\">" *
                  "Active Spatial Units ($n_show ranked by density) \u2014 " *
                  "<tspan fill=\"#38bdf8\">\u25a0 Observed</tspan> " *
                  "<tspan fill=\"#f43f5e\">\u25a0 Predicted</tspan></text>")
        write(io, "</svg>")
        String(take!(io))
    end


    brier  = Float64.(ppc.brier_scores)
    kl     = Float64.(ppc.kl_divergences)
    obs_d  = Float64.(ppc.observed_dist)
    pred_d = hasproperty(ppc, :predicted_dist_mean) ?
             Float64.(ppc.predicted_dist_mean) : Float64[]

    sm = ppc.summary
    brier_m  = round(sm.brier_mean;     digits = 6)
    brier_lo = round(sm.brier_lower_ci; digits = 6)
    brier_hi = round(sm.brier_upper_ci; digits = 6)
    kl_m     = round(sm.kl_mean;        digits = 6)
    kl_lo    = round(sm.kl_lower_ci;    digits = 6)
    kl_hi    = round(sm.kl_upper_ci;    digits = 6)
    n_obs    = sm.n_observations
    n_draws  = sm.n_draws

    svg_brier = _trace_svg(brier, "Brier Score (draw)",    "#38bdf8")
    svg_kl    = _trace_svg(kl,    "KL Divergence (draw)",  "#f43f5e")
    svg_dist  = _dist_svg(obs_d, pred_d)

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>$species — Posterior Predictive Check</title>
<style>
  :root {
    --bg:          #0f172a;
    --surface:     #1e293b;
    --border:      #334155;
    --text:        #f1f5f9;
    --muted:       #94a3b8;
    --accent:      #38bdf8;
    --font-main:   'Outfit', system-ui, sans-serif;
    --font-mono:   'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: var(--font-main);
    padding: 28px 36px;
  }
  h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 6px; }
  .subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }
  .kpi-row {
    display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 28px;
  }
  .kpi {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 20px; min-width: 180px;
  }
  .kpi-label { font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }
  .kpi-value {
    font-size: 1.25rem; font-weight: 700;
    font-family: var(--font-mono); color: var(--accent);
  }
  .kpi-ci {
    font-size: 0.72rem; color: var(--muted);
    font-family: var(--font-mono); margin-top: 2px;
  }
  .section-title {
    font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--muted); margin: 22px 0 10px;
  }
  .traces { display: flex; gap: 20px; flex-wrap: wrap; }
  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .dist-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin-top: 20px;
  }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
</head>
<body>
<h1>$species — Posterior Predictive Check</h1>
<div class="subtitle">$n_obs observations · $n_draws MCMC draws evaluated</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Brier Score (MSE)</div>
    <div class="kpi-value">$brier_m</div>
    <div class="kpi-ci">95% CI [$brier_lo, $brier_hi]</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">KL Divergence</div>
    <div class="kpi-value">$kl_m</div>
    <div class="kpi-ci">95% CI [$kl_lo, $kl_hi]</div>
  </div>
</div>

<div class="section-title">Per-Draw Traces</div>
<div class="traces">
  <div class="panel">$svg_brier</div>
  <div class="panel">$svg_kl</div>
</div>

<div class="section-title">Recapture Probability Distribution</div>
<div class="dist-panel">$svg_dist</div>
</body></html>
"""
    write(filepath, html)
    return filepath
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
    # The ensemble is its own diagnostic and is listed separately in `diagnostics`,
    # so it must not be gated on `:validation` as well. Gating both on `validation`
    # meant `--diagnostics=bayesian_ensemble` on its own returned here and did
    # nothing at all, without a word of output.
    wants_validation  = wants_diagnostic(params, :validation)
    wants_ensemble    = wants_diagnostic(params, :bayesian_ensemble)
    (wants_validation || wants_ensemble) || return nothing
    verbose = params.verbose
    out_dir = params.output_dir
    mkpath(out_dir)

    pa = wants_validation ?
        (verbose && println("\n[Phase 5c] Running Validation Mark-Recapture Analyses...");
         run_validation_analyses(loaded, fitted, kernels, params, out_dir)) :
        nothing

    ensemble_res = if wants_ensemble
        verbose && println(
            "\n[Phase 5d] Running Bayesian ensemble path & corridor propagation..."
        )
        reconstruct_paths_bayesian_ensemble(loaded, fitted, params)
    else
        nothing
    end

    # `pa` is `nothing` when only the ensemble was requested, so each field is
    # taken from it only when the validation analyses actually ran.
    return (
        path_uncertainty         = isnothing(pa) ? nothing : pa.path_uncertainty,
        connectivity_matrix      = isnothing(pa) ? nothing : pa.connectivity_matrix,
        connectivity_uncertainty = isnothing(pa) ? nothing : pa.connectivity_uncertainty,
        posterior_predictive     = isnothing(pa) ? nothing : pa.posterior_predictive,
        summary_file             = isnothing(pa) ? nothing : pa.summary_file,
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
5. `compute_advanced_diagnostics`      -- circuit theory
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
`circuit`, `parameters`, `depth_range`.
"""
function run_movement_analysis(
    params = movement_parameters_default()
)::NamedTuple

    out_dir = params.output_dir
    mkpath(out_dir)

    # Panel failures are collected per run, so a previous run's dead panels are
    # never reported against this one.
    empty!(PANEL_SKIPS)

    results_checkpoint = joinpath(out_dir, "movement_results_checkpoint.jld2")

    # --figures-only: load the full results checkpoint and re-render dashboards
    # without re-running any computation.
    if params.figures_only
        isfile(results_checkpoint) || error(
            "[figures-only] Full results checkpoint not found: $results_checkpoint\n" *
            "Run the pipeline at least once without --figures-only to create it."
        )
        params.verbose && println(
            "\n[figures-only] Loading results checkpoint: $results_checkpoint"
        )
        rc = JLD2.load(results_checkpoint)
        loaded_r             = rc["loaded"]
        kernels_r            = rc["kernels"]
        path_res_r           = rc["path_res"]
        diagnostics_r        = rc["diagnostics"]
        validation_r         = rc["validation"]
        agent_trajectories_r = rc["agent_trajectories"]

        # Force render_html on so figures are actually written.
        params_render = merge(params, (render_html = true,))

        params_render.verbose && println("\n[figures-only] Regenerating dashboards...")
        export_dashboards(
            loaded_r, kernels_r, path_res_r, diagnostics_r,
            params_render, validation_r, agent_trajectories_r,
            agent_space_use_r
        )
        params_render.verbose && println("\n[figures-only] Done.")

        return (
            data               = loaded_r.data,
            models             = nothing,
            chains             = nothing,
            P_kernel           = kernels_r.P_kernel,
            paths              = path_res_r.paths,
            corridors          = path_res_r.corridors,
            stochastic_paths   = path_res_r.stochastic_paths,
            domain_bottlenecks = path_res_r.domain_bottlenecks,
            circuit            = diagnostics_r.circuit,
            validation_analyses = validation_r,
            agent_trajectories = agent_trajectories_r,
            movement_stats     = nothing,
            phenology          = nothing,
            trait_models       = nothing,
            parameters         = (
                alpha     = kernels_r.alpha_hat,
                residence = kernels_r.rho_hat,
                gamma     = kernels_r.gamma_hat,
            ),
            depth_range        = loaded_r.parsed_depth_range,
        )
    end

    checkpoint_file = joinpath(out_dir, "movement_checkpoint.jld2")
    resume_from_checkpoint = params.resume_from_checkpoint

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
    
    agent_trajectories = nothing
    agent_space_use = nothing
    if :agent in params.model_modes
        if params.verbose
            println("\n[Phase 2b] Projecting synthetic agents forward...")
        end
        n_sim_agents = params.n_agent_projections
        # Use observed release sites to start agents (column is :release, not :release_unit)
        release_nodes = Int.(loaded.obs_df.release)
        maximum(release_nodes) <= size(kernels.P_kernel, 1) || throw(BoundsError(
            kernels.P_kernel,
            "release unit $(maximum(release_nodes)) exceeds the fitted kernel",
        ))
        # The projection horizon is a modelling choice: how far to carry an untagged
        # animal forward. Telemetry does not identify it. The `k` column is a gap
        # ratio, delta_t over the nominal step, and it is 1 for every regularly
        # sampled interval -- so inheriting it blindly collapses every agent to a
        # single step and destroys the space-use question the projection exists
        # to answer. It is used only where it genuinely varies.
        durations = if hasproperty(loaded.obs_df, :k) && !isnothing(loaded.obs_df.k)
            k = max.(1, round.(Int, collect(loaded.obs_df.k)))
            (length(unique(k)) > 1 && any(>(1), k)) ? k : fill(params.agent_horizon, length(release_nodes))
        else
            fill(params.agent_horizon, length(release_nodes))
        end

        # _resolve_centroids returns (planar_km, lonlat, mesh_drawing). Headings
        # want one coordinate list in one space, so pick planar km when it exists
        # and fall back to lon/lat.
        agent_planar, agent_lonlat, _ =
            _resolve_centroids(loaded.mesh, size(kernels.P_kernel, 1))
        agent_centroids = agent_planar === nothing ? agent_lonlat : agent_planar

        agent_trajectories = forward_project_agents(
            release_nodes, durations;
            n_agents = n_sim_agents,
            transition_kernel = sparse(kernels.P_kernel),
            centroids = agent_centroids,
            persistence = params.persistence,
            seed = params.seed,
        )
        # Expected space use under the projected kernel: how likely an untagged
        # animal is to reach each unit, and how long it stays there once it does.
        agent_space_use = forward_space_use(agent_trajectories, loaded.n_spatial)
        if params.verbose
            println("  Projected $(n_sim_agents) agents over $(params.agent_horizon) steps.")
            top = findmax(agent_space_use.visit_probability)
            println("  Highest projected use: unit $(top[2]) (p = $(round(top[1]; digits = 3)))")
        end
    end
    
    path_res    = reconstruct_paths_and_diagnostics(loaded, kernels, params)
    diagnostics = compute_advanced_diagnostics(loaded, path_res, params)
    validation  = execute_validation_analyses(loaded, fitted, kernels, params)
    dashboards  = export_dashboards(
        loaded, kernels, path_res, diagnostics, params, validation, agent_trajectories,
    agent_space_use
    )

    # Write a full results checkpoint so --figures-only can regenerate dashboards
    # without re-running any computation.  Written after every successful run so
    # it always reflects the most recent results.
    try
        params.verbose && println(
            "\n[Checkpoint] Saving full results to: $results_checkpoint"
        )
        JLD2.save(
            results_checkpoint,
            "loaded",             loaded,
            "kernels",            kernels,
            "path_res",           path_res,
            "diagnostics",        diagnostics,
            "validation",         validation,
            "agent_trajectories", agent_trajectories,
        )
    catch e
_record_panel_skip("Results checkpoint write", e); params.verbose && println(
"  (Results checkpoint write skipped: $(_error_note(e)))"
            )
    end

    # Reported unconditionally, not under `verbose`. A panel that fails on every
    # run must not be able to do so invisibly.
    report_panel_skips()

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
        validation_analyses = validation,
        agent_trajectories = agent_trajectories,
          agent_space_use    = agent_space_use,
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
          # Panels that failed this run, so a caller does not have to read stdout
          # to find out that an output is missing.
          panel_skips       = copy(PANEL_SKIPS),
      )
    end
