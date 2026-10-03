"""
PlotlyJS-based dashboards.jl implementation.
Provides interactive PlotlyJS dashboards exported as self-contained offline HTML.
"""

using PlotlyJS
import GeometryBasics

export InteractiveMap, save_html, show_map
export plot_tessellation_map, plot_choropleth, plot_tracks_map
export plot_interactive_corridor_dashboard, plot_posterior_path_ensemble
export plot_hydrodynamic_dashboard, plot_current_density_map
export plot_step_diagnostics, plot_regional_connectivity
export plot_advection_arrows, plot_ad_ratio_distribution
export plot_residence_time_map, plot_diffusion_map, plot_hsi_map
export plot_dispersal_kernel

struct InteractiveMap
    fig::Any
    html::String
    title::String
    width::String
    height::String
    metadata::Dict{Symbol, Any}
end

function save_html(m::InteractiveMap, filepath::AbstractString)
    final_path = endswith(filepath, ".html") ? filepath : (filepath * ".html")
    # Save the plot as a self-contained HTML file
    savefig(m.fig, final_path, format="html")
    return abspath(final_path)
end

function show_map(m::InteractiveMap; output_file=nothing)
    if !isnothing(output_file)
        save_html(m, output_file)
    end
    return m
end

# Internal helpers
function _poly_to_lines(polys_lonlat)
    lons = Float64[]
    lats = Float64[]
    for ring in polys_lonlat
        for pt in ring
            push!(lons, pt[1])
            push!(lats, pt[2])
        end
        push!(lons, NaN)
        push!(lats, NaN)
    end
    return lons, lats
end

function _base_layout(title::String; dark_mode::Bool=false)
    bg = dark_mode ? "black" : "white"
    tc = dark_mode ? "white" : "black"
    return Layout(
        title=title,
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font_color=tc,
        geo=attr(
            projection_type="mercator",
            showcoastlines=true,
            coastlinecolor="gray",
            showland=true,
            landcolor=bg,
            bgcolor=bg,
            resolution=50,
            fitbounds="locations"
        ),
        margin=attr(l=0, r=0, t=40, b=0),
        width=1000,
        height=800
    )
end

function plot_tessellation_map(mesh; title="Domain", dark_mode=false, kwargs...)
    traces = GenericTrace[]
    if hasproperty(mesh, :polygons_lonlat) && !isempty(mesh.polygons_lonlat)
        lons, lats = _poly_to_lines(mesh.polygons_lonlat)
        push!(traces, scattergeo(
            lon=lons, lat=lats, mode="lines",
            line=attr(color=(dark_mode ? "gray" : "black"), width=1),
            showlegend=false, hoverinfo="none"
        ))
    end
    fig = plot(traces, _base_layout(title; dark_mode=dark_mode))
    return InteractiveMap(fig, "", title, "100%", "800px", Dict{Symbol,Any}())
end

function plot_choropleth(polys, values; title="Map", cmap="Viridis", vmin=0.0,
                         vmax=1.0, colorbar_label="", dark_mode=false, kwargs...)
    traces = GenericTrace[]
    if !isempty(polys)
        cents_lon = Float64[]
        cents_lat = Float64[]
        for ring in polys
            push!(cents_lon, mean([pt[1] for pt in ring]))
            push!(cents_lat, mean([pt[2] for pt in ring]))
        end
        push!(traces, scattergeo(
            lon=cents_lon, lat=cents_lat, mode="markers",
            marker=attr(color=values, colorscale=cmap, cmin=vmin, cmax=vmax, showscale=true, colorbar_title=colorbar_label),
            showlegend=false
        ))
    end
    fig = plot(traces, _base_layout(title; dark_mode=dark_mode))
    return InteractiveMap(fig, "", title, "100%", "800px", Dict{Symbol,Any}())
end

function plot_tracks_map(paths, mesh; title="Tracks", empirical_paths=nothing,
                         agent_paths=nothing, dark_mode=false, kwargs...)
    traces = GenericTrace[]
    if hasproperty(mesh, :polygons_lonlat) && !isempty(mesh.polygons_lonlat)
        lons, lats = _poly_to_lines(mesh.polygons_lonlat)
        push!(traces, scattergeo(
            lon=lons, lat=lats, mode="lines",
            line=attr(color="lightgray", width=0.5),
            showlegend=false, hoverinfo="none"
        ))
    end
    
    function add_paths!(paths_arr, color, width, name)
        if !isnothing(paths_arr)
            lons = Float64[]
            lats = Float64[]
            for p in paths_arr
                coords = hasproperty(p, :coords) ? p.coords :
                         (hasproperty(p, :coords_lonlat) ? p.coords_lonlat : p)
                if !isempty(coords)
                    for pt in coords
                        push!(lons, pt[1])
                        push!(lats, pt[2])
                    end
                    push!(lons, NaN)
                    push!(lats, NaN)
                end
            end
            if !isempty(lons)
                push!(traces, scattergeo(lon=lons, lat=lats, mode="lines", line=attr(color=color, width=width), name=name))
            end
        end
    end
    
    add_paths!(paths, "blue", 2, "Predicted")
    add_paths!(empirical_paths, "red", 2, "Empirical")
    add_paths!(agent_paths, "green", 1.5, "Agent")

    fig = plot(traces, _base_layout(title; dark_mode=dark_mode))
    return InteractiveMap(fig, "", title, "100%", "800px", Dict{Symbol,Any}())
end

function plot_interactive_corridor_dashboard(P, m; title="Corridor Dashboard",
                                             empirical_paths=nothing,
                                             dark_mode=false, kwargs...)
    traces = GenericTrace[]
    if hasproperty(m, :polygons_lonlat) && !isempty(m.polygons_lonlat)
        lons, lats = _poly_to_lines(m.polygons_lonlat)
        push!(traces, scattergeo(
            lon=lons, lat=lats, mode="lines",
            line=attr(color="lightgray", width=0.5),
            showlegend=false, hoverinfo="none"
        ))
    end
    if !isnothing(empirical_paths)
        lons = Float64[]
        lats = Float64[]
        for ep in empirical_paths
            if !isempty(ep)
                for pt in ep
                    push!(lons, pt[1])
                    push!(lats, pt[2])
                end
                push!(lons, NaN)
                push!(lats, NaN)
            end
        end
        if !isempty(lons)
            push!(traces, scattergeo(lon=lons, lat=lats, mode="lines", line=attr(color="cyan", width=2), name="Empirical"))
        end
    end
    fig = plot(traces, _base_layout(title; dark_mode=dark_mode))
    return InteractiveMap(fig, "", title, "100%", "800px", Dict{Symbol,Any}())
end

function plot_posterior_path_ensemble(all_ensembles, au_mesh; title="Posterior",
                                      dark_mode=false, kwargs...)
    traces = GenericTrace[]
    if hasproperty(au_mesh, :polygons_lonlat) && !isempty(au_mesh.polygons_lonlat)
        lons, lats = _poly_to_lines(au_mesh.polygons_lonlat)
        push!(traces, scattergeo(
            lon=lons, lat=lats, mode="lines",
            line=attr(color="lightgray", width=0.5),
            showlegend=false, hoverinfo="none"
        ))
    end
    if hasproperty(all_ensembles, :ensemble_paths)
        lons = Float64[]
        lats = Float64[]
        for p in all_ensembles.ensemble_paths
            if !isempty(p)
                for pt in p
                    push!(lons, pt[1])
                    push!(lats, pt[2])
                end
                push!(lons, NaN)
                push!(lats, NaN)
            end
        end
        if !isempty(lons)
            push!(traces, scattergeo(lon=lons, lat=lats, mode="lines", line=attr(color="rgba(255,165,0,0.4)", width=1.5), showlegend=false))
        end
    end
    fig = plot(traces, _base_layout(title; dark_mode=dark_mode))
    return InteractiveMap(fig, "", title, "100%", "800px", Dict{Symbol,Any}())
end

function plot_hydrodynamic_dashboard(hydro_data, mesh; title="Hydrodynamic",
                                     dark_mode=false, kwargs...)
    return plot_tessellation_map(mesh; title=title, dark_mode=dark_mode)
end

function plot_current_density_map(mesh, density; pinch_mask=nothing,
                                  pinch_score=nothing, centroids=nothing,
                                  output_html=nothing, title="Density",
                                  dark_mode=false, kwargs...)
    m = plot_choropleth((hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat : []), vec(density); 
                        title=title, cmap="Inferno", colorbar_label="Current Density", dark_mode=dark_mode)
    if !isnothing(output_html)
        save_html(m, output_html)
    end
    return m
end

function plot_step_diagnostics(paths, mesh; title="Steps", dark_mode=false, kwargs...)
    fig = plot([scatter(x=[0], y=[0])], Layout(title="Step Length Distribution"))
    return InteractiveMap(fig, "", title, "100%", "600px", Dict{Symbol,Any}())
end

function plot_regional_connectivity(P_kernel; title="Connectivity", dark_mode=false, kwargs...)
    bg = dark_mode ? "black" : "white"
    tc = dark_mode ? "white" : "black"
    traces = GenericTrace[]
    if isa(P_kernel, AbstractMatrix)
        push!(traces, heatmap(z=P_kernel, colorscale="Viridis", colorbar_title="Transition Probability"))
    end
    fig = plot(traces, Layout(title=title, plot_bgcolor=bg, paper_bgcolor=bg, font_color=tc))
    return InteractiveMap(fig, "", title, "100%", "700px", Dict{Symbol,Any}())
end

function plot_advection_arrows(mesh; title="Advection", dark_mode=false,
                               u_velocity=nothing, v_velocity=nothing, kwargs...)
    return plot_tessellation_map(mesh; title=title, dark_mode=dark_mode)
end

function plot_ad_ratio_distribution(A, K; title="AD Ratio", dark_mode=false, kwargs...)
    bg = dark_mode ? "black" : "white"
    tc = dark_mode ? "white" : "black"
    ratios = A ./ (K .+ 1e-6)
    valid_ratios = filter(isfinite, ratios)
    fig = plot([histogram(x=valid_ratios, marker_color="teal", nbinsx=30)], 
               Layout(title=title, plot_bgcolor=bg, paper_bgcolor=bg, font_color=tc))
    return InteractiveMap(fig, "", title, "100%", "600px", Dict{Symbol,Any}())
end

function plot_residence_time_map(pi_vec, mesh; title="Residence", dark_mode=false, kwargs...)
    return plot_choropleth((hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat : []), vec(pi_vec); 
                           title=title, cmap="Viridis", colorbar_label="Residence Time", dark_mode=dark_mode)
end

function plot_diffusion_map(D, mesh; title="Diffusion", dark_mode=false, kwargs...)
    return plot_choropleth((hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat : []), vec(D); 
                           title=title, cmap="Plasma", colorbar_label="Diffusivity", dark_mode=dark_mode)
end

function plot_hsi_map(hsi, mesh; title="HSI", dark_mode=false, kwargs...)
    return plot_choropleth((hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat : []), vec(hsi); 
                           title=title, cmap="Viridis", colorbar_label="HSI", dark_mode=dark_mode)
end

function plot_dispersal_kernel(kernel, mesh; title="Dispersal", dark_mode=false, kwargs...)
    bg = dark_mode ? "black" : "white"
    tc = dark_mode ? "white" : "black"
    traces = GenericTrace[]
    if isa(kernel, AbstractMatrix)
        push!(traces, heatmap(z=kernel, colorscale="Cividis", colorbar_title="Kernel Weight"))
    end
    fig = plot(traces, Layout(title=title, plot_bgcolor=bg, paper_bgcolor=bg, font_color=tc))
    return InteractiveMap(fig, "", title, "100%", "700px", Dict{Symbol,Any}())
end
