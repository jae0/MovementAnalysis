"""
    MovementAnalysisPlottingExt

Optional plotting backend for MovementAnalysis.

Raster/`Plots`-style diagnostics call `plot` and `savefig`, which the base
package deliberately does not depend on: `Plots` pulls in GR and is much heavier
than the rest of the dependency tree, and the package's primary output is
self-contained Leaflet HTML. Installing `Plots` in the user's environment loads
this extension automatically and supplies the real implementations.

Without `Plots` the base methods remain defined and raise a clear
`ArgumentError` naming the missing dependency, rather than failing later with an
obscure `UndefVarError: plot not defined`.
"""
module MovementAnalysisPlottingExt

using MovementAnalysis
using Plots
using Statistics: mean

# Concrete arity is more specific than the base `args...` fallback, so these
# definitions add methods rather than overwriting the base ones.

function MovementAnalysis.plot_posterior_predictive_check(ppc, output_dir)
    mkpath(output_dir)
    plot_file = joinpath(output_dir, "posterior_predictive_diagnostics.png")

    p1 = plot(ppc.brier_scores; label = "Brier Score", xlabel = "Draw",
              ylabel = "Score", title = "Posterior Predictive: Brier Score",
              legend = :topright)
    p2 = plot(ppc.kl_divergences; label = "KL Divergence", xlabel = "Draw",
              ylabel = "Divergence", title = "Posterior Predictive: KL Divergence",
              legend = :topright)
    p3 = plot(1:length(ppc.observed_dist), ppc.observed_dist;
              label = "Observed", xlabel = "Spatial Unit", ylabel = "Probability",
              title = "Recapture Probability Distribution")

    plot(p1, p2, p3; layout = (3, 1), size = (800, 900))
    savefig(plot_file)
    return plot_file
end

function MovementAnalysis._ad_ratio_histogram_via_plots(
    advection_field::AbstractVector{<:Real},
    diffusion_field::AbstractVector{<:Real}
)
    ratios = advection_field ./ (mean(diffusion_field) .+ 1e-6)
    plt = Plots.histogram(
        ratios, bins = 25,
        title = "Advection-to-Diffusion Ratio (Péclet-like)",
        xlabel = "Ratio (Advection / Diffusion)", ylabel = "Frequency",
        label = "Spatial Units", color = :plum, linecolor = :white
    )
    Plots.vline!(
        plt, [1.0], color = :red, linestyle = :dash, linewidth = 2.0,
        label = "Equilibrium Threshold"
    )
    return plt
end

end # module
