"""
    MovementAnalysis

Standalone Spatial Animal Movement, Trajectory Reconstruction, and Ecological
Connectivity Analysis Engine. Provides mark-recapture telemetry ingestion,
stochastic transition kernel estimation, explicit Turing.jl probabilistic
telemetry calibration, A* least-cost trajectory and corridor routing, electrical
circuit theory pinchpoint analysis, Chebyshev Spectral Graph Wavelets (SGWT),
and interactive Leaflet HTML dashboard generation.
"""
module MovementAnalysis

using LinearAlgebra
using SparseArrays
using Statistics
using StatsBase
using Distributions
using Random
using Dates
using DataFrames
using Printf

using Graphs
using NearestNeighbors
using LibGEOS
using CoordRefSystems
using Unitful

using FFTW
using Wavelets
using WaveletsExt

using JLD2
using DuckDB
using Turing

# Source modules in logical dependency order
include("spatial_utils.jl")
include("ssa_movement.jl")
include("turing_models.jl")
include("movement.jl")
include("circuit.jl")
include("graph_wavelets.jl")
include("dashboards.jl")
include("pipeline.jl")
include("agent_movement.jl")

# -----------------------------------------------------------------------------
# Public API Exports
# -----------------------------------------------------------------------------

export
    # Spatial Mesh & Partitioning Utilities
    build_hex_mesh_planar,
    map_point_to_units,
    map_to_units,
    assign_spatial_units,
    load_open_bathymetry,
    extract_hydrodynamic_dataset,
    compute_network_transfer_matrix,
    get_polygon_area,
    summarize_sample_matrix,
    reshard_spatial_field,

    # Continuous-Time SSA Movement Models (Master Equation)
    SSAMovementParams,
    calculate_ssa_utility,
    construct_ssa_generator,
    calculate_ssa_transition_matrix,
    simulate_gillespie_trajectories,
    generate_ssa_movement_data,

    # Probabilistic Turing Models
    pure_telemetry_turing_model,
    joint_survey_telemetry_turing_model,
    ssa_telemetry_turing_model,
    joint_survey_ssa_telemetry_turing_model,

    # Agent-Based Model Alternative
    CrabAgent,
    simulate_agent_trajectories,

    # Telemetry & Mark-Recapture Data Structures
    TelemetryData,
    prepare_movement_data,
    validate_telemetry,
    snowcrab_movement_data,
    haversine_distance,
    lonlat_to_xy_km,
    xy_km_to_lonlat,
    filter_dead_tags,
    summarize_tag_activity,
    tag_to_study_id,

    # Kernel Construction & Transition Probabilities
    construct_stochastic_transition_kernel,
    construct_dynamic_transition_kernels,
    calculate_multistep_transition,
    compute_directed_adjacency,
    resolvent_transition,
    power_transition,
    simulate_correlated_density_vector,
    generate_ADR_simulation_bundle,
    generate_movement_data,

    # Path & Corridor Trajectory Reconstruction
    predict_path,
    astar_least_cost_path,
    astar_predict_path,
    astar_stochastic_least_cost_path,
    astar_stochastic_predict_path,
    astar_multiresolution_path,
    construct_adaptive_multiresolution_domain,
    StochasticAStarResult,
    AStar,
    get_astar_paths,
    smooth_marine_path,
    predict_corridor,
    predict_dynamic_path,
    predict_dynamic_corridor,
    sample_markov_bridge,
    viterbi_hmm_path_smoothing,
    forward_backward_state_probabilities,
    simulate_posterior_trajectories,
    simulate_mechanistic_trajectories,

    # Movement Ecology & Demographic Analysis
    compute_movement_statistics,
    analyze_seasonal_movement_phenology,
    model_trait_movement_associations,
    export_movement_summary_csv,
    path_credible_intervals,
    export_path_uncertainty_summary,
    reconstruct_paths_bayesian_ensemble,
    compute_stock_connectivity_matrix,
    compute_connectivity_credible_intervals,
    export_connectivity_matrix,
    export_connectivity_uncertainty,
    posterior_predictive_check,
    export_posterior_predictive_check,
    plot_posterior_predictive_check,
    run_priority_analyses,

    # Electrical Circuit Theory & Ecological Pinchpoints
    Circuit,
    PosteriorCircuitResult,
    build_circuit_laplacian,
    effective_resistance_matrix,
    solve_circuit_voltage,
    pairwise_effective_resistance,
    current_density_map,
    identify_ecological_pinchpoints,
    identify_stochastic_pinchpoints,
    posterior_circuit_inference,
    get_circuit_paths,
    resistance_covariance_matrix,

    # Spectral Graph Wavelet Transform (SGWT)
    GraphWavelet,
    SpectralGraphWaveletResult,
    build_normalized_laplacian,
    compute_laplacian_spectral_bounds,
    chebyshev_polynomial_coefficients,
    apply_graph_spectral_filter,
    spectral_graph_wavelet_transform,
    inverse_spectral_graph_wavelet_transform,
    denoise_spatial_signal_wavelet,
    graph_wavelet_basis_matrix,

    # Interactive HTML / Leaflet Visualization
    LeafletMap,
    save_html,
    utm_to_lonlat,
    lonlat_to_utm,
    leaflet_choropleth,
    leaflet_spatial_map,
    leaflet_spatial_graph,
    leaflet_hsi_map,
    leaflet_diffusion_map,
    leaflet_residence_time_map,
    leaflet_advection_arrows,
    leaflet_velocity_field,
    leaflet_tracks_map,
    leaflet_render_paths,
    leaflet_spacetime_map,
    leaflet_movement_dashboard,
    leaflet_interactive_corridor_dashboard,
    leaflet_dispersal_kernel,
    leaflet_step_diagnostics,
    leaflet_regional_connectivity,
    leaflet_ad_ratio_distribution,
    leaflet_hydrodynamic_dashboard,
    leaflet_current_density_map,
    leaflet_graph_wavelet_dashboard,
    export_movement_posterior_dashboard,
    export_movement_flow_dashboard,
    export_movement_summary_dashboard,

    # Pipeline Orchestration
    movement_parameters_default,
    movement_parameters_snowcrab,
    load_movement_data,
    fit_movement_models,
    extract_transition_kernels,
    reconstruct_paths_and_diagnostics,
    compute_advanced_diagnostics,
    export_dashboards,
    execute_priority_analyses,
    run_movement_analysis

function __init__()
    Random.seed!(42)
end

end # module MovementAnalysis
