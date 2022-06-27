import ColorSchemes
using HDF5: h5open
using MCMCChainsStorage
import Random
import Logging
import ParetoSmooth
using Distributions: Exponential, Uniform
using Turing: Turing, @model, NUTS, filldist, MCMCThreads
using Statistics: mean, std
using PyFormattedStrings: @f_str
using Parameters: @with_kw


include("merge_chains.jl")
import .MergeChains

include("model_comparison.jl")
import .ModelComparison

# compute_waic, compute_z, compare, plot

###



##


@model function diffusion_1D_simple(Δ)

    # prior d
    d ~ Exponential(0.1)

    # likelihood
    for i = 1:length(Δ)
        Δ[i] ~ Rayleigh_from_d(d)
    end
    # Δ ~ filldist(Rayleigh_from_d(d), length(Δ))

    return
end




@model function diffusion_2D_simple(Δ)

    # prior d
    Δd ~ filldist(Exponential(0.1), 2)
    d = cumsum(Δd) # ordering

    dists = [Rayleigh_from_d(d_i) for d_i in d]

    # prior θ
    θ₁ ~ Uniform(0, 1)
    θ = [θ₁, 1 - θ₁]

    # likelihood
    for i = 1:length(Δ)
        Δ[i] ~ MixtureModel(dists, θ)
    end
    # Δ ~ filldist(MixtureModel(dists, θ), length(Δ))

    return (; d, θ)
end



@model function diffusion_3D_simple(Δ)

    # prior d
    Δd ~ filldist(Exponential(0.1), 3)
    d = cumsum(Δd) # ordering

    dists = [Rayleigh_from_d(d_i) for d_i in d]

    # prior θ
    Δθ ~ filldist(Uniform(0, 1), 3)
    # Δθ ~ filldist(Exponential(0.1), 3)
    θ = cumsum(Δθ) # ordering
    θ = θ / sum(θ) # normalization

    # likelihood
    for i = 1:length(Δ)
        Δ[i] ~ MixtureModel(dists, θ)
    end
    # Δ ~ filldist(distribution, length(Δ))

    return (; d, θ)

end

###



@model function diffusion_MSD(x, y, sy)

    R_inf ~ Exponential(0.1)
    d ~ Uniform(0, 1)
    σ ~ Uniform(0, 1)

    for i = 1:length(x)
        ỹ = f_MSD(x[i], R_inf, d, σ)
        y[i] ~ Normal(ỹ, sy[i])
    end

end


@model function bayesian_OLS(x, y, σ)
    # Our prior belief
    a ~ Normal(0.1, 0.1)
    b ~ Normal(0.1, 0.1)

    y_hat = a .* x .+ b

    y ~ Turing.MvNormal(y_hat, σ .^ 2)
end



@with_kw mutable struct Fit
    name::String
    model::Turing.AbstractMCMC.AbstractModel
    alg::Turing.Inference.InferenceAlgorithm = NUTS(0.65)
    N_samples::Int = 1000
    N_chains::Int = 1
    chains = missing
    WAIC = missing
end


function get_chain_name(fit::Fit)
    f"{fit.name}__{fit.N_samples}__samples__{fit.N_chains}__chains"
end


function get_waic_name(fit::Fit)
    f"{fit.name}__{fit.N_samples}__samples__{fit.N_chains}__waic"
end



function run_inference(fit::Fit)

    if fit.N_chains == 1
        return Turing.sample(fit.model, fit.alg, fit.N_samples)
    else
        return Turing.sample(fit.model, fit.alg, MCMCThreads(), fit.N_samples, fit.N_chains)
    end
end



function add_chains!(
    fit::Fit;
    forced::Bool = false,
    hide_warnings::Bool = false,
    merge_chains::Bool = true,
    save_chains::Bool = true,
)

    chain_name = get_chain_name(fit)
    filename = f"chains/{chain_name}.h5"

    if isfile(filename) && !forced
        println(f"Loading chains for {fit.name}")
        chains = h5open(filename, "r") do f
            read(f, Turing.Chains)
        end
        fit.chains = chains
        return nothing
    end

    println(f"Running Bayesian inference on {fit.name}, please wait.")
    Random.seed!(1)


    if !hide_warnings
        chains = run_inference(fit)
    else
        logger = Logging.SimpleLogger(Logging.Error)
        chains = Logging.with_logger(logger) do
            run_inference(fit)
        end
    end

    if merge_chains
        chains = MergeChains.merge(fit.model, chains)
    end

    if save_chains
        println(f"Saving {fit.name}")
        h5open(filename, "w") do f
            write(f, chains)
        end
    end

    fit.chains = chains
    return nothing

end



function pointwise_log_likelihoods(model::Turing.Model, chains::Turing.Chains)
    return ParetoSmooth.pointwise_log_likelihoods(model, chains)
end



function pointwise_log_likelihoods(fit::Fit; hide_warnings::Bool = true)

    if !hide_warnings
        log_likelihood = ParetoSmooth.pointwise_log_likelihoods(fit.model, fit.chains)
    else
        logger = Logging.SimpleLogger(Logging.Error)
        log_likelihood = Logging.with_logger(logger) do
            ParetoSmooth.pointwise_log_likelihoods(fit.model, fit.chains)
        end
    end

    return log_likelihood
end



function add_waic!(
    fit::Fit;
    forced::Bool = false,
    hide_warnings::Bool = true,
    save_waic::Bool = true,
)

    waic_name = get_waic_name(fit)
    filename = f"chains/{waic_name}.jld2"

    if isfile(filename) && !forced
        println(f"Loading WAIC for {fit.name}")
        waic = ModelComparison.load(filename)
        fit.WAIC = waic
        return nothing
    end

    if fit.chains isa Missing
        add_chains!(fit; hide_warnings = hide_warnings, forced = forced)
    end

    println(f"Computing WAIC for {fit.name}")
    log_likelihood = pointwise_log_likelihoods(fit; hide_warnings = hide_warnings)
    waic = ModelComparison.compute_waic(log_likelihood)

    if save_waic
        println(f"Saving WAIC for {fit.name}")
        ModelComparison.save(waic, filename)
    end

    fit.WAIC = waic
    return nothing
end



##

function get_colors()
    names = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "pink", "grey"]
    colors = ColorSchemes.Set1_9
    # d_colors = Dict(names .=> colors)
    return colors
end


function get_variables_in_group(chains::Turing.Chains, variable::Union{Symbol,String})
    return namesingroup(chains, variable)
end

function get_variables_in_group(chains::Turing.Chains, variables::Tuple)
    return reduce(vcat, get_variables_in_group(chains, variable) for variable in variables)
end


function get_variables_in_group(fit::Fit, variables::Tuple)
    return get_variables_in_group(fit.chains, variables)
end


function plot_chains(chains::Turing.Chains; variables = nothing, resolution = (1_000, 1200))

    if isnothing(variables)
        variables = names(chains, :parameters)
    end

    colors = get_colors()
    n_chains = length(Turing.chains(chains))
    n_samples = length(chains)

    fig = CairoMakie.Figure(; resolution = resolution)

    # left part of the plot // traces
    for (i, variable) in enumerate(variables)
        ax = CairoMakie.Axis(fig[i, 1]; ylabel = string(variable))
        for chain = 1:n_chains
            values = chains[:, variable, chain]
            CairoMakie.lines!(ax, 1:n_samples, values; label = string(chain))
        end

        if i == length(variables)
            ax.xlabel = "Iteration"
        end

    end

    # right part of the plot // density
    for (i, variable) in enumerate(variables)
        ax = CairoMakie.Axis(
            fig[i, 2];
            ylabel = string(variable),
            limits = (nothing, nothing, 0, nothing),
        )
        for chain = 1:n_chains
            values = chains[:, variable, chain]
            CairoMakie.density!(
                ax,
                values;
                label = string(chain),
                strokewidth = 3,
                strokecolor = (colors[chain], 0.8),
                color = (colors[chain], 0),
            )
        end

        CairoMakie.hideydecorations!(ax, grid = false)
        ax.title = string(variable)
        if i == length(variables)
            ax.xlabel = "Parameter estimate"
        end
    end

    return fig

end

function plot_chains(fit::Fit; variables = nothing, resolution = (1_000, 1200))
    return plot_chains(fit.chains; variables = variables, resolution = resolution)
end



function compute_U_left(chains::Turing.Chains)
    return compute_U_left(mean(chains[:Θ]))
end


function model_comparison(fits, waic_names::Vector{String})
    return ModelComparison.compare([fit.WAIC for fit in fits], waic_names)
end


function plot_model_comparison(df_comparison)
    return ModelComparison.plot(df_comparison)
end




function compute_WAICs_1D_2D_3D(
    df_Δ,
    model_names;
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
    forced = false,
)

    fit_1D = Fit(
        name = model_names[1],
        model = diffusion_1D_simple(df_Δ.Δ),
        N_samples = N_samples,
        N_chains = N_chains,
    )
    add_waic!(fit_1D; hide_warnings = hide_warnings, forced = forced)
    # fit_1D.WAIC

    fit_2D = Fit(
        name = model_names[2],
        model = diffusion_2D_simple(df_Δ.Δ),
        N_samples = N_samples,
        N_chains = N_chains,
    )
    add_chains!(fit_2D; hide_warnings = hide_warnings, forced = forced)
    add_waic!(fit_2D; hide_warnings = hide_warnings, forced = forced)
    # fit_2D.WAIC


    fit_3D = Fit(
        name = model_names[3],
        model = diffusion_3D_simple(df_Δ.Δ),
        N_samples = N_samples,
        N_chains = N_chains,
    )
    add_chains!(fit_3D; hide_warnings = hide_warnings, forced = forced)
    add_waic!(fit_3D; hide_warnings = hide_warnings, forced = forced)

    return fit_1D, fit_2D, fit_3D

end


function compute_and_plot_WAICs(
    df_Δ,
    name;
    suffix = "simple",
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
    forced = false,
    ic = :waic,
)

    model_names = [f"{name}_{i}D_{suffix}" for i = 1:3]
    fits = compute_WAICs_1D_2D_3D(
        df_Δ,
        model_names,
        N_samples = N_samples,
        N_chains = N_chains,
        hide_warnings = hide_warnings,
        forced = forced,
    )

    waic_names = [f"{name} {i}D" for i = 1:3]
    df_comparison = model_comparison(fits, waic_names)

    fig_comparison_waic = plot_model_comparison(df_comparison)
    return df_comparison, fig_comparison_waic

end
