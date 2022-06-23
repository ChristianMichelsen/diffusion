import ColorSchemes
using Turing: MCMCThreads
using HDF5: h5open
using MCMCChainsStorage
import Random
import Logging

###


@model function diffusion_2D_simple(Δ)

    # prior d
    Δds ~ filldist(Exponential(0.1), 2)
    ds = cumsum(Δds)

    dists = [diffusion_1D(d) for d in ds]

    # prior θ
    Θ ~ Uniform(0, 1)
    w = [Θ, 1 - Θ]

    # mixture distribution
    distribution = MixtureModel(dists, w)

    # likelihood
    Δ ~ filldist(distribution, length(Δ))

    return (; ds)
end



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

    y ~ Turing.MvNormal(y_hat, σ.^2)
end


function get_unique_name(model_name::String, N_samples::Int, N_chains::Int = 1)
    return f"{model_name}__{N_samples}__samples__{N_chains}__chains"
end


function run_inference(model, alg, N_samples, N_chains)

    if N_chains == 1
        return Turing.sample(model, alg, N_samples)
    else
        return Turing.sample(model, alg, MCMCThreads(), N_samples, N_chains)
    end

end

function get_chains(;
    name::String,
    model::Turing.AbstractMCMC.AbstractModel,
    alg::Turing.Inference.InferenceAlgorithm = NUTS(0.65),
    N_samples::Int = 1000,
    N_chains::Int = 1,
    forced::Bool = false,
    hide_warnings::Bool = false,
    merge_chains::Bool = true,
)

    model_name = get_unique_name(name, N_samples, N_chains)
    filename = f"chains/{model_name}.h5"

    if isfile(filename) && !forced
        println(f"Loading {name}")
        chains = h5open(filename, "r") do f
            read(f, Turing.Chains)
        end
        return chains

    else

        println(f"Running Bayesian inference on {name}, please wait.")
        Random.seed!(1)


        if !hide_warnings
            chains = run_inference(model, alg, N_samples, N_chains)
        else
            logger = Logging.SimpleLogger(Logging.Error)
            chains = Logging.with_logger(logger) do
                run_inference(model, alg, N_samples, N_chains)
            end
        end

        if merge_chains
            chains = get_merged_chains(model, chains)
        end

        println(f"Saving {name}")
        h5open(filename, "w") do f
            write(f, chains)
        end

        return chains

    end

end


##

function get_generated_quantities(model::Turing.Model, chains::Turing.Chains)
    chains_params = Turing.MCMCChains.get_sections(chains, :parameters)
    generated_quantities = Turing.generated_quantities(model, chains_params)
    return generated_quantities
end


function get_generated_quantities(dict::Dict)
    return get_generated_quantities(dict[:model], dict[:chains])
end


""" Get the number of dimensions (K) for the specific variable """
function get_K(dict::Dict, variable::Union{Symbol,String})
    K = length(first(dict[:generated_quantities])[variable])
    return K
end


function get_variables(dict::Dict)
    return dict[:generated_quantities] |> first |> keys
end


function get_N_samples(dict::Dict)
    return length(dict[:chains])
end


function get_N_chains(dict::Dict)
    return length(Turing.chains(dict[:chains]))
end


function generated_quantities_to_chain(dict::Dict, variable::Union{Symbol,String})

    K = get_K(dict, variable)

    matrix = zeros(dict[:N_samples], K, dict[:N_chains])
    for chain = 1:dict[:N_chains]
        for (i, xi) in enumerate(dict[:generated_quantities][:, chain])
            matrix[i, :, chain] .= xi[variable]
        end
    end

    if K == 1
        chain_names = [Symbol("$variable")]
    else
        chain_names = [Symbol("$variable[$i]") for i = 1:K]
    end
    generated_chain = Turing.Chains(matrix, chain_names, info = dict[:chains].info)

    return generated_chain

end


function generated_quantities_to_chains(dict::Dict)
    return hcat(
        [generated_quantities_to_chain(dict, variable) for variable in dict[:variables]]...,
    )
end


function merge_generated_chains(dict::Dict)
    return hcat(
        dict[:chains],
        Turing.setrange(dict[:generated_chains], range(dict[:chains])),
    )
end


function get_merged_chains(model::Turing.Model, chains::Turing.Chains)

    dict = Dict{Symbol,Any}(:model => model, :chains => chains)

    dict[:generated_quantities] = get_generated_quantities(dict)

    if dict[:generated_quantities] isa Matrix{Nothing}
        return chains
    end

    dict[:variables] = get_variables(dict)
    dict[:N_samples] = get_N_samples(dict)
    dict[:N_chains] = get_N_chains(dict)

    dict[:generated_chains] = generated_quantities_to_chains(dict)
    return merge_generated_chains(dict)

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


function compute_U_left(chains::Turing.Chains)
    return compute_U_left(mean(chains[:Θ]))
end

