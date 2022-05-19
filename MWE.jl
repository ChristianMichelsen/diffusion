using Turing: Turing, Model, @model, NUTS, sample, filldist, MCMCThreads, namesingroup
using Distributions: Exponential, Uniform, Rayleigh, MixtureModel

save_plots = false


function generate_data()
    x1 = rand(Exponential(0.1), 1000)
    x2 = rand(Exponential(0.5), 500)
    return vcat(x1, x2)
end

x = generate_data();


using CairoMakie

function plot_chains(chns)

    params = sort(names(chns, :parameters))

    n_chains = length(Turing.chains(chns))
    n_samples = length(chns)

    fig = Figure(; resolution = (1_000, 800))

    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 1]; ylabel = string(param))
        for chain = 1:n_chains
            values = chns[:, param, chain]
            lines!(ax, 1:n_samples, values; label = string(chain))
        end

        hideydecorations!(ax; label = false)
        if i < length(params)
            hidexdecorations!(ax; grid = false)
        else
            ax.xlabel = "Iteration"
        end
    end

    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 2]; ylabel = string(param))
        for chain = 1:n_chains
            values = chns[:, param, chain]
            density!(ax, values; label = string(chain))
        end

        hideydecorations!(ax)
        if i < length(params)
            hidexdecorations!(ax; grid = false)
        else
            ax.xlabel = "Parameter estimate"
        end
    end

    axes = [only(contents(fig[i, 2])) for i = 1:length(params)]
    linkxaxes!(axes...)

    axislegend(first(axes))

    fig

end



@model function no_ordering(x)

    # prior d
    rs ~ filldist(Exponential(0.1), 2)

    dists = [Exponential(r) for r in rs]

    # prior θ
    Θ ~ Uniform(0, 1)
    w = [Θ, 1 - Θ]
    f = w[2]

    # mixture distribution
    distribution = MixtureModel(dists, w)

    # likelihood
    x ~ filldist(distribution, length(x))

end

model_no_ordering = no_ordering(x);
chains_no_ordering = sample(model_no_ordering, NUTS(0.65), MCMCThreads(), 100, 4);


plot_chains(chains_no_ordering)
save_plots && save("chains_no_ordering.png", plot_chains(chains_no_ordering))


@model function with_ordering(x)

    # prior d
    Δr ~ filldist(Exponential(0.1), 2)
    rs = cumsum(Δr)

    dists = [Exponential(r) for r in rs]

    # prior θ
    Θ ~ Uniform(0, 1)
    w = [Θ, 1 - Θ]
    f = w[2]

    # mixture distribution
    distribution = MixtureModel(dists, w)

    # likelihood
    x ~ filldist(distribution, length(x))

    return (; rs, f)
end


model_with_ordering = with_ordering(x)
chains_with_ordering = sample(model_with_ordering, NUTS(0.65), MCMCThreads(), 100, 4)

plot_chains(chains_with_ordering)
save_plots && save("chains_with_ordering.png", plot_chains(chains_with_ordering))


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


function get_generated_quantities(model::Turing.Model, chains::Turing.Chains)
    chains_params = Turing.MCMCChains.get_sections(chains, :parameters)
    generated_quantities = Turing.generated_quantities(model, chains_params)
    return generated_quantities
end

function get_generated_quantities(dict::Dict)
    return get_generated_quantities(dict[:model], dict[:chains])
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
    dict[:variables] = get_variables(dict)
    dict[:N_samples] = get_N_samples(dict)
    dict[:N_chains] = get_N_chains(dict)

    dict[:generated_chains] = generated_quantities_to_chains(dict)
    return merge_generated_chains(dict)

end

merged_chains = get_merged_chains(model_with_ordering, chains_with_ordering)

variables = [namesingroup(merged_chains, :rs)..., :Θ]
plot_chains(merged_chains[variables])
save_plots && save("merged_chains.png", plot_chains(merged_chains[variables]))

# model = model_with_ordering
# chains = chains_with_ordering