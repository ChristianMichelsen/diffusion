import ColorSchemes
using Turing: MCMCThreads


function initialize_fit(; model, alg = NUTS(0.65), N_samples = 1000, N_chains = 1)
    d_fit =
        Dict(:model => model, :alg => alg, :N_samples => N_samples, :N_chains => N_chains)
    return d_fit
end


function fit!(d_fit::Dict)

    if d_fit[:N_chains] == 1
        chains = sample(d_fit[:model], d_fit[:alg], d_fit[:N_samples])
    else
        chains = sample(
            d_fit[:model],
            d_fit[:alg],
            MCMCThreads(),
            d_fit[:N_samples],
            d_fit[:N_chains],
        )
    end

    d_fit[:chains] = chains
end

function get_chains(d_fit::Dict)
    return d_fit[:chains]
end

function get_colors()
    names = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "pink", "grey"]
    colors = ColorSchemes.Set1_9
    # d_colors = Dict(names .=> colors)
    return colors
end


function plot_chains(chains::Turing.MCMCChains.Chains, resolution = (1_000, 1200))

    params = names(chains, :parameters)

    colors = get_colors()
    n_chains = length(Turing.chains(chains))
    n_samples = length(chains)

    fig = Figure(; resolution = resolution)

    # left part of the plot // traces
    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 1]; ylabel = string(param))
        for chain = 1:n_chains
            values = chains[:, param, chain]
            lines!(ax, 1:n_samples, values; label = string(chain))
        end

        if i == length(params)
            ax.xlabel = "Iteration"
        end

    end

    # right part of the plot // density
    for (i, param) in enumerate(params)
        ax =
            Axis(fig[i, 2]; ylabel = string(param), limits = (nothing, nothing, 0, nothing))
        for chain = 1:n_chains
            values = chains[:, param, chain]
            density!(
                ax,
                values;
                label = string(chain),
                strokewidth = 3,
                strokecolor = (colors[chain], 0.8),
                color = (colors[chain], 0),
            )
        end

        hideydecorations!(ax, grid = false)
        ax.title = string(param)
        if i == length(params)
            ax.xlabel = "Parameter estimate"
        end
    end

    return fig

end


function plot_chains(d_fit::Dict, resolution = (1_000, 1200))
    chains = get_chains(d_fit)
    return plot_chains(chains, resolution)
end