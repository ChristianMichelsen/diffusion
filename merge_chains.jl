
module MergeChains

export merge


import Turing


function get_generated_quantities(model::Turing.Model, chains::Turing.Chains)
    chains_params = Turing.MCMCChains.get_sections(chains, :parameters)
    generated_quantities = Turing.generated_quantities(model, chains_params)
    return generated_quantities
end


function generated_quantities_to_chain(
    generated_quantities::AbstractMatrix,
    chains::Turing.Chains,
    variable::Union{Symbol,String},
)

    # The number of dimensions (K) for the specific variable
    K = length(first(generated_quantities)[variable])
    N_samples = length(chains)
    N_chains = length(Turing.chains(chains))

    matrix = zeros(N_samples, K, N_chains)
    for chain = 1:N_chains
        for (i, xi) in enumerate(generated_quantities[:, chain])
            matrix[i, :, chain] .= xi[variable]
        end
    end

    if K == 1
        chain_names = [Symbol("$variable")]
    else
        chain_names = [Symbol("$variable[$i]") for i = 1:K]
    end
    generated_chain = Turing.Chains(matrix, chain_names, info = chains.info)

    return generated_chain
end


function generated_quantities_to_chain(
    generated_quantities::AbstractMatrix,
    chains::Turing.Chains,
    variables::Tuple,
)
    func = variable -> generated_quantities_to_chain(generated_quantities, chains, variable)
    return hcat(func.(variables)...)
end


function merge_generated_chains(chains::Turing.Chains, generated_chains::Turing.Chains)
    return hcat(chains, Turing.setrange(generated_chains, range(chains)))
end


function merge(model::Turing.Model, chains::Turing.Chains)

    generated_quantities = get_generated_quantities(model, chains)

    if generated_quantities isa Matrix{Nothing}
        return chains
    end

    variables = generated_quantities |> first |> keys
    generated_chains =
        generated_quantities_to_chain(generated_quantities, chains, variables)

    chains_merged = merge_generated_chains(chains, generated_chains)

    return chains_merged

end

end # module