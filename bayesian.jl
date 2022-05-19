
include("utils.jl")


##

const τ = 0.02
const L_MAX = 10

##


# We have three potential candicates for the WT list
WT1_files = get_list(
    "Sir3-Halo-WT_Single_Molecules_Data_Set_04/Sir3-Halo-WT_Single_Molecules_Data_Set_04",
)
WT2_files = get_list("Sir3-Halo-WT_Single_Molecules_Data_Set_04_Judith_Cleaned")
WT3_files = get_list("WT_L1")


# Three other data sets: Focus for Sir3, Rad52 single (for comparison) and Delta - for the Sir2D4D mutant
focus_files = get_list("Sir3_Halo_WT_Focus_cleaned/Sir3_Halo_WT_Focus_cropped")
rad_files = get_list("Rad52_MathiasAllCells")
delta_files = get_list("Sir3-Halo-Sir2DSir4D_Judith_Cleaned")


df_WT1 = load_cells(WT1_files)
df_WT2 = load_cells(WT2_files)
df_WT3 = load_cells(WT3_files)

df_focus = load_cells(focus_files)
df_rad = load_cells(rad_files)
df_delta = load_cells(delta_files)

df_Δ_WT1 = compute_dist(df_WT1)
df_Δ_WT2 = compute_dist(df_WT2)
df_Δ_WT3 = compute_dist(df_WT3)

df_Δ_focus = compute_dist(df_focus)
df_Δ_rad = compute_dist(df_rad)\
df_Δ_delta = compute_dist(df_delta)

##

using Turing: Turing, Model, @model, NUTS, sample, filldist, arraydist, namesingroup
using Distributions: Exponential, truncated, Uniform
using DataFrames: levels
using CairoMakie



include("utils_bayesian.jl")


# df_Δ = df_Δ_WT1
# Δ = df_Δ_WT1.Δ
# groups = levels(df_Δ_WT1.group)
# group = groups[1]


# @model function diffusion_2D_naive(Δ)

#     # prior d
#     ds ~ arraydist([
#         truncated(Exponential(0.1), 0.0001, 0.075),
#         truncated(Exponential(0.1), 0.075, 1),
#     ],)

#     # dists = [Rayleigh(d_to_σ(d)) for d in ds]
#     dists = [diffusion_1D(d) for d in ds]

#     # prior θΘ
#     Θ ~ Uniform(0, 1)
#     w = [Θ, 1 - Θ]

#     # mixture distribution
#     distribution = MixtureModel(dists, w)

#     # likelihood
#     Δ ~ filldist(distribution, length(Δ))

# end

# fit_WT1_2D_naive = initialize_fit(model = diffusion_2D_naive(df_Δ_WT1.Δ));
# fit!(fit_WT1_2D_naive)
# plot_chains(fit_WT1_2D_naive)
# get_chains(fit_WT1_2D_naive)



@model function diffusion_2D_naive_ordered(Δ)

    # prior d
    Δds ~ filldist(Exponential(0.1), 2)
    ds = cumsum(Δds)
    # ds ~ arraydist([Dirac(d) for d in cumsum(Δds)])


    # dists = [Rayleigh(d_to_σ(d)) for d in ds]
    dists = [diffusion_1D(d) for d in ds]

    # prior θΘ
    Θ ~ Uniform(0, 1)
    w = [Θ, 1 - Θ]
    f = w[2]

    # mixture distribution
    distribution = MixtureModel(dists, w)

    # likelihood
    Δ ~ filldist(distribution, length(Δ))

    return (; ds, f)
end



fit_WT1_2D_naive_ordered =
    initialize_fit(model = diffusion_2D_naive_ordered(df_Δ_WT1.Δ), N_chains = 4);
fit!(fit_WT1_2D_naive_ordered)
plot_chains(fit_WT1_2D_naive_ordered)


chains = get_chains(fit_WT1_2D_naive_ordered)
model = diffusion_2D_naive_ordered(df_Δ_WT1.Δ)



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

dict = Dict{Symbol,Any}(:model => model, :chains => chains)


dict[:generated_quantities] = get_generated_quantities(dict);

dict[:variables] = get_variables(dict)

dict[:N_samples] = get_N_samples(dict)
dict[:N_chains] = get_N_chains(dict)



function generated_quantities_to_chain(dict::Dict, variable::Union{Symbol,String})

    K = get_K(dict, variable)

    matrix = zeros(dict[:N_samples], K, dict[:N_chains])
    for chain in 1:dict[:N_chains]
        for (i, xi) in enumerate(dict[:generated_quantities][:, chain])
            matrix[i, :, chain] .= xi[variable]
        end
    end

    if K == 1
        chain_names = [Symbol("$variable")]
    else
        chain_names = [Symbol("$variable[$i]") for i = 1:K]
    end
    generated_chain = Turing.Chains(matrix, chain_names, info=dict[:chains].info)

    return generated_chain

end



function generated_quantities_to_chains(dict::Dict)
    return hcat(
        [generated_quantities_to_chain(dict, variable) for variable in dict[:variables]]...,
    )
end

dict[:generated_chains] = generated_quantities_to_chains(dict)


function merge_generated_chains(dict::Dict)
    return hcat(dict[:chains], Turing.setrange(dict[:generated_chains], range(dict[:chains])))
end

merge_generated_chains(dict)




