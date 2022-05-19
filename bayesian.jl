
using PyFormattedStrings: @f_str

include("utils.jl")


##

const τ = 0.02
const L_MAX = 10

const N_samples = 1000
const N_chains = 4


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
df_Δ_rad = compute_dist(df_rad)
df_Δ_delta = compute_dist(df_delta)

##

using Turing: Turing, Model, @model, NUTS, filldist, arraydist, namesingroup
using Distributions: Exponential, truncated, Uniform
using DataFrames: levels
using CairoMakie



include("utils_bayesian.jl")


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


chains_WT1_2D_simple = get_chains(
    name = "WT1_2D_simple",
    model = diffusion_2D_simple(df_Δ_WT1.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
)

variables = [get_variables_in_group(chains_WT1_2D_simple, :ds)..., :Θ]
plot_chains(chains_WT1_2D_simple; variables = variables)


