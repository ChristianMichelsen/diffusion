
using PyFormattedStrings: @f_str
using StatsBase: mean, std, median
using Distributions: Normal
using MCMCChains: corner
using StatsPlots
import CairoMakie

include("utils.jl")


##

const τ = 0.02
const L_MAX = 10

const N_samples = 1000
const N_chains = 4
const xM = collect(range(τ, L_MAX * τ, L_MAX))


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
fig_WT1_2D_simple = plot_chains(chains_WT1_2D_simple; variables = variables)
save("figs/WT1_2D_simple.pdf", fig_WT1_2D_simple)



# chains_WT1_2D_simple[variables]

Din_WT1 = mean(chains_WT1_2D_simple[Symbol("ds[1]")])
Derr_WT1 = std(chains_WT1_2D_simple[Symbol("ds[1]")])

println(f"Din_WT1 = {Din_WT1:.4f}")
println(f"Derr_WT1 = {Derr_WT1:.4f}")

##

df_MSD_WT1 = compute_MSD(WT1_files, L_MAX, Din_WT1, Derr_WT1)


@model function diffusion_MSD(x, y, sy)

    R_inf ~ Exponential(0.1)
    d ~ Uniform(0, 1)
    σ ~ Uniform(0, 1)

    for i = 1:length(x)
        ỹ = f_MSD(x[i], R_inf, d, σ)
        y[i] ~ Normal(ỹ, sy[i])
    end

end

chains_WT1_MSD = get_chains(
    name = "WT1_MSD",
    model = diffusion_MSD(xM, df_MSD_WT1[:, "mean"], df_MSD_WT1[:, "sdom"]),
    N_samples = N_samples,
    N_chains = N_chains,
    # hide_warnings = true,
)
fig_WT1_MSD = plot_chains(chains_WT1_MSD)
save("figs/WT1_MSD.pdf", fig_WT1_MSD)


R_inf_WT1 = sqrt(2) * mean(chains_WT1_MSD[:R_inf])
DCon1_WT1 = mean(chains_WT1_MSD[:d])

fx_WT1 = fit_polynomial(xM[1:3], df_MSD_WT1[1:3, "mean"], 1)
DCon2_WT1 = fx_WT1[1] / 2.0
println(f"DCon1_WT1 = {DCon1_WT1:.4f}")
println(f"DCon2_WT1 = {DCon2_WT1:.4f}")

fig_WT1_MSD_corner = corner(chains_WT1_MSD)
save("figs/WT1_MSD_corner.pdf", fig_WT1_MSD_corner)


########

chains_focus_2D_simple = get_chains(
    name = "focus_2D_simple",
    model = diffusion_2D_simple(df_Δ_focus.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
)

variables = [get_variables_in_group(chains_focus_2D_simple, :ds)..., :Θ]
fig_focus_2D_simple = plot_chains(chains_focus_2D_simple; variables = variables)
save("figs/focus_2D_simple.pdf", fig_focus_2D_simple)



Din_focus = chains_focus_2D_simple[Symbol("ds[1]")]
df_MSD_focus = compute_MSD(focus_files, L_MAX, mean(Din_focus), std(Din_focus))

fx_focus = fit_polynomial(xM[1:3], df_MSD_focus[1:3, "mean"], 1)
Db_focus = fx_focus[1] / 2.0
println(f"Db_focus = {Db_focus:.4f}")

########

chains_delta_2D_simple = get_chains(
    name = "delta_2D_simple",
    model = diffusion_2D_simple(df_Δ_delta.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
)

variables = [get_variables_in_group(chains_delta_2D_simple, :ds)..., :Θ]
fig_delta_2D_simple = plot_chains(chains_delta_2D_simple; variables = variables)
save("figs/delta_2D_simple.pdf", fig_delta_2D_simple)



DoutF_delta = mean(chains_delta_2D_simple[Symbol("ds[2]")])

########

chains_WT2_2D_simple = get_chains(
    name = "WT2_2D_simple",
    model = diffusion_2D_simple(df_Δ_WT2.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
)

variables = [get_variables_in_group(chains_WT2_2D_simple, :ds)..., :Θ]
fig_WT2_2D_simple = plot_chains(chains_WT2_2D_simple; variables = variables)
save("figs/WT2_2D_simple.pdf", fig_WT2_2D_simple)


Din_WT2 = mean(chains_WT2_2D_simple[Symbol("ds[1]")])
Derr_WT2 = std(chains_WT2_2D_simple[Symbol("ds[1]")])

#####



U_left = compute_U_left(chains_WT1_2D_simple)
U_right = compute_U_right(DCon2_WT1, Db_focus, DoutF_delta)

println("U_{left}=", U_left)
println("U_{right}=", U_right)











#######


df_Δ_WT1_good_groups = filter(:nrow => >(100), df_Δ_WT1)
df_Δ_WT1_good_groups

unique(df_Δ_WT1_good_groups[:, [:group, :id, :cell]])


@model function diffusion_2D_groups(Δ, group)

    groups = levels(group)
    N_groups = length(levels(group))

    # prior d
    Δds ~ filldist(Exponential(0.1), 2)
    ds = cumsum(Δds)

    dists = [diffusion_1D(d) for d in ds]

    # prior θ
    Θ ~ filldist(Uniform(0, 1), N_groups)

    for (i, g) in enumerate(groups)

        group_mask = (g .== group)
        w = [Θ[i], 1 - Θ[i]]

        # mixture distribution
        distribution = MixtureModel(dists, w)

        # likelihood
        Δ[group_mask] ~ filldist(distribution, sum(group_mask))

    end

    return (; ds)
end


chains_WT1_2D_groups = get_chains(
    name = "WT1_2D_groups_100",
    model = diffusion_2D_groups(df_Δ_WT1_good_groups.Δ, df_Δ_WT1_good_groups.group),
    N_samples = N_samples,
    N_chains = N_chains,
    # hide_warnings = true,
)


variables = [
    get_variables_in_group(chains_WT1_2D_groups, :ds)...,
    get_variables_in_group(chains_WT1_2D_groups, :Θ)...,
]

plot_chains(chains_WT1_2D_groups; variables = variables[begin:begin+5])

fig_WT1_2D_groups_100 =
    plot_chains(chains_WT1_2D_groups; variables = variables, resolution = (1000, 10000))
save("figs/WT1_2D_groups_100.pdf", fig_WT1_2D_groups_100)


##


# chains = chains_WT1_2D_groups

function compute_d_combined!(d_combined, ds, Θ)
    for i in eachindex(Θ)
        d_combined[i] = ds[i, 1] * Θ[i] + ds[i, 2] * (1 - Θ[i])
    end
end

function compute_d_combined(chains::Turing.Chains)
    ds = Array(chains[get_variables_in_group(chains, :ds)])
    Θs = Array(chains[get_variables_in_group(chains, :Θ)])
    d_combined = zeros(eltype(Θs), size(Θs))
    for i in axes(Θs, 2)
        compute_d_combined!(view(d_combined, :, i), ds, view(Θs, :, i))
    end
    return d_combined
end

d_combined_WT1_2D_groups = compute_d_combined(chains_WT1_2D_groups)

mean(eachrow(d_combined_WT1_2D_groups)) .< 0.045876509391691855
