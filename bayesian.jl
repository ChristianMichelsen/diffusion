using PyFormattedStrings: @f_str
using StatsBase: mean, std, median
using Distributions: Normal
using MCMCChains: corner
using StatsPlots
import CairoMakie
using Turing: Turing, Model, @model, NUTS, filldist, arraydist, namesingroup
using Distributions: Exponential, truncated, Uniform
using DataFrames: levels


###############################################################################
#
#                    ███████ ███████ ████████ ██    ██ ██████
#                    ██      ██         ██    ██    ██ ██   ██
#                    ███████ █████      ██    ██    ██ ██████
#                         ██ ██         ██    ██    ██ ██
#                    ███████ ███████    ██     ██████  ██
#
###############################################################################

include("utils.jl")
include("utils_bayesian.jl")

fontsize_theme = CairoMakie.Theme(fontsize = 20)
CairoMakie.set_theme!(fontsize_theme)

##

const τ = 0.02
const L_MAX = 10

const N_samples = 1000
const N_chains = 4
const xM = collect(range(τ, L_MAX * τ, L_MAX))
const min_N_rows = 100

##


###############################################################################
#
#                    ██████   █████  ████████  █████
#                    ██   ██ ██   ██    ██    ██   ██
#                    ██   ██ ███████    ██    ███████
#                    ██   ██ ██   ██    ██    ██   ██
#                    ██████  ██   ██    ██    ██   ██
#
###############################################################################


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

###############################################################################
#
#                        ██     ██ ████████  ██
#                        ██     ██    ██    ███
#                        ██  █  ██    ██     ██
#                        ██ ███ ██    ██     ██
#                         ███ ███     ██     ██
#
###############################################################################


chains_WT1_2D_simple = get_chains(
    name = "WT1_2D_simple",
    model = diffusion_2D_simple(df_Δ_WT1.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
)

variables = [get_variables_in_group(chains_WT1_2D_simple, :ds)..., :Θ]
fig_WT1_2D_simple = plot_chains(chains_WT1_2D_simple; variables = variables)
CairoMakie.save("figs/WT1_2D_simple.pdf", fig_WT1_2D_simple)

Din_WT1 = mean(chains_WT1_2D_simple[Symbol("ds[1]")])
Derr_WT1 = std(chains_WT1_2D_simple[Symbol("ds[1]")])

println(f"Din_WT1 = {Din_WT1:.4f}")
println(f"Derr_WT1 = {Derr_WT1:.4f}")

##

df_MSD_WT1 = compute_MSD(WT1_files, L_MAX, Din_WT1, Derr_WT1)

chains_WT1_MSD = get_chains(
    name = "WT1_MSD",
    model = diffusion_MSD(xM, df_MSD_WT1[:, "mean"], df_MSD_WT1[:, "sdom"]),
    N_samples = N_samples,
    N_chains = N_chains,
    # hide_warnings = true,
)
fig_WT1_MSD = plot_chains(chains_WT1_MSD)
CairoMakie.save("figs/WT1_MSD.pdf", fig_WT1_MSD)


# R_inf_WT1 = sqrt(2) * mean(chains_WT1_MSD[:R_inf])
DCon1_WT1_chains = chains_WT1_MSD[:d]
DCon1_WT1 = mean(DCon1_WT1_chains)

chains_WT1_polynomial = get_chains(
    name = "WT1_polynomial",
    model = bayesian_OLS(xM[1:3], df_MSD_WT1[1:3, "mean"], df_MSD_WT1[1:3, "sdom"]),
    N_samples = N_samples,
    N_chains = N_chains,
    merge_chains = false,
)


DCon2_WT1_chains = chains_WT1_polynomial[:a] ./ 2
DCon2_WT1 = mean(DCon2_WT1_chains)

# fx_WT1 = fit_polynomial(xM[1:3], df_MSD_WT1[1:3, "mean"], 1)
# DCon2_WT1 = fx_WT1[1] / 2.0
println(f"DCon1_WT1 = {DCon1_WT1:.4f}")
println(f"DCon2_WT1 = {DCon2_WT1:.4f}")

fig_WT1_MSD_corner = corner(chains_WT1_MSD)
CairoMakie.save("figs/WT1_MSD_corner.pdf", fig_WT1_MSD_corner)


###############################################################################
#
#                    ███████  ██████   ██████ ██    ██ ███████
#                    ██      ██    ██ ██      ██    ██ ██
#                    █████   ██    ██ ██      ██    ██ ███████
#                    ██      ██    ██ ██      ██    ██      ██
#                    ██       ██████   ██████  ██████  ███████
#
###############################################################################


chains_focus_2D_simple = get_chains(
    name = "focus_2D_simple",
    model = diffusion_2D_simple(df_Δ_focus.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
)

variables = [get_variables_in_group(chains_focus_2D_simple, :ds)..., :Θ]
fig_focus_2D_simple = plot_chains(chains_focus_2D_simple; variables = variables)
CairoMakie.save("figs/focus_2D_simple.pdf", fig_focus_2D_simple)


Din_focus = chains_focus_2D_simple[Symbol("ds[1]")]
df_MSD_focus = compute_MSD(focus_files, L_MAX, mean(Din_focus), std(Din_focus))



chains_focus_polynomial = get_chains(
    name = "focus_polynomial",
    model = bayesian_OLS(xM[1:3], df_MSD_focus[1:3, "mean"], df_MSD_focus[1:3, "sdom"]),
    N_samples = N_samples,
    N_chains = N_chains,
    merge_chains = false,
)

Db_focus_chains = chains_focus_polynomial[:a] ./ 2
Db_focus = mean(Db_focus_chains)

# fx_focus = fit_polynomial(xM[1:3], df_MSD_focus[1:3, "mean"], 1)
# Db_focus = fx_focus[1] / 2.0

println(f"Db_focus = {Db_focus:.4f}")

###############################################################################
#
#                    ██████  ███████ ██      ████████  █████
#                    ██   ██ ██      ██         ██    ██   ██
#                    ██   ██ █████   ██         ██    ███████
#                    ██   ██ ██      ██         ██    ██   ██
#                    ██████  ███████ ███████    ██    ██   ██
#
###############################################################################


chains_delta_2D_simple = get_chains(
    name = "delta_2D_simple",
    model = diffusion_2D_simple(df_Δ_delta.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
    hide_warnings = true,
)

variables = [get_variables_in_group(chains_delta_2D_simple, :ds)..., :Θ]
fig_delta_2D_simple = plot_chains(chains_delta_2D_simple; variables = variables)
CairoMakie.save("figs/delta_2D_simple.pdf", fig_delta_2D_simple)


DoutF_delta_chains = chains_delta_2D_simple[Symbol("ds[2]")]
DoutF_delta = mean(DoutF_delta_chains)

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
CairoMakie.save("figs/WT2_2D_simple.pdf", fig_WT2_2D_simple)


Din_WT2 = mean(chains_WT2_2D_simple[Symbol("ds[1]")])
Derr_WT2 = std(chains_WT2_2D_simple[Symbol("ds[1]")])

#####


U_left = compute_U_left(chains_WT1_2D_simple)
U_right = compute_U_right(DCon2_WT1, Db_focus, DoutF_delta)

println("U_{left}=", U_left)
println("U_{right}=", U_right)


##

U_lefts = compute_U_left.(chains_WT1_2D_simple[:Θ])
fig_U_left = plot_U_direction(U_lefts, "left")
CairoMakie.save("figs/U_left.pdf", fig_U_left)

U_rights = compute_U_right.(DCon2_WT1_chains, Db_focus_chains, DoutF_delta_chains)
fig_U_right = plot_U_direction(U_rights, "right")
CairoMakie.save("figs/U_right.pdf", fig_U_right)

fig_Us = plot_Us([U_lefts, U_rights], ["left", "right"])
CairoMakie.save("figs/Us.pdf", fig_Us)
