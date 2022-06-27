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

savefigs = true
# savefigs = false
# forced = true
forced = false
do_waic_comparison = true # can be quite slow

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


###############################################################################
#
#                        ██     ██ ████████  ██
#                        ██     ██    ██    ███
#                        ██  █  ██    ██     ██
#                        ██ ███ ██    ██     ██
#                         ███ ███     ██     ██
#
###############################################################################


fit_WT1 = Fit(
    name = "WT1_2D_simple",
    model = diffusion_2D_simple(df_Δ_WT1.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
)
add_chains!(fit_WT1; hide_warnings = true, forced = forced)

variables = get_variables_in_group(fit_WT1, (:d, :θ))
fig_WT1_2D_simple = plot_chains(fit_WT1; variables = variables)
savefigs && CairoMakie.save("figs/WT1_2D_simple.pdf", fig_WT1_2D_simple)


Din_WT1_chains = fit_WT1.chains[Symbol("d[1]")];
Din_WT1 = mean(Din_WT1_chains)
Derr_WT1 = std(Din_WT1_chains)
println(f"Din_WT1 = {Din_WT1:.4f} ± {Derr_WT1:.4f}")

##

df_MSD_WT1 = compute_MSD(WT1_files, L_MAX, Din_WT1, Derr_WT1)


fit_WT1_MSD = Fit(
    name = "WT1_MSD",
    model = diffusion_MSD(xM, df_MSD_WT1[:, "mean"], df_MSD_WT1[:, "sdom"]),
    N_samples = N_samples,
    N_chains = N_chains,
)
add_chains!(fit_WT1_MSD; hide_warnings = true, forced = forced)


fig_WT1_MSD = plot_chains(fit_WT1_MSD)
savefigs && CairoMakie.save("figs/WT1_MSD.pdf", fig_WT1_MSD)

fig_WT1_MSD_corner = corner(fit_WT1_MSD.chains)
savefigs && CairoMakie.save("figs/WT1_MSD_corner.pdf", fig_WT1_MSD_corner)


# R_inf_WT1 = sqrt(2) * mean(fit_WT1_MSD.chains[:R_inf])
DCon1_WT1_chains = fit_WT1_MSD.chains[:d];
DCon1_WT1 = mean(DCon1_WT1_chains)
DCon1_WT1_err = std(DCon1_WT1_chains)

fit_WT1_polynomial = Fit(
    name = "WT1_polynomial",
    model = bayesian_OLS(xM[1:3], df_MSD_WT1[1:3, "mean"], df_MSD_WT1[1:3, "sdom"]),
    N_samples = N_samples,
    N_chains = N_chains,
)
add_chains!(fit_WT1_polynomial; merge_chains = false, forced = forced)


DCon2_WT1_chains = fit_WT1_polynomial.chains[:a] ./ 2;
DCon2_WT1 = mean(DCon2_WT1_chains)
DCon2_WT1_err = std(DCon2_WT1_chains)

# fx_WT1 = fit_polynomial(xM[1:3], df_MSD_WT1[1:3, "mean"], 1)
# DCon2_WT1 = fx_WT1[1] / 2.0
println(f"DCon1_WT1 = {DCon1_WT1:.4f} ± {DCon1_WT1_err:.4f}")
println(f"DCon2_WT1 = {DCon2_WT1:.4f} ± {DCon2_WT1_err:.4f}")


if do_waic_comparison
    df_comparison_WT1, fig_comparison_waic_WT1 = compute_and_plot_WAICs(
        df_Δ_WT1,
        "WT1",
        N_samples = N_samples,
        N_chains = N_chains,
        hide_warnings = true,
        forced = forced,
    )

    savefigs && CairoMakie.save("figs/WT1_comparison_waic.pdf", fig_comparison_waic_WT1)
end

###############################################################################
#
#                    ███████  ██████   ██████ ██    ██ ███████
#                    ██      ██    ██ ██      ██    ██ ██
#                    █████   ██    ██ ██      ██    ██ ███████
#                    ██      ██    ██ ██      ██    ██      ██
#                    ██       ██████   ██████  ██████  ███████
#
###############################################################################


fit_focus = Fit(
    name = "focus_2D_simple",
    model = diffusion_2D_simple(df_Δ_focus.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
)
add_chains!(fit_focus; hide_warnings = true, forced = forced)


variables = get_variables_in_group(fit_focus.chains, (:d, :θ))
fig_focus_2D_simple = plot_chains(fit_focus; variables = variables)
savefigs && CairoMakie.save("figs/focus_2D_simple.pdf", fig_focus_2D_simple)


Din_focus = fit_focus.chains[Symbol("d[1]")];
df_MSD_focus = compute_MSD(focus_files, L_MAX, mean(Din_focus), std(Din_focus))


fit_focus_polynomial = Fit(
    name = "focus_polynomial",
    model = bayesian_OLS(xM[1:3], df_MSD_focus[1:3, "mean"], df_MSD_focus[1:3, "sdom"]),
    N_samples = N_samples,
    N_chains = N_chains,
    # merge_chains = false,
    # forced = forced,
)
add_chains!(fit_focus_polynomial; merge_chains = false, forced = forced)


Db_focus_chains = fit_focus_polynomial.chains[:a] ./ 2;
Db_focus = mean(Db_focus_chains)
Db_focus_err = std(Db_focus_chains)

# fx_focus = fit_polynomial(xM[1:3], df_MSD_focus[1:3, "mean"], 1)
# Db_focus = fx_focus[1] / 2.0

println(f"Db_focus = {Db_focus:.4f} ± {Db_focus_err:.4f}")


if do_waic_comparison
    df_comparison_focus, fig_comparison_waic_focus = compute_and_plot_WAICs(
        df_Δ_focus,
        "focus",
        N_samples = N_samples,
        N_chains = N_chains,
        hide_warnings = true,
        forced = forced,
    )

    savefigs && CairoMakie.save("figs/focus_comparison_waic.pdf", fig_comparison_waic_focus)
end

###############################################################################
#
#                    ██████  ███████ ██      ████████  █████
#                    ██   ██ ██      ██         ██    ██   ██
#                    ██   ██ █████   ██         ██    ███████
#                    ██   ██ ██      ██         ██    ██   ██
#                    ██████  ███████ ███████    ██    ██   ██
#
###############################################################################


fit_delta = Fit(
    name = "delta_2D_simple",
    model = diffusion_2D_simple(df_Δ_delta.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
)
add_chains!(fit_delta; hide_warnings = true, forced = forced)


variables = get_variables_in_group(fit_delta.chains, (:d, :θ))
fig_delta_2D_simple = plot_chains(fit_delta; variables = variables)
savefigs && CairoMakie.save("figs/delta_2D_simple.pdf", fig_delta_2D_simple)


DoutF_delta_chains = fit_delta.chains[Symbol("d[2]")];
DoutF_delta = mean(DoutF_delta_chains)


if do_waic_comparison
    df_comparison_delta, fig_comparison_waic_delta = compute_and_plot_WAICs(
        df_Δ_delta,
        "delta",
        N_samples = N_samples,
        N_chains = N_chains,
        hide_warnings = true,
        forced = forced,
    )

    savefigs && CairoMakie.save("figs/delta_comparison_waic.pdf", fig_comparison_waic_delta)
end

###############################################################################
#
#                    ██     ██ ████████ ██████
#                    ██     ██    ██         ██
#                    ██  █  ██    ██     █████
#                    ██ ███ ██    ██    ██
#                     ███ ███     ██    ███████
#
###############################################################################



fit_WT2 = Fit(
    name = "WT2_2D_simple",
    model = diffusion_2D_simple(df_Δ_WT2.Δ),
    N_samples = N_samples,
    N_chains = N_chains,
)
add_chains!(fit_WT2; hide_warnings = true, forced = forced)
# add_waic!(fit_WT2; hide_warnings = true, forced = forced)


variables = get_variables_in_group(fit_WT2.chains, (:d, :θ))
fig_WT2_2D_simple = plot_chains(fit_WT2; variables = variables)
savefigs && CairoMakie.save("figs/WT2_2D_simple.pdf", fig_WT2_2D_simple)

# Din_WT2 = mean(fit_WT2.chains[Symbol("d[1]")])
# Derr_WT2 = std(fit_WT2.chains[Symbol("d[1]")])

if do_waic_comparison
    df_comparison_WT2, fig_comparison_waic_WT2 = compute_and_plot_WAICs(
        df_Δ_WT2,
        "WT2",
        N_samples = N_samples,
        N_chains = N_chains,
        hide_warnings = true,
        forced = forced,
    )

    savefigs && CairoMakie.save("figs/WT2_comparison_waic.pdf", fig_comparison_waic_WT2)
end

###############################################################################
#
#                    ██    ██
#                    ██    ██
#                    ██    ██
#                    ██    ██
#                     ██████
#
###############################################################################


# U_left = compute_U_left(chains_WT1_2D_simple)
# U_right = compute_U_right(DCon2_WT1, Db_focus, DoutF_delta)

U_lefts = compute_U_left.(fit_WT1.chains[Symbol("θ[1]")]);
U_rights = compute_U_right.(DCon2_WT1_chains, Db_focus_chains, DoutF_delta_chains);


println(f"U left = {mean(U_lefts):.3f} ± {std(U_lefts):.3f}")
println(f"U right = {mean(U_rights):.3f} ± {std(U_rights):.3f}")


##

fig_U_left = plot_U_direction(U_lefts, "left")
savefigs && CairoMakie.save("figs/U_left.pdf", fig_U_left)

fig_U_right = plot_U_direction(U_rights, "right")
savefigs && CairoMakie.save("figs/U_right.pdf", fig_U_right)

fig_Us = plot_Us([U_lefts, U_rights], ["left", "right"])
savefigs && CairoMakie.save("figs/Us.pdf", fig_Us)
