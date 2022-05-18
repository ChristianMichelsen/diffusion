using Glob: glob
using NaturalSort: natural
using DataFrames:
    DataFrame,
    select,
    select!,
    groupby,
    groupindices,
    transform,
    combine,
    groupcols,
    disallowmissing!,
    disallowmissing,
    nrow,
    SubDataFrame,
    GroupedDataFrame
import CSV
# using Pipe: @pipe
using Chain: @chain
using Optim: optimize, TwiceDifferentiable, Optim, hessian!
using OrderedCollections: LittleDict
using LinearAlgebra: diag
using StatsBase: StatsBase, cov2cor, mean, std
using PyFormattedStrings
import Polynomials
using Distributions: Rayleigh, fit_mle, MixtureModel, logpdf, pdf
using CairoMakie

##

include("utils.jl")
include("utils_frequentist.jl")


##

const τ = 0.02
const L_MAX = 10

const xM = collect(range(τ, τ * L_MAX, L_MAX))


##


d_p0 = LittleDict("d1" => 0.04, "d2" => 0.16, "f" => 0.6)
d_lower = LittleDict("d1" => 0.0, "d2" => 0.0, "f" => 0.0)
d_upper = LittleDict("d1" => 10.0, "d2" => 10.0, "f" => 1.0)


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

Δ_WT1 = compute_dist(df_WT1).Δ
Δ_WT2 = compute_dist(df_WT2).Δ
Δ_WT3 = compute_dist(df_WT3).Δ

Δ_focus = compute_dist(df_focus).Δ
Δ_rad = compute_dist(df_rad).Δ
Δ_delta = compute_dist(df_delta).Δ

##



fit_WT1 =
    fit(make_closure(Δ_WT1, diffusion_nll_2D), diffusion_nll_2D, d_p0, d_lower, d_upper)

Din_WT1 = fit_WT1.μ[1]
Derr_WT1 = fit_WT1.σ[1]

println(f"Din_WT1 = {Din_WT1:.4f}")
println(f"Derr_WT1 = {Derr_WT1:.4f}")



df_MSD_WT1 = compute_MSD(WT1_files, L_MAX, Din_WT1, Derr_WT1)

p0_MSD = LittleDict("R_inf" => 0.2, "d" => 0.1, "σ" => 0.002)
lower_MSD = LittleDict("R_inf" => 0.0, "d" => 0.0, "σ" => 0.0)
upper_MSD = LittleDict("R_inf" => 1.0, "d" => 1.0, "σ" => 1.0)

χ²_closure = make_χ²_closure(xM, df_MSD_WT1[!, "mean"], df_MSD_WT1[!, "sdom"], f_MSD)
fit_WT1_MSD = fit(χ²_closure, f_MSD, p0_MSD, lower_MSD, upper_MSD)


R_inf_WT1 = sqrt(2) * fit_WT1_MSD.μ[1]
DCon1_WT1 = fit_WT1_MSD.μ[2]

fx_WT1 = Polynomials.fit(xM[1:3], df_MSD_WT1[1:3, "mean"], 1)
DCon2_WT1 = fx_WT1[1] / 2.0

println(f"DCon1_WT1 = {DCon1_WT1:.4f}")
println(f"DCon2_WT1 = {DCon2_WT1:.4f}")

##


fit_focus =
    fit(make_closure(Δ_focus, diffusion_nll_2D), diffusion_nll_2D, d_p0, d_lower, d_upper)

Din_focus = fit_focus.μ[1]
Derr_focus = fit_focus.σ[1]

df_MSD_focus = compute_MSD(focus_files, L_MAX, Din_focus, Derr_focus)

fx_focus = Polynomials.fit(xM[1:3], df_MSD_focus[1:3, "mean"], 1)
Db_focus = fx_focus[1] / 2.0
println(f"Db_focus = {Db_focus:.4f}")


##


fit_delta =
    fit(make_closure(Δ_delta, diffusion_nll_2D), diffusion_nll_2D, d_p0, d_lower, d_upper)

DoutF_delta = fit_delta.μ[2]

##


fit_WT2 =
    fit(make_closure(Δ_WT2, diffusion_nll_2D), diffusion_nll_2D, d_p0, d_lower, d_upper)

Din_WT2 = fit_WT2.μ[1]
Derr_WT2 = fit_WT2.σ[1]

##



########### Here we do the actual computations for the inputs of the nucleus. The geometrical calculations are to include the fact that we can only see a part of the entire nucleus

U_left = compute_U_left(fit_WT1)
U_right = compute_U_right(DCon2_WT1, Db_focus, DoutF_delta)

println("U_{left}=", U_left)
println("U_{right}=", U_right)

##

Δs = [Δ_WT1, Δ_focus, Δ_delta]
names = ["WT1", "Focus", "Delta"]
colors = [:green, :blue, :red]


for (Δ, name, color) in zip(Δs, names, colors)
    fig = plot_and_fit(Δ, name, color)
    display(fig)
end