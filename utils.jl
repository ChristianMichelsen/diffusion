using Glob: glob
using NaturalSort: natural
import CSV
# using Pipe: @pipe
using Chain: @chain
using Distributions: Rayleigh, fit_mle, MixtureModel, logpdf, pdf
import Polynomials
using LinearAlgebra: I
using PyFormattedStrings: @f_str
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


##


function get_list(folder)
    return sort(glob("Cell*", "../Python/" * folder), lt = natural)
end


function extract_cell_number(file)
    return @chain file basename filter(isdigit, _) parse(Int, _)
end

function add_group_id!(df, cols)
    df.group = groupby(df, cols) |> groupindices
    disallowmissing!(df)
end

function add_group_id(df, cols)
    df.group = groupby(df, cols) |> groupindices
    df = disallowmissing(df)
    return df
end



function load_single_cell(file; add_group_ids = false)
    df = CSV.File(file, header = [:id, :x, :y, :t]) |> DataFrame
    df.cell .= extract_cell_number(file)
    if add_group_ids
        add_group_id!(df, :id)
    end
    return df
end


function load_cells(files)
    df = @chain files map(load_single_cell, _) vcat(_...) add_group_id([:cell, :id])
    return df
end



##


function compute_group_dist(data, k::Integer = 1)
    Δ = @chain begin
        data[1:end-k, :] .- data[k+1:end, :]
        _ .^ 2
        eachcol
        sum
        sqrt.(_)
    end
    return Δ
end

function compute_group_dist(g::SubDataFrame, k::Integer = 1)
    return compute_group_dist(g[:, [:x, :y]], k)
end

function compute_group_dist(x, y, k::Integer = 1)
    return compute_group_dist(hcat(x, y), k)
end


function compute_dist(gdf::GroupedDataFrame, k = 1)
    return combine(
        gdf,
        (g -> g[begin+k:end, [:id, :cell]]),
        nrow,
        [:x, :y] => ((x, y) -> compute_group_dist(x, y, k)) => :Δ,
    )
end

function compute_dist(df::DataFrame, k = 1)
    return compute_dist(groupby(df, :group), k)
end


##


function d_to_σ(d)
    σ = sqrt(2 * d * τ)
    return σ
end

function σ_to_d(σ)
    d = σ^2 / (2 * τ)
    return d
end


function Rayleigh_from_d(d)
    return Rayleigh(d_to_σ(d))
end

# function diffusion_1D(d)
#     return Rayleigh(d_to_σ(d))
# end


# function diffusion_2D(d1, d2, f)
#     prior = [f, 1 - f]
#     rayleighs = [diffusion_1D(d) for d in [d1, d2]]
#     distribution = MixtureModel(rayleighs, prior)
#     return distribution
# end

##


function append_to_squared_dists_and_N!(squared_dists, Ns, g, L_MAX)
    for k = 1:L_MAX
        squared_dist = compute_group_dist(g, k) .^ 2
        N = length(squared_dist)
        append!(squared_dists[k], squared_dist)
        Ns[k] += N
    end
end



function make_df_MSD(squared_dists, Ns, L_MAX)

    MSD_mean = zeros(L_MAX)
    MSD_std = zeros(L_MAX)
    MSD_N = Ns

    for L = 1:L_MAX
        MSD_mean[L] = mean(squared_dists[L])
        MSD_std[L] = std(squared_dists[L])
    end
    MSD_sdom = MSD_std ./ sqrt.(MSD_N)

    d_MSD = ("mean" => MSD_mean, "std" => MSD_std, "N" => MSD_N, "sdom" => MSD_sdom)
    df_MSD = DataFrame(d_MSD...)

    return df_MSD
end


function compute_MSD(files, L_MAX, Dintmp, Dstmp, Typ = 0)
    """extracts the MSD for the slow diffusion coefficient (if typ = 0, which is standard)"""

    squared_dists = [Float64[] for _ = 1:L_MAX]
    Ns = zeros(Int, L_MAX)

    for file in files

        df = CSV.File(file, header = [:id, :x, :y, :t]) |> DataFrame

        # g = groupby(df, :id) |> first

        for g in groupby(df, :id)

            # no distances
            if nrow(g) == 1
                continue
            end

            Δ = compute_group_dist(g)

            D = sum(Δ .^ 2) / (4 * 0.02 * length(Δ))
            bool1 = (Typ == 0) && (length(Δ) > L_MAX) && (D < Dintmp + 3 * Dstmp)
            bool2 = (Typ != 0) && (length(Δ) > L_MAX) && (D > Dintmp + 3 * Dstmp)

            if bool1 || bool2
                append_to_squared_dists_and_N!(squared_dists, Ns, g, L_MAX)
            end
        end
    end

    df_MSD = make_df_MSD(squared_dists, Ns, L_MAX)
    return df_MSD
end


function compute_U_right(DCon2_WT1, Db_focus, DoutF_delta)
    U_right = log((DCon2_WT1 - Db_focus) / (DoutF_delta - Db_focus))
    return U_right
end


function compute_U_left(pp::Float64)
    r0 = 1.0
    h = 0.85
    Vcap = pi * h^2 / 3 * (3 * r0 - h)
    RR = 0.13
    V0 = 4 * pi / 3 - 2 * Vcap
    VF = 8 * V0 / (4 * pi / 3) * 4 * pi / 3 * RR^3
    U_left = -log(pp * (V0 - VF) / ((1 - pp) * VF))
    return U_left
end

function compute_U_left(fit_WT1::NamedTuple)
    return compute_U_left(fit_WT1.μ[3])
end



function plot_U_direction(U_lefts, direction; resolution = (1_000, 600))

    n_chains = size(U_lefts, 2)
    colors = get_colors()

    fig = CairoMakie.Figure(; resolution = resolution)

    μ = mean(U_lefts)
    σ = std(U_lefts)
    μ_sdom = sdom(U_lefts)

    title =
        f"Density plot of U ({direction}). \n Summary: μ = {μ:.4f}, σ = {σ:.4f}, μ_sdom = {μ_sdom:.4f}"

    ax = CairoMakie.Axis(
        fig[1, 1];
        title = title,
        xlabel = f"U ({direction})",
        ylabel = "Density",
        limits = (nothing, nothing, 0, nothing),
    )

    for chain = 1:n_chains
        CairoMakie.density!(
            ax,
            U_lefts[:, chain];
            label = f"Chain {chain}",
            strokewidth = 3,
            strokecolor = (colors[chain], 0.8),
            color = (colors[chain], 0),
        )
    end

    CairoMakie.axislegend(ax)
    return fig

end


##


##

function f_MSD(xM, R_inf, d, σ)
    return 4 * σ^2 + R_inf^2 * (1 - exp(-4 * d * xM / R_inf^2))
end


# function fit_polynomial(x, y, order = 1)
#     fx = Polynomials.fit(x, y, order)
#     return fx
# end

function sdom(x)
    return std(x) / sqrt(length(x))
end


##

function plot_Us(Us, directions; resolution = (1_000, 600))

    N_Us = size(Us, 1)
    colors = get_colors()

    fig = CairoMakie.Figure(; resolution = resolution)

    # μ = mean(U_lefts)
    # σ = std(U_lefts)
    # μ_sdom = sdom(U_lefts)

    title = "Density plots of Us"

    ax = CairoMakie.Axis(
        fig[1, 1];
        title = title,
        xlabel = "U",
        ylabel = "Density",
        limits = (nothing, nothing, 0, nothing),
    )


    for (i, U) in enumerate(Us)

        U = Us[i]
        n_chains = size(U, 2)

        for chain = 1:n_chains
            CairoMakie.density!(
                ax,
                U[:, chain];
                label = directions[i],
                strokewidth = 3,
                strokecolor = (colors[i], 0.8),
                color = (colors[i], 0),
            )
        end
    end

    CairoMakie.axislegend(ax; merge = true)
    return fig

end



##

function plot_MSD(df_MSD; title = "")
    fig = CairoMakie.Figure(; resolution = (1_000, 600))
    ax = CairoMakie.Axis(fig[1, 1]; ylabel = "MSD", title = title)
    xs = 1:size(df_MSD, 1)
    CairoMakie.errorbars!(ax, xs, df_MSD[:, "mean"], df_MSD[:, "sdom"]; label = "mean")
    ax.xlabel = "Iteration"
    return fig
end