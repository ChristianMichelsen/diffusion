
module ModelComparison

export compute_waic, compute_z, compare, plot, save, load



import StatsFuns
using Statistics: mean, std
import DataFrames
import CairoMakie

struct WAIC{F,V}
    waic::F
    waic_i::V
    lppd::F
    penalty::F
    std::F
end

import JLD2
# JLD2.writeas(::Type{WAIC}) = Float64
# JLD2.wconvert(::Type{Float64}, b::WAIC) = b.x
# JLD2.rconvert(::Type{WAIC}, x::Float64) = WAIC(x)


function save(waic::WAIC, filename)
    JLD2.jldsave(filename; waic)
end

function load(filename)
    JLD2.jldopen(filename, "r") do file
        return file["waic"]
    end
end


"""
    Modified from github.com/StatisticalRethinkingJulia/StatsModelComparisons.jl
"""
function waic_2D_array(ll::AbstractArray)

    n_samples, n_obs = size(ll)
    pD = zeros(n_obs)

    lpd = reshape(StatsFuns.logsumexp(ll .- log(n_samples); dims = 1), n_obs)

    var2(x) = mean(x .^ 2) .- mean(x)^2
    for i = 1:n_obs
        pD[i] = var2(ll[:, i])
    end

    waic_vec = (-2) .* (lpd - pD)
    waic = sum(waic_vec)
    lpd = sum(lpd)
    pD = sum(pD)

    local se
    try
        se = sqrt(n_obs * var2(waic_vec))
    catch e
        println(e)
        se = missing
    end

    # return (WAIC=waics, WAIC_i=waic_vec, lppd=lpd, penalty=pD, std_err=se)
    return WAIC(waic, waic_vec, lpd, pD, se)
end


function reshape_pointwise_log_likes(pointwise_log_likes::AbstractArray)
    dims = size(pointwise_log_likes)
    lwpt = reshape(pointwise_log_likes, dims[1], dims[2] * dims[3])
    ll = permutedims(lwpt, (2, 1))
    return ll
end

"""
    Use waic on pointwise_log_likes from Turing
"""
function compute_waic(pointwise_log_likes::AbstractArray)
    ll = reshape_pointwise_log_likes(pointwise_log_likes)
    return waic_2D_array(ll)
end


function compute_z(waic_1::WAIC, waic_2::WAIC)
    N = length(waic_1.waic_i)
    @assert N == length(waic_2.waic_i)

    Δ_std = sqrt(N) * std(waic_2.waic_i - waic_1.waic_i)
    Δ_mean = waic_2.waic - waic_1.waic

    z = abs(Δ_mean / Δ_std)

    return z, Δ_mean, Δ_std
end


function compare(waics, waic_names::Vector{String})

    ordered = sortperm(waics; by = w -> w.waic, rev = false)

    zs = Float64[]
    Δ_means = Float64[]
    Δ_stds = Float64[]
    for i in ordered
        z, Δ_mean, Δ_std = compute_z(waics[i], waics[ordered[1]])
        push!(zs, z)
        push!(Δ_means, Δ_mean)
        push!(Δ_stds, Δ_std)
    end

    df_comparison = DataFrames.DataFrame(
        name = waic_names[ordered],
        WAIC = [w.waic for w in waics[ordered]],
        std = [w.std for w in waics[ordered]],
        lppd = [w.lppd for w in waics[ordered]],
        pWAIC = [w.penalty for w in waics[ordered]],
        z = zs,
        Δ_mean = Δ_means,
        Δ_std = Δ_stds,
    )
    df_comparison[1, :Δ_mean] = NaN
    df_comparison[1, :Δ_std] = NaN

    return df_comparison
end


function plot(df_comparison, ic = :waic)

    # ic = information criterion

    if ic == :waic
        ic_out_of_sample = df_comparison.WAIC
        ic_in_sample = -2 * df_comparison.lppd
        ic_name = "WAIC (lower is better)"
    elseif ic == :log
        ic_out_of_sample = -0.5 * df_comparison.WAIC
        ic_in_sample = df_comparison.lppd
        ic_name = "Log score (higher is better)"
    end


    ys = size(df_comparison, 1):-1:1

    f = CairoMakie.Figure(resolution = (1000, 300))
    ax = CairoMakie.Axis(
        f[1, 1],
        xlabel = ic_name,
        # ylabel = "Model",
        limits = (nothing, nothing, minimum(ys) - 0.3, maximum(ys) + 0.3),
        title = "Model Comparison",
        yticks = (ys, df_comparison.name),
    )


    CairoMakie.errorbars!(
        ax,
        ic_out_of_sample,
        ys,
        df_comparison.std,
        color = :black,
        whiskerwidth = 10,
        direction = :x,
    ) # same low and high error


    CairoMakie.vlines!(
        ax,
        ic_out_of_sample[1],
        # linestyle = :dash,
        color = :black,
        linewidth = 0.75,
    )


    CairoMakie.scatter!(
        ax,
        ic_out_of_sample,
        ys,
        markersize = 15,
        color = :black,
        marker = :circle,
    )
    CairoMakie.scatter!(ax, ic_in_sample, ys, markersize = 15, color = :black, marker = :x)

    mask_Δ = 2:size(df_comparison, 1)
    CairoMakie.errorbars!(
        ax,
        ic_out_of_sample[mask_Δ], # df_comparison[mask_Δ, :WAIC]
        ys[mask_Δ] .+ 0.3,
        df_comparison[mask_Δ, :Δ_std],
        color = :grey,
        whiskerwidth = 7,
        direction = :x,
    ) # same low and high error

    CairoMakie.scatter!(
        ax,
        ic_out_of_sample[mask_Δ], # df_comparison[mask_Δ, :WAIC]
        # df_comparison[mask_Δ, :WAIC],
        ys[mask_Δ] .+ 0.3,
        marker = :utriangle,
        markersize = 15,
        color = :grey,
    )

    for i in mask_Δ
        z = round(df_comparison[i, :z], digits = 2)
        x = ic_out_of_sample[i] # df_comparison[i, :WAIC]
        y = ys[i] + 0.5
        CairoMakie.text!(
            "z = $z",
            position = (x, y),
            align = (:center, :center),
            color = :grey,
            textsize = 16,
        )
    end
    return f

end


end