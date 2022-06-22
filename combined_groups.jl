

# #######


# df_Δ_WT1_good_groups = filter(:nrow => >(min_N_rows), df_Δ_WT1)
# df_Δ_WT1_good_groups
# unique(df_Δ_WT1_good_groups[:, [:group, :id, :cell]])


# @model function diffusion_2D_groups(Δ, group)

#     groups = levels(group)
#     N_groups = length(levels(group))

#     # prior d
#     Δds ~ filldist(Exponential(0.1), 2)
#     ds = cumsum(Δds)

#     dists = [diffusion_1D(d) for d in ds]

#     # prior θ
#     Θ ~ filldist(Uniform(0, 1), N_groups)

#     for (i, g) in enumerate(groups)

#         group_mask = (g .== group)
#         w = [Θ[i], 1 - Θ[i]]

#         # mixture distribution
#         distribution = MixtureModel(dists, w)

#         # likelihood
#         Δ[group_mask] ~ filldist(distribution, sum(group_mask))

#     end

#     return (; ds)
# end


# chains_WT1_2D_groups = get_chains(
#     name = f"WT1_2D_groups_{min_N_rows}",
#     model = diffusion_2D_groups(df_Δ_WT1_good_groups.Δ, df_Δ_WT1_good_groups.group),
#     N_samples = N_samples,
#     N_chains = N_chains,
#     # hide_warnings = true,
# )


# variables = [
#     get_variables_in_group(chains_WT1_2D_groups, :ds)...,
#     get_variables_in_group(chains_WT1_2D_groups, :Θ)...,
# ]

# plot_chains(chains_WT1_2D_groups; variables = variables[begin:begin+5])

# fig_WT1_2D_groups =
#     plot_chains(chains_WT1_2D_groups; variables = variables, resolution = (1000, 10000))
# save(f"figs/WT1_2D_groups_{min_N_rows}.pdf", fig_WT1_2D_groups)


# ##


# # chains = chains_WT1_2D_groups

# function compute_d_combined!(d_combined, ds, Θ)
#     for i in eachindex(Θ)
#         d_combined[i] = ds[i, 1] * Θ[i] + ds[i, 2] * (1 - Θ[i])
#     end
# end

# function compute_d_combined(chains::Turing.Chains)
#     ds = Array(chains[get_variables_in_group(chains, :ds)])
#     Θs = Array(chains[get_variables_in_group(chains, :Θ)])
#     d_combined = zeros(eltype(Θs), size(Θs))
#     for i in axes(Θs, 2)
#         compute_d_combined!(view(d_combined, :, i), ds, view(Θs, :, i))
#     end
#     return d_combined
# end

# d_combined_WT1_2D_groups = compute_d_combined(chains_WT1_2D_groups)
# mean(eachrow(d_combined_WT1_2D_groups)) .< 0.045876509391691855


# ####################

# df_Δ_focus_good_groups = filter(:nrow => >(min_N_rows), df_Δ_focus);
# unique(df_Δ_focus_good_groups[:, [:group, :id, :cell]])


# chains_focus_2D_groups = get_chains(
#     name = f"focus_2D_groups_{min_N_rows}",
#     model = diffusion_2D_groups(df_Δ_focus_good_groups.Δ, df_Δ_focus_good_groups.group),
#     N_samples = N_samples,
#     N_chains = N_chains,
#     # hide_warnings = true,
# )



# variables = [
#     get_variables_in_group(chains_focus_2D_groups, :ds)...,
#     get_variables_in_group(chains_focus_2D_groups, :Θ)...,
# ]

# plot_chains(chains_focus_2D_groups; variables = variables[begin:begin+5])

# chains_focus_2D_groups[get_variables_in_group(chains_focus_2D_groups, :Θ)]


# ####################

# df_Δ_delta_good_groups = filter(:nrow => >(min_N_rows), df_Δ_delta)
# unique(df_Δ_delta_good_groups[:, [:group, :id, :cell]])


# chains_delta_2D_groups = get_chains(
#     name = f"delta_2D_groups_{min_N_rows}",
#     model = diffusion_2D_groups(df_Δ_delta_good_groups.Δ, df_Δ_delta_good_groups.group),
#     N_samples = N_samples,
#     N_chains = N_chains,
#     # hide_warnings = true,
# )


# variables = [
#     get_variables_in_group(chains_delta_2D_groups, :ds)...,
#     get_variables_in_group(chains_delta_2D_groups, :Θ)...,
# ]

# plot_chains(chains_delta_2D_groups; variables = variables[begin:begin+5])

# fig_delta_2D_groups =
#     plot_chains(chains_delta_2D_groups; variables = variables, resolution = (1000, 10000))
# save(f"figs/delta_2D_groups_{min_N_rows}.pdf", fig_delta_2D_groups)

