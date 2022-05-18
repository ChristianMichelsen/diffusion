
function diffusion_nll_1D(Δ, d)
    # return Δ / (2 * d * τ) * exp(-(Δ^2) / (4 * d * τ))
    distribution = diffusion_1D(d)
    return -sum(logpdf.(distribution, Δ))
end


function diffusion_nll_2D(Δ, d1, d2, f)
    # return f * diffusion_1D(Δ, d1) + (1 - f) * diffusion_1D(Δ, d2)
    distribution = diffusion_2D(d1, d2, f)
    return -sum(logpdf.(distribution, Δ))
end


function make_closure(Δ, func)
    return p -> func(Δ, p...)
end



function StatsBase.cov2cor(X::Matrix)
    return cov2cor(X, sqrt.(diag(X)))
end

function get_limits(d::AbstractDict)
    return collect(values(d))
end

function fit(nll, func, p0)
    p0 = get_limits(p0)

    func = TwiceDifferentiable(nll, p0; autodiff = :forward)
    res = optimize(func, p0)
    return get_fit_results(res, func)
end

function fit(nll, func, p0, lower, upper)

    p0 = get_limits(p0)
    lower = get_limits(lower)
    upper = get_limits(upper)

    func = TwiceDifferentiable(nll, p0; autodiff = :forward)
    res = optimize(func, lower, upper, p0)
    return get_fit_results(res, func)
end

function get_fit_results(res, func)
    parameters = Optim.minimizer(res)
    numerical_hessian = hessian!(func, parameters)
    var_cov_matrix = inv(numerical_hessian)
    return (
        res = res,
        μ = parameters,
        σ = sqrt.(diag(var_cov_matrix)),
        Σ = var_cov_matrix,
        ρ = cov2cor(var_cov_matrix),
    )
end



function f_MSD(xM, R_inf, d, σ)
    return 4 * σ^2 + R_inf^2 * (1 - exp(-4 * d * xM / R_inf^2))
end


function make_χ²_closure(xM, y, err, func)
    return p -> sum(@. ((func(xM, p...) - y)^2 / err^2))
end


function get_fit_parameter_string(parameter, index, fit)
    return f"{parameter} = {fit.μ[index]:.3f} +/- {fit.σ[index]:.3f}"
end

function plot_and_fit(Δ, name, color)

    fit_plot =
        fit(make_closure(Δ, diffusion_nll_2D), diffusion_nll_2D, d_p0, d_lower, d_upper)


    D1_str = get_fit_parameter_string("D1", 1, fit_plot)
    D2_str = get_fit_parameter_string("D2", 2, fit_plot)
    f_str = get_fit_parameter_string("f", 3, fit_plot)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Distance [um]", ylabel = "PDF value [um]")
    ylims!(ax, low = 0)

    hist!(
        ax,
        Δ,
        bins = 50,
        color = color,
        normalization = :pdf,
        strokewidth = 1,
        strokecolor = :black,
    )

    x = range(0, 0.5, 1000)
    y = pdf.(diffusion_2D(fit_plot.μ...), x)

    lines!(ax, x, y, color = :black, linewidth = 3)

    kws = Dict(:align => (:left, :center), :space => :relative)

    xpos = 0.42
    text!(f"Two Diffusion coefficients, {name}", position = (xpos, 0.9); kws...)
    text!(D1_str, position = (xpos, 0.8); kws...)
    text!(D2_str, position = (xpos, 0.75); kws...)
    text!(f_str, position = (xpos, 0.7); kws...)
    fig

    return fig

end

