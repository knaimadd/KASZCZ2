using Plots, CSV, DataFrames, LsqFit, Statistics, StatsBase, StatsPlots, Measures, HypothesisTests, ARCHModels, Distributions

wroc = CSV.File("wroc.csv") |> DataFrame

T = wroc[:,3]

t0 = findfirst(T .== 20100101)
ts = findfirst(T .== 20191231)
tr = findfirst(T .== 20201231)

T = wroc[t0:ts,3]
X = wroc[t0:ts,4]*0.1
Q = wroc[t0:ts,5]
Xp = wroc[ts+1:tr,4]*0.1
Tp = ts+1:tr
##
times=[findfirst(T .== 20100101),
findfirst(T .== 20130101),
findfirst(T .== 20160101),
findfirst(T .== 20190101)]

p = plot(X, label=false, xlabel="data pomiaru", ylabel="temperatura [°C]",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10,  
xticks=(times, ["01.01.2010r.", "01.01.2013r.", "01.01.2016r.", "01.01.2019r."]))

savefig(p, "szereg.png")
##
p = boxplot(X, ylabel="temperatura [°C]", label=false,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10)

savefig(p, "boxplot.png")
##
ts = 0:length(X)-1

lags = ts[1:10:end-10]
p = scatter(lags, autocor(X, lags), ylabel="ACF", xlabel="przesunięcie", label=false,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10)
p = plot!(lags, autocor(X, lags), line=:stem, c=1, label=false)

savefig(p, "ACF1.png")
##
p = plot(0:40, pacf(X, 0:40), line=:stem, c=1, lw=2, label=false)
p = scatter!(0:40, pacf(X, 0:40), ylabel="PACF", xlabel="przesunięcie", label=false,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10,
ms=5, c=1)

savefig(p, "PACF1.png")

##
linear(x, p) = x*p[1] .+ p[2]
sinusoidla(x, p) = p[3]*sin.(x*p[1] .+ p[2])
##
ts = 1:length(T)

fitlin = curve_fit(linear, ts, X, [1/30, 0])
pl = fitlin.param
Xl = X .- linear(ts, pl)
##
r(x) = round(x, digits=3)
##
p = plot(X, label="badany szereg czsowy", ylabel = "Yₜ", xlabel="numer obserwacji",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10,)
p = plot!(linear(ts, pl), label="y = $(r(pl[1]))x + $(r(pl[2]))", lw=2)

savefig(p, "liniowe.png")
##
p = scatter(lags, autocor(Xl, lags), ylabel="ACF", xlabel="przesunięcie", label=false,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10)
p = plot!(lags, autocor(Xl, lags), line=:stem, c=1, label=false)

savefig(p, "ACF2.png")
##
fitsin = curve_fit(sinusoidla, ts, Xl, [0.017, -20, 10])
ps = fitsin.param
Xs = Xl .- sinusoidla(ts, ps)
##
p = plot(Xl, label="badany szereg czsowy", ylabel="Yₜ - m(t)", xlabel="numer obserwacji",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)
p = plot!(sinusoidla(ts, ps), lw=2.5, label="y=$(r(ps[3]))sin($(r(ps[1]))x $(r(ps[2])))")

savefig(p, "przekLin.png")
##
p = plot(Xs, label=false, ylabel="Yₜ - m(t) - s(t)", xlabel="numer obserwacji",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)

savefig(p, "przekSin")
##
p = plot(0:50, autocor(Xs, 0:50), line=:stem, c=1, label=false, lw=2)
p = scatter!(0:50, autocor(Xs, 0:50), ylabel="ACF", xlabel="przesunięcie", label=false,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, ms=5, c=1)

savefig(p, "ACF3.png")
##
p = plot(0:50, pacf(Xs, 0:50), line=:stem, c=1, label=false, lw=2)
p = scatter!(0:50, pacf(Xs, 0:50), ylabel="PACF", xlabel="przesunięcie", label=false,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, ms=5, c=1)

savefig(p, "PACF3.png")
##
ADFTest(Xs, :constant, 3)
##

##
fit_arma(df, p, q) = fit(ARMA{p, q}, df)

n = length(X)
pMax = 8
qMax = 8
AIC = Matrix{Float64}(undef, pMax+1, qMax+1) 
BIC = Matrix{Float64}(undef, pMax+1, qMax+1) 
HQIC = Matrix{Float64}(undef, pMax+1, qMax+1)
for p in 0:pMax, q in 0:qMax
    model = fit_arma(Xs, p, q)
    k = p+q
    L = loglikelihood(model)
    AIC[p+1, q+1] = -2*L + 2k
    BIC[p+1, q+1] = -2*L + k*log(n)
    HQIC[p+1, q+1] = -2*L + 2k*log(log(n)) 
end
##
min1 = findmin(AIC)[2]
min2 = findmin(BIC)[2]
min3 = findmin(HQIC)[2]
##
p1 = heatmap(AIC, yflip=true, xlabel="q", ylabel="p", c=cgrad(:default)[25:end],
title = "AIC", xticks=(1:9, string.(0:8)), yticks=(1:9, string.(0:8)), aspect_ratio=:equal, legend=false)
p1 = scatter!([min1[2]], [min1[1]], xlim=[0.5,9.5], ylim=[0.5,9.5],
c=:lightgreen, marker=:x, label="minimum", markersize=5)

p2 = heatmap(BIC, yflip=true, xlabel="q", ylabel="p", c=cgrad(:default)[25:end],
title = "BIC", xticks=(1:9, string.(0:8)), yticks=(1:9, string.(0:8)), aspect_ratio=:equal, legend=:none)
p2 = scatter!([min2[2]], [min2[1]], xlim=[0.5,9.5], ylim=[0.5,9.5],
c=:lightgreen, marker=:x, label=false, markersize=5)

p3 = heatmap(HQIC, yflip=true, xlabel="q", ylabel="p", c=cgrad(:default)[25:end],
title = "HQIC", xticks=(1:9, string.(0:8)), yticks=(1:9, string.(0:8)), aspect_ratio=:equal, legend=:none)
p3 = scatter!([min3[2]], [min3[1]], xlim=[0.5,9.5], ylim=[0.5,9.5],
c=:lightgreen, marker=:x, label=false, markersize=5)
##
l = @layout[grid(1,3) a{0.03w}]

p4 = heatmap((LinRange(minimum(AIC), maximum(AIC), 101)).*ones(101,1), legend=:none, xticks=:none, yticks=(1:20:101, string.(round.(Int, LinRange(minimum(AIC), maximum(AIC), 6)))),
ytickfontsize=9, c=cgrad(:default)[25:end])

p = plot(p1, p2, p3, p4, layout=l, size=(900, 320), margin=2.5mm, titlefontsize=16,
labelfontsize=13)

savefig(p, "heatmap")

##
heatmap([1 0 1;
1 0 2], yflip=true)
##
model1 = fit_arma(Xs, 1, 3)
##
model = fit_arma(Xs, 1, 3)
##
c = coeftable(model)
##
sim = simulate(model).data
##
res = residuals(model)
##
p = scatter(res, c=:red, ms=3, label=false, xlabel="numer obserwacji", ylabel="residua",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10)

savefig(p, "residua")
##
xs = LinRange(-4, 4, 10^3)
histogram(res, normed=true)
plot!(xs, pdf.(PGeneralizedGaussian(mean(res), 1.3, 1.9), xs),)
##
mean(res)
##
var(res)
##
length(res)
##
OneSampleTTest(res, 0)
##
LeveneTest(res[1:1826], res[1827:end])
##
LeveneTest(res[1:913], res[914:1826], res[1827:2738], res[2739:end])
##
EqualVarianceTTest(res[1:1826], res[1827:end])
##
lags=0:50
p = scatter(lags, autocor(res, lags), ylabel="ACF", xlabel="przesunięcie", label=false,
size=(900, 600), margin=3mm, labelfontsize=15, c=:red, tickfontsize=10, ms=5)
p = plot!(lags, autocor(res, lags), line=:stem, c=:red, label=false)

savefig(p, "resACF")
##
LjungBoxTest(res, 5, 4)
##
xs = LinRange(-4, 4, 10^3)
p = histogram(res, normed=true, xlabel="x", ylabel="f(x)", label="histogram przybliżający gęstość residuów")
p = plot!(xs, pdf.(Normal(mean(res), std(res)), xs), lw=3, label="gęstość f(x) rozkładu N(-0.0006, 1,0003)",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)

savefig(p, "resHist")
##
p = plot(xs, ecdf(res)(xs), lw=3.3, xlabel="x", ylabel="F(x)", label="dystrybuanta empiryczna residuów")
p = plot!(xs, cdf.(Normal(mean(res), std(res)), xs), lw=2.5, label="dystrybuanta F(x) rozkładu N(-0.0006, 1.0003)",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)

savefig(p, "resDist")
##
using Pingouin

Pingouin.normality(res)
##
xs = LinRange(-4, 4, 10^3)
p = histogram(res, normed=true, xlabel="x", ylabel="f(x)", label="histogram przybliżający gęstość residuów")
p = plot!(xs, pdf.(PGeneralizedGaussian(mean(res), 1.32, 1.85), xs), lw=3, label="gęstość f(x) wyestymowanego rozkładu",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)

savefig(p, "gHist")
##
ExactOneSampleKSTest(res, PGeneralizedGaussian(mean(res), 1.32, 1.85))
##
OneSampleADTest(res, PGeneralizedGaussian(mean(res), 1.32, 1.85))
##
modelAlt = fit(ARMA{1, 3}, Xs, dist=StdGED)

coeftable(modelAlt)
coeftable(model)
##
m = 10^3
n = 60

B = Matrix{Float64}(undef, m, n+1)

for j in 1:m
    B[j, :] = autocor(simulate(model).data, 0:n)
end
##
Bq95 = [quantile(B[:, i], 0.95) for i in 1:n+1]
Bq05 = [quantile(B[:, i], 0.05) for i in 1:n+1]
##
p = plot(0:n, Bq05, fillrange=Bq95, alpha=0.3, label=false, c=:gray,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)
p = plot!(0:n, Bq05, c=3, label="kwantyl 0.05 ACF", lw=2)
p = plot!(0:n, Bq95, c=2, label="kwantyl 0.95 ACF", lw=2)
p = scatter!(0:n, autocor(Xs, 0:n), c=1, label="ACF danych", xlabel="przesunięcie", ylabel="ACF")

savefig(p, "ACFprzedzial")
##
m = 10^3
n = 60

A = Matrix{Float64}(undef, m, n+1)

for j in 1:m
    A[j, :] = pacf(simulate(model).data, 0:n)
end
##
q95 = [quantile(A[:, i], 0.95) for i in 1:n+1]
q05 = [quantile(A[:, i], 0.05) for i in 1:n+1]
##
p = plot(0:n, q05, fillrange=q95, alpha=0.3, label=false, c=:gray,
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)
p = plot!(0:n, q05, c=3, label="kwantyl 0.05 PACF", lw=2)
p = plot!(0:n, q95, c=2, label="kwantyl 0.95 PACF", lw=2)
p = scatter!(0:n, pacf(Xs, 0:n), c=1, label="PACF danych", xlabel="przesunięcie", ylabel="PACF")

savefig(p, "PACFprzedzial")
##
n = length(Xs)
m = 10^4

B = Matrix{Float64}(undef, m, n)

for j in 1:m
    B[j, :] = simulate(model).data
end
##
k05 = [quantile(B[:, i], 0.05) for i in 1:n]
k10 = [quantile(B[:, i], 0.1) for i in 1:n]
k20 = [quantile(B[:, i], 0.2) for i in 1:n]
k80 = [quantile(B[:, i], 0.80) for i in 1:n]
k90 = [quantile(B[:, i], 0.90) for i in 1:n]
k95 = [quantile(B[:, i], 0.95) for i in 1:n]
##
p = plot(Xs, xlabel="numer obserwacji", ylabel="Xₜ", label="dane",
size=(900, 600), margin=3mm, labelfontsize=15, tickfontsize=10, legendfontsize=10)
p = plot!(k05, lw=0.7, label="kwantyl rzędu 0.05")
p = plot!(k10, lw=0.7, label="kwantyl rzędu 0.10")
p = plot!(k20, lw=0.7, label="kwantyl rzędu 0.20")
p = plot!(k80, lw=0.7, label="kwantyl rzędu 0.80")
p = plot!(k90, lw=0.7, label="kwantyl rzędu 0.90", c=:red)
p = plot!(k95, lw=0.7, label="kwantyl rzędu 0.95")

savefig(p, "kwantyl")
##
Xs
L = 1:length(T)

d = DataFrame([Xs, L], [:t, :val])

CSV.write("dane.csv", d)
##

function genFtr(model, n)
    X = Vector{Float64}(undef, n)
    data = model.data
    res = residuals(model)
    ϕ = coef(model)[3]
    θ = coef(model)[4:end]
    new_res = randn(n)
    X[1] = ϕ .* data[end] + sum(θ .* res[end-2:end])
    X[2] = ϕ .* X[1] + θ[1] * new_res[1] + sum(θ[2:end] .* res[end-1:end])
    X[3] = ϕ .* X[2] + sum(θ[1:2] .* new_res[1:2]) + θ[3] .* res[end]
    for i in 4:n
        X[i] = ϕ * X[i-1] + sum(θ .* new_res[i-3:i-1])
    end
    return X
end

function strange(model, h)
    new = fit(MA{h}, model.data)
    σ = std(residuals(new))
    ψ = coef(new)[end-h:end]
    return σ^2*sum(ψ.^2)
end
##
scatter(genFtr(model, 366))
