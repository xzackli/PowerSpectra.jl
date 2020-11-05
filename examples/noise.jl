using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools
using Distributions, Random
import ThreadPools: @qthreads
using BenchmarkTools
hp = pyimport("healpy")
nmt = pyimport("pymaster")

##
nside = 256
mask1 = Map{Float64, RingOrder}(ones(nside2npix(nside)))
mask2 = Map{Float64, RingOrder}(ones(nside2npix(nside)))
beam1 = SpectralVector(ones(3 * nside))
beam2 = SpectralVector(ones(3 * nside))

theory = CSV.read("notebooks/data/theory.csv")
noise = CSV.read("notebooks/data/noise.csv")
nltt = convert(Vector{Float64}, noise.nltt)

nalm0 = generate_correlated_noise(nside, 10 * 2π / 180, nltt)
pow = alm2cl(nalm0)
x = Float64.(collect(eachindex(pow)))
y = log.(pow)

clf()
plot(pow)
n0 = 1.5394030890788515 / nside
plot(nltt)
yscale("log")
gcf()


##
function get_sim(nside, nltt, factorized_mcm)
    pixwin = true
    signal = hp.synfast(theory.cltt, 
        nside, verbose=false, pixwin=pixwin, new=true)

    n1 = generate_correlated_noise(nside, 10 * 2π / 180, nltt)
    n2 = generate_correlated_noise(nside, 10 * 2π / 180, nltt)
    # n1 = hp.synfast(noise.nltt, 
    #     nside, verbose=false, pixwin=pixwin, new=true)
    # n2 = hp.synfast(noise.nltt, 
    #     nside, verbose=false, pixwin=pixwin, new=true)
    map1 = Map{Float64, RingOrder}(nside)
    map1.pixels .= hp.sphtfunc.smoothing(signal .+ n1, beam_window=beam1, verbose=false)
    map2 = Map{Float64, RingOrder}(nside)
    map2.pixels .= hp.sphtfunc.smoothing(signal .+ n2, beam_window=beam2, verbose=false)
    Clhat = compute_spectra(map1 * mask1, map2 * mask2, factorized_mcm, beam1, beam2)
    return Clhat
end

function generate_sim_array(nsims)
    result = Array{Float64, 2}(undef, (3 * nside, nsims))
    for i in 1:nsims
        Clhat = get_sim(nside, nltt, factorized_mcm)
        result[:,i] .= Clhat
    end
    return result
end

##
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
m_143_hm1 = Field("143_hm1", mask1, flat_mask, beam1)
m_143_hm2 = Field("143_hm2", mask2, flat_mask, beam2)
workspace = SpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
@time mcm = compute_mcm(workspace, "143_hm1", "143_hm2")
@time factorized_mcm = cholesky(Hermitian(mcm.parent));



##
sims = generate_sim_array(30)

##

##

σₚ = std(alm2map(nalm0, nside))
meanpow = mean(sims, dims=2)[:,1]

clf()
plot(meanpow[3:512])
n0 = 1.5394030890788515 / nside
# plot(n0)
plot(nltt[3:512])
# axhline(n0)
# plot(ss.savgol_filter(ys, 51, 1))
yscale("log")
gcf()

##

import AngularPowerSpectra: TT

theory = CSV.read("notebooks/data/theory.csv")
noise = CSV.read("notebooks/data/noise.csv")

cltt = SpectralVector(convert(Vector, theory.cltt))
nltt = SpectralVector(convert(Vector, noise.nltt))

spectra = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (TT, "143_hm1", "143_hm1") => cltt .+ nltt,
    (TT, "143_hm1", "143_hm2") => cltt,
    (TT, "143_hm2", "143_hm1") => cltt,
    (TT, "143_hm2", "143_hm2") => cltt .+ nltt)
@time C = compute_covmat(workspace, spectra, factorized_mcm, factorized_mcm,
                         m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);


##


clf()
varpow = var(sims, dims=2)
plot(varpow[3:512])
plot(diag(C.parent)[3:512])
yscale("log")
gcf()

##

# m = Map{Float64,RingOrder}(nside)
# clf(); 
# randn!(m.pixels)
# σₚ = 1.0
# npix = nside2npix(nside)
# Ωₚ = 4π / npix
# plot(alm2cl(map2alm(m))[1:512], label="realization")
# axhline(npix * σₚ^2 * Ωₚ^2 / 4π, color="C2", label="analytic")
# title("White Noise")
# xlabel(raw"Multipole moment, $\ell$")
# # yscale("log")
# gcf()

##
