## load in the data
using AngularPowerSpectra
using Healpix

data_dir = "/home/zequnl/.julia/dev/AngularPowerSpectra/notebooks/data/"
# flatmap = readMapFromFITS(data_dir * "mask.fits", 1, Float64)
flatmap = Map{Float64, RingOrder}(ones(nside2npix(1024)))  # FLAT MASK

m_143_hm1 = Field("143_hm1", flatmap, flatmap)
m_143_hm2 = Field("143_hm2", flatmap, flatmap)
workspace = SpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)

AngularPowerSpectra.effective_weights!(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
AngularPowerSpectra.W_spectra!(workspace)

##
using BenchmarkTools
@btime cov($workspace, $m_143_hm1, $m_143_hm2, $m_143_hm1, $m_143_hm2)

##
@btime AngularPowerSpectra.compute_mcm!($workspace, "143_hm1", "143_hm2")

##

# using PyPlot
# plt.clf()
# plt.imshow(log10.(C))
# plt.gcf()
