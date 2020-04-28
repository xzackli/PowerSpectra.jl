using PyPlot
plt.plot()
plt.gcf()
## load in the data
using PSPlanck
using Healpix

data_dir = "/home/zequnl/.julia/dev/PSPlanck/notebooks/data/"
# m = Healpix.readMapFromFITS(data_dir * "mask.fits", 1, Float64)
flatmap = Map{Float64, RingOrder}(ones(nside2npix(32)))  # FLAT MASK
# flat_alm = map2alm(m)

m_143_hm1 = Field("143_hm1", flatmap, flatmap)
m_143_hm2 = Field("143_hm2", flatmap, flatmap)


w = cov(m_143_hm1, m_143_hm2)
# mollview(m)

##
res = PSPlanck.wigner3j²(Float64, 6,6,6,0,0)

##
##

plt.clf()
plt.plot(alm2cl(x.I))
plt.gcf()
