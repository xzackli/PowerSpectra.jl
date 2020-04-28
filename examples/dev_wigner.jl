## load in the data
using PSPlanck
using Healpix

data_dir = "/home/zequnl/.julia/dev/PSPlanck/notebooks/data/"
flatmap = readMapFromFITS(data_dir * "mask.fits", 1, Float64)
# flatmap = Map{Float64, RingOrder}(ones(nside2npix(512)))  # FLAT MASK

m_143_hm1 = Field("143_hm1", flatmap, flatmap)
m_143_hm2 = Field("143_hm2", flatmap, flatmap)
workspace = CovarianceWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)

PSPlanck.w_coefficients!(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
PSPlanck.W_spectra!(workspace)

## 
# using ProfileVega  # NO THREADING
PSPlanck.__init__()
@time begin
    c = cov(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2; lmax=256)
    GC.gc()
end

##
# PSPlanck.__init__()
@time cov(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2; lmax=256)

##
@time 


##

##
using PyPlot
using LinearAlgebra

plt.clf()
plt.plot( diag(c) )
plt.gcf()
##

plt.clf()
plt.imshow( log10.(c) )
plt.colorbar()
plt.gcf()
