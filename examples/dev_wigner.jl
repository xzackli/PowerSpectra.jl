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
    c = cov(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2; lmax=150)
    GC.gc()
end

##
# PSPlanck.__init__()
@time cov(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2; lmax=256)

##
using PSPlanck

function test1(n)
    for i in 1:n
        for j in reverse(1:n)
            for k in abs(i-j):(i+j)
                PSPlanck.wigner3j²(Float64,i,j,k,0,0)
            end
        end
    end
end
##
PSPlanck.__init__()
@time test1(200)


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



##

