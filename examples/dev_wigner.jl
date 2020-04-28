
## load in the data
using PSPlanck
using Healpix

data_dir = "/home/zequnl/.julia/dev/PSPlanck/notebooks/data/"
flatmap = Healpix.readMapFromFITS(data_dir * "mask.fits", 1, Float64)
# flatmap = Map{Float64, RingOrder}(ones(nside2npix(512)))  # FLAT MASK


m_143_hm1 = Field("143_hm1", flatmap, flatmap)
m_143_hm2 = Field("143_hm2", flatmap, flatmap)

# w = cov(m_143_hm1, m_143_hm2)
@time begin
wcoeff = PSPlanck.w_coefficients(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
field_names = [m_143_hm1.name, m_143_hm2.name, m_143_hm1.name, m_143_hm2.name]
W_spec = PSPlanck.W_spectra(Float64, field_names, wcoeff)
end

##

using LinearAlgebra
import ThreadPools: @qthreads

function test(n, W)
    x = zeros(Float64, (n,n))
    W_arr1 = W[
        PSPlanck.∅∅, PSPlanck.∅∅, "143_hm1", "143_hm1", 
        "143_hm2", "143_hm2", PSPlanck.TT, PSPlanck.TT]
    W_arr2 = W[
        PSPlanck.∅∅, PSPlanck.∅∅, "143_hm1", "143_hm2", 
        "143_hm1", "143_hm2", PSPlanck.TT, PSPlanck.TT]
    W_arr3 = W[
        PSPlanck.∅∅, PSPlanck.II, "143_hm1", "143_hm1", 
        "143_hm2", "143_hm2", PSPlanck.TT, PSPlanck.TT]

    ij = [(i,j) for i in 1:n for j in i:n]
    @qthreads for (i,j) in ij
        x[i, j] = (
            PSPlanck.ΞTT(W_arr1, i, j) +
            PSPlanck.ΞTT(W_arr2, i, j) + 
            PSPlanck.ΞTT(W_arr3, i, j) 
        )
    end
    return Symmetric(x)
end

##
@time begin
    test(128, W_spec)
    GC.gc()
end

# mollview(m)
##
using PyPlot
using LinearAlgebra

plt.clf()
plt.plot( (diag(test(150, W_spec))) )
plt.gcf()
##

plt.clf()
plt.imshow( log.((test(150, W_spec))) )
plt.gcf()
