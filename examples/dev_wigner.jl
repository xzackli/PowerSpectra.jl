## load in the data
using AngularPowerSpectrum
using Healpix

data_dir = "/home/zequnl/.julia/dev/AngularPowerSpectrum/notebooks/data/"
flatmap = readMapFromFITS(data_dir * "mask.fits", 1, Float64)
# flatmap = Map{Float64, RingOrder}(ones(nside2npix(512)))  # FLAT MASK

m_143_hm1 = Field("143_hm1", flatmap, flatmap)
m_143_hm2 = Field("143_hm2", flatmap, flatmap)
workspace = CovarianceWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)

AngularPowerSpectrum.w_coefficients!(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
AngularPowerSpectrum.W_spectra!(workspace)

## 
# using ProfileVega  # NO THREADING
import WignerFamilies: wigner3j_f!, WignerF, WignerSymbolVector, swap_triangular, get_wigner_array
import UnsafeArrays: uview
import ThreadPools: @qthreads
import Base.Threads: @threads

function test(::Type{T}, n) where {T}

    thread_buffers = Vector{Vector{T}}(undef, Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        thread_buffers[i] = Vector{T}(undef, 2*n+1)
    end
    
    @qthreads for j₁ in 0:n
        tid = Threads.threadid()
        for j₂ in 0:j₁
            w = WignerF(T, j₁, j₂, 0, 0)
            buffer = uview(thread_buffers[tid], 1:length(w.nₘᵢₙ:w.nₘₐₓ))
            w3j = WignerSymbolVector(buffer, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j)
        end
    end
end

@time test(Float64, 3000)


##

@time begin
    c = cov(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2; lmax=150)
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

