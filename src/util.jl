

function binning_matrix(left_bins, right_bins, weight_function_ℓ; lmax=nothing)
    nbins = length(left_bins)
    lmax = isnothing(lmax) ? right_bins[end] : lmax
    P = zeros(nbins, lmax)
    for b in 1:nbins
        weights = weight_function_ℓ.(left_bins[b]:right_bins[b])
        norm = sum(weights)
        P[b, left_bins[b]+1:right_bins[b]+1] .= weights ./ norm
    end
    return P
end


# function generate_correlated_noise(nside, σ, nltt)
#     noisemap = Map{Float64, RingOrder}(nside)
#     res = noisemap.resolution

#     rings = Array{RingInfo, 1}(undef, Threads.nthreads())
#     cov_buffers = Array{Array{Float64, 2}, 1}(undef, Threads.nthreads())
#     Threads.@threads for i in 1:Threads.nthreads()
#         rings[i] = RingInfo(0, 0, 0, 0, 0)
#         cov_buffers[i] = Array{Float64, 2}(undef, (4nside, 4nside))
#     end

#     @threads for ring_index in 1:(4nside - 1)
#         thid = Threads.threadid()
#         ring = rings[thid]
#         getringinfo!(res, ring_index, ring)
#         npix = ring.numOfPixels
#         pixsize = 2π / npix
#         lastpix = ring.firstPixIdx + npix - 1
#         ring_vector = view(noisemap.pixels, ring.firstPixIdx:lastpix)
#         cov_pp = view(cov_buffers[thid], 1:npix, 1:npix)

#         for m ∈ 1:npix, n ∈ 1:npix
#             Δϕ = abs(m - n) * pixsize
#             Δϕ = min(Δϕ, 2π - Δϕ)
#             cov_pp[m, n] = exp(-0.5 * (Δϕ / σ))
#         end
        
#         dist = MvNormal(zeros(npix), Matrix(Hermitian(cov_pp)))
#         rand!(dist, ring_vector)
#     end

#     alms = map2alm(noisemap)

#     n0 = 1.5394030890788515 / nside
#     for l in 0:alms.lmax
#         for m in 0:l
#             index = almIndex(alms, l, m)
#             alms.alm[index] *= sqrt(l / n0 * nltt[l+1])
#         end
#     end

#     alm2map(alms, nside)
#     # alms
# end



# function get_ell_array(lmax)
#     nalm = numberOfAlms(lmax, lmax)
#     ell_alm_array = zeros(Int, nalm)
#     zero_alm = Alm(lmax, lmax, Zeros{Complex{Float64}}(nalm))

#     for l in 0:lmax
#         for m in 0:l 
#             index = almIndex(zero_alm, l, m)
#             ell_alm_array[index] = l
#         end
#     end
#     return ell_alm_array
# end

