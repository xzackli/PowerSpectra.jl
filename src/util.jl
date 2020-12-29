

function binning_matrix(left_bins, right_bins, weight_function_ℓ; ℓₘₐₓ=nothing)
    nbins = length(left_bins)
    ℓₘₐₓ = isnothing(ℓₘₐₓ) ? right_bins[end] : ℓₘₐₓ
    P = zeros(nbins, ℓₘₐₓ)
    for b in 1:nbins
        weights = weight_function_ℓ.(left_bins[b]:right_bins[b])
        norm = sum(weights)
        P[b, left_bins[b]+1:right_bins[b]+1] .= weights ./ norm
    end
    return P
end


function read_commented_header(filename; delim=" ", strip_spaces=true)
    header = CSV.read(filename, DataFrame; header=false, delim=delim, ignorerepeated=true, limit=1, type=String)
    if strip_spaces
        headers = [String(strip(header[1,"Column$(i)"])) for i in 1:ncol(header)] 
        
    else
        headers = [header[1,"Column$(i)"] for i in 1:ncol(header)] 
    end
    if headers[1] == "#"   # skip the #
        headers = headers[2:end]
    elseif headers[1][1] == '#'
        headers[1] = String(strip(headers[1][2:end]))
    end

    table = CSV.read(filename, DataFrame; comment="#", header=headers, delim=delim, ignorerepeated=true)
    return table
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
#     for l in 0:alms.ℓₘₐₓ
#         for m in 0:l
#             index = almIndex(alms, l, m)
#             alms.alm[index] *= sqrt(l / n0 * nltt[l+1])
#         end
#     end

#     alm2map(alms, nside)
#     # alms
# end



# function get_ell_array(ℓₘₐₓ)
#     nalm = numberOfAlms(ℓₘₐₓ, ℓₘₐₓ)
#     ell_alm_array = zeros(Int, nalm)
#     zero_alm = Alm(ℓₘₐₓ, ℓₘₐₓ, Zeros{Complex{Float64}}(nalm))

#     for l in 0:ℓₘₐₓ
#         for m in 0:l 
#             index = almIndex(zero_alm, l, m)
#             ell_alm_array[index] = l
#         end
#     end
#     return ell_alm_array
# end

