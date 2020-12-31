

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


function read_commented_header(filename; delim=" ", strip_spaces=true)
    header = CSV.read(filename, DataFrame; header=false, delim=delim, ignorerepeated=true,
        limit=1, type=String)
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

    table = CSV.read(filename, DataFrame; comment="#", header=headers, delim=delim,
        ignorerepeated=true)
    return table
end


"""
convenience functions for interacting with Alm using a[ℓ, m] indices
"""
@inline function Base.setindex!(A::Alm{T,AA}, val, ℓ::Int, m::Int) where {T,AA}
    i = almIndex(A, ℓ, m)
    A.alm[i] = val
end
@inline function Base.getindex(A::Alm{T,AA}, ℓ::Int, m::Int) where {T, AA}
    i = almIndex(A, ℓ, m)
    A.alm[i]
end


"""
    synalm([rng=GLOBAL_RNG], Cl::AbstractArray{T,3}, nside::Int) where T

# Arguments:
- `Cl::AbstractArray{T,3}`: array with dimensions of comp, comp, ℓ
- `nside::Int`: healpix resolution

# Returns:
- `Vector{Alm{T}}`: spherical harmonics realizations for each component

# Examples
```julia
nside = 16
C0 = [3.  2.;  2.  5.]
Cl = repeat(C0, 1, 1, 3nside)  # spectra constant with ℓ
synalm(Cl, nside)
```
"""
function synalm(rng::AbstractRNG, Cl::AbstractArray{T,3}, nside::Int) where T
    Cl_ℓmax = size(Cl,3)
    ncomp = size(Cl,1)
    @assert size(Cl,1) == size(Cl,2)
    @assert ncomp > 0
    lmax = 3nside-1

    # covariance buffer between components
    𝐂 = Array{T,2}(undef, (ncomp, ncomp))
    # alm array to return
    alms = Alm{Complex{T}}[Alm{Complex{T}}(3nside-1, 3nside-1) for i in 1:ncomp]

    # first we synthesize just a unit normal for alms. we'll adjust the magnitudes later
    # it is faster to perform randn! on arrays than one at a time
    for comp in 1:ncomp
        randn!(rng, alms[comp].alm)
    end

    a_buffer = zeros(Complex{T}, ncomp)
    for ℓ in 0:lmax
        for m in 0:ℓ
            # build the 𝐂 matrix for ℓ. only necessary to copy the upper triangle
            for cᵢ in 1:ncomp, cⱼ in cᵢ:ncomp
                𝐂[cᵢ, cⱼ] = Cl[cᵢ, cⱼ, ℓ+1]
            end
            cholesky!(Hermitian(𝐂))
            i_alm = almIndex(alms[1], ℓ, m)  # compute alm index
            for comp in 1:ncomp  # copy over the random variates into buffer
                a_buffer[comp] = alms[comp].alm[i_alm]
            end
            lmul!(LowerTriangular(𝐂'), a_buffer)  # transform
            for comp in 1:ncomp  # copy buffer back into the alms
                alms[comp].alm[i_alm] = a_buffer[comp]
            end
        end
    end

    return alms
end
synalm(Cl::AbstractArray{T,3}, nside::Int) where T = synalm(Random.default_rng(), Cl, nside)
