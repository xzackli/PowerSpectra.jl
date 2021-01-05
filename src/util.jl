

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

# convenience functions for interacting with Alm using a[ℓ, m] indices
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
alms = synalm(Cl, nside)
```
"""
function synalm(rng::AbstractRNG, Cl::AbstractArray{T,3}, nside::Int) where T
    ncomp = size(Cl,1)
    @assert ncomp > 0
    alms = [Alm{Complex{T}}(3nside-1, 3nside-1) for i in 1:ncomp]
    synalm!(rng, Cl, alms)
    return alms
end
synalm(Cl::AbstractArray{T,3}, nside::Int) where T = synalm(Random.default_rng(), Cl, nside)


"""
    synalm!([rng=GLOBAL_RNG], Cl::AbstractArray{T,3}, alms::Vector{Alm{Complex{T}}}) where T

In-place synthesis of spherical harmonic coefficients, given spectra.

# Arguments:
- `Cl::AbstractArray{T,3}`: array with dimensions of comp, comp, ℓ
- `alms::Vector`: array of Alm to fill

# Examples
```julia
nside = 16
C0 = [3.  2.;  2.  5.]
Cl = repeat(C0, 1, 1, 3nside)  # spectra constant with ℓ
alms = [Alm{Complex{Float64}}(3nside-1, 3nside-1) for i in 1:2]
synalm!(Cl, alms)
```
"""
function synalm!(rng::AbstractRNG, Cl::AbstractArray{T,3}, alms::Vector) where {T}
    # This implementation could be 1.2x faster by storing the cholesky factorization, but
    # typically you also perform two SHTs with each synalm, which dominates the cost.

    ncomp = size(Cl,1)
    @assert ncomp > 0
    @assert size(Cl,1) == size(Cl,2)
    @assert size(alms,1) > 0
    lmax = alms[1].lmax

    # first we synthesize just a unit normal for alms. we'll adjust the magnitudes later
    for comp in 1:ncomp
        randn!(rng, alms[comp].alm)
    end
    𝐂 = Array{T,2}(undef, (ncomp, ncomp))  # covariance for this given ℓ
    h𝐂 = Hermitian(Array{T,2}(undef, (ncomp, ncomp)))  # hermitian buffer
    alm_out = zeros(Complex{T}, ncomp)
    alm_in = zeros(Complex{T}, ncomp)

    for ℓ in 0:lmax
        # build the 𝐂 matrix for ℓ
        for cᵢ in 1:ncomp, cⱼ in 1:ncomp
            𝐂[cᵢ, cⱼ] = Cl[cᵢ, cⱼ, ℓ+1]
        end

        if iszero(𝐂)
            for m in 0:ℓ
                i_alm = almIndex(alms[1], ℓ, m)  # compute alm index
                for comp in 1:ncomp  # copy buffer back into the alms
                    alms[comp].alm[i_alm] = zero(T)
                end
            end
        else
            h𝐂 .= Hermitian(𝐂)
            cholesky_factorizable = isposdef!(h𝐂)
            if !cholesky_factorizable
                h𝐂 .= sqrt(𝐂)
                for m in 0:ℓ
                    i_alm = almIndex(alms[1], ℓ, m)  # compute alm index
                    for comp in 1:ncomp  # copy over the random variates into buffer
                        alm_in[comp] = alms[comp].alm[i_alm]
                    end
                    mul!(alm_out, h𝐂, alm_in)
                    for comp in 1:ncomp  # copy buffer back into the alms
                        alms[comp].alm[i_alm] = alm_out[comp]
                    end
                end
            else
                # cholesky!(h𝐂)  # we already cholesky'd by calling isposdef!
                for m in 0:ℓ
                    i_alm = almIndex(alms[1], ℓ, m)  # compute alm index
                    for comp in 1:ncomp  # copy over the random variates into buffer
                        alm_in[comp] = alms[comp].alm[i_alm]
                    end
                    lmul!(LowerTriangular(h𝐂'), alm_in)  # transform
                    for comp in 1:ncomp  # copy buffer back into the alms
                        alms[comp].alm[i_alm] = alm_in[comp]
                    end
                end
            end
        end
    end
end
synalm!(Cl::AbstractArray{T,3}, alms::Vector) where T = synalm!(Random.default_rng(), Cl, alms)
