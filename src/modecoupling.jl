
# Projector function for TT. Goes into the mode-coupling matrix.
function Ξ_TT(𝐖::SpectralVector{T, AA},
              w3j²₀₀::WignerSymbolVector,
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₀₀), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j²₀₀), lastindex(𝐖))
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₀₀[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end


# Projector function for EE. Goes into the mode-coupling matrix.
# Note that w3j² refers to the square of ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
function Ξ_EE(𝐖::SpectralVector{T, AA},
              w3j²₂₂::WignerSymbolVector,
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₂₂), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j²₂₂), lastindex(𝐖))
    if isodd(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₂₂[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end

# Projector function for EE. Goes into the mode-coupling matrix.
# Note that w3j² refers to the square of ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
function Ξ_EB(𝐖::SpectralVector{T, AA},
              w3j²₂₂::WignerSymbolVector,
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₂₂), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j²₂₂), lastindex(𝐖))
    if iseven(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₂₂[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end


# Projector function for TE. Goes into the mode-coupling matrix.
# Note that w3j₀₀₂₂ refers to ( ℓ ℓ₂ ℓ₃ 0 0 0 ) × ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
function Ξ_TE(𝐖::SpectralVector{T, AA},
              w3j₀₀₂₂::WignerSymbolVector,
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j₀₀₂₂), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j₀₀₂₂), lastindex(𝐖))
    if isodd(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j₀₀₂₂[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end

# use a view of a memory buffer and fill with wigner 3j
function fill_3j!(buffer::Array{T,N}, ℓ₁, ℓ₂, m₁, m₂) where {T,N}
    w = WignerF(T, ℓ₁, ℓ₂, m₁, m₂)  # set up the wigner recurrence
    buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
    w3j = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
    wigner3j_f!(w, w3j)  # deposit symbols into buffer
    return w3j
end

# inner MCM loop TT
function inner_mcm⁰⁰!(𝐌::SpectralArray{T,2},
                      Vᵢⱼ::SpectralVector{T}) where {T}
    @assert axes(𝐌, 1) == axes(𝐌, 2)
    lmin, lmax = first(axes(𝐌, 1)), last(axes(𝐌, 1))
    thread_buffers = get_thread_buffers(T, 2lmax+1)

    @qthreads for ℓ₁ in lmin:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w3j²₀₀ = fill_3j!(buffer, ℓ₁, ℓ₂, 0, 0)
            w3j²₀₀.symbols .= w3j²₀₀.symbols .^ 2  # square the symbols
            Ξ = Ξ_TT(Vᵢⱼ, w3j²₀₀, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ
        end
    end
    return 𝐌
end


# inner MCM loop TE and TB
function inner_mcm⁰²!(𝐌::SpectralArray{T,2}, Vᵢⱼ::SpectralVector{T}) where {T}
    @assert axes(𝐌, 1) == axes(𝐌, 2)
    lmin, lmax = first(axes(𝐌, 1)), last(axes(𝐌, 1))
    thread_buffers_0 = get_thread_buffers(T, 2lmax+1)
    thread_buffers_2 = get_thread_buffers(T, 2lmax+1)
    @qthreads for ℓ₁ in lmin:lmax
        tid = Threads.threadid()
        buffer0 = thread_buffers_0[tid]
        buffer2 = thread_buffers_2[tid]
        for ℓ₂ in ℓ₁:lmax
            w3j₀₀ = fill_3j!(buffer0, ℓ₁, ℓ₂, 0, 0)
            w3j₂₂ = fill_3j!(buffer2, ℓ₁, ℓ₂, -2, 2)
            w3j₀₀₂₂ = w3j₀₀
            w3j₀₀₂₂.symbols .*= w3j₂₂.symbols
            Ξ = Ξ_TE(Vᵢⱼ, w3j₀₀₂₂, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ
        end
    end
    return 𝐌
end


# inner MCM loop for spin 2, called "EE" in Planck notation
function inner_mcm⁺⁺!(𝐌::SpectralArray{T,2}, Vᵢⱼ::SpectralVector{T}) where {T}
    @assert axes(𝐌, 1) == axes(𝐌, 2)
    lmin, lmax = first(axes(𝐌, 1)), last(axes(𝐌, 1))
    thread_buffers = get_thread_buffers(T, 2lmax+1)

    @qthreads for ℓ₁ in lmin:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w3j²₂₂ = fill_3j!(buffer, ℓ₁, ℓ₂, -2, 2)
            w3j²₂₂.symbols .= w3j²₂₂.symbols .^ 2  # square the symbols
            Ξ = Ξ_EE(Vᵢⱼ, w3j²₂₂, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ
        end
    end
    return 𝐌
end


# inner MCM loop for spin 2
function inner_mcm⁻⁻!(𝐌::SpectralArray{T,2}, Vᵢⱼ::SpectralVector{T}) where {T}
    @assert axes(𝐌, 1) == axes(𝐌, 2)
    lmin, lmax = first(axes(𝐌, 1)), last(axes(𝐌, 1))
    thread_buffers = get_thread_buffers(T, 2lmax+1)

    @qthreads for ℓ₁ in lmin:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w3j²₂₂ = fill_3j!(buffer, ℓ₁, ℓ₂, -2, 2)
            w3j²₂₂.symbols .= w3j²₂₂.symbols .^ 2  # square the symbols
            Ξ = Ξ_EB(Vᵢⱼ, w3j²₂₂, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ
        end
    end
    return 𝐌
end


@doc raw"""
    mcm(spec::Symbol, alm₁::Alm{T}, alm₂::Alm{T}; lmax=nothing)

Compute the mode-coupling matrix. See the [Spectral Analysis](@ref)
section in the documentation for examples. These are used by applying the 
linear solve operator `\` to a `SpectralArray{T,1}`.

Choices for `spec`:

* `:TT`, identical to `M⁰⁰`
* `:TE`, identical to `:ET`, `:TB`, `:BT`, `:M⁰²`, `:M²⁰`
* `:EE_BB`, returns coupling matrix for stacked EE and BB vectors
* `:EB_BE`, returns coupling matrix for stacked EB and BE vectors
* `:M⁺⁺`, sub-block of spin-2 mode-coupling matrices
* `:M⁻⁻`, sub-block of spin-2 mode-coupling matrices

# Arguments:
- `spec::Symbol`: cross-spectrum of the mode-coupling matrix
- `alm₁::Alm{T}`: first mask's spherical harmonic coefficients
- `alm₂::Alm{T}`: second mask's spherical harmonic coefficients

# Keywords
- `lmin=0`: minimum multiple for mode-coupling matrix
- `lmax=nothing`: maximum multipole for mode-coupling matrix

# Returns:
- the mode coupling matrix. for single symbols, this returns a 
    `SpectralArray{T,2}`. if spec is `:EE_BB` or `:EB_BE`, returns a 
    `BlockSpectralMatrix{T}` with 2×2 blocks.
"""
function mcm(spec::Symbol, alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}};
             lmin=0, lmax=nothing) where T
    if isnothing(lmax)  # use alm lmax if an lmax is not specified
        lmax = min(alm₁.lmax, alm₂.lmax)
    end
    Vᵢⱼ = SpectralVector(alm2cl(alm₁, alm₂)[1:(lmax+1)])  # zero-indexed
    if spec ∈ (:TT, :M⁰⁰)
        𝐌 = spectralzeros(lmin:lmax, lmin:lmax)
        return inner_mcm⁰⁰!(𝐌, Vᵢⱼ)
    elseif spec ∈ (:TE, :ET, :TB, :BT, :M⁰², :M²⁰)
        𝐌 = spectralzeros(lmin:lmax, lmin:lmax)
        return inner_mcm⁰²!(𝐌, Vᵢⱼ)
    elseif spec == :M⁺⁺
        𝐌 = spectralzeros(lmin:lmax, lmin:lmax)
        return inner_mcm⁺⁺!(𝐌, Vᵢⱼ)
    elseif spec == :M⁻⁻
        𝐌 = spectralzeros(lmin:lmax, lmin:lmax)
        return inner_mcm⁻⁻!(𝐌, Vᵢⱼ)
    elseif spec == :EE_BB
        𝐌⁺⁺ = spectralzeros(lmin:lmax, lmin:lmax)
        𝐌⁻⁻ = spectralzeros(lmin:lmax, lmin:lmax)
        inner_mcm⁺⁺!(𝐌⁺⁺, Vᵢⱼ)
        inner_mcm⁻⁻!(𝐌⁻⁻, Vᵢⱼ)
        return [ 𝐌⁺⁺  𝐌⁻⁻;
                 𝐌⁻⁻  𝐌⁺⁺ ]
    elseif spec == :EB_BE
        𝐌⁺⁺ = spectralzeros(lmin:lmax, lmin:lmax)
        𝐌⁻⁻ = spectralzeros(lmin:lmax, lmin:lmax)
        inner_mcm⁺⁺!(𝐌⁺⁺, Vᵢⱼ)
        inner_mcm⁻⁻!(𝐌⁻⁻, Vᵢⱼ)
        return [ 𝐌⁺⁺   (-𝐌⁻⁻);
                (-𝐌⁻⁻)   𝐌⁺⁺ ]
    end
    throw(ArgumentError("$(spec) not a valid spectrum."))
end


function mcm(spec::Tuple{Symbol,Symbol}, alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}};
             lmin=0, lmax=nothing) where T
    if isnothing(lmax)  # use alm lmax if an lmax is not specified
        lmax = min(alm₁.lmax, alm₂.lmax)
    end
    Vᵢⱼ = SpectralVector(alm2cl(alm₁, alm₂)[1:(lmax+1)])  # zero-indexed
    if spec == (:EE_BB, :EB_BE)
        𝐌⁺⁺ = spectralzeros(lmin:lmax, lmin:lmax)
        𝐌⁻⁻ = spectralzeros(lmin:lmax, lmin:lmax)
        inner_mcm⁺⁺!(𝐌⁺⁺, Vᵢⱼ)
        inner_mcm⁻⁻!(𝐌⁻⁻, Vᵢⱼ)
        EE_BB = [ 𝐌⁺⁺  𝐌⁻⁻;
                  𝐌⁻⁻  𝐌⁺⁺ ]  
        EB_BE = [ 𝐌⁺⁺   (-𝐌⁻⁻);
                 (-𝐌⁻⁻)   𝐌⁺⁺ ]
        return EE_BB, EB_BE
    end
end


# convenience function
mcm(spec::Symbol, m₁::HealpixMap, m₂::HealpixMap; lmin=0, lmax=nothing) =
    mcm(spec, map2alm(m₁), map2alm(m₂); lmin=lmin, lmax=lmax)
function mcm(spec::Symbol, f₁::CovField, f₂::CovField; lmin=0, lmax=nothing)
    mask1 = (string(spec)[1] == 'T') ? f₁.maskT : f₁.maskP
    mask2 = (string(spec)[2] == 'T') ? f₂.maskT : f₂.maskP
    mcm(spec, map2alm(mask1), map2alm(mask2); lmin=lmin, lmax=lmax)
end


"""Scale a map."""
function scale!(m::HealpixMap, s::Number)
    m .*= s
end
function scale!(m::PolarizedHealpixMap, sT::Number, sP::Number)
    m.i .*= sT
    m.q .*= sP
    m.u .*= sP
end
scale!(m::PolarizedHealpixMap, s::Number) = scale!(m, s, s)



"""
    mask!(m::HealpixMap, mask)
    mask!(m::PolarizedHealpixMap, maskT, maskP)

Mask a map or polarized map in place.

# Arguments:
- `m::Union{HealpixMap,PolarizedHealpixMap}`: map or polarized map to mask
- `maskT::HealpixMap`: mask for first map's intensity
- `maskP::HealpixMap`: mask for first map's polarization
"""
function mask!(m::HealpixMap, mask)
    m .*= mask
    return m
end
function mask!(m::PolarizedHealpixMap, maskT, maskP)
    m.i .*= maskT
    m.q .*= maskP
    m.u .*= maskP
    return m
end
mask!(m::PolarizedHealpixMap, mask) = mask!(m, mask, mask)

"""
    master(map₁::PolarizedHealpixMap, maskT₁::HealpixMap, maskP₁::HealpixMap,
           map₂::PolarizedHealpixMap, maskT₂::HealpixMap, maskP₂::HealpixMap; already_masked=false)

Perform a mode-decoupling calculation for two polarized maps, along with masks to apply.
Returns spectra for ``TT``, ``TE``, ``ET``, ``EE``, ``EB``, ``BE``, and ``BB``.

# Arguments:
- `map₁::PolarizedHealpixMap`: the first IQU map
- `maskT₁::HealpixMap`: mask for first map's intensity
- `maskP₁::HealpixMap`: mask for first map's polarization
- `map₂::PolarizedHealpixMap`: the second IQU map
- `maskT₂::HealpixMap`: mask for second map's intensity
- `maskP₂::HealpixMap`: mask for second map's polarization

# Keywords
- `already_masked::Bool=false`: are the input maps already multiplied with the masks?
- `lmin::Int=0`: minimum multipole

# Returns: 
- `Dict{Symbol,SpectralVector}`: spectra `Dict`, indexed with `:TT`, `:TE`, `:ET`, etc.
"""
function master(map₁::PolarizedHealpixMap, maskT₁::HealpixMap, maskP₁::HealpixMap,
                map₂::PolarizedHealpixMap, maskT₂::HealpixMap, maskP₂::HealpixMap; 
                already_masked::Bool=false, lmin::Int=0)
    if already_masked
        maskedmap₁, maskedmap₂ = map₁, map₂
    else
        maskedmap₁ = deepcopy(map₁)
        maskedmap₂ = deepcopy(map₂)
        mask!(maskedmap₁, maskT₁, maskP₁)
        mask!(maskedmap₂, maskT₂, maskP₂)
    end
    return maskedalm2spectra(map2alm(maskedmap₁), map2alm(maskT₁), map2alm(maskP₁),
                             map2alm(maskedmap₂), map2alm(maskT₂), map2alm(maskP₂); 
                             lmin=lmin)
end

"""Construct a NamedTuple with T,E,B names for the alms."""
function name_alms(alms::Vector)
    return (T=alms[1], E=alms[2], B=alms[3])
end

"""Compute spectra from alms of masked maps and alms of the masks themselves."""
function maskedalm2spectra(maskedmap₁vec::Vector{A}, maskT₁::A, maskP₁::A,
                           maskedmap₂vec::Vector{A}, maskT₂::A, maskP₂::A;
                           lmin=0) where {CT, A <: Alm{CT}}
    ## add TEB names
    maskedmap₁ = name_alms(maskedmap₁vec)
    maskedmap₂ = name_alms(maskedmap₂vec)
    spectra = Dict{Symbol, SpectralVector}()

    ## spectra that are independent
    for (X, Y) in ((:T,:T), (:T,:E), (:E,:T), (:T,:B), (:B,:T))
        spec = Symbol(X, Y)  # join X and Y 

        ## select temp or pol mask
        maskX = (X == :T) ? maskT₁ : maskP₁
        maskY = (Y == :T) ? maskT₂ : maskP₂

        ## compute mcm
        M = mcm(spec, maskX, maskY; lmin=lmin)
        pCl = SpectralVector(alm2cl(maskedmap₁[X], maskedmap₂[Y]))[IdentityRange(lmin:end)]
        Cl = M \ pCl
        spectra[spec] = Cl  # store the result
    end

    M_EE_BB, M_EB_BE = mcm((:EE_BB, :EB_BE), maskP₁, maskP₂; lmin=lmin)
    ## EE and BB have to be decoupled together
    pCl_EE = SpectralVector(alm2cl(maskedmap₁[:E], maskedmap₂[:E]))[IdentityRange(lmin:end)]
    pCl_BB = SpectralVector(alm2cl(maskedmap₁[:B], maskedmap₂[:B]))[IdentityRange(lmin:end)]
    ## apply the 2×2 block mode-coupling matrix to the stacked EE and BB spectra
    @spectra Cl_EE, Cl_BB = M_EE_BB \ [pCl_EE; pCl_BB]
    spectra[:EE] = Cl_EE
    spectra[:BB] = Cl_BB

    ## EB and BE have to be decoupled together
    pCl_EB = SpectralVector(alm2cl(maskedmap₁[:E], maskedmap₂[:B]))[IdentityRange(lmin:end)]
    pCl_BE = SpectralVector(alm2cl(maskedmap₁[:B], maskedmap₂[:E]))[IdentityRange(lmin:end)]
    ## apply the 2×2 block mode-coupling matrix to the stacked EB and BE spectra
    @spectra Cl_EB, Cl_BE = M_EB_BE \ [pCl_EB; pCl_BE]
    spectra[:EB] = Cl_EB
    spectra[:BE] = Cl_BE

    return spectra
end

