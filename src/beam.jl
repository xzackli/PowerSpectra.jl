
@doc raw"""
    Ξsum(alm₁, alm₂, w3j₁, w3j₂)

Sum over ``\ell`` and ``m`` of two ``a_{\ell m}`` and nontrivial Wigner-3j vectors. This is 
a step in computing the ``\mathbf{\Xi}`` matrix. The ``\rho`` factors are not 
in this summation, as they can be pulled out.

```math
\begin{aligned}
(\Xi \mathrm{sum}) &= \sum_{\ell^{\prime} m^{\prime}} \,  W_{\ell'}^{\nu_1,\nu_2,s_1,s_2,j_1,j_2} \times \begin{pmatrix} \ell & \ell^{\prime} & \ell^{\prime\prime} \\
     -s_1 & s_1+\nu_1  & -\nu_1 \end{pmatrix} \begin{pmatrix}
     \ell & \ell^{\prime} & \ell^{\prime\prime} \\ -s_2 & s_2+\nu_2  & -\nu_2 \end{pmatrix}
\end{aligned}
```
"""
function Ξsum(W::SpectralVector{T},
              w3j₁::WignerSymbolVector{T, Int},
              w3j₂::WignerSymbolVector{T, Int}) where {T<:Number}

    ℓ_start = max(firstindex(w3j₁), firstindex(w3j₂))
    ℓ_end = min(lastindex(w3j₁), lastindex(w3j₂))
    Σ = zero(T)
    @inbounds @simd for ℓ = ℓ_start:ℓ_end  # sum over nontrivial Wigner 3j symbols
        Σ += W[ℓ] * w3j₁[ℓ] * w3j₂[ℓ]  # accumulate in ℓ. we pull 3j out of m sum
    end
    return Σ
end

@doc raw"""
    quickpolW(alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}})

Computes a scaled spectrum of the scanned pattern.

```math
W_{\ell'}^{\nu_1,\nu_2,s_1,s_2,j_1,j_2} = \sum_{m^\prime=-\ell^\prime}^{\ell^\prime}  \left(_{s_1+\nu_1}\tilde{\omega}^{(j_1)}_{\ell^\prime m^\prime}\right)
    \left(_{s_2+\nu_2}\tilde{\omega}^{(j_2)}_{\ell^\prime m^\prime}\right)^*
```
"""
function quickpolW(alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}}) where T<:Number
    mmax = min(alm₁.mmax, alm₂.mmax)
    lmax = min(alm₁.lmax, alm₂.lmax)
    cl = SpectralVector(zeros(T, lmax + 1))
    for ℓ = 0:lmax  # fill in cl for each ℓ
        for m = 1:min(ℓ, mmax)  # loop over m (symmetric, but skip m=0)
            index = almIndex(alm₁, ℓ, m)
            cl[ℓ] += 2 * real(alm₁.alm[index] * conj(alm₂.alm[index]))
        end
        index0 = almIndex(alm₁, ℓ, 0)  # now do m=0
        cl[ℓ] += real(alm₁.alm[index0] * conj(alm₂.alm[index0]))
    end
    return cl
end

# loop over nonzero elements of a banded SpectralArray
function specrowrange(𝚵::SpectralArray, r)
    start = max(2,BandedMatrices.rowstart(𝚵.parent,r+1)-1)
    stop = BandedMatrices.rowstop(𝚵.parent,r+1)-1
    return start:stop
end

@doc raw"""
    quickpolΞ!(𝚵::AA, ν₁, ν₂, s₁, s₂, ω₁, ω₂)

This computes the ``\Xi_{\ell^{\prime \prime},\ell}`` matrix. It assumes ``\rho`` has been
absorbed into the ``\omega`` terms.

 - `ω₁`: effective scan weights with spin s₁ + ν₁
 - `ω₂`: effective scan weights with spin s₂ + ν₂
"""
function quickpolΞ!(𝚵::AA, ν₁, ν₂, s₁, s₂, ω₁::Alm, ω₂::Alm,
                    buf1::Array{Array{T,1},1}, 
                    buf2::Array{Array{T,1},1}) where {T, AA<:SpectralArray{T,2}}
    # make some basic checks
    size(𝚵,1) != size(𝚵,2) && throw(ArgumentError("𝚵 is not square."))
    lmax = lastindex(𝚵,1)  # indexed 0:lmax
    W = quickpolW(ω₁, ω₂)

    @qthreads for ℓ″ = 2:lmax
        tid = Threads.threadid()
        buffer1 = buf1[tid]
        buffer2 = buf2[tid]
        for ℓ = specrowrange(𝚵, ℓ″)
            # wigner families over ℓ′
            wF₁ = WignerF(T, ℓ, ℓ″, -s₁, -ν₁)  # set up the wigner recurrence problem
            wF₂ = WignerF(T, ℓ, ℓ″, -s₂, -ν₂)  # set up the wigner recurrence problem
            bufferview1 = uview(buffer1, 1:length(wF₁.nₘᵢₙ:wF₁.nₘₐₓ))  # preallocated buffer
            bufferview2 = uview(buffer2, 1:length(wF₂.nₘᵢₙ:wF₂.nₘₐₓ))  # preallocated buffer
            w3j₁ = WignerSymbolVector(bufferview1, wF₁.nₘᵢₙ:wF₁.nₘₐₓ)
            w3j₂ = WignerSymbolVector(bufferview2, wF₂.nₘᵢₙ:wF₂.nₘₐₓ)
            wigner3j_f!(wF₁, w3j₁)  # deposit symbols into buffer
            wigner3j_f!(wF₂, w3j₂)  # deposit symbols into buffer
            𝚵[ℓ″, ℓ] = Ξsum(W, w3j₁, w3j₂)
        end
    end

    sgn = (-1)^(s₁ + s₂ + ν₁ + ν₂)
    𝚵 .*= sgn
    return 𝚵
end
function quickpolΞ!(𝚵::AA, ν₁, ν₂, s₁, s₂,
                    ω₁::Alm, ω₂::Alm) where {T, AA<:SpectralArray{T,2}, SV<:SpectralVector}
    lmax = lastindex(𝚵,1)  # indexed 0:lmax
    buf1 = get_thread_buffers(T, 2lmax+1)
    buf2 = get_thread_buffers(T, 2lmax+1)
    quickpolΞ!(𝚵, ν₁, ν₂, s₁, s₂, ω₁, ω₂, buf1, buf2)
    return 𝚵
end

@doc raw"""
    kᵤ([T=Float64], u)

Defined only for u ∈ {-2, 0, 2}.
"""
function kᵤ(T::Type, u)
    if iszero(u)
        return one(T)
    elseif abs(u) == 2
        return T(1//2)
    end

    throw(ArgumentError("Defined only for u ∈ {-2, 0, 2}."))
end
kᵤ(u) = kᵤ(Float64, u)
