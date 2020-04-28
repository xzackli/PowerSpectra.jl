module PSPlanck

import PyCall
import Healpix: Map, Alm, RingOrder, alm2cl, map2alm, numberOfAlms
import Combinatorics: permutations, combinations, with_replacement_combinations
import DataStructures: DefaultDict
import ThreadSafeDicts: ThreadSafeDict

using WignerSymbols
WignerSymbols.MAX_J[] = 1000


export mollview
export cov, Field

include("field.jl")
include("squarewigner.jl")


function __init__()
    global wigner_caches
    resize!(empty!(wigner_caches), Threads.nthreads())
    empty!(Wigner3j)
    empty!(Wigner6j)
end


# function wigner3j²(T::Type{<:Real}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁)
#     w3j = WignerSymbols.wigner3j(j₁, j₂, j₃, m₁, m₂, m₃)  # get a RationalRoot
#     return convert(T, abs(w3j.signedsquare))
# end
# wigner3j²(j₁::Integer, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁) = wigner3j²(
#     Rational{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃)


@enum MapType ∅∅ II QQ UU TT PP TP

function w_coefficients(m_i::Field{T}, m_j::Field{T}, 
                        m_p::Field{T}, m_q::Field{T}) where {T <: Real}
    # generate coefficients w
    fields = Field{T}[m_i, m_j, m_p, m_q]
    names = [m_i.name, m_j.name, m_p.name, m_q.name]

    map_buffer = Map{T, RingOrder}(zeros(T, size(m_i.maskT.pixels)))  # reuse pixel buffer

    # XX, i, j, YY
    w = Dict{Tuple{MapType, String, String, MapType}, Healpix.Alm{Complex{T}}}()

    for (i, name_i) in enumerate(names)
        for (j, name_j) in enumerate(names)
            map_buffer.pixels .= fields[i].maskT.pixels
            map_buffer.pixels .*= fields[j].maskT.pixels
            w[∅∅, name_i, name_j, TT] = map2alm(map_buffer)  # allocate new alms

            if i == j  # δᵢⱼ here, so we don't create arrays for this
                map_buffer.pixels .*= (fields[j].maskT.pixels.^2)
                w[II, name_i, name_i, TT] = map2alm(map_buffer)
            end
        end
    end
    return w
end

# X, Y, i, j, p, q, α, β
const WIndex = Tuple{MapType, MapType, String, String, String, String, MapType, MapType}

function W_spectra(T::Type{<:Real}, names, w)
    # generate a list of jobs
    weight_indices = WIndex[]
    for (i, j, p, q) in permutations(names)
        push!(weight_indices, (∅∅, ∅∅, i, j, p, q, TT, TT))
        i == j && push!(weight_indices, (II, ∅∅, i, j, p, q, TT, TT))
        p == q && push!(weight_indices, (∅∅, II, i, j, p, q, TT, TT))
        i == j && p == q && push!(weight_indices, (II, II, i, j, p, q, TT, TT))
    end

    # use a thread safe dict to put it together
    W = ThreadSafeDict{WIndex, Array{T}}()
    Threads.@threads for (X, Y, i, j, p, q) in weight_indices
        W[X, Y, i, j, p, q, TT, TT] = alm2cl(w[X, i, j, TT], w[Y, p, q, TT])
    end

    # copy results over to a default dict
    Wdefault = DefaultDict{WIndex, Array{T}}(zero(T))
    for k in keys(W)
        Wdefault[k] = W[k]
    end
    return Wdefault
end

"""
Temperature-only covariance matrix.
"""
function cov(m_i::Field{T}, m_j::Field{T}, 
             m_p::Field{T}, m_q::Field{T}) where {T <: Real}

    wcoeff = w_coefficients(m_i, m_j, m_p, m_q)
    names = [m_i.name, m_j.name, m_p.name, m_q.name]
    W = W_spectra(T, names, wcoeff)
    
end
function cov(m_i::Field{T}, m_j::Field{T}) where {T <: Real}
    cov(m_i, m_j, m_i, m_j)
end

function ΞTT(W_arr::Array{T,1}, ℓ₁::Integer, ℓ₂::Integer) where T
    Ξ = zero(T)
    for ℓ₃ in abs(ℓ₁ - ℓ₂):(ℓ₁ + ℓ₂)
        Ξ += (2ℓ₃ + 1)/4π * wigner3j²(T, ℓ₁, ℓ₂, ℓ₃, 0, 0, 0) * W_arr[ℓ₃+1]
    end
    return Ξ
end

end
