module PSPlanck

import PyCall
import Healpix: Map, Alm, RingOrder, alm2cl, map2alm
import Combinatorics: permutations, combinations, with_replacement_combinations
import DataStructures: DefaultDict
import WignerSymbols
WignerSymbols.MAX_J[] = 1000

export mollview
export cov, Field

include("field.jl")

"""
    wigner3j²(T::Type{<:Real}=Rational{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁)

Square of the Wigner-3j symbol.
"""
function wigner3j²(T::Type{<:Real}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁)
    w3j = WignerSymbols.wigner3j(j₁, j₂, j₃, m₁, m₂, m₃)  # get a RationalRoot
    return convert(T, abs(w3j.signedsquare))
end
wigner3j²(j₁::Integer, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁) = wigner3j²(
    Rational{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃)


@enum MapType ∅∅ II QQ UU TT PP

function w_coefficients(m_i::Field{T}, m_j::Field{T}, 
                        m_p::Field{T}, m_q::Field{T}) where {T <: Real}
    # generate coefficients w
    fields = Field{T}[m_i, m_j, m_p, m_q]
    names = [m_i.name, m_j.name, m_p.name, m_q.name]

    nside = map_buffer.resolution.nside
    lmax = 3 * nside - 1
    zero_alm = Alm{Complex{T}}(zeros(numberOfAlms(lmax)))
    w = DefaultDict{Tuple{MapType, String, String}, Healpix.Alm{Complex{T}}}(zero_alm)
    
    map_buffer = Map{T, RingOrder}(zeros(T, size(m_i.maskT.pixels)))  # reuse pixel buffer

    for (i, name_i) in enumerate(names)
        for (j, name_j) in enumerate(names)
            map_buffer.pixels .= fields[i].maskT.pixels
            map_buffer.pixels .*= fields[j].maskT.pixels
            w[∅∅, name_i, name_j] = map2alm(map_buffer)  # allocate new alms

            if i == j  # δᵢⱼ here, so we just don't create arrays for this
                map_buffer.pixels .*= (fields[j].maskT.pixels.^2)
                w[II, name_i, name_i] = map2alm(map_buffer)
            end

        end
    end
    return w
end

"""
Temperature-only covariance matrix.
"""
function cov(m_i::Field{T}, m_j::Field{T}, 
             m_p::Field{T}, m_q::Field{T}) where {T <: Real}

    W = Dict{Tuple{MapType, MapType, String, String, String, String}, Array{T,1}}()
    w = w_coefficients(m_i, m_j, m_p, m_q)

    names = [m_i.name, m_j.name, m_p.name, m_q.name]
    for (i,j,p,q) in permutations(names)
        for (XX, YY) in with_replacement_combinations([∅∅, II], 2)
            W[XX, YY, i, j, p, q] = alm2cl(w[XX, i, j], w[YY, p, q])
        end
    end

    
    
end

"""
Temperature-only variance.
"""
function cov(m_i::Field{T}, m_j::Field{T}) where {T <: Real}
    cov(m_i, m_j, m_i, m_j)
end


# function get_W(i::SkyField{T}, j::SkyField{T}, 
#                p::SkyField{T}, q::SkyField{T}) where {T}
#     W = Dict{Tuple{WeightType,WeightType,Int,Int,Int,Int},SkyField{T}}
#     W[∅∅, i, j, p, q] = Healpix.alm2cl()
# end

# function cov(m₁::T, m₂::T,
#              ℓ₁::Integer, ℓ₂::Integer) where {T <: Healpix.Alm{Complex{Float64}}}
#     return sqrt()
# end

# function W_functions(m₁::T, m₂::T, σ₁::T, σ₂::T) where {T <: Healpix.Map{Float64, Healpix.RingOrder}}
    
    
#     w = (
#         (:∅∅, 1),
#     )
#     w_∅∅_TT₁ = map2alm(m₁),
#     w_II_TT₁ = map2alm(m₁ .* σ₁),
#     w_∅∅_TT₂ = map2alm(m₂),
#     w_II_TT₂ = map2alm(m₂ .* σ₂),

#     # W functions with key (α, i, j)
#     return (
#         (:∅∅, :∅∅) = alm2cl(w_∅∅_TT, w_∅∅_TT),
#         ∅∅_TT = alm2cl(w_∅∅_TT, w_II_TT),
#     )
# end


function cov(m₁::T, m₂::T, σ₁::T, σ₂::T) where {T <: Healpix.Map{Float64, Healpix.RingOrder}}

    W = W_functions(m₁, m₂, σ₁, σ₂)

    # return cov(Healpix.map2alm(mask₁), Healpix.map2alm(mask₂))
end


end