module PSPlanck

import Healpix
using WignerSymbols
using RationalRoots

export cov


# function cov(m₁::T, m₂::T,
#              ℓ₁::Integer, ℓ₂::Integer) where {T <: Healpix.Alm{Complex{Float64}}}
#     return sqrt()
# end

function W_functions(m₁::T, m₂::T, σ₁::T, σ₂::T) where {T <: Healpix.Map{Float64, Healpix.RingOrder}}
    
    
    w = (
        (:∅∅, 1,
    )
    w_∅∅_TT₁ = map2alm(m₁),
    w_II_TT₁ = map2alm(m₁ .* σ₁),
    w_∅∅_TT₂ = map2alm(m₂),
    w_II_TT₂ = map2alm(m₂ .* σ₂),

    # W functions with key (α, i, j)
    return (
        (:∅∅, :∅∅) = alm2cl(w_∅∅_TT, w_∅∅_TT),
        ∅∅_TT = alm2cl(w_∅∅_TT, w_II_TT),
    )
end

function Ξ_TT(T::Type{<:Real}, X, Y, ℓ₁::Int, ℓ₂::Int)

    Ξ_temp = zero(T)
    ℓmin = abs(ℓ₁ - ℓ₂)
    ℓmax = ℓ₁ + ℓ₂
    for ℓ₃ in ℓmin:ℓmax
        Ξ_temp = Ξ_temp + (
            (2 * ℓ₃ + 1) / (4π) * 
            wigner3j(T, ℓ₁, ℓ₂, ℓ₃, 0, 0, 0)^2 
        )
    end

    return Ξ_temp
end

function cov(m₁::T, m₂::T, σ₁::T, σ₂::T) where {T <: Healpix.Map{Float64, Healpix.RingOrder}}

    W = W_functions(m₁, m₂, σ₁, σ₂)

    # return cov(Healpix.map2alm(mask₁), Healpix.map2alm(mask₂))
end


end