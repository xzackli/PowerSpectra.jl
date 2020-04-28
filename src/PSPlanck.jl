module PSPlanck

import Base.Threads: @threads
import PyCall
import Healpix: Map, Alm, RingOrder, alm2cl, map2alm, numberOfAlms
import Combinatorics: permutations, combinations, with_replacement_combinations
import DataStructures: DefaultDict
import ThreadSafeDicts: ThreadSafeDict
import ThreadPools: @qthreads
import Random: shuffle!
using WignerSymbols
WignerSymbols.MAX_J[] = 1000

export mollview
export cov, Field, CovarianceWorkspace

include("field.jl")
include("squarewigner.jl")

function __init__()
    global wigner_caches
    resize!(empty!(wigner_caches), Threads.nthreads())
    empty!(Wigner3j)
    empty!(Wigner6j)
end

@enum MapType ∅∅ II QQ UU TT PP TP

# X, Y, i, j, α, p, q, β
const WIndex = Tuple{MapType, MapType, String, String, MapType, String, String, MapType}

struct CovarianceWorkspace{T <: Real}
    field_names::NTuple{4, String}
    w_coeff::DefaultDict{Tuple{MapType, String, String, MapType}, Healpix.Alm{Complex{T}}}
    W_spectra::DefaultDict{WIndex, Vector{T}}
end

function CovarianceWorkspace(m_i::Field{T}, m_j::Field{T}, 
                             m_p::Field{T}, m_q::Field{T}) where {T}
    field_names = (m_i.name, m_j.name, m_p.name, m_q.name)
    lmax = 3 * m_i.maskT.resolution.nside - 1
    zero_alm = Alm{Complex{T}}(lmax, lmax)
    zero_cl = alm2cl(zero_alm)
    return CovarianceWorkspace{T}(
        field_names, 
        DefaultDict{Tuple{MapType, String, String, MapType}, Healpix.Alm{Complex{T}}}(zero_alm),
        DefaultDict{WIndex, Vector{T}}(zero_cl)
    )
end
"""
Compute the effective weight map copefficients and store them in the workspace.
"""
function w_coefficients!(workspace::CovarianceWorkspace{T},
                         m_i::Field{T}, m_j::Field{T}, 
                         m_p::Field{T}, m_q::Field{T}) where {T <: Real}
    # generate coefficients w
    fields = Field{T}[m_i, m_j, m_p, m_q]
    names = [m_i.name, m_j.name, m_p.name, m_q.name]

    map_buffer = Map{T, RingOrder}(zeros(T, size(m_i.maskT.pixels)))  # reuse pixel buffer

    # XX, i, j, YY
    w = workspace.w_coeff

    for (i, name_i) in enumerate(names)
        for (j, name_j) in enumerate(names)
            if ((∅∅, name_i, name_j, TT) ∉ keys(w))
                map_buffer.pixels .= fields[i].maskT.pixels
                map_buffer.pixels .*= fields[j].maskT.pixels
                w[∅∅, name_i, name_j, TT] = map2alm(map_buffer)  # allocate new alms
            end

            if i == j  # δᵢⱼ here, so we don't create arrays for this
                if ((II, name_i, name_i, TT) ∉ keys(w))
                    map_buffer.pixels .= fields[i].maskT.pixels
                    map_buffer.pixels .*= fields[j].maskT.pixels
                    map_buffer.pixels .*= fields[j].σTT.pixels.^2
                    w[II, name_i, name_i, TT] = map2alm(map_buffer)
                end
            end
        end
    end
end


function W_spectra!(workspace::CovarianceWorkspace{T}) where {T}
    # generate a list of jobs
    weight_indices = WIndex[]
    for (i, j, p, q) in permutations(workspace.field_names)
        push!(weight_indices, (∅∅, ∅∅, i, j, TT, p, q, TT))
        i == j && push!(weight_indices, (II, ∅∅, i, j, TT, p, q, TT))
        p == q && push!(weight_indices, (∅∅, II, i, j, TT, p, q, TT))
        i == j && p == q && push!(weight_indices, (II, II, i, j, TT, p, q, TT))
    end

    # use a thread safe dict to put it together
    W = workspace.W_spectra
    @threads for (X, Y, i, j, α, p, q, β) in weight_indices
        if (X, Y, i, j, α, p, q, β) ∉ keys(W)
            w1 = workspace.w_coeff[X, i, j, TT]
            w2 = workspace.w_coeff[Y, p, q, TT]
            if(typeof(w1) <: Alm && typeof(w2) <: Alm)
                W[X, Y, i, j, α, p, q, β] = alm2cl(w1, w2)
            end
        end
    end

    for k in keys(W)  # copy results over to workspace
        workspace.W_spectra[k] = W[k]
    end
end

# """
# Projector function for temperature.
# """
# function ΞTT(W_arr::Vector{T}, ℓ₁::Integer, ℓ₂::Integer) where T
#     Ξ = zero(T)
#     for ℓ₃ in abs(ℓ₁ - ℓ₂):(ℓ₁ + ℓ₂)
#         Ξ += (2ℓ₃ + 1) * wigner3j²(T, ℓ₁, ℓ₂, ℓ₃, 0, 0, 0) * W_arr[ℓ₃+1]
#     end
#     return Ξ/4π
# end

"""
    cov(workspace::CovarianceWorkspace{T}, m_i::Field{T}, m_j::Field{T}, 
        m_p::Field{T}=m_i, m_q::Field{T}=m_j; band=5) where {T <: Real}

Compute the covariance matrix between Cℓ₁(i,j) and Cℓ₂(p,q) for temperature.

# Arguments
- `m_i::Field{T}`: the array to search
- `m_j::Field{T}`: the value to search for

# Keywords
- `band::Integer`: compute the banded covariance matrix. Set to 0 for just the diagonal.

# Returns
- `Symmetric{Array{T,2}}`: covariance
"""
function cov(workspace::CovarianceWorkspace{T}, 
             m_i::Field{T}, m_j::Field{T}, m_p::Field{T}, m_q::Field{T};
             lmax=nothing, band=nothing) where {T <: Real}

    w_coefficients!(workspace, m_i, m_j, m_p, m_q)
    W_spectra!(workspace)

    lmax = isnothing(lmax) ? (m_i.maskT.resolution.nside - 1) : lmax
    band = isnothing(band) ? lmax : band

    i, j, p, q = workspace.field_names
    W = workspace.W_spectra
    W_arr = (
        W[∅∅, ∅∅, i, p, TT, j, q, TT],
        W[∅∅, ∅∅, i, q, TT, j, p, TT],
        W[∅∅, TT, i, p, TT, j, q, TT],
        W[∅∅, TT, j, q, TT, i, p, TT],
        W[∅∅, TT, i, q, TT, j, p, TT],
        W[∅∅, TT, j, p, TT, i, q, TT],
        W[TT, TT, i, p, TT, j, q, TT],
        W[TT, TT, i, q, TT, j, p, TT]
    )
    C = zeros(T, (lmax, lmax))
    _covTT!(C, lmax, W_arr, band)
    return C
end
cov(m_i::Field{T}, m_j::Field{T}) where {T <: Real} = cov(m_i, m_j, m_i, m_j)

function ΞTT_term(ℓ₃::Integer, W0::Vector{T}, wig::T) where T
    return (2ℓ₃ + 1) * wig * W0[ℓ₃+1]
end

# inner loop 
function _covTT!(C::AbstractArray{T,2}, lmax::Integer, 
                 W_arr::NTuple{8,Vector{T}}, band::Integer) where {T}

    ℓpairs = [(ℓ₁,ℓ₂) for ℓ₁ in 1:lmax for ℓ₂ in ℓ₁:min(ℓ₁+band,lmax)]
    reverse!(ℓpairs)
    
    @qthreads for (ℓ₁,ℓ₂) in ℓpairs
        Ξ = zeros(T, 8)
        @inbounds for ℓ₃ in abs(ℓ₁ - ℓ₂):(ℓ₁ + ℓ₂)
            wig = wigner3j²(T, ℓ₁, ℓ₂, ℓ₃, 0, 0, 0)
            @inbounds for term_index in 1:8
                Ξ[term_index] += ΞTT_term(ℓ₃, W_arr[term_index], wig)
            end
        end

        for term_index in 1:8
            C[ℓ₁, ℓ₂] += Ξ[term_index]/T(4π)  # still need to add Cl
        end
        C[ℓ₂, ℓ₁] = C[ℓ₁, ℓ₂]
    end
    return C
end


end
