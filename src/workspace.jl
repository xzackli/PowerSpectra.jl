
abstract type AbstractField{T} end

struct Field{T} <: AbstractField{T}
    name::String
    maskT::Map{T}
    σ²II::Map{T}
    beam::SpectralVector{T}
end

struct PolarizedField{T} <: AbstractField{T}
    name::String
    maskT::Map{T}
    maskP::Map{T}
    σ²II::Map{T}
    σ²QQ::Map{T}
    σ²UU::Map{T}
    beamT::SpectralVector{T}
    beamP::SpectralVector{T}
end

function Base.show(io::IO, ::MIME"text/plain", x::Field{T}) where T
    println("Field " * x.name, ": ", typeof(x.maskT), " ", size(x.maskT.nside))
    println("maskT [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
    println("σ²II   [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
end


@enum MapType ∅∅ II QQ UU TT PP TP PT TE ET EE

# index for the mask spectra V
const VIndex = Tuple{MapType, String, String}

# Index for the covariance's weighted mask spectra W, indexed X, Y, i, j, α, p, q, β
const WIndex = Tuple{MapType, MapType, String, String, MapType, String, String, MapType}

struct SpectralWorkspace{T <: Real}
    field_names::NTuple{2, String}
    lmax::Int

    mask_alm::Dict{Tuple{String, MapType}, Alm{Complex{T}}}  # T and P alms for i and j
end


function SpectralWorkspace(m_i::PolarizedField{T}, m_j::PolarizedField{T}; lmax::Int=0) where {T}
    field_names = (m_i.name, m_j.name)
    (m_i.maskT.resolution.nside != m_i.maskP.resolution.nside) && throw(
        ArgumentError("m_i temperature and polarization nside do not match."))
    (m_j.maskT.resolution.nside != m_j.maskP.resolution.nside) && throw(
        ArgumentError("m_j temperature and polarization nside do not match."))
    lmax = iszero(lmax) ? 3 * m_i.maskT.resolution.nside - 1 : lmax

    masks = Dict{Tuple{String, MapType}, Alm{Complex{T}}}(
        (m_i.name, TT) => map2alm(m_i.maskT), 
        (m_j.name, TT) => map2alm(m_j.maskT), 
        (m_i.name, PP) => map2alm(m_i.maskP), 
        (m_j.name, PP) => map2alm(m_j.maskP))

    return SpectralWorkspace{T}(field_names, lmax, masks)
end


struct CovarianceWorkspace{T <: Real}
    field_names::NTuple{4, String}
    lmax::Int

    # for mode coupling matrices
    mask_alm::Dict{Tuple{String, MapType}, Alm{Complex{T}}}

    # for covariances
    effective_weights::ThreadSafeDict{Tuple{MapType, String, String, MapType}, Alm{Complex{T}}}
    W_spectra::ThreadSafeDict{WIndex, SpectralVector{T}}
end

# function SpectralWorkspace(m_i::Field{T}, m_j::Field{T}, m_p::Field{T}, m_q::Field{T};
#                            lmax::Int=0) where {T}
#     field_names = (m_i.name, m_j.name, m_p.name, m_q.name)
#     lmax = iszero(lmax) ? 3 * m_i.maskT.resolution.nside - 1 : lmax

#     zero_alm = Alm(lmax, lmax, Zeros{Complex{T}}(numberOfAlms(lmax, lmax)))
#     zero_cl = SpectralVector(Zeros{T}(lmax+1))

#     masks = Dict{Tuple{String, MapType}, Alm{Complex{T}}}(
#         (m_i.name, TT) => map2alm(m_i.maskT), 
#         (m_j.name, TT) => map2alm(m_j.maskT), 
#         (m_p.name, TT) => map2alm(m_p.maskT), 
#         (m_q.name, TT) => map2alm(m_q.maskT)
#     )

#     return SpectralWorkspace{T}(
#         field_names,
#         lmax,
#         masks,
#         ThreadSafeDict{Tuple{MapType, String, String, MapType}, Alm{Complex{T}}}(),
#         ThreadSafeDict{WIndex, SpectralVector{T}}()
#     )
# end



# function PolarizedSpectralWorkspace(m_i::PolarizedField{T}, m_j::PolarizedField{T}, 
#                                     m_p::PolarizedField{T}, m_q::PolarizedField{T};
#                                     lmax::Int=0) where {T}
#     field_names = (m_i.name, m_j.name, m_p.name, m_q.name)
#     lmax = iszero(lmax) ? 3 * m_i.maskT.resolution.nside - 1 : lmax

#     zero_alm = Alm(lmax, lmax, Zeros{Complex{T}}(numberOfAlms(lmax, lmax)))
#     zero_cl = SpectralVector(Zeros{T}(lmax+1))

#     masks = Dict{Tuple{String, MapType}, Alm{Complex{T}}}(
#         (m_i.name, TT) => map2alm(m_i.maskT), 
#         (m_j.name, TT) => map2alm(m_j.maskT), 
#         (m_p.name, TT) => map2alm(m_p.maskT), 
#         (m_q.name, TT) => map2alm(m_q.maskT),
#         (m_i.name, PP) => map2alm(m_i.maskP), 
#         (m_j.name, PP) => map2alm(m_j.maskP), 
#         (m_p.name, PP) => map2alm(m_p.maskP), 
#         (m_q.name, PP) => map2alm(m_q.maskP)
#     )

#     return SpectralWorkspace{T}(
#         field_names,
#         lmax,
#         masks,
#         ThreadSafeDict{Tuple{MapType, String, String, MapType}, Alm{Complex{T}}}(),
#         ThreadSafeDict{WIndex, SpectralVector{T}}()
#     )
# end


"""
Allocate Vector{T} of a given size for each thread.
"""
function get_thread_buffers(::Type{T}, size) where {T}
    thread_buffers = Vector{Vector{T}}(undef, Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        thread_buffers[i] = Vector{T}(undef, size)
    end
    return thread_buffers
end