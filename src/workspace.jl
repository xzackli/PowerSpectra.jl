
struct Field{T}
    name::String
    maskT::Map{T}
    σTT::Map{T}
    beam::SpectralVector{T}
end

struct PolarizedField{T}
    name::String
    maskT::Map{T}
    maskP::PolarizedMap{T}
    σTT::Map{T}
    σPP::PolarizedMap{T}
end

function Base.show(io::IO, ::MIME"text/plain", x::Field{T}) where T
    println("Field " * x.name, ": ", typeof(x.maskT), " ", size(x.maskT.nside))
    println("maskT [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
    println("σII   [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
end


@enum MapType ∅∅ II QQ UU TT PP TP

# index for the mask spectra V
const VIndex = Tuple{MapType, String, String}

# Index for the covariance's weighted mask spectra W, indexed X, Y, i, j, α, p, q, β
const WIndex = Tuple{MapType, MapType, String, String, MapType, String, String, MapType}

struct SpectralWorkspace{T <: Real}
    field_names::NTuple{4, String}
    lmax::Int

    # for mode coupling matrices
    masks::Dict{String, Alm{Complex{T}}}
    V_spectra::Dict{VIndex, SpectralVector{T}}

    # for covariances
    effective_weights::DefaultDict{Tuple{MapType, String, String, MapType}, Alm{Complex{T}}}
    W_spectra::DefaultDict{WIndex, SpectralVector{T}}
end

function SpectralWorkspace(m_i::Field{T}, m_j::Field{T}, m_p::Field{T}, m_q::Field{T};
                           lmax::Int=0) where {T}
    field_names = (m_i.name, m_j.name, m_p.name, m_q.name)
    lmax = iszero(lmax) ? 3 * m_i.maskT.resolution.nside - 1 : lmax

    zero_alm = Alm(lmax, lmax, Zeros{Complex{T}}(numberOfAlms(lmax, lmax)))
    zero_cl = SpectralVector(Zeros{T}(lmax+1))

    masks = Dict{String, Alm{Complex{T}}}(
        m_i.name => map2alm(m_i.maskT), 
        m_j.name => map2alm(m_j.maskT), 
        m_p.name => map2alm(m_p.maskT), 
        m_q.name => map2alm(m_q.maskT)
    )

    return SpectralWorkspace{T}(
        field_names,
        lmax,
        masks,
        Dict{VIndex, SpectralVector{T}}(),
        DefaultDict{Tuple{MapType, String, String, MapType}, Alm{Complex{T}}}(zero_alm),
        DefaultDict{WIndex, SpectralVector{T}}(zero_cl)
    )
end

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