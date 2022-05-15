
abstract type AbstractField{T} end

# struct CovField{T, AAT, AAP, AAσ, AA_BEAM} <: AbstractField{T}
#     name::String
#     maskT::HealpixMap{T, RingOrder, AAT}
#     maskP::HealpixMap{T, RingOrder, AAP}
#     σ²::PolarizedHealpixMap{T, RingOrder, AAσ}
#     beamT::SpectralVector{T, AA_BEAM}
#     beamP::SpectralVector{T, AA_BEAM}
# end


struct CovField{T, MTT, MTP, AAσ, AA_BEAM} <: AbstractField{T}
    name::String
    maskT::MTT
    maskP::MTP
    σ²::AAσ
    beamT::SpectralVector{T, AA_BEAM}
    beamP::SpectralVector{T, AA_BEAM}
end


"""
    CovField(name, maskT, maskP,
        σ²II::HealpixMap{T, O, AA}, σ²QQ::HealpixMap{T, O, AA}, σ²UU::HealpixMap{T, O, AA},
        beamT::SpectralVector{T}, beamP::SpectralVector{T})

Create a structure for describing the information needed for a covariance
involving this field.

# Arguments:
- `name::String`: name of this field
- `maskT::HealpixMap{T}`: temperature mask
- `maskP::HealpixMap{T}`: polarization mask
- `σ²::PolarizedHealpixMap{T}`: pixel variances
- `beamT::SpectralVector{T}`: temperature beam
- `beamP::SpectralVector{T}`: polarization beam
"""
function CovField end


function CovField(name::String, maskT::HealpixMap{T}, maskP::HealpixMap{T}, σ²::PolarizedHealpixMap) where T
    nside = maskT.resolution.nside
    return CovField(name, maskT, maskP, σ², 
        SpectralVector(ones(3nside)), SpectralVector(ones(3nside)))
end

function CovField(name::String, maskT::HealpixMap{T, O, AA}, maskP::HealpixMap{T, O, AA},
        σ²II::HealpixMap{T, O, AA}, σ²QQ::HealpixMap{T, O, AA}, σ²UU::HealpixMap{T, O, AA}) where {T, O, AA}
    σ² = PolarizedHealpixMap{T, O, AA}(σ²II, σ²QQ, σ²UU)
    return CovField(name, maskT, maskP, σ²)
end

function CovField(name::String, maskT::HealpixMap{T, O, AA}, maskP::HealpixMap{T, O, AA}) where {T, O, AA}
    nside = maskT.resolution.nside
    one_beam = SpectralVector(ones(3nside))
    zero_map = HealpixMap{T, O, AA}(zeros(nside2npix(nside)))
    σ² = PolarizedHealpixMap{T, O, AA}(zero_map, zero_map, zero_map)
    return CovField(name, maskT, maskP, σ², one_beam, one_beam)
end


# converts i.e. :TP => (:TT, :PP)
function split_maptype(XY::Symbol)
    a, b = String(XY)
    return Symbol(a,a), Symbol(b,b)
end

# index for the mask spectra V
const SpectrumName = Tuple{Symbol, String, String}

# Index for the covariance's weighted mask spectra W, indexed X, Y, i, j, α, p, q, β
const WIndex = Tuple{Symbol, Symbol, String, String, Symbol, String, String, Symbol}


struct CovarianceWorkspace{T <: Real, MT, WT, EWT, WST}
    field_names::NTuple{4, String}
    lmax::Int
    mask_p::MT                  #::Dict{Tuple{String, Symbol}, HealpixMap{T,RingOrder}}
    weight_p::WT                 #::Dict{Tuple{String, Symbol}, HealpixMap{T,RingOrder}}
    effective_weights::EWT       #::Dict{Tuple{Symbol, String, String, Symbol}, Alm{Complex{T}}}
    W_spectra::WST               #::Dict{WIndex, SpectralVector{T}}
end

function CovarianceWorkspace(T::Type, field_names::NTuple{4, String}, lmax::Int, mask_p::MT, 
        weight_p::WT, effective_weights::EWT, W_spectra::WST) where {MT, WT, EWT, WST}
    CovarianceWorkspace{T, MT, WT, EWT, WST}(field_names, lmax, mask_p, weight_p, 
        effective_weights, W_spectra)
end


"""
    CovarianceWorkspace(m_i, m_j, m_p, m_q; lmax::Int=0)

Inputs and cache for covariance calculations. A covariance matrix relates the masks of
four fields and spins. This structure caches various cross-spectra between masks and
noise-weighted masks. By default, operates at Float64 precision.

# Arguments:
- `m_i::CovField{T}`: map i
- `m_j::CovField{T}`: map j
- `m_p::CovField{T}`: map p
- `m_q::CovField{T}`: map q

# Keywords
- `lmax::Int=0`: maximum multipole to compute covariance matrix
"""
function CovarianceWorkspace(m_i::CovField{TC,MTT}, m_j::CovField{TC,MTT},
                             m_p::CovField{TC,MTT}, m_q::CovField{TC,MTT}; lmax::Int=0) where {TC,MTT}
    field_names = (m_i.name, m_j.name, m_p.name, m_q.name)  # for easy access
    lmax = iszero(lmax) ? nside2lmax(m_i.maskT.resolution.nside) : lmax  # set an lmax if not specified
    mask_p = Dict{Tuple{String, Symbol},MTT}(
        (m_i.name, :TT) => m_i.maskT, (m_j.name, :TT) => m_j.maskT,
        (m_p.name, :TT) => m_p.maskT, (m_q.name, :TT) => m_q.maskT,
        (m_i.name, :PP) => m_i.maskP, (m_j.name, :PP) => m_j.maskP,
        (m_p.name, :PP) => m_p.maskP, (m_q.name, :PP) => m_q.maskP)
    weight_p = Dict{Tuple{String, Symbol},MTT}(
        (m_i.name, :II) => m_i.σ².i, (m_i.name, :QQ) => m_i.σ².q, (m_i.name, :UU) => m_i.σ².u,
        (m_j.name, :II) => m_j.σ².i, (m_j.name, :QQ) => m_j.σ².q, (m_j.name, :UU) => m_j.σ².u,
        (m_p.name, :II) => m_p.σ².i, (m_p.name, :QQ) => m_p.σ².q, (m_p.name, :UU) => m_p.σ².u,
        (m_q.name, :II) => m_q.σ².i, (m_q.name, :QQ) => m_q.σ².q, (m_q.name, :UU) => m_q.σ².u)

    return CovarianceWorkspace(
        Float64,
        field_names,
        lmax,
        mask_p,
        weight_p,
        Dict{Tuple{Symbol, String, String, Symbol}, Alm{ComplexF64, Vector{ComplexF64}}}(),
        Dict{WIndex, SpectralVector{Float64, Vector{Float64}}}())
end


pixsize(m::HealpixMap) = 4π / m.resolution.numOfPixels
pixsize(m::Enmap) = pixareamap(m)


function effective_weight_alm!(workspace::CovarianceWorkspace{T}, A, i, j, α) where T
    if (A, i, j, α) in keys(workspace.effective_weights)
        return workspace.effective_weights[(A, i, j, α)]
    end

    X, Y = split_maptype(α)

    m_iX = workspace.mask_p[i, X]
    m_jY = workspace.mask_p[j, Y]
    map_buffer = similar(m_iX)

    if A == :∅∅
        parent(map_buffer) .= parent(m_iX) .* parent(m_jY)  # healpix doesn't broadcast properly
        w_result = map2alm(map_buffer)
        workspace.effective_weights[A, i, j, α] = w_result
        return w_result
    elseif (A in (:II, :QQ, :UU))
        if i == j
            parent(map_buffer) .= parent(m_iX) .* parent(m_jY)
            Ω_p = pixsize(map_buffer)
            parent(map_buffer) .*= parent(workspace.weight_p[i, A]) .* Ω_p
            w_result = map2alm(map_buffer)
            workspace.effective_weights[A, i, j, α] = w_result
            return w_result
        end
    end

    # otherwise return zero
    lmax = workspace.lmax
    return Alm(lmax, lmax, zeros(ComplexF64, numberOfAlms(lmax, lmax)))
end


function window_function_W!(workspace::CovarianceWorkspace{T}, X, Y, i, j, α, p, q, β) where T
    # check if it's already computed
    if (X, Y, i, j, α, p, q, β) in keys(workspace.W_spectra)
        return workspace.W_spectra[(X, Y, i, j, α, p, q, β)]
    end

    # TT turns into II
    if X == :TT
        wterms_X = (:II,)
    elseif X == :PP
        wterms_X = (:QQ, :UU)
    else
        wterms_X = (X,)
    end

    if Y == :TT
        wterms_Y = (:II,)
    elseif Y == :PP
        wterms_Y = (:QQ, :UU)
    else
        wterms_Y = (Y,)
    end

    result = zeros(T, workspace.lmax+1)

    # Planck 2015 eq. C.11 - C.16
    for wX in wterms_X
        for wY in wterms_Y
            result .+= alm2cl(
                effective_weight_alm!(workspace, wX, i, j, α),
                effective_weight_alm!(workspace, wY, p, q, β))[begin:workspace.lmax+1]
        end
    end
    norm = one(T) / (length(wterms_X) * length(wterms_Y))
    result .*= norm
    result = SpectralVector(result)

    workspace.W_spectra[X, Y, i, j, α, p, q, β] = result
    return result
end


# Allocate Vector{T} of a given size for each thread.
function get_thread_buffers(::Type{T}, size) where {T}
    thread_buffers = Vector{Vector{T}}(undef, Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        thread_buffers[i] = Vector{T}(undef, size)
    end
    return thread_buffers
end
