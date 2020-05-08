
"""
Compute the effective weight map copefficients and store them in the workspace.
"""
function w_coefficients!(workspace::SpectralWorkspace{T},
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


function W_spectra!(workspace::SpectralWorkspace{T}) where {T}
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
                W[X, Y, i, j, α, p, q, β] = SpectralVector(alm2cl(w1, w2))
            end
        end
    end

    for k in keys(W)  # copy results over to workspace
        workspace.W_spectra[k] = W[k]
    end
end

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
function cov(workspace::SpectralWorkspace{T}, 
             m_i::Field{T}, m_j::Field{T}, m_p::Field{T}, m_q::Field{T};
             lmax=nothing, band=nothing) where {T <: Real}

    w_coefficients!(workspace, m_i, m_j, m_p, m_q)
    W_spectra!(workspace)

    lmax = isnothing(lmax) ? (m_i.maskT.resolution.nside - 1) : lmax
    band = isnothing(band) ? lmax : band

    i, j, p, q = workspace.field_names
    W = workspace.W_spectra

    C = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    loop_covTT!(C, lmax, band,
        W[∅∅, ∅∅, i, p, TT, j, q, TT],
        W[∅∅, ∅∅, i, q, TT, j, p, TT],
        W[∅∅, TT, i, p, TT, j, q, TT],
        W[∅∅, TT, j, q, TT, i, p, TT],
        W[∅∅, TT, i, q, TT, j, p, TT],
        W[∅∅, TT, j, p, TT, i, q, TT],
        W[TT, TT, i, p, TT, j, q, TT],
        W[TT, TT, i, q, TT, j, p, TT])
    return C
end
cov(m_i::Field{T}, m_j::Field{T}) where {T <: Real} = cov(m_i, m_j, m_i, m_j)


"""
Projector function for temperature.
"""
function ΞTT(W_arr::SpectralVector{T, AA}, w3j²::WignerSymbolVector{T, Int}, ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    @inbounds for ℓ₃ in eachindex(w3j²)
        Ξ += (2ℓ₃ + 1) * w3j²[ℓ₃] * W_arr[ℓ₃]
    end
    return Ξ/4π
end
ΞTT(W_arr::SpectralVector{T, AA}, w3j²::WignerSymbolVector{T, Int}, 
    ℓ₁::Int, ℓ₂::Int) where {T, AA<:Zeros{T,1,Tuple{Base.OneTo{Int}}}} = zero(T)

# inner loop 
function loop_covTT!(C::SpectralArray{T,2}, lmax::Integer, band::Integer,
                     W1, W2, W3, W4, W5, W6, W7, W8) where {T}

    thread_buffers = Vector{Vector{T}}(undef, Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        thread_buffers[i] = Vector{T}(undef, 2*lmax+1)
    end
    
    @qthreads for ℓ₁ in 0:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:min(ℓ₁+band,lmax)
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            C[ℓ₁, ℓ₂] = (
                ΞTT(W1, w3j², ℓ₁, ℓ₂) + ΞTT(W2, w3j², ℓ₁, ℓ₂) + ΞTT(W3, w3j², ℓ₁, ℓ₂) +
                ΞTT(W4, w3j², ℓ₁, ℓ₂) + ΞTT(W5, w3j², ℓ₁, ℓ₂) + ΞTT(W6, w3j², ℓ₁, ℓ₂) +
                ΞTT(W7, w3j², ℓ₁, ℓ₂) + ΞTT(W8, w3j², ℓ₁, ℓ₂))
            C[ℓ₂, ℓ₁] = C[ℓ₁, ℓ₂]
        end
    end
end
