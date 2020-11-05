
"""
Compute the effective weight map copefficients and store them in the workspace.
"""
function effective_weights!(workspace::SpectralWorkspace{T},
                         m_i::Field{T}, m_j::Field{T}, 
                         m_p::Field{T}, m_q::Field{T}) where {T <: Real}
    # generate coefficients w
    fields = Field{T}[m_i, m_j, m_p, m_q]
    names = [m_i.name, m_j.name, m_p.name, m_q.name]

    map_buffer = Map{T, RingOrder}(zeros(T, size(m_i.maskT.pixels)))  # reuse pixel buffer

    # XX, i, j, YY
    w = workspace.effective_weights

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
    W = ThreadSafeDict{WIndex, SpectralVector{T}}()
    @threads for (X, Y, i, j, α, p, q, β) in weight_indices
        if (X, Y, i, j, α, p, q, β) ∉ keys(W)
            w1 = workspace.effective_weights[X, i, j, TT]
            w2 = workspace.effective_weights[Y, p, q, TT]
            if(typeof(w1) <: Alm && typeof(w2) <: Alm)

                # PP turns into QQ and UU pair
                wterms_X = (X == :PP) ? (:QQ, :UU) : (X,)
                wterms_Y = (Y == :PP) ? (:QQ, :UU) : (Y,)
                result = zeros(T, workspace.lmax+1)

                # Planck 2015 eq. C.11
                for wX in wterms_X
                    for wY in wterms_Y
                        result .+= alm2cl(
                            workspace.effective_weights[wX, i, j, α], 
                            workspace.effective_weights[wY, p, q, β])
                    end
                end
                norm = one(T) / (length(wterms_X) * length(wterms_Y))
                result .*= norm
                W[X, Y, i, j, α, p, q, β] = SpectralVector(result)
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
function compute_covmat(workspace::SpectralWorkspace{T}, 
             spectra, factorized_mcm_XY, factorized_mcm_ZW,
             m_i::Field{T}, m_j::Field{T}, m_p::Field{T}, m_q::Field{T};
             lmax=0) where {T <: Real}

    effective_weights!(workspace, m_i, m_j, m_p, m_q)
    W_spectra!(workspace)

    lmax = iszero(lmax) ? workspace.lmax : lmax
    i, j, p, q = workspace.field_names
    W = workspace.W_spectra

    C = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    loop_covTT!(C, lmax, 
        spectra[TT,i,p], spectra[TT,j,q], spectra[TT,i,q], spectra[TT,j,p],
        W[∅∅, ∅∅, i, p, TT, j, q, TT],
        W[∅∅, ∅∅, i, q, TT, j, p, TT],
        W[∅∅, TT, i, p, TT, j, q, TT],
        W[∅∅, TT, j, q, TT, i, p, TT],
        W[∅∅, TT, i, q, TT, j, p, TT],
        W[∅∅, TT, j, p, TT, i, q, TT],
        W[TT, TT, i, p, TT, j, q, TT],
        W[TT, TT, i, q, TT, j, p, TT])

    rdiv!(C.parent, factorized_mcm_ZW)
    ldiv!(factorized_mcm_XY, C.parent)

    # beam_covTT!(C, m_i.beam, m_j.beam, m_p.beam, m_q.beam)
    return C
end
# cov(m_i::Field{T}, m_j::Field{T}) where {T <: Real} = cov(m_i, m_j, m_i, m_j)


function beam_covTT!(C, Bl_i, Bl_j, Bl_p, Bl_q)
    for ℓ ∈ axes(C,1), ℓp ∈ axes(C,2)
        C[ℓ, ℓp] *= 1.0 / (Bl_i[ℓ] * Bl_j[ℓ] * Bl_p[ℓp] * Bl_q[ℓp])
    end
end

# inner loop 
function loop_covTT!(C::SpectralArray{T,2}, lmax::Integer, 
                     TTip::SpectralVector{T}, TTjq::SpectralVector{T}, 
                     TTiq::SpectralVector{T}, TTjp::SpectralVector{T},
                     W1, W2, W3, W4, W5, W6, W7, W8) where {T}

    thread_buffers = get_thread_buffers(T, 2 * lmax + 1)
    
    @qthreads for ℓ₁ in 0:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            C[ℓ₁, ℓ₂] = (
                sqrt(TTip[ℓ₁] * TTip[ℓ₂] * TTjq[ℓ₁] * TTjq[ℓ₂]) * ΞTT(W1, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTiq[ℓ₁] * TTiq[ℓ₂] * TTjp[ℓ₁] * TTjp[ℓ₂]) * ΞTT(W2, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTip[ℓ₁] * TTip[ℓ₂]) * ΞTT(W3, w3j², ℓ₁, ℓ₂) +
                sqrt(TTjq[ℓ₁] * TTjq[ℓ₂]) * ΞTT(W4, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTiq[ℓ₁] * TTiq[ℓ₂]) * ΞTT(W5, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTjp[ℓ₁] * TTjp[ℓ₂]) * ΞTT(W6, w3j², ℓ₁, ℓ₂) +
                ΞTT(W7, w3j², ℓ₁, ℓ₂) + 
                ΞTT(W8, w3j², ℓ₁, ℓ₂)
            )
            C[ℓ₂, ℓ₁] = C[ℓ₁, ℓ₂]
        end
    end
end
