
"""
Compute all effective weight map coefficients and store them in the workspace.
"""
function effective_weights_w!(workspace::SpectralWorkspace{T},
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
                    map_buffer.pixels .*= fields[j].σ²II.pixels
                    w[II, name_i, name_i, TT] = map2alm(map_buffer)
                end
            end
        end
    end
end


"""
Compute all effective weight map coefficients and store them in the workspace.
"""
function effective_weights_w!(workspace::SpectralWorkspace{T},
                              m_i::PolarizedField{T}, m_j::PolarizedField{T}, 
                              m_p::PolarizedField{T}, m_q::PolarizedField{T}) where {T <: Real}
    # generate coefficients w
    fields = [m_i, m_j, m_p, m_q]
    names = [m_i.name, m_j.name, m_p.name, m_q.name]
    lmax = workspace.lmax
    map_buffer = Map{T, RingOrder}(zeros(T, size(m_i.maskT.pixels)))  # reuse pixel buffer
    zero_alm = Alm(lmax, lmax, Zeros{Complex{T}}(numberOfAlms(lmax, lmax)))
    N_pix = map_buffer.resolution.numOfPixels
    Ω_p = 4π / N_pix
    w = workspace.effective_weights

    ell_array = get_ell_array(lmax)

    for (i, name_i) in enumerate(names)
        for (j, name_j) in enumerate(names)


            # OTHER PROJECTORS ARE DISABLED FOR SPEED TEMPORARILY 
            
            # Planck 2015 C.17 - C.20
            # if ((∅∅, name_i, name_j, TT) ∉ keys(w))  # ∅∅ TT
            #     map_buffer.pixels .= fields[i].maskT.pixels
            #     map_buffer.pixels .*= fields[j].maskT.pixels
            #     w[∅∅, name_i, name_j, TT] = map2alm(map_buffer)  # allocate new alms
            # end

            # if ((∅∅, name_i, name_j, TP) ∉ keys(w))  # ∅∅ TP
            #     map_buffer.pixels .= fields[i].maskT.pixels
            #     map_buffer.pixels .*= fields[j].maskP.pixels
            #     w[∅∅, name_i, name_j, TP] = map2alm(map_buffer)  # allocate new alms
            # end
            # if ((∅∅, name_i, name_j, PT) ∉ keys(w))  # ∅∅ TP
            #     map_buffer.pixels .= fields[i].maskP.pixels
            #     map_buffer.pixels .*= fields[j].maskT.pixels
            #     w[∅∅, name_i, name_j, PT] = map2alm(map_buffer)  # allocate new alms
            # end

            if name_i == name_j  # δᵢⱼ here, so we don't create arrays for this. C.21 - C.23
                # if ((II, name_i, name_i, TT) ∉ keys(w))  # II TT
                #     map_buffer.pixels .= fields[i].maskT.pixels
                #     map_buffer.pixels .*= fields[j].maskT.pixels
                #     map_buffer.pixels .*= fields[j].σ²II.pixels .* Ω_p
                #     w[II, name_i, name_i, TT] = map2alm(map_buffer)
                # end
                if ((QQ, name_i, name_i, PP) ∉ keys(w))  # QQ PP
                    map_buffer.pixels .= fields[i].maskP.pixels
                    map_buffer.pixels .*= fields[j].maskP.pixels
                    map_buffer.pixels .*= fields[j].σ²QQ.pixels .* Ω_p
                    w[QQ, name_i, name_i, PP] = map2alm(map_buffer)
                end
                if ((UU, name_i, name_i, PP) ∉ keys(w))  # UU PP
                    map_buffer.pixels .= fields[i].maskP.pixels
                    map_buffer.pixels .*= fields[j].maskP.pixels
                    map_buffer.pixels .*= fields[j].σ²UU.pixels .* Ω_p
                    w[UU, name_i, name_i, PP] = map2alm(map_buffer)
                end
            else
                w[II, name_i, name_j, TT] = zero_alm
                w[QQ, name_i, name_j, PP] = zero_alm
                w[UU, name_i, name_j, PP] = zero_alm
            end
            if ((∅∅, name_i, name_j, PP) ∉ keys(w))  # ∅∅ PP
                map_buffer.pixels .= fields[i].maskP.pixels
                map_buffer.pixels .*= fields[j].maskP.pixels
                w[∅∅, name_i, name_j, PP] = map2alm(map_buffer)  # allocate new alms
            end
        end

    end
end


function window_function_W!(workspace::SpectralWorkspace{T}, X, Y, i, j, α, p, q, β) where {T}
    # check if it's already computed
    if (X, Y, i, j, α, p, q, β) in keys(workspace.W_spectra)
        return workspace.W_spectra[(X, Y, i, j, α, p, q, β)]
    end

    # PP turns into QQ and UU pair
    wterms_X = (X == PP) ? (QQ, UU) : (X,)
    wterms_Y = (Y == PP) ? (QQ, UU) : (Y,)
    result = zeros(T, workspace.lmax+1)

    # Planck 2015 eq. C.11 - C.16
    for wX in wterms_X
        for wY in wterms_Y
            result .+= alm2cl(
                workspace.effective_weights[wX, i, j, α], 
                workspace.effective_weights[wY, p, q, β])
        end
    end
    norm = one(T) / (length(wterms_X) * length(wterms_Y))

    result .*= norm
    result = SpectralVector(result)

    workspace.W_spectra[X, Y, i, j, α, p, q, β] = result
    return result
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
function compute_covmat_TT(workspace::SpectralWorkspace{T}, 
             spectra, factorized_mcm_XY, factorized_mcm_ZW,
             m_i::Field{T}, m_j::Field{T}, m_p::Field{T}, m_q::Field{T};
             lmax=0) where {T <: Real}

    effective_weights_w!(workspace, m_i, m_j, m_p, m_q)

    lmax = iszero(lmax) ? workspace.lmax : lmax
    i, j, p, q = workspace.field_names
    W = workspace.W_spectra

    C = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    loop_covTT!(C, lmax, 
        spectra[TT,i,p], spectra[TT,j,q], spectra[TT,i,q], spectra[TT,j,p],
        window_function_W!(workspace, ∅∅, ∅∅, i, p, TT, j, q, TT),
        window_function_W!(workspace, ∅∅, ∅∅, i, q, TT, j, p, TT),
        window_function_W!(workspace, ∅∅, TT, i, p, TT, j, q, TT),
        window_function_W!(workspace, ∅∅, TT, j, q, TT, i, p, TT),
        window_function_W!(workspace, ∅∅, TT, i, q, TT, j, p, TT),
        window_function_W!(workspace, ∅∅, TT, j, p, TT, i, q, TT),
        window_function_W!(workspace, TT, TT, i, p, TT, j, q, TT),
        window_function_W!(workspace, TT, TT, i, q, TT, j, p, TT))

    rdiv!(C.parent, factorized_mcm_ZW)
    ldiv!(factorized_mcm_XY, C.parent)

    # beam_covTT!(C, m_i.beam, m_j.beam, m_p.beam, m_q.beam)
    return C
end


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
    
    @threads for ℓ₁ in 0:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            C[ℓ₁, ℓ₂] = (
                sqrt(TTip[ℓ₁] * TTip[ℓ₂] * TTjq[ℓ₁] * TTjq[ℓ₂]) * Ξ_TT(W1, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTiq[ℓ₁] * TTiq[ℓ₂] * TTjp[ℓ₁] * TTjp[ℓ₂]) * Ξ_TT(W2, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTip[ℓ₁] * TTip[ℓ₂]) * Ξ_TT(W3, w3j², ℓ₁, ℓ₂) +
                sqrt(TTjq[ℓ₁] * TTjq[ℓ₂]) * Ξ_TT(W4, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTiq[ℓ₁] * TTiq[ℓ₂]) * Ξ_TT(W5, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTjp[ℓ₁] * TTjp[ℓ₂]) * Ξ_TT(W6, w3j², ℓ₁, ℓ₂) +
                Ξ_TT(W7, w3j², ℓ₁, ℓ₂) + 
                Ξ_TT(W8, w3j², ℓ₁, ℓ₂)
            )
            C[ℓ₂, ℓ₁] = C[ℓ₁, ℓ₂]
        end
    end
end




function compute_covmat_EE(workspace::SpectralWorkspace{T}, 
             spectra, rescaling_coefficients, factorized_mcm_XY, factorized_mcm_ZW,
             m_i::PolarizedField{T}, m_j::PolarizedField{T}, m_p::PolarizedField{T}, m_q::PolarizedField{T};
             lmax=0) where {T <: Real}

    effective_weights_w!(workspace, m_i, m_j, m_p, m_q)

    lmax = iszero(lmax) ? workspace.lmax : lmax
    i, j, p, q = workspace.field_names
    W = workspace.W_spectra

    r_ℓ_ip = rescaling_coefficients[EE, i, p]
    r_ℓ_jq = rescaling_coefficients[EE, j, q]
    r_ℓ_iq = rescaling_coefficients[EE, i, q]
    r_ℓ_jp = rescaling_coefficients[EE, j, p]

    C = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    loop_covEE!(C, lmax, 
        spectra[EE,i,p], spectra[EE,j,q], spectra[EE,i,q], spectra[EE,j,p],
        r_ℓ_ip, r_ℓ_jq, r_ℓ_iq, r_ℓ_jp,
        window_function_W!(workspace, ∅∅, ∅∅, i, p, PP, j, q, PP),
        window_function_W!(workspace, ∅∅, ∅∅, i, q, PP, j, p, PP),

        window_function_W!(workspace, ∅∅, PP, i, p, PP, j, q, PP),
        window_function_W!(workspace, ∅∅, PP, j, q, PP, i, p, PP),
        window_function_W!(workspace, ∅∅, PP, i, q, PP, j, p, PP),
        window_function_W!(workspace, ∅∅, PP, j, p, PP, i, q, PP),

        window_function_W!(workspace, PP, PP, i, p, PP, j, q, PP),
        window_function_W!(workspace, PP, PP, i, q, PP, j, p, PP))

    rdiv!(C.parent, factorized_mcm_ZW)
    ldiv!(factorized_mcm_XY, C.parent)

    return C
end


# inner loop 
function loop_covEE!(C::SpectralArray{T,2}, lmax::Integer,
                     EEip::SpectralVector{T}, EEjq::SpectralVector{T}, 
                     EEiq::SpectralVector{T}, EEjp::SpectralVector{T},
                     r_ℓ_ip::SpectralVector{T}, r_ℓ_jq::SpectralVector{T}, 
                     r_ℓ_iq::SpectralVector{T}, r_ℓ_jp::SpectralVector{T},
                     W1, W2, W3, W4, W5, W6, W7, W8) where {T}

    thread_buffers = get_thread_buffers(T, 2 * lmax + 1)
    
    @qthreads for ℓ₁ in 2:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax 
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            C[ℓ₁, ℓ₂] = (
                sqrt(EEip[ℓ₁] * EEip[ℓ₂] * EEjq[ℓ₁] * EEjq[ℓ₂]) * Ξ_EE(W1, w3j², ℓ₁, ℓ₂) + 
                sqrt(EEiq[ℓ₁] * EEiq[ℓ₂] * EEjp[ℓ₁] * EEjp[ℓ₂]) * Ξ_EE(W2, w3j², ℓ₁, ℓ₂) +
                sqrt(EEip[ℓ₁] * EEip[ℓ₂]) * Ξ_EE(W3, w3j², ℓ₁, ℓ₂) * r_ℓ_jq[ℓ₁] * r_ℓ_jq[ℓ₂] +
                sqrt(EEjq[ℓ₁] * EEjq[ℓ₂]) * Ξ_EE(W4, w3j², ℓ₁, ℓ₂) * r_ℓ_ip[ℓ₁] * r_ℓ_ip[ℓ₂] + 
                sqrt(EEiq[ℓ₁] * EEiq[ℓ₂]) * Ξ_EE(W5, w3j², ℓ₁, ℓ₂) * r_ℓ_jp[ℓ₁] * r_ℓ_jp[ℓ₂]  + 
                sqrt(EEjp[ℓ₁] * EEjp[ℓ₂]) * Ξ_EE(W6, w3j², ℓ₁, ℓ₂) * r_ℓ_iq[ℓ₁] * r_ℓ_iq[ℓ₂]  + 
                Ξ_EE(W7, w3j², ℓ₁, ℓ₂) * r_ℓ_ip[ℓ₁] * r_ℓ_jq[ℓ₁] * r_ℓ_ip[ℓ₂] * r_ℓ_jq[ℓ₂] + 
                Ξ_EE(W8, w3j², ℓ₁, ℓ₂) * r_ℓ_iq[ℓ₁] * r_ℓ_jp[ℓ₁] * r_ℓ_iq[ℓ₂] * r_ℓ_jp[ℓ₂]
            )

            C[ℓ₂, ℓ₁] = C[ℓ₁, ℓ₂]
        end
    end
end
