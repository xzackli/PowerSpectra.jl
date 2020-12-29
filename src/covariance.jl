
function decouple_covmat!(𝐂::SpectralArray{T,2}, mcm_adj_XY::F, mcm_adj_ZW::F) where {T <: Real, F<:Factorization{T}}
    rdiv!(𝐂.parent', mcm_adj_ZW)
    rdiv!(𝐂.parent, mcm_adj_XY)
    return 𝐂
end


function compute_coupled_covmat_TTTT(workspace::CovarianceWorkspace{T}, spectra, 
                                     rescaling_coefficients; ℓₘₐₓ=0) where {T <: Real}

    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    i, j, p, q = workspace.field_names

    r_ℓ_ip = rescaling_coefficients[TT, i, p]
    r_ℓ_jq = rescaling_coefficients[TT, j, q]
    r_ℓ_iq = rescaling_coefficients[TT, i, q]
    r_ℓ_jp = rescaling_coefficients[TT, j, p]

    𝐂 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    loop_covTTTT!(𝐂, ℓₘₐₓ, 
        spectra[TT,i,p], spectra[TT,j,q], spectra[TT,i,q], spectra[TT,j,p],
        r_ℓ_ip, r_ℓ_jq, r_ℓ_iq, r_ℓ_jp,
        window_function_W!(workspace, ∅∅, ∅∅, i, p, TT, j, q, TT),
        window_function_W!(workspace, ∅∅, ∅∅, i, q, TT, j, p, TT),
        window_function_W!(workspace, ∅∅, TT, i, p, TT, j, q, TT),
        window_function_W!(workspace, ∅∅, TT, j, q, TT, i, p, TT),
        window_function_W!(workspace, ∅∅, TT, i, q, TT, j, p, TT),
        window_function_W!(workspace, ∅∅, TT, j, p, TT, i, q, TT),
        window_function_W!(workspace, TT, TT, i, p, TT, j, q, TT),
        window_function_W!(workspace, TT, TT, i, q, TT, j, p, TT))
        
    return 𝐂
end


# inner loop 
function loop_covTTTT!(𝐂::SpectralArray{T,2}, ℓₘₐₓ::Integer, 
                     TTip::SpectralVector{T}, TTjq::SpectralVector{T}, 
                     TTiq::SpectralVector{T}, TTjp::SpectralVector{T},
                     r_ℓ_ip::SpectralVector{T}, r_ℓ_jq::SpectralVector{T}, 
                     r_ℓ_iq::SpectralVector{T}, r_ℓ_jp::SpectralVector{T},
                     W1, W2, W3, W4, W5, W6, W7, W8) where {T}

    thread_buffers = get_thread_buffers(T, 2 * ℓₘₐₓ + 1)
    
    @qthreads for ℓ₁ in 0:ℓₘₐₓ
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            𝐂[ℓ₁, ℓ₂] = (
                sqrt(TTip[ℓ₁] * TTip[ℓ₂] * TTjq[ℓ₁] * TTjq[ℓ₂]) * Ξ_TT(W1, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTiq[ℓ₁] * TTiq[ℓ₂] * TTjp[ℓ₁] * TTjp[ℓ₂]) * Ξ_TT(W2, w3j², ℓ₁, ℓ₂) + 
                sqrt(TTip[ℓ₁] * TTip[ℓ₂]) * Ξ_TT(W3, w3j², ℓ₁, ℓ₂) * r_ℓ_jq[ℓ₁] * r_ℓ_jq[ℓ₂] +
                sqrt(TTjq[ℓ₁] * TTjq[ℓ₂]) * Ξ_TT(W4, w3j², ℓ₁, ℓ₂) * r_ℓ_ip[ℓ₁] * r_ℓ_ip[ℓ₂] + 
                sqrt(TTiq[ℓ₁] * TTiq[ℓ₂]) * Ξ_TT(W5, w3j², ℓ₁, ℓ₂) * r_ℓ_jp[ℓ₁] * r_ℓ_jp[ℓ₂]  + 
                sqrt(TTjp[ℓ₁] * TTjp[ℓ₂]) * Ξ_TT(W6, w3j², ℓ₁, ℓ₂) * r_ℓ_iq[ℓ₁] * r_ℓ_iq[ℓ₂]  + 
                Ξ_TT(W7, w3j², ℓ₁, ℓ₂) * r_ℓ_ip[ℓ₁] * r_ℓ_jq[ℓ₁] * r_ℓ_ip[ℓ₂] * r_ℓ_jq[ℓ₂] + 
                Ξ_TT(W8, w3j², ℓ₁, ℓ₂) * r_ℓ_iq[ℓ₁] * r_ℓ_jp[ℓ₁] * r_ℓ_iq[ℓ₂] * r_ℓ_jp[ℓ₂])
            𝐂[ℓ₂, ℓ₁] = 𝐂[ℓ₁, ℓ₂]
        end
    end
end


function compute_coupled_covmat_EEEE(workspace::CovarianceWorkspace{T}, spectra, 
                                     rescaling_coefficients; ℓₘₐₓ=0) where {T <: Real}

    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    i, j, p, q = workspace.field_names

    r_ℓ_ip = rescaling_coefficients[EE, i, p]
    r_ℓ_jq = rescaling_coefficients[EE, j, q]
    r_ℓ_iq = rescaling_coefficients[EE, i, q]
    r_ℓ_jp = rescaling_coefficients[EE, j, p]

    𝐂 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    loop_covEEEE!(𝐂, ℓₘₐₓ, 
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

    return 𝐂
end


# inner loop 
function loop_covEEEE!(𝐂::SpectralArray{T,2}, ℓₘₐₓ::Integer,
                     EEip::SpectralVector{T}, EEjq::SpectralVector{T}, 
                     EEiq::SpectralVector{T}, EEjp::SpectralVector{T},
                     r_ℓ_ip::SpectralVector{T}, r_ℓ_jq::SpectralVector{T}, 
                     r_ℓ_iq::SpectralVector{T}, r_ℓ_jp::SpectralVector{T},
                     W1, W2, W3, W4, W5, W6, W7, W8) where {T}

    thread_buffers = get_thread_buffers(T, 2 * ℓₘₐₓ + 1)
    
    @qthreads for ℓ₁ in 0:ℓₘₐₓ
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ 
            w = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            𝐂[ℓ₁, ℓ₂] = (
                sqrt(EEip[ℓ₁] * EEip[ℓ₂] * EEjq[ℓ₁] * EEjq[ℓ₂]) * Ξ_EE(W1, w3j², ℓ₁, ℓ₂) + 
                sqrt(EEiq[ℓ₁] * EEiq[ℓ₂] * EEjp[ℓ₁] * EEjp[ℓ₂]) * Ξ_EE(W2, w3j², ℓ₁, ℓ₂) +
                sqrt(EEip[ℓ₁] * EEip[ℓ₂]) * Ξ_EE(W3, w3j², ℓ₁, ℓ₂) * r_ℓ_jq[ℓ₁] * r_ℓ_jq[ℓ₂] +
                sqrt(EEjq[ℓ₁] * EEjq[ℓ₂]) * Ξ_EE(W4, w3j², ℓ₁, ℓ₂) * r_ℓ_ip[ℓ₁] * r_ℓ_ip[ℓ₂] + 
                sqrt(EEiq[ℓ₁] * EEiq[ℓ₂]) * Ξ_EE(W5, w3j², ℓ₁, ℓ₂) * r_ℓ_jp[ℓ₁] * r_ℓ_jp[ℓ₂]  + 
                sqrt(EEjp[ℓ₁] * EEjp[ℓ₂]) * Ξ_EE(W6, w3j², ℓ₁, ℓ₂) * r_ℓ_iq[ℓ₁] * r_ℓ_iq[ℓ₂]  + 
                Ξ_EE(W7, w3j², ℓ₁, ℓ₂) * r_ℓ_ip[ℓ₁] * r_ℓ_jq[ℓ₁] * r_ℓ_ip[ℓ₂] * r_ℓ_jq[ℓ₂] + 
                Ξ_EE(W8, w3j², ℓ₁, ℓ₂) * r_ℓ_iq[ℓ₁] * r_ℓ_jp[ℓ₁] * r_ℓ_iq[ℓ₂] * r_ℓ_jp[ℓ₂])

            𝐂[ℓ₂, ℓ₁] = 𝐂[ℓ₁, ℓ₂]
        end
    end
end


function compute_coupled_covmat_TTTE(workspace::CovarianceWorkspace{T}, spectra, 
                                     rescaling_coefficients; ℓₘₐₓ=0) where {T <: Real}

    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    i, j, p, q = workspace.field_names
    W = workspace.W_spectra

    r_ℓ_ip = rescaling_coefficients[TT, i, p]
    r_ℓ_jp = rescaling_coefficients[TT, j, p]

    𝐂 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    loop_covTTTE!(𝐂, ℓₘₐₓ, 
        spectra[TT,i,p], spectra[TT,j,p], spectra[TE,i,q], spectra[TE,j,q],
        r_ℓ_ip, r_ℓ_jp,
        window_function_W!(workspace, ∅∅, ∅∅, i, p, TT, j, q, TP),
        window_function_W!(workspace, ∅∅, ∅∅, i, q, TP, j, p, TT),
        window_function_W!(workspace, ∅∅, TT, j, q, TP, i, p, TT),
        window_function_W!(workspace, ∅∅, TT, i, q, TP, j, p, TT))

    return 𝐂
end


# inner loop 
function loop_covTTTE!(𝐂::SpectralArray{T,2}, ℓₘₐₓ::Integer,
                     TTip::SpectralVector{T}, TTjp::SpectralVector{T}, 
                     TEiq::SpectralVector{T}, TEjq::SpectralVector{T},
                     r_ℓ_ip::SpectralVector{T}, r_ℓ_jp::SpectralVector{T},
                     W1, W2, W3, W4) where {T}

    thread_buffers = get_thread_buffers(T, 2 * ℓₘₐₓ + 1)
    
    @qthreads for ℓ₁ in 2:ℓₘₐₓ
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ 
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            𝐂[ℓ₁, ℓ₂] = (
                sqrt(TTip[ℓ₁] * TTip[ℓ₂]) * (TEjq[ℓ₁] + TEjq[ℓ₂]) * Ξ_TT(W1, w3j², ℓ₁, ℓ₂) +
                sqrt(TTjp[ℓ₁] * TTjp[ℓ₂]) * (TEiq[ℓ₁] + TEiq[ℓ₂]) * Ξ_TT(W2, w3j², ℓ₁, ℓ₂) +
                (TEjq[ℓ₁] + TEjq[ℓ₂]) * Ξ_TT(W3, w3j², ℓ₁, ℓ₂)  * r_ℓ_ip[ℓ₁] * r_ℓ_ip[ℓ₂]  +
                (TEiq[ℓ₁] + TEiq[ℓ₂]) * Ξ_TT(W4, w3j², ℓ₁, ℓ₂)  * r_ℓ_jp[ℓ₁] * r_ℓ_jp[ℓ₂] 
            ) / 2

            𝐂[ℓ₂, ℓ₁] = 𝐂[ℓ₁, ℓ₂]
        end
    end
end


function compute_coupled_covmat_TETE(workspace::CovarianceWorkspace{T}, spectra, 
                                     rescaling_coefficients; ℓₘₐₓ=0) where {T <: Real}

    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    i, j, p, q = workspace.field_names
    W = workspace.W_spectra

    r_TT_ip = rescaling_coefficients[TT, i, p]
    r_PP_jq = rescaling_coefficients[EE, j, q]

    𝐂 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    loop_covTETE!(𝐂, ℓₘₐₓ, 
        spectra[TT,i,p], spectra[EE,j,q], spectra[TE,i,q], spectra[TE,j,p], 
        r_TT_ip, r_PP_jq, 
        window_function_W!(workspace, ∅∅, ∅∅, i, p, TT, j, q, PP),
        window_function_W!(workspace, ∅∅, ∅∅, i, q, TP, j, p, PT),
        window_function_W!(workspace, ∅∅, PP, i, p, TT, j, q, PP),
        window_function_W!(workspace, ∅∅, TT, j, q, PP, i, p, TT),
        window_function_W!(workspace, TT, PP, i, p, TT, j, q, PP))

    return 𝐂
end


# inner loop 
function loop_covTETE!(𝐂::SpectralArray{T,2}, ℓₘₐₓ::Integer,
                     TTip::SpectralVector{T}, EEjq::SpectralVector{T}, 
                     TEiq::SpectralVector{T}, TEjp::SpectralVector{T},
                     r_TT_ip::SpectralVector{T}, r_PP_jq::SpectralVector{T}, 
                     W1, W2, W3, W4, W5) where {T}

    thread_buffers_0 = get_thread_buffers(T, 2*ℓₘₐₓ+1)
    thread_buffers_2 = get_thread_buffers(T, 2*ℓₘₐₓ+1)
    
    @qthreads for ℓ₁ in 2:ℓₘₐₓ
        buffer0 = thread_buffers_0[Threads.threadid()]
        buffer2 = thread_buffers_2[Threads.threadid()]

        for ℓ₂ in ℓ₁:ℓₘₐₓ 
            w00 = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            w22 = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view_0 = uview(buffer0, 1:(w00.nₘₐₓ - w00.nₘᵢₙ + 1))  # preallocated buffer
            buffer_view_2 = uview(buffer2, 1:(w22.nₘₐₓ - w22.nₘᵢₙ + 1))  # preallocated buffer
            w3j_00 = WignerSymbolVector(buffer_view_0, w00.nₘᵢₙ:w00.nₘₐₓ)
            w3j_22 = WignerSymbolVector(buffer_view_2, w22.nₘᵢₙ:w22.nₘₐₓ)
            wigner3j_f!(w00, w3j_00)  # deposit symbols into buffer
            wigner3j_f!(w22, w3j_22)  # deposit symbols into buffer

            # varied over ℓ₃
            w3j_00_22 = w3j_22  # buffer 2
            w3j_00_22.symbols .*= w3j_00.symbols   # buffer2 = (buffer 2) * (buffer 1)
            w3j_00_00 = w3j_00
            w3j_00_00.symbols .*= w3j_00.symbols # (buffer 1) = (buffer 1) * (buffer 1)


            𝐂[ℓ₁, ℓ₂] = (
                sqrt(TTip[ℓ₁] * TTip[ℓ₂] * EEjq[ℓ₁] * EEjq[ℓ₂]) * Ξ_TE(W1, w3j_00_22, ℓ₁, ℓ₂) +
                0.5 * (TEiq[ℓ₁] * TEjp[ℓ₂] + TEjp[ℓ₁] * TEiq[ℓ₂]) * Ξ_TT(W2, w3j_00_00, ℓ₁, ℓ₂) +
                sqrt(TTip[ℓ₁] * TTip[ℓ₂]) * Ξ_TE(W3, w3j_00_22, ℓ₁, ℓ₂) * r_PP_jq[ℓ₁] * r_PP_jq[ℓ₂] +
                sqrt(EEjq[ℓ₁] * EEjq[ℓ₂]) * Ξ_TE(W4, w3j_00_22, ℓ₁, ℓ₂) * r_TT_ip[ℓ₁] * r_TT_ip[ℓ₂] +
                Ξ_TE(W5, w3j_00_22, ℓ₁, ℓ₂) * r_TT_ip[ℓ₁] * r_TT_ip[ℓ₂] * r_PP_jq[ℓ₁] * r_PP_jq[ℓ₂]
            )

            𝐂[ℓ₂, ℓ₁] = 𝐂[ℓ₁, ℓ₂]
        end
    end
end


function compute_coupled_covmat_TEEE(workspace::CovarianceWorkspace{T}, spectra, 
                                     rescaling_coefficients; ℓₘₐₓ=0, planck=true) where {T <: Real}

    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    i, j, p, q = workspace.field_names

    r_EE_jq = rescaling_coefficients[EE, j, q]
    r_EE_jp = rescaling_coefficients[EE, j, p]

    𝐂 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))

    if planck
        loop_covTEEE_planck!(𝐂, ℓₘₐₓ, 
            spectra[EE,j,q], spectra[EE,j,p], spectra[TE,i,p], spectra[TE,i,q],
            r_EE_jq, r_EE_jp,
            window_function_W!(workspace, ∅∅, ∅∅, i, p, TP, j, q, PP),
            window_function_W!(workspace, ∅∅, ∅∅, i, q, TP, j, p, PP),
            window_function_W!(workspace, ∅∅, PP, i, p, TP, j, q, PP),
            window_function_W!(workspace, ∅∅, PP, i, q, TP, j, p, PP))
    else
        loop_covTEEE!(𝐂, ℓₘₐₓ, 
            spectra[EE,j,q], spectra[EE,j,p], spectra[TE,i,p], spectra[TE,i,q],
            r_EE_jq, r_EE_jp,
            window_function_W!(workspace, ∅∅, ∅∅, i, p, TP, j, q, PP),
            window_function_W!(workspace, ∅∅, ∅∅, i, q, TP, j, p, PP),
            window_function_W!(workspace, ∅∅, PP, i, p, TP, j, q, PP),
            window_function_W!(workspace, ∅∅, PP, i, q, TP, j, p, PP))
    end

    return 𝐂
end


# inner loop 
function loop_covTEEE!(𝐂::SpectralArray{T,2}, ℓₘₐₓ::Integer,
                      EEjq::SpectralVector{T}, EEjp::SpectralVector{T}, 
                      TEip::SpectralVector{T}, TEiq::SpectralVector{T},
                     r_EE_jq::SpectralVector{T}, r_EE_jp::SpectralVector{T}, 
                     W1, W2, W3, W4) where {T}

    thread_buffers_0 = get_thread_buffers(T, 2*ℓₘₐₓ+1)
    thread_buffers_2 = get_thread_buffers(T, 2*ℓₘₐₓ+1)
    @qthreads for ℓ₁ in 2:ℓₘₐₓ
        buffer0 = thread_buffers_0[Threads.threadid()]
        buffer2 = thread_buffers_2[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ 
            w00 = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            w22 = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view_0 = uview(buffer0, 1:(w00.nₘₐₓ - w00.nₘᵢₙ + 1))  # preallocated buffer
            buffer_view_2 = uview(buffer2, 1:(w22.nₘₐₓ - w22.nₘᵢₙ + 1))  # preallocated buffer
            w3j_00 = WignerSymbolVector(buffer_view_0, w00.nₘᵢₙ:w00.nₘₐₓ)
            w3j_22 = WignerSymbolVector(buffer_view_2, w22.nₘᵢₙ:w22.nₘₐₓ)
            wigner3j_f!(w00, w3j_00)  # deposit symbols into buffer
            wigner3j_f!(w22, w3j_22)  # deposit symbols into buffer
            # varied over ℓ₃
            w3j² = w3j_22  # buffer 2
            w3j².symbols .*= w3j_00.symbols   # buffer2 = (buffer 2) * (buffer 1)



            𝐂[ℓ₁, ℓ₂] = (
                sqrt(EEjq[ℓ₁] * EEjq[ℓ₂]) * (TEip[ℓ₁] + TEip[ℓ₂]) * Ξ_EE(W1, w3j², ℓ₁, ℓ₂) +
                sqrt(EEjp[ℓ₁] * EEjp[ℓ₂]) * (TEiq[ℓ₁] + TEiq[ℓ₂]) * Ξ_EE(W2, w3j², ℓ₁, ℓ₂) + 
                (TEip[ℓ₁] + TEip[ℓ₂]) * Ξ_EE(W3, w3j², ℓ₁, ℓ₂) * r_EE_jq[ℓ₁] * r_EE_jq[ℓ₂] + 
                (TEiq[ℓ₁] + TEiq[ℓ₂]) * Ξ_EE(W4, w3j², ℓ₁, ℓ₂) * r_EE_jp[ℓ₁] * r_EE_jp[ℓ₂]
            ) / 2

            𝐂[ℓ₂, ℓ₁] = 𝐂[ℓ₁, ℓ₂]
        end
    end
end


# inner loop 
function loop_covTEEE_planck!(𝐂::SpectralArray{T,2}, ℓₘₐₓ::Integer,
                      EEjq::SpectralVector{T}, EEjp::SpectralVector{T}, 
                      TEip::SpectralVector{T}, TEiq::SpectralVector{T},
                     r_EE_jq::SpectralVector{T}, r_EE_jp::SpectralVector{T}, 
                     W1, W2, W3, W4) where {T}

    thread_buffers = get_thread_buffers(T, 2 * ℓₘₐₓ + 1)
    @qthreads for ℓ₁ in 0:ℓₘₐₓ
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ 
            w = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            𝐂[ℓ₁, ℓ₂] = (
                sqrt(EEjq[ℓ₁] * EEjq[ℓ₂]) * (TEip[ℓ₁] + TEip[ℓ₂]) * Ξ_EE(W1, w3j², ℓ₁, ℓ₂) +
                sqrt(EEjp[ℓ₁] * EEjp[ℓ₂]) * (TEiq[ℓ₁] + TEiq[ℓ₂]) * Ξ_EE(W2, w3j², ℓ₁, ℓ₂) + 
                (TEip[ℓ₁] + TEip[ℓ₂]) * Ξ_EE(W3, w3j², ℓ₁, ℓ₂) * r_EE_jq[ℓ₁] * r_EE_jq[ℓ₂] + 
                (TEiq[ℓ₁] + TEiq[ℓ₂]) * Ξ_EE(W4, w3j², ℓ₁, ℓ₂) * r_EE_jp[ℓ₁] * r_EE_jp[ℓ₂]
            ) / 2

            𝐂[ℓ₂, ℓ₁] = 𝐂[ℓ₁, ℓ₂]
        end
    end
end


function compute_coupled_covmat_TTEE(workspace::CovarianceWorkspace{T}, spectra, 
                                     rescaling_coefficients; ℓₘₐₓ=0) where {T <: Real}

    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    i, j, p, q = workspace.field_names
    W = workspace.W_spectra

    𝐂 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    loop_covTTEE!(𝐂, ℓₘₐₓ, 
        spectra[TE,i,p], spectra[TE,i,q], spectra[TE,j,q], spectra[TE,j,p],
        window_function_W!(workspace, ∅∅, ∅∅, i, p, TP, j, q, TP),
        window_function_W!(workspace, ∅∅, ∅∅, i, q, TP, j, p, TP),
    )

    return 𝐂
end


# inner loop 
function loop_covTTEE!(𝐂::SpectralArray{T,2}, ℓₘₐₓ::Integer,
                      TEip::SpectralVector{T}, TEiq::SpectralVector{T},
                      TEjq::SpectralVector{T}, TEjp::SpectralVector{T}, 
                     W1, W2) where {T}

    thread_buffers = get_thread_buffers(T, 2 * ℓₘₐₓ + 1)
    
    @qthreads for ℓ₁ in 2:ℓₘₐₓ
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ 
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j² = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²)  # deposit symbols into buffer
            w3j².symbols .= w3j².symbols .^ 2  # square the symbols
            𝐂[ℓ₁, ℓ₂] = (
                (TEip[ℓ₁] * TEjq[ℓ₂] + TEjq[ℓ₁] * TEip[ℓ₂]) * Ξ_TT(W1, w3j², ℓ₁, ℓ₂) + 
                (TEiq[ℓ₁] * TEjp[ℓ₂] + TEjp[ℓ₁] * TEiq[ℓ₂]) * Ξ_TT(W2, w3j², ℓ₁, ℓ₂) 
            ) / 2

            𝐂[ℓ₂, ℓ₁] = 𝐂[ℓ₁, ℓ₂]
        end
    end
end

