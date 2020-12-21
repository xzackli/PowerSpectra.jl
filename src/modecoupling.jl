
"""
Projector function for TT. Goes into the mode-coupling matrix.
"""
function Ξ_TT(W_arr::SpectralVector{T, AA}, 
             w3j²_00::WignerSymbolVector{T, Int}, 
             ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²_00), firstindex(W_arr))
    ℓ₃_end = min(lastindex(w3j²_00), lastindex(W_arr))
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²_00[ℓ₃] * W_arr[ℓ₃]
    end
    return Ξ / (4π)
end
Ξ_TT(W_arr::SpectralVector{T, AA}, w3j²_00::WignerSymbolVector{T, Int}, 
    ℓ₁::Int, ℓ₂::Int) where {T, AA<:Zeros} = zero(T)


"""
Projector function for EE. Goes into the mode-coupling matrix.

Note that w3j² refers to the square of ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
"""
function Ξ_EE(W_arr::SpectralVector{T, AA}, 
                w3j²_22::WignerSymbolVector{T, Int}, 
                ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²_22), firstindex(W_arr))
    ℓ₃_end = min(lastindex(w3j²_22), lastindex(W_arr))
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * (1 + (-1)^(ℓ₁ + ℓ₂ + ℓ₃))^2 * w3j²_22[ℓ₃] * W_arr[ℓ₃]
    end
    return Ξ / (16π)
end
Ξ_EE(W_arr::SpectralVector{T, AA}, w3j²_22::WignerSymbolVector{T, Int}, 
    ℓ₁::Int, ℓ₂::Int) where {T, AA<:Zeros} = zero(T)


"""
Projector function for TE. Goes into the mode-coupling matrix.

Note that w3j_00_mul_22 refers to ( ℓ ℓ₂ ℓ₃ 0 0 0 ) × ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
"""
function Ξ_TE(W_arr::SpectralVector{T, AA}, 
              w3j_00_mul_22::WignerSymbolVector{T, Int}, 
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j_00_mul_22), firstindex(W_arr))
    ℓ₃_end = min(lastindex(w3j_00_mul_22), lastindex(W_arr))
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * (1 + (-1)^(ℓ₁ + ℓ₂ + ℓ₃)) * w3j_00_mul_22[ℓ₃] * W_arr[ℓ₃]
    end
    return Ξ / (8π)
end
Ξ_TE(W_arr::SpectralVector{T, AA}, w3j²_22::WignerSymbolVector{T, Int}, 
    ℓ₁::Int, ℓ₂::Int) where {T, AA<:Zeros} = zero(T)


# inner MCM loop TT
function loop_mcm_TT!(mcm::SpectralArray{T,2}, lmax::Integer, 
                      Vij::SpectralVector{T}) where {T}
    thread_buffers = get_thread_buffers(T, 2*lmax+1)
    
    @qthreads for ℓ₁ in 2:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in 2:lmax
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j²_00 = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²_00)  # deposit symbols into buffer
            w3j²_00.symbols .= w3j²_00.symbols .^ 2  # square the symbols
            mcm[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ_TT(Vij, w3j²_00, ℓ₁, ℓ₂)
        end
    end
    mcm[0,0] = one(T)
    mcm[1,1] = one(T)
    return mcm
end

function compute_mcm_TT(workspace::SpectralWorkspace{T}, 
                        name_i::String, name_j::String; lmax::Int=0) where {T}
    lmax = iszero(lmax) ? workspace.lmax : lmax
    Vij = SpectralVector(alm2cl(workspace.masks[name_i, TT], workspace.masks[name_j, TT]))
    workspace.V_spectra[TT, name_i, name_j] = Vij
    mcm = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    return loop_mcm_TT!(mcm, lmax, Vij)
end


# inner MCM loop
function loop_mcm_EE!(mcm::SpectralArray{T,2}, lmax::Integer, 
                      Vij::SpectralVector{T}) where {T}
    thread_buffers = get_thread_buffers(T, 2*lmax+1)
    
    lmin = 2
    @qthreads for ℓ₁ in lmin:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in lmin:lmax
            w = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j²_22 = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²_22)  # deposit symbols into buffer
            w3j²_22.symbols .= w3j²_22.symbols .^ 2  # square the symbols
            mcm[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ_EE(Vij, w3j²_22, ℓ₁, ℓ₂)
        end
    end
    mcm[0,0] = one(T)
    mcm[1,1] = one(T)
    return mcm
end

function compute_mcm_EE(workspace::SpectralWorkspace{T}, 
                     name_i::String, name_j::String; lmax::Int=0) where {T}
    
    lmax = iszero(lmax) ? workspace.lmax : lmax
    Vij = SpectralVector(alm2cl(workspace.masks[name_i, PP], workspace.masks[name_j, PP]))
    workspace.V_spectra[PP, name_i, name_j] = Vij
    mcm = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    return loop_mcm_EE!(mcm, lmax, Vij)
end


## TE
# inner MCM loop
function loop_mcm_TE!(mcm::SpectralArray{T,2}, lmax::Integer, 
                      thread_buffers_0, thread_buffers_2,
                      Vij::SpectralVector{T}) where {T}
    
    @qthreads for ℓ₁ in 2:lmax
        buffer0 = thread_buffers_0[Threads.threadid()]
        buffer2 = thread_buffers_2[Threads.threadid()]

        for ℓ₂ in ℓ₁:lmax
            w00 = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            w22 = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view_0 = view(buffer0, 1:(w00.nₘₐₓ - w00.nₘᵢₙ + 1))  # preallocated buffer
            buffer_view_2 = view(buffer2, 1:(w22.nₘₐₓ - w22.nₘᵢₙ + 1))  # preallocated buffer
            w3j_00 = WignerSymbolVector(buffer_view_0, w00.nₘᵢₙ:w00.nₘₐₓ)
            w3j_22 = WignerSymbolVector(buffer_view_2, w22.nₘᵢₙ:w22.nₘₐₓ)
            wigner3j_f!(w00, w3j_00)  # deposit symbols into buffer
            wigner3j_f!(w22, w3j_22)  # deposit symbols into buffer

            w3j_00_22 = w3j_00
            w3j_00_22.symbols .*= w3j_22.symbols
            mcm[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ_TE(Vij, w3j_00_22, ℓ₁, ℓ₂)
        end
    end
    mcm[0,0] = one(T)
    mcm[1,1] = one(T)
    return mcm
end

function compute_mcm_TE(workspace::SpectralWorkspace{T}, 
                     name_i::String, name_j::String; lmax::Int=0) where {T}
    
    lmax = iszero(lmax) ? workspace.lmax : lmax
    thread_buffers_0 = get_thread_buffers(T, 2*lmax+1)
    thread_buffers_2 = get_thread_buffers(T, 2*lmax+1)

    Vij = SpectralVector(alm2cl(workspace.masks[name_i, TT], workspace.masks[name_j, PP]))
    workspace.V_spectra[TP, name_i, name_j] = Vij
    mcm = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    return loop_mcm_TE!(mcm, lmax, thread_buffers_0, thread_buffers_2, Vij)
end

function compute_mcm_ET(workspace::SpectralWorkspace{T}, 
                     name_i::String, name_j::String; lmax::Int=0) where {T}
    
    lmax = iszero(lmax) ? workspace.lmax : lmax
    thread_buffers_0 = get_thread_buffers(T, 2*lmax+1)
    thread_buffers_2 = get_thread_buffers(T, 2*lmax+1)

    Vij = SpectralVector(alm2cl(workspace.masks[name_i, PP], workspace.masks[name_j, TT]))
    workspace.V_spectra[TP, name_i, name_j] = Vij
    mcm = SpectralArray(zeros(T, (lmax+1, lmax+1)))
    return loop_mcm_TE!(mcm, lmax, thread_buffers_0, thread_buffers_2, Vij)
end


function compute_spectra(map_1::Map{T}, map_2::Map{T}, 
                         factorized_mcm,
                         Bℓ_1::SpectralVector{T}, Bℓ_2::SpectralVector{T}) where T
    Cl_hat = alm2cl(map2alm(map_1), map2alm(map_2))
    # Cl_hat[1:2] .= 0.0
    ldiv!(factorized_mcm, Cl_hat)
    return Cl_hat ./ (Bℓ_1.parent .* Bℓ_2.parent)
end


function compute_spectra(alm_1::Alm{Complex{T},Array{Complex{T},1}}, alm_2::Alm{Complex{T},Array{Complex{T},1}}, 
                         factorized_mcm,
                         Bℓ_1::SpectralVector{T}, Bℓ_2::SpectralVector{T}) where T
    Cl_hat = alm2cl(alm_1, alm_2)
    Cl_hat[1:2] .= zero(T)
    ldiv!(factorized_mcm, Cl_hat)
    return Cl_hat ./ (Bℓ_1.parent .* Bℓ_2.parent)
end

function binning_matrix(left_bins, right_bins, weight_function_ℓ; lmax=nothing)
    nbins = length(left_bins)
    lmax = isnothing(lmax) ? right_bins[end] : lmax
    P = zeros(nbins, lmax)
    for b in 1:nbins
        weights = weight_function_ℓ.(left_bins[b]:right_bins[b])
        norm = sum(weights)
        P[b, left_bins[b]+1:right_bins[b]+1] .= weights ./ norm
    end
    return P
end
