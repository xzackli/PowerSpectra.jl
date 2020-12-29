
"""
Projector function for TT. Goes into the mode-coupling matrix.
"""
function Ξ_TT(W_arr::SpectralVector{T, AA}, 
             w3j²₀₀::WignerSymbolVector{T, Int}, 
             ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₀₀), firstindex(W_arr))
    ℓ₃_end = min(lastindex(w3j²₀₀), lastindex(W_arr))
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₀₀[ℓ₃] * W_arr[ℓ₃]
    end
    return Ξ / (4π)
end


"""
Projector function for EE. Goes into the mode-coupling matrix.

Note that w3j² refers to the square of ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
"""
function Ξ_EE(W_arr::SpectralVector{T, AA}, 
                w3j²₂₂::WignerSymbolVector{T, Int}, 
                ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₂₂), firstindex(W_arr))
    ℓ₃_end = min(lastindex(w3j²₂₂), lastindex(W_arr))
    if isodd(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₂₂[ℓ₃] * W_arr[ℓ₃]
    end
    return Ξ / (4π)
end


"""
Projector function for TE. Goes into the mode-coupling matrix.

Note that w3j₀₀₂₂ refers to ( ℓ ℓ₂ ℓ₃ 0 0 0 ) × ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
"""
function Ξ_TE(W_arr::SpectralVector{T, AA}, 
              w3j₀₀₂₂::WignerSymbolVector{T, Int}, 
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j₀₀₂₂), firstindex(W_arr))
    ℓ₃_end = min(lastindex(w3j₀₀₂₂), lastindex(W_arr))
    if isodd(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j₀₀₂₂[ℓ₃] * W_arr[ℓ₃]
    end
    return Ξ / (4π)
end


# inner MCM loop TT
function loop_mcm_TT!(𝐌::SpectralArray{T,2}, ℓₘₐₓ::Integer, 
                      Vᵢⱼ::SpectralVector{T}) where {T}
    thread_buffers = get_thread_buffers(T, 2ℓₘₐₓ+1)
    
    @qthreads for ℓ₁ in 2:ℓₘₐₓ
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ
            w = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j²₀₀ = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²₀₀)  # deposit symbols into buffer
            w3j²₀₀.symbols .= w3j²₀₀.symbols .^ 2  # square the symbols
            Ξ = Ξ_TT(Vᵢⱼ, w3j²₀₀, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ
        end
    end
    𝐌[0,0] = one(T)
    𝐌[1,1] = one(T)
    return 𝐌
end

function compute_mcm_TT(workspace::SpectralWorkspace{T}, 
                        name_i::String, name_j::String; ℓₘₐₓ::Int=0) where {T}
    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    Vᵢⱼ = SpectralVector(alm2cl(workspace.mask_alm[name_i, TT], workspace.mask_alm[name_j, TT]))
    𝐌 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    return loop_mcm_TT!(𝐌, ℓₘₐₓ, Vᵢⱼ)
end


# inner MCM loop
function loop_mcm_EE!(𝐌::SpectralArray{T,2}, ℓₘₐₓ::Integer, 
                      Vᵢⱼ::SpectralVector{T}) where {T}
    thread_buffers = get_thread_buffers(T, 2ℓₘₐₓ+1)
    
    @qthreads for ℓ₁ in 2:ℓₘₐₓ
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:ℓₘₐₓ
            w = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
            w3j²₂₂ = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
            wigner3j_f!(w, w3j²₂₂)  # deposit symbols into buffer
            w3j²₂₂.symbols .= w3j²₂₂.symbols .^ 2  # square the symbols
            Ξ = Ξ_EE(Vᵢⱼ, w3j²₂₂, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ
        end
    end
    𝐌[0,0] = one(T)
    𝐌[1,1] = one(T)
    return 𝐌
end

function compute_mcm_EE(workspace::SpectralWorkspace{T}, 
                     name_i::String, name_j::String; ℓₘₐₓ::Int=0) where {T}
    
    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    Vᵢⱼ = SpectralVector(alm2cl(
        workspace.mask_alm[name_i, PP], 
        workspace.mask_alm[name_j, PP]))
    𝐌 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    return loop_mcm_EE!(𝐌, ℓₘₐₓ, Vᵢⱼ)
end


## TE
# inner MCM loop
function loop_mcm_TE!(𝐌::SpectralArray{T,2}, ℓₘₐₓ::Integer, 
                      thread_buffers_0, thread_buffers_2,
                      Vᵢⱼ::SpectralVector{T}) where {T}
    
    @qthreads for ℓ₁ in 2:ℓₘₐₓ
        buffer0 = thread_buffers_0[Threads.threadid()]
        buffer2 = thread_buffers_2[Threads.threadid()]

        for ℓ₂ in ℓ₁:ℓₘₐₓ
            w₀₀ = WignerF(T, ℓ₁, ℓ₂, 0, 0)  # set up the wigner recurrence
            w₂₂ = WignerF(T, ℓ₁, ℓ₂, -2, 2)  # set up the wigner recurrence
            buffer_view_0 = uview(buffer0, 1:(w₀₀.nₘₐₓ - w₀₀.nₘᵢₙ + 1))  # preallocated buffer
            buffer_view_2 = uview(buffer2, 1:(w₂₂.nₘₐₓ - w₂₂.nₘᵢₙ + 1))  # preallocated buffer
            w3j₀₀ = WignerSymbolVector(buffer_view_0, w₀₀.nₘᵢₙ:w₀₀.nₘₐₓ)
            w3j₂₂ = WignerSymbolVector(buffer_view_2, w₂₂.nₘᵢₙ:w₂₂.nₘₐₓ)
            wigner3j_f!(w₀₀, w3j₀₀)  # deposit symbols into buffer
            wigner3j_f!(w₂₂, w3j₂₂)  # deposit symbols into buffer

            w3j₀₀₂₂ = w3j₀₀
            w3j₀₀₂₂.symbols .*= w3j₂₂.symbols
            Ξ = Ξ_TE(Vᵢⱼ, w3j₀₀₂₂, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ

        end
    end
    𝐌[0,0] = one(T)
    𝐌[1,1] = one(T)
    return 𝐌
end

function compute_mcm_TE(workspace::SpectralWorkspace{T}, 
                     name_i::String, name_j::String; ℓₘₐₓ::Int=0) where {T}
    
    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    thread_buffers_0 = get_thread_buffers(T, 2ℓₘₐₓ+1)
    thread_buffers_2 = get_thread_buffers(T, 2ℓₘₐₓ+1)

    Vᵢⱼ = SpectralVector(alm2cl(
        workspace.mask_alm[name_i, TT], 
        workspace.mask_alm[name_j, PP]))
    𝐌 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    return loop_mcm_TE!(𝐌, ℓₘₐₓ, thread_buffers_0, thread_buffers_2, Vᵢⱼ)
end

function compute_mcm_ET(workspace::SpectralWorkspace{T}, 
                     name_i::String, name_j::String; ℓₘₐₓ::Int=0) where {T}
    
    ℓₘₐₓ = iszero(ℓₘₐₓ) ? workspace.ℓₘₐₓ : ℓₘₐₓ
    thread_buffers_0 = get_thread_buffers(T, 2ℓₘₐₓ+1)
    thread_buffers_2 = get_thread_buffers(T, 2ℓₘₐₓ+1)

    Vᵢⱼ = SpectralVector(alm2cl(
        workspace.mask_alm[name_i, PP], 
        workspace.mask_alm[name_j, TT]))
    𝐌 = SpectralArray(zeros(T, (ℓₘₐₓ+1, ℓₘₐₓ+1)))
    return loop_mcm_TE!(𝐌, ℓₘₐₓ, thread_buffers_0, thread_buffers_2, Vᵢⱼ)
end


"""
Compute a mode-coupling matrix.
"""
function mcm(workspace::SpectralWorkspace{T}, spec::MapType, f1_name::String, f2_name::String) where {T}
    if spec == TT
        return compute_mcm_TT(workspace, f1_name, f2_name)
    elseif spec == TE
        return compute_mcm_TE(workspace, f1_name, f2_name)
    elseif spec == ET
        return compute_mcm_ET(workspace, f1_name, f2_name)
    elseif spec == EE
        return compute_mcm_EE(workspace, f1_name, f2_name)
    else
        throw(ArgumentError("Spectrum requested is not yet implemented."))
    end
end
function mcm(workspace::SpectralWorkspace{T}, spec::MapType, 
             f1::PolarizedField{T}, f2::PolarizedField{T}) where {T}
    return mcm(workspace, spec, f1.name, f2.name)
end
function mcm(spec::MapType, f1::PolarizedField{T}, f2::PolarizedField{T}) where {T}
    workspace = SpectralWorkspace(f1, f2)
    return mcm(workspace, spec, f1, f2)
end


function spectra_from_masked_maps(map_1::Map{T}, map_2::Map{T}, 
                         factorized_mcm,
                         Bℓ_1::SpectralVector{T}, Bℓ_2::SpectralVector{T}) where T
    Cl_hat = alm2cl(map2alm(map_1), map2alm(map_2))
    Cl_hat[1:2] .= zero(T)  # set monopole and dipole to zero
    ldiv!(factorized_mcm, Cl_hat)
    return Cl_hat ./ (Bℓ_1.parent .* Bℓ_2.parent)
end


function spectra_from_masked_maps(
        alm_1::Alm{Complex{T},Array{Complex{T},1}}, alm_2::Alm{Complex{T},Array{Complex{T},1}}, 
        factorized_mcm,
        Bℓ_1::SpectralVector{T}, Bℓ_2::SpectralVector{T}) where T
    Cl_hat = alm2cl(alm_1, alm_2)
    Cl_hat[1:2] .= zero(T)  # set monopole and dipole to zero
    ldiv!(factorized_mcm, Cl_hat)
    return Cl_hat ./ (Bℓ_1.parent .* Bℓ_2.parent)
end

