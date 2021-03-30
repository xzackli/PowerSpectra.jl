
# Projector function for TT. Goes into the mode-coupling matrix.
function Ξ_TT(𝐖::SpectralVector{T, AA},
              w3j²₀₀::WignerSymbolVector{T, Int},
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₀₀), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j²₀₀), lastindex(𝐖))
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₀₀[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end


# Projector function for EE. Goes into the mode-coupling matrix.
# Note that w3j² refers to the square of ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
function Ξ_EE(𝐖::SpectralVector{T, AA},
              w3j²₂₂::WignerSymbolVector{T, Int},
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₂₂), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j²₂₂), lastindex(𝐖))
    if isodd(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₂₂[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end

# Projector function for EE. Goes into the mode-coupling matrix.
# Note that w3j² refers to the square of ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
function Ξ_EB(𝐖::SpectralVector{T, AA},
              w3j²₂₂::WignerSymbolVector{T, Int},
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j²₂₂), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j²₂₂), lastindex(𝐖))
    if iseven(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j²₂₂[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end


# Projector function for TE. Goes into the mode-coupling matrix.
# Note that w3j₀₀₂₂ refers to ( ℓ ℓ₂ ℓ₃ 0 0 0 ) × ( ℓ ℓ₂ ℓ₃ 0 -2 2 )
function Ξ_TE(𝐖::SpectralVector{T, AA},
              w3j₀₀₂₂::WignerSymbolVector{T, Int},
              ℓ₁::Int, ℓ₂::Int) where {T, AA}
    Ξ = zero(T)
    ℓ₃_start = max(firstindex(w3j₀₀₂₂), firstindex(𝐖))
    ℓ₃_end = min(lastindex(w3j₀₀₂₂), lastindex(𝐖))
    if isodd(ℓ₁ + ℓ₂ + ℓ₃_start)
        ℓ₃_start += 1
    end
    @inbounds @simd for ℓ₃ ∈ ℓ₃_start:2:ℓ₃_end
        Ξ += (2ℓ₃ + 1) * w3j₀₀₂₂[ℓ₃] * 𝐖[ℓ₃]
    end
    return Ξ / (4π)
end

# use a view of a memory buffer and fill with wigner 3j
function fill_3j!(buffer::Array{T,N}, ℓ₁, ℓ₂, m₁, m₂) where {T,N}
    w = WignerF(T, ℓ₁, ℓ₂, m₁, m₂)  # set up the wigner recurrence
    buffer_view = uview(buffer, 1:length(w.nₘᵢₙ:w.nₘₐₓ))  # preallocated buffer
    w3j = WignerSymbolVector(buffer_view, w.nₘᵢₙ:w.nₘₐₓ)
    wigner3j_f!(w, w3j)  # deposit symbols into buffer
    return w3j
end

# inner MCM loop TT
function inner_mcm⁰⁰!(𝐌::SpectralArray{T,2}, lmax::Integer,
                      Vᵢⱼ::SpectralVector{T}) where {T}
    thread_buffers = get_thread_buffers(T, 2lmax+1)

    @qthreads for ℓ₁ in 2:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w3j²₀₀ = fill_3j!(buffer, ℓ₁, ℓ₂, 0, 0)
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


# inner MCM loop TE and TB
function inner_mcm⁰²!(𝐌::SpectralArray{T,2}, lmax::Integer,
                      Vᵢⱼ::SpectralVector{T}) where {T}
    thread_buffers_0 = get_thread_buffers(T, 2lmax+1)
    thread_buffers_2 = get_thread_buffers(T, 2lmax+1)
    @qthreads for ℓ₁ in 2:lmax
        tid = Threads.threadid()
        buffer0 = thread_buffers_0[tid]
        buffer2 = thread_buffers_2[tid]
        for ℓ₂ in ℓ₁:lmax
            w3j₀₀ = fill_3j!(buffer0, ℓ₁, ℓ₂, 0, 0)
            w3j₂₂ = fill_3j!(buffer2, ℓ₁, ℓ₂, -2, 2)
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


# inner MCM loop for spin 2, called "EE" in Planck notation
function inner_mcm⁺⁺!(𝐌::SpectralArray{T,2}, lmax::Integer,
                      Vᵢⱼ::SpectralVector{T}) where {T}
    thread_buffers = get_thread_buffers(T, 2lmax+1)

    @qthreads for ℓ₁ in 2:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w3j²₂₂ = fill_3j!(buffer, ℓ₁, ℓ₂, -2, 2)
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


# inner MCM loop for spin 2
function inner_mcm⁻⁻!(𝐌::SpectralArray{T,2}, lmax::Integer,
                      Vᵢⱼ::SpectralVector{T}) where {T}
    thread_buffers = get_thread_buffers(T, 2lmax+1)

    @qthreads for ℓ₁ in 2:lmax
        buffer = thread_buffers[Threads.threadid()]
        for ℓ₂ in ℓ₁:lmax
            w3j²₂₂ = fill_3j!(buffer, ℓ₁, ℓ₂, -2, 2)
            w3j²₂₂.symbols .= w3j²₂₂.symbols .^ 2  # square the symbols
            Ξ = Ξ_EB(Vᵢⱼ, w3j²₂₂, ℓ₁, ℓ₂)
            𝐌[ℓ₁, ℓ₂] = (2ℓ₂ + 1) * Ξ
            𝐌[ℓ₂, ℓ₁] = (2ℓ₁ + 1) * Ξ
        end
    end
    𝐌[0,0] = one(T)
    𝐌[1,1] = one(T)
    return 𝐌
end


"""
    mcm(spec::Symbol, alm₁::Alm{T}, alm₂::Alm{T}; lmax=nothing)

# Arguments:
- `spec::Symbol`: cross-spectrum, i.e. `:TE`
- `alm₁::Alm{T}`: first mask's spherical harmonic coefficients
- `alm₂::Alm{T}`: second mask's spherical harmonic coefficients

# Keywords
- `lmax=nothing`: maximum multipole for mode-coupling matrix

# Returns:
- `SpectralArray{T,2}`: the index where `val` is located in the `array`
"""
function mcm(spec::Symbol, alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}}; lmax=nothing) where T
    if isnothing(lmax)
        lmax = min(alm₁.lmax, alm₂.lmax)
    end
    Vᵢⱼ = SpectralVector(alm2cl(alm₁, alm₂))
    𝐌 = SpectralArray(zeros(T, (lmax+1, lmax+1)))

    if spec == :TT
        inner_mcm⁰⁰!(𝐌, lmax, Vᵢⱼ)
    elseif spec ∈ (:TE, :ET, :TB, :BT)
        inner_mcm⁰²!(𝐌, lmax, Vᵢⱼ)
    elseif spec == :EE
        inner_mcm⁺⁺!(𝐌, lmax, Vᵢⱼ)
    end
end

# convenience function
mcm(spec::Symbol, m₁::Map, m₂::Map; lmax=nothing) =
    mcm(spec, map2alm(m₁), map2alm(m₂); lmax=lmax)

# Workspace mode-coupling routines

# function compute_mcm_TT(workspace::SpectralWorkspace{T},
#                         name_i::String, name_j::String; lmax::Int=0) where {T}
#     lmax = iszero(lmax) ? workspace.lmax : lmax
#     Vᵢⱼ = SpectralVector(alm2cl(workspace.mask_alm[name_i, :TT], workspace.mask_alm[name_j, :TT]))
#     𝐌 = SpectralArray(zeros(T, (lmax+1, lmax+1)))
#     return loop_mcm_TT!(𝐌, lmax, Vᵢⱼ)
# end

# function compute_mcm_EE(workspace::SpectralWorkspace{T},
#                         name_i::String, name_j::String; lmax::Int=0) where {T}

#     lmax = iszero(lmax) ? workspace.lmax : lmax
#     Vᵢⱼ = SpectralVector(alm2cl(
#         workspace.mask_alm[name_i, :PP],
#         workspace.mask_alm[name_j, :PP]))
#     𝐌 = SpectralArray(zeros(T, (lmax+1, lmax+1)))
#     return loop_mcm_EE!(𝐌, lmax, Vᵢⱼ)
# end

# function compute_mcm_TE(workspace::SpectralWorkspace{T},
#                         name_i::String, name_j::String; lmax::Int=0) where {T}

#     lmax = iszero(lmax) ? workspace.lmax : lmax
#     thread_buffers_0 = get_thread_buffers(T, 2lmax+1)
#     thread_buffers_2 = get_thread_buffers(T, 2lmax+1)

#     Vᵢⱼ = SpectralVector(alm2cl(
#         workspace.mask_alm[name_i, :TT],
#         workspace.mask_alm[name_j, :PP]))
#     𝐌 = SpectralArray(zeros(T, (lmax+1, lmax+1)))
#     return loop_mcm_TE!(𝐌, lmax, thread_buffers_0, thread_buffers_2, Vᵢⱼ)
# end


# function compute_mcm_ET(workspace::SpectralWorkspace{T},
#                      name_i::String, name_j::String; lmax::Int=0) where {T}

#     lmax = iszero(lmax) ? workspace.lmax : lmax
#     thread_buffers_0 = get_thread_buffers(T, 2lmax+1)
#     thread_buffers_2 = get_thread_buffers(T, 2lmax+1)

#     Vᵢⱼ = SpectralVector(alm2cl(
#         workspace.mask_alm[name_i, :PP],
#         workspace.mask_alm[name_j, :TT]))
#     𝐌 = SpectralArray(zeros(T, (lmax+1, lmax+1)))
#     return loop_mcm_TE!(𝐌, lmax, thread_buffers_0, thread_buffers_2, Vᵢⱼ)
# end

# function compute_mcm_EB(workspace::SpectralWorkspace{T},
#                         name_i::String, name_j::String; lmax::Int=0) where {T}

#     lmax = iszero(lmax) ? workspace.lmax : lmax
#     Vᵢⱼ = SpectralVector(alm2cl(
#         workspace.mask_alm[name_i, :PP],
#         workspace.mask_alm[name_j, :PP]))
#     𝐌 = SpectralArray(zeros(T, (lmax+1, lmax+1)))
#     return loop_mcm_EB!(𝐌, lmax, Vᵢⱼ)
# end


# """
#     mcm(workspace::SpectralWorkspace{T}, spec::Symbol, f1_name::String, f2_name::String) where {T}

# # Arguments:
# - `workspace::SpectralWorkspace{T}`: stores the SHTs of the masks
# - `spec::String`: the spectrum to compute, such as "TT", "TE", or "EE"
# - `f1_name::String`: the name of the first field
# - `f2_name::String`: the name of the second field

# # Returns:
# - `SpectralArray{T,2}`: zero-indexed array containing the mode-coupling matrix

# # Examples
# ```julia
# m1 = CovField("field1", mask1_T, mask1_P)
# m2 = CovField("field2", mask2_T, mask2_P)
# workspace = SpectralWorkspace(m1, m2)
# 𝐌 = mcm(workspace, spec, "field1", "field2")
# ```
# """
# function mcm(workspace::SpectralWorkspace{T}, spec::String,
#              f1_name::String, f2_name::String) where {T}
#     if spec == "TT"
#         return compute_mcm_TT(workspace, f1_name, f2_name)
#     elseif spec == "TE"
#         return compute_mcm_TE(workspace, f1_name, f2_name)
#     elseif spec == "ET"
#         return compute_mcm_ET(workspace, f1_name, f2_name)
#     elseif spec == "EE"
#         return compute_mcm_EE(workspace, f1_name, f2_name)
#     elseif spec == "EB"
#         return compute_mcm_EB(workspace, f1_name, f2_name)
#     else
#         throw(ArgumentError("Spectrum requested is not implemented."))
#     end
# end
# function mcm(workspace::SpectralWorkspace{T}, spec::String,
#              f1::CovField{T}, f2::CovField{T}) where {T}
#     return mcm(workspace, spec, f1.name, f2.name)
# end
# function mcm(spec::String, f1::CovField{T}, f2::CovField{T}) where {T}
#     workspace = SpectralWorkspace(f1, f2)
#     return mcm(workspace, spec, f1, f2)
# end


# EXPERIMENTAL
# EE and BB with coupling between them!
# function mcm22(workspace, f1::CovField{T}, f2::CovField{T}) where {T}
#     M_EE = parent(mcm(workspace, "EE", f1.name, f2.name))
#     M_EB = parent(mcm(workspace, "EB", f1.name, f2.name))
#     num_ell = size(M_EE,1)
#     M22 = zeros(2num_ell, 2num_ell)

#     M22[1:num_ell,1:num_ell] .= M_EE
#     M22[num_ell+1:2num_ell,num_ell+1:2num_ell] .= M_EE
#     M22[1:num_ell,num_ell+1:2num_ell] .= M_EB
#     M22[num_ell+1:2num_ell,1:num_ell] .= M_EB

#     return M22  # probably need to do pivoted qr as this may be nearly rank deficient
# end
# function mcm22(workspace, f1_name::String, f2_name::String) where {T}
#     M_EE = parent(mcm(workspace, "EE", f1_name, f2_name))
#     M_EB = parent(mcm(workspace, "EB", f1_name, f2_name))
#     num_ell = size(M_EE,1)
#     M22 = zeros(2num_ell, 2num_ell)

#     M22[1:num_ell,1:num_ell] .= M_EE
#     M22[num_ell+1:2num_ell,num_ell+1:2num_ell] .= M_EE
#     M22[1:num_ell,num_ell+1:2num_ell] .= M_EB
#     M22[num_ell+1:2num_ell,1:num_ell] .= M_EB

#     return M22  # probably need to do pivoted qr as this may be nearly rank deficient
# end

# mcm22(f1, f2) = mcm22(SpectralWorkspace(f1, f2), f1, f2)

# i.e.
# ĉ_EE = alm2cl(a1[2], a2[2])
# ĉ_BB = alm2cl(a1[3], a2[3])
# ctot = qr(M22, Val(true)) \ vcat(ĉ_EE, ĉ_BB)
# c_EE = ctot[1:num_ell]
# c_BB = ctot[num_ell+1:2num_ell];


"""
    map2cl(...)

# Arguments:
- `map_1::Map{T}`: masked map
- `map_2::Map{T}`: masked map
- `factorized_mcm::Factorization`: lu(mode coupling matrix)
- `Bℓ_1::SpectralVector{T}`: beam associated with first map
- `Bℓ_2::SpectralVector{T}`: beam associated with second map

# Returns:
- `Array{T,1}`: spectrum
"""
function map2cl(
        map_1::Map{T}, map_2::Map{T}, factorized_mcm::Factorization,
        Bℓ_1::SpectralVector{T}, Bℓ_2::SpectralVector{T}) where T
    return alm2cl(map2alm(map_1), map2alm(map_2), factorized_mcm, Bℓ_1, Bℓ_2)
end

function map2cl(
        map_1::Map{T}, map_2::Map{T}, factorized_mcm::Factorization) where T
    Cl_hat = alm2cl(map2alm(map_1), map2alm(map_2))
    return alm2cl(map2alm(map_1), map2alm(map_2), factorized_mcm)
end


function alm2cl(
        alm_1::Alm{Complex{T},Array{Complex{T},1}}, alm_2::Alm{Complex{T},Array{Complex{T},1}},
        factorized_mcm::Factorization, Bℓ_1::SpectralVector{T}, Bℓ_2::SpectralVector{T}) where T
    Cl_hat = alm2cl(alm_1, alm_2, factorized_mcm)
    return Cl_hat ./ (parent(Bℓ_1) .* parent(Bℓ_2))
end


function alm2cl(alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}}, factorized_mcm::Factorization) where {T<:Number}
    Cl_hat = alm2cl(alm₁, alm₂)
    Cl_hat[1:2] .= zero(T)  # set monopole and dipole to zero
    ldiv!(factorized_mcm, Cl_hat)
    return Cl_hat
end

function alm2cl(alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}}, mcm::AbstractArray) where {T<:Number}
    return alm2cl(alm₁, alm₂, lu(mcm))
end

function alm2cl(alm₁::Alm{Complex{T}}, alm₂::Alm{Complex{T}}, mcm::OffsetArray) where {T<:Number}
    return alm2cl(alm₁, alm₂, lu(parent(mcm)))
end



function alm2cl(a1_E_B::Tuple{Alm, Alm}, a2_E_B::Tuple{Alm, Alm}, mcm)
    ĉ_EE = alm2cl(a1_E_B[1], a2_E_B[1])
    ĉ_BB = alm2cl(a1_E_B[2], a2_E_B[2])
    num_ell = size(ĉ_EE, 1)
    ctot = qr(mcm, Val(true)) \ vcat(ĉ_EE, ĉ_BB)
    c_EE = ctot[1:num_ell]
    c_BB = ctot[num_ell+1:2num_ell]
    return c_EE, c_BB
end


"""
    mask!(m::Map{T}, mask::Map{T}) where T

Convenience function for applying a mask to a map.
"""
function mask!(m::Map{T}, mask::Map{T}) where T
    m.pixels .*= mask.pixels
end

function mask!(m::PolarizedMap{T}, maskT::Map{T}, maskP::Map{T}) where T
    m.i.pixels .*= maskT.pixels
    m.q.pixels .*= maskP.pixels
    m.u.pixels .*= maskP.pixels
end
