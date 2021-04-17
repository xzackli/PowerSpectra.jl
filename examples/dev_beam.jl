using Zygote
using FFTW
using LinearAlgebra

filter(x) = norm(eachindex(x)) .> 2  # remove some k mode

# bizarre k-space filter function
function f(x)
    y = fft(x)
    z = y .* filter(y)
    return sum(abs2.(ifft(z)))
end

a = rand(8,8)
f'(a)
##
using PowerSpectra
using PowerSpectra: get_thread_buffers, quickpol𝚵!
using Healpix
using WignerFamilies, Random

T = Float64
lmax = 10
buf1 = get_thread_buffers(T, 2lmax+1)
buf2 = get_thread_buffers(T, 2lmax+1)
𝐁 = zeros(lmax, lmax)

ω₁ = Alm(lmax, 4)
ω₂ = Alm(lmax, 4)
rand!(ω₁.alm)
rand!(ω₂.alm)

b₁ = SpectralVector(ones(50))
b₂ = SpectralVector(ones(50))

PowerSpectra.Ξsum(ω₁, ω₂, wigner3j_f(4,6,-2,0), wigner3j_f(4,6,-2,0) )

##

#             ν₁, ν₂, u₁, u₂, s₁, s₂, ω₁, ω₂, b₁, b₂
quickpol𝚵!(𝐁,  0,  0,  0,  0,  2,  2, ω₁, ω₂, b₁, b₂, buf1, buf2)

##
