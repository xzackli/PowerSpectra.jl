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
