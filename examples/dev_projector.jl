##
using AngularPowerSpectra
using Healpix
using PyPlot
# using RationalRoots
# using BenchmarkTools
# using WignerSymbols



## 
for i in 1:5
    println(i)
end

##
clf()

x = collect(1:0.1:8)
plt.plot(x, sin.(x))
plt.xlabel("x axis")

gcf()
##

using WignerSymbols

@doc raw"""
    cov(...)

Evaluates the covariance between power spectra.

```math
\mathrm{Cov} ( C_{\ell}^{X_1 X_2 \, i, j}, C_{\ell'}^{Y_1 Y_2 \, p, q})
```

For example, 
```cov(:T, :T, :hm1_143, :hm2_143, 
       :T, :T, :hm1_143, :hm2_143)
````
"""
function cov(X₁, X₂, i, j, Y₁, Y₂, p, q)
    
end

function Ξ_TT(T::Type{<:Real}, X, Y, ℓ₁::Int, ℓ₂::Int)

    Ξ_temp = zero(T)
    ℓmin = abs(ℓ₁ - ℓ₂)
    ℓmax = ℓ₁ + ℓ₂
    for ℓ₃ in ℓmin:ℓmax
        Ξ_temp = Ξ_temp + (
            (2 * ℓ₃ + 1) / (4π) * 
            wigner3j(T, ℓ₁, ℓ₂, ℓ₃, 0, 0, 0)^2 
        )
    end

    return Ξ_temp
end

## 
Ξ_TT(Float64, (:T,:T), (:T,:T), 5, 6)

##
(:T,:T) == (:T,:T)

##
w = (
    ∅∅ = 0,
)

w[:∅∅]