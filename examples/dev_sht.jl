using PlanckCov
using WignerSymbols

##
PlanckCov.greet()
print("oh yeahhh")

## 
x = wigner3j(1,1,0, 0,0)

##
using PyPlot
##
clf()

x = collect(1:0.1:8)
plt.plot(x, sin.(x))

gcf()
##
function Ξ_TT(T::Type{<:Real}, X, Y, ℓ₁, ℓ₂, ℓ_max::Int)

    Ξ_temp = zero(T)
    for ℓ₃ in 1:ℓ_max::Int
        Ξ_temp = Ξ_temp + (
            (2 * ℓ₃ + 1) / (4π) * 
            wigner3j(T, ℓ₁, ℓ₂, ℓ₃, 0, 0, 0)^2 
        )
    end

    return Ξ_temp
end

## 


##
@code_warntype Ξ_TT(Float64, "TT", "TT", 5, 6, 10)