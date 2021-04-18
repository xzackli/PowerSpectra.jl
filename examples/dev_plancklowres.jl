##
using Healpix
using PowerSpectra
using Plots

nside = 256
m₁ = PowerSpectra.planck256_polmap("100", "hm1")
m₂ = PowerSpectra.planck256_polmap("100", "hm2")
maskT₁ = PowerSpectra.planck256_mask("100", "hm1", :T)
maskP₁ = PowerSpectra.planck256_mask("100", "hm1", :P)
maskT₂ = PowerSpectra.planck256_mask("100", "hm2", :T)
maskP₂ = PowerSpectra.planck256_mask("100", "hm2", :P)

# convert to μK
scale!(m₁, 1e6)
scale!(m₂, 1e6)

## utility function for doing TEB mode decoupling
Cl = master(m₁, maskT₁, maskP₁, 
            m₂, maskT₂, maskP₂)
lmax = nside2lmax(256)
print(keys(Cl))

##
spec = :TE
bl = PowerSpectra.planck_beam_bl("100", "hm1", "100", "hm2", spec, spec; lmax=lmax)
ℓ = eachindex(bl)
plot( (ℓ.^2 / (2π)) .*  Cl[spec] ./ bl.^2, label="\$D_{\\ell}\$", xlim=(0,2nside) )
theory = PowerSpectra.planck_theory_Dl()
plot!(theory[spec], label="theory $(spec)")

##
spec = :EE
bl = PowerSpectra.planck_beam_bl("100", "hm1", "100", "hm2", spec, spec; lmax=lmax)
ℓ = eachindex(bl)
plot( (ℓ.^2 / (2π)) .*  Cl[spec] ./ bl.^2, label="\$D_{\\ell}\$", xlim=(0,2nside) )
theory = PowerSpectra.planck_theory_Dl()
plot!(theory[spec], label="theory $(spec)")


##