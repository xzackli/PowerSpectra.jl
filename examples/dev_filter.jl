using FastTransforms
using Plots

# F = sphrandn(Float64, 512, 1023); # convenience method
# heatmap(F, yflip = true, c=:vik)

##
F = sphrandn(Float64, 512, 1023); # convenience method
F[F .> 0] .= 1.0
P = plan_sph2fourier(F);
G = P * F

# F = ones(512, 1023)
# S = fourier2sph(F)
# heatmap(S, yflip = true, c=:vik)

##
heatmap(G, yflip = true, c=:vik)

##
