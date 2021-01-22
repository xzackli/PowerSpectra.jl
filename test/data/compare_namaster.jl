## load in the data
ENV["OMP_NUM_THREADS"] = 6
using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
# using BenchmarkTools
hp = pyimport("healpy")
nmt = pyimport("pymaster")

##
data_dir = "/home/zequnl/.julia/dev/AngularPowerSpectra/test/data/"
# mask = readMapFromFITS(data_dir * "mask.fits", 1, Float64)
nside = 256
lmax = 3 * nside - 1

np = pyimport("numpy")
nawrapper_dir = "/home/zequnl/Projects/nawrapper/notebooks/"
cls = np.genfromtxt("$(nawrapper_dir)/data/example_cls.txt", unpack=true)
ell, cltt, clte, clee, clbb = cls[1,:], cls[2,:], cls[3,:], cls[4,:], cls[5,:]
nw = pyimport("nawrapper")
nltt = nw.get_Nl(theta_fwhm=20.0, sigma_T=150, l_max=lmax+1)
nlee = nltt ./ 100

n1 = hp.synfast(
        (nltt, nlee, nlee,
         np.zeros_like(cltt), np.zeros_like(cltt), np.zeros_like(cltt)),
        nside, verbose=false, pixwin=false, new=true)

##
flat_beam = SpectralVector(ones(3*nside))
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
mask1 = readMapFromFITS("$(data_dir)/example_mask_1.fits", 1, Float64)
mask2 = readMapFromFITS("$(data_dir)/example_mask_2.fits", 1, Float64)

##
flat_beam = SpectralVector(ones(3*nside))
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
m1 = PolarizedField("143_hm1", mask1, mask1, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
m2 = PolarizedField("143_hm2", mask2, mask2, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
workspace = SpectralWorkspace(m1, m2)

npix = nside2npix(nside)
map1_ = PolarizedMap{Float64, RingOrder, Vector{Float64}}(
    n1[1,:],
    n1[2,:],
    n1[3,:])
map2_ = deepcopy(map1_)


map1, map2 = deepcopy(map1_), deepcopy(map2_)
mask!(map1, mask1, mask1)
mask!(map2, mask2, mask2)
a1 = map2alm(map1)
a2 = map2alm(map2)

M_EE = mcm(workspace, "EE", "143_hm1", "143_hm2").parent
M_EB = mcm(workspace, "EB", "143_hm1", "143_hm2").parent
M = M_EE - M_EB * inv(M_EE) * M_EB
M[1,1], M[2,2] = 1.0, 1.0
# Cl_hat = alm2cl(a1[2], a2[2], lu(M.parent), flat_beam, flat_beam)
Cl_hat = alm2cl(a1[2], a2[2], M)

##
(M₊ - M₋ * inv(M₊) * M₋) .- (M_EE - M_EB * inv(M_EE) * M_EB)

##
M₊ .- M_EE[3:end,3:end]
##

M₋ * inv(M₊) * M₋ .- M_EB[3:end,3:end] * inv(M_EE[3:end,3:end]) * M_EB[3:end,3:end]


##


# num_ell = size(M_EE,1)
# M22 = zeros(2num_ell, 2num_ell)

# M22[1:num_ell,1:num_ell] .= M_EE
# M22[num_ell+1:2num_ell,num_ell+1:2num_ell] .= M_EE
# M22[1:num_ell,num_ell+1:2num_ell] .= M_EB
# M22[num_ell+1:2num_ell,1:num_ell] .= M_EB

##
M22 = AngularPowerSpectra.mcm22(m1, m2)
ĉ_EE = alm2cl(a1[2], a2[2])
ĉ_BB = alm2cl(a1[3], a2[3])
ctot = qr(M22, Val(true)) \ vcat(ĉ_EE, ĉ_BB)
c_EE = ctot[1:num_ell]
c_BB = ctot[num_ell+1:2num_ell];

##


##

map1, map2 = deepcopy(map1_), deepcopy(map2_)
b = nmt.NmtBin.from_nside_linear(nside, 1)
# f_1 = nmt.NmtField(mask1.pixels, [map1_.i.pixels])
# f_2 = nmt.NmtField(mask2.pixels, [map2_.i.pixels])
f_1 = nmt.NmtField(mask1.pixels, [map1.q.pixels, map1.u.pixels])
f_2 = nmt.NmtField(mask2.pixels, [map2.q.pixels, map2.u.pixels])

w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f_1, f_2, b)
cl_coupled = nmt.compute_coupled_cell(f_1, f_2)
# cl_coupled[2:end, :] .= 0.0
cl_decoupled = w.decouple_cell(cl_coupled)

cl_coupled[2:end, :] .= 0.0
cl_decoupled_EEONLY = w.decouple_cell(cl_coupled)

##
clf()
plot(( c_EE[3:end] ./ cl_decoupled[1,:]), "-")
plot(( c_BB[3:end] ./ cl_decoupled[4,:]), "-")
xlim(0,150)
# plot(cl_decoupled[4,:])
# xlim(0,150)
# ylim(0,0.0001)
# plot(nlee)
# plot(cl_decoupled)
# yscale("log")
gcf()

##

##
clf()
# plot(cl_coupled[1,3:end] ./ Cl_hat[3:end] )

plot(aps_decoupled[1:end] ./ cl_decoupled_EEONLY[1,1:end], "-" )
# ylim(0,2)
xlim(0,100)
# ylim(-1e-2, 1e-2)
# plot(cl_nmt[1,:])
# plot(Cl_hat[3:end])
# xlim(0,100)

# yscale("log")
gcf()

##

M_EE = mcm(workspace, "EE", "143_hm1", "143_hm2")
M_EB = mcm(workspace, "EB", "143_hm1", "143_hm2")

##
# mcm_aps = M.parent
M₊ = w.get_coupling_matrix()[1:4:4lmax+1, 1:4:4lmax+1][3:end,3:end]
M₋ = w.get_coupling_matrix()[1:4:4lmax+1, 4:4:4lmax+4][3:end,3:end]

MCM_nmt = M₊ - M₋ * inv(M₊) * M₋
##

# maximum(abs.(M₊ .- M_EE.parent[3:end,3:end]))
clf()
plt.imshow(M₊ ./ M_EE.parent[3:end,3:end])
gcf()

##
clf()
cl_bespoke = deepcopy(cl_coupled[1,3:end])
# linalg = pyimport("scipy.linalg")
# cl_bespoke = linalg.lu_solve(linalg.lu_factor(
#     mcm_nmt[3:end,3:end]
#     ), b=cl_bespoke)

@time ldiv!(lu(MCM_nmt), cl_bespoke)

# plot(cl_bespoke)
# plot( cl_decoupled_EEONLY[1,:])

# ylim(0,0.00004)
plot(cl_bespoke ./ cl_decoupled_EEONLY[1,:], "-")
# ylim(0.8,1.2)
xlim(0,30)
gcf()

##
clf()
plot(diag(M₋,4) ./ diag(M₊,4))
gcf()
##
