using AngularPowerSpectra
using Healpix
using CSV
using Test
using DelimitedFiles
using LinearAlgebra
using PyCall
using PyPlot

hp = pyimport("healpy")
nmt = pyimport("pymaster")

##

nside = 256
map1 = readMapFromFITS("test/example_map.fits", 1, Float64)
mask = readMapFromFITS("test/example_mask.fits", 1, Float64)
flat_beam = SpectralVector(ones(3*nside))
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
@time mcm = compute_mcm_TT(workspace, "143_hm1", "143_hm2")
@time factorized_mcm = lu(Hermitian(mcm.parent))

f_0 = nmt.NmtField(mask.pixels, [map1.pixels])

ells = collect(0:3 * nside-1)  # Array of multipoles
weights = ones(size(ells))  # Array of weights
bpws = collect(0:3nside-1)
b = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=weights)

w = nmt.NmtWorkspace()
@time w.compute_coupling_matrix(f_0, f_0, b)
reference_mcm = Hermitian(w.get_coupling_matrix())
cl_coupled = nmt.compute_coupled_cell(f_0, f_0)
cl_00 = w.decouple_cell(cl_coupled)[1,:]


##

##

# @test all(reference_mcm .≈ mcm.parent)



##
using LinearAlgebra
nocv = alm2cl(map2alm(map1))
pseudo_cl = alm2cl(map2alm(map1 * mask))

clf()
plt.plot(pseudo_cl ./ cl_coupled[1,:])
gcf()


linalg = pyimport("scipy.linalg")
# pseudo_cl = linalg.lu_solve(linalg.lu_factor((), b=cl_coupled[1, :])
ldiv!(lu(mcm.parent), pseudo_cl)
# pseudo_cl .= max.(0.0, pseudo_cl)

##
clf()
plt.imshow((reference_mcm .- mcm.parent))
plt.colorbar()
gcf()

##
using DataFrames
data_dir = "/home/zequnl/.julia/dev/AngularPowerSpectra/notebooks/data/"
theory = CSV.File(data_dir * "theory.csv") |> DataFrame



##
using DelimitedFiles

extra = [
#     2 4 3
    5 9  7
    10 14 12
    15 19 17
    20 24 22
    25 29  27
]
binning = vcat(extra, readdlm("/tigress/zequnl/cmb/software/PSpipe/project/Planck/planck_spectra/binused.dat", Int))[1:88, :]
# binning = readdlm("/tigress/zequnl/cmb/software/PSpipe/project/Planck/planck_spectra/binused.dat", Int)
lb = (binning[:,1] .+ binning[:,2]) ./ 2
P = binning_matrix(binning[:,1], binning[:,2], ℓ -> ℓ; lmax=3nside);


##
clf()
plt.plot(lb, P * (nocv .* collect(0:767) .* hp.pixwin(nside)))
plt.plot(lb, P * (cl_00 .* b.get_effective_ells()), alpha=0.5)
plt.plot(lb, P * (pseudo_cl .* collect(0:767)), alpha=0.5)
plt.xlim(0,100)

gcf()

##
clf()
plt.imshow( log10.(w.get_coupling_matrix()) )
gcf()

##
clf()
plt.imshow( (w.get_coupling_matrix() .- Symmetric(w.get_coupling_matrix()))[1:20, 1:20]   )
plt.colorbar()
gcf()

##
np = pyimport("numpy")
np.where( (w.get_coupling_matrix() .- Symmetric(w.get_coupling_matrix())) .< 0.06 )