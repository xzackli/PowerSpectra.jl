import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.io import ascii

# This script showcases the ability of namaster to compute Gaussian
# estimates of the covariance matrix.
# A similar example for flat-sky fields can be found in
# test/sample_covariance_flat.py

# HEALPix map resolution
nside = 256

# We start by creating some synthetic masks and maps with contaminants.
# Here we will focus on the auto-correlation of a spin-1 field.
# a) Read and apodize mask
mask1_T = hp.read_map("mask1_T.fits", verbose=False)
mask1_P = hp.read_map("mask1_P.fits", verbose=False)
mask2_T = hp.read_map("mask2_T.fits", verbose=False)
mask2_P = hp.read_map("mask2_P.fits", verbose=False)

# Let's now create a fictitious theoretical power spectrum to generate
# Gaussian realizations:
larr = np.arange(3*nside)
theory = ascii.read("theory.csv")
noise = ascii.read("noise.csv")
cl_tt = theory["cltt"]
cl_ee = theory["clee"]
cl_bb = 0*cl_tt
cl_te = theory["clte"]
cl_tb = 0*cl_tt
cl_eb = 0*cl_tt

nl_tt = noise["nltt"]
nl_ee = noise["nlee"]


# This routine generates a spin-0 and a spin-2 Gaussian random field based
# on these power spectra
def get_sample_field(maskT, maskP):
    mp_t, mp_q, mp_u = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te],
                                  nside, verbose=False)
    return nmt.NmtField(maskT, [mp_t]), nmt.NmtField(maskP, [mp_q, mp_u])


# We also copy this function from sample_workspaces.py. It computes
# power spectra given a pair of fields and a workspace.
def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)

    return cl_decoupled


# Let's generate one particular sample and its power spectrum.
print("Field")
f0_1, f2_1 = get_sample_field(mask1_T, mask1_P)
f0_2, f2_2 = get_sample_field(mask2_T, mask2_P)

# We will use 20 multipoles per bandpower.
b = nmt.NmtBin.from_nside_linear(nside, 1)
print("Workspace")
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(f0_1, f0_2, b)
w02 = nmt.NmtWorkspace()
w02.compute_coupling_matrix(f0_1, f2_2, b)
w20 = nmt.NmtWorkspace()
w20.compute_coupling_matrix(f2_1, f0_2, b)
w22 = nmt.NmtWorkspace()
w22.compute_coupling_matrix(f2_1, f2_2, b)
n_ell = 3 * nside

np.save("mcmTT", w00.get_coupling_matrix().reshape([3*nside, 1, 3*nside, 1])[:, 0, :, 0])
np.save("mcmTE", w02.get_coupling_matrix().reshape([3*nside, 2, 3*nside, 2])[:, 0, :, 0])
np.save("mcmET", w20.get_coupling_matrix().reshape([3*nside, 2, 3*nside, 2])[:, 0, :, 0])
np.save("mcmEE", w22.get_coupling_matrix().reshape([3*nside, 4, 3*nside, 4])[:, 0, :, 0])
# import sys
# sys.exit()

# Let's now compute the Gaussian estimate of the covariance!
print("Covariance")
# First we generate a NmtCovarianceWorkspace object to precompute
# and store the necessary coupling coefficients
# This is the time-consuming operation
# Note that you only need to do this once,
# regardless of spin
cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0_1, f0_2, f0_1, f0_2)
covar_00_00 = nmt.gaussian_covariance(cw,
                                      0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_tt + nl_tt],  # TT
                                      [cl_tt],  # TT
                                      [cl_tt],  # TT
                                      [cl_tt + nl_tt],  # TT
                                      w00, wb=w00, coupled=True).reshape(
                                          [n_ell, 1, n_ell, 1])
covar_TT_TT = covar_00_00[:, 0, :, 0]
np.save("covar_TT_TT", covar_TT_TT)



cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0_1, f0_2, f0_1, f2_2)
covar_00_02 = nmt.gaussian_covariance(cw, 0, 0, 0, 2,  # Spins of the 4 fields
                                      [cl_tt + nl_tt],  # TT
                                      [cl_te, cl_tb],  # TE, TB
                                      [cl_tt],  # TT
                                      [cl_te, cl_tb],  # TE, TB
                                      w00, wb=w02, coupled=True).reshape(
                                          [n_ell, 1, n_ell, 2])
covar_TT_TE = covar_00_02[:, 0, :, 0]
np.save("covar_TT_TE", covar_TT_TE)


cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0_1, f2_2, f0_1, f2_2)
covar_02_02 = nmt.gaussian_covariance(cw, 0, 2, 0, 2,  # Spins of the 4 fields
                                      [cl_tt + nl_tt],  # TT
                                      [cl_te, cl_tb],  # TE, TB
                                      [cl_te, cl_tb],  # ET, BT
                                      [cl_ee + nl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      w02, wb=w02, coupled=True).reshape(
                                          [n_ell, 2, n_ell, 2])
covar_TE_TE = covar_02_02[:, 0, :, 0]
covar_TE_TB = covar_02_02[:, 0, :, 1]
covar_TB_TE = covar_02_02[:, 1, :, 0]
covar_TB_TB = covar_02_02[:, 1, :, 1]
np.save("covar_TE_TE", covar_TE_TE)


cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0_1, f0_2, f2_1, f2_2)
covar_00_22 = nmt.gaussian_covariance(cw, 0, 0, 2, 2,  # Spins of the 4 fields
                                      [cl_te, cl_tb],  # TE, TB
                                      [cl_te, cl_tb],  # TE, TB
                                      [cl_te, cl_tb],  # TE, TB
                                      [cl_te, cl_tb],  # TE, TB
                                      w00, wb=w22, coupled=True).reshape(
                                          [n_ell, 1, n_ell, 4])
covar_TT_EE = covar_00_22[:, 0, :, 0]
covar_TT_EB = covar_00_22[:, 0, :, 1]
covar_TT_BE = covar_00_22[:, 0, :, 2]
covar_TT_BB = covar_00_22[:, 0, :, 3]
np.save("covar_TT_EE", covar_TT_EE)


cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0_1, f2_2, f2_1, f2_2)
covar_02_22 = nmt.gaussian_covariance(cw, 0, 2, 2, 2,  # Spins of the 4 fields
                                      [cl_te, cl_tb],  # TE, TB
                                      [cl_te, cl_tb],  # TE, TB
                                      [cl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      [cl_ee + nl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      w02, wb=w22, coupled=True).reshape(
                                          [n_ell, 2, n_ell, 4])
covar_TE_EE = covar_02_22[:, 0, :, 0]
covar_TE_EB = covar_02_22[:, 0, :, 1]
covar_TE_BE = covar_02_22[:, 0, :, 2]
covar_TE_BB = covar_02_22[:, 0, :, 3]
covar_TB_EE = covar_02_22[:, 1, :, 0]
covar_TB_EB = covar_02_22[:, 1, :, 1]
covar_TB_BE = covar_02_22[:, 1, :, 2]
covar_TB_BB = covar_02_22[:, 1, :, 3]
np.save("covar_TE_EE", covar_TE_EE)


cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f2_1, f2_2, f2_1, f2_2)
covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                      [cl_ee + nl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      [cl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      [cl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      [cl_ee + nl_ee, cl_eb,
                                       cl_eb, cl_bb],  # EE, EB, BE, BB
                                      w22, wb=w22, coupled=True).reshape(
                                          [n_ell, 4, n_ell, 4])
covar_EE_EE = covar_22_22[:, 0, :, 0]
covar_EE_EB = covar_22_22[:, 0, :, 1]
covar_EE_BE = covar_22_22[:, 0, :, 2]
covar_EE_BB = covar_22_22[:, 0, :, 3]
covar_EB_EE = covar_22_22[:, 1, :, 0]
covar_EB_EB = covar_22_22[:, 1, :, 1]
covar_EB_BE = covar_22_22[:, 1, :, 2]
covar_EB_BB = covar_22_22[:, 1, :, 3]
covar_BE_EE = covar_22_22[:, 2, :, 0]
covar_BE_EB = covar_22_22[:, 2, :, 1]
covar_BE_BE = covar_22_22[:, 2, :, 2]
covar_BE_BB = covar_22_22[:, 2, :, 3]
covar_BB_EE = covar_22_22[:, 3, :, 0]
covar_BB_EB = covar_22_22[:, 3, :, 1]
covar_BB_BE = covar_22_22[:, 3, :, 2]
covar_BB_BB = covar_22_22[:, 3, :, 3]
np.save("covar_EE_EE", covar_EE_EE)