module AngularPowerSpectra

import ThreadPools: @qthreads
import Base.Threads: @threads
import UnsafeArrays: uview, UnsafeArray
import ThreadSafeDicts: ThreadSafeDict
import DataStructures: DefaultDict
import Combinatorics: permutations, combinations, with_replacement_combinations
import Healpix: Map, PolarizedMap, Alm, RingOrder, alm2cl, map2alm, numberOfAlms,
    RingInfo, getringinfo!, almIndex, alm2map
import WignerFamilies: wigner3j_f!, WignerF, WignerSymbolVector, get_wigner_array, 
    swap_triangular
import FillArrays: Zeros
import OffsetArrays: OffsetArray, OffsetVector
import LinearAlgebra: ldiv!, rdiv!, Hermitian
# import Distributions: MvNormal
using Random
# import LoopVectorization: @avx

export compute_mcm_TT, compute_spectra, beam_cov
export compute_covmat_TTTT, compute_covmat_EEEE
export Field, SpectralWorkspace, SpectralVector, SpectralArray
export PolarizedField
export CovarianceWorkspace
export compute_mcm_EE
export compute_mcm_TE, compute_mcm_ET
export binning_matrix
export generate_correlated_noise

include("util.jl")
include("spectralarray.jl")
include("workspace.jl")
include("modecoupling.jl")
include("covariance.jl")


end
