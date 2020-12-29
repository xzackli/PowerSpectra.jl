module AngularPowerSpectra

import ThreadPools: @qthreads
import Base.Threads: @threads
import UnsafeArrays: uview, UnsafeArray
import ThreadSafeDicts: ThreadSafeDict
# import DataStructures: DefaultDict
import Combinatorics: permutations, combinations, with_replacement_combinations
import Healpix: Map, PolarizedMap, Alm, RingOrder, alm2cl, map2alm, numberOfAlms,
    RingInfo, getringinfo!, almIndex, alm2map, nside2npix
import WignerFamilies: wigner3j_f!, WignerF, WignerSymbolVector, get_wigner_array
import FillArrays: Zeros
import OffsetArrays: OffsetArray, OffsetVector
import LinearAlgebra: ldiv!, rdiv!, Factorization
# import Distributions: MvNormal
using Random
using CSV, DataFrames
# import LoopVectorization: @avx

export mcm, decouple_covmat!, spectra_from_masked_maps
export compute_covmat_TTTT, compute_covmat_EEEE
export Field, SpectralWorkspace, SpectralVector, SpectralArray

export PolarizedField
export CovarianceWorkspace

export binning_matrix
export read_commented_header

include("util.jl")
include("spectralarray.jl")
include("workspace.jl")
include("modecoupling.jl")
include("covariance.jl")


end
