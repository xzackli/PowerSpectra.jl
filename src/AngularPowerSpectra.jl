module AngularPowerSpectra

import ThreadPools: @qthreads
import Base.Threads: @threads
import Base: copyto!
import UnsafeArrays: uview, UnsafeArray
import ThreadSafeDicts: ThreadSafeDict
import DataStructures: DefaultDict
import Combinatorics: permutations, combinations, with_replacement_combinations
import Healpix: Map, PolarizedMap, Alm, RingOrder, alm2cl, map2alm, numberOfAlms,
    RingInfo, getringinfo!, almIndex, alm2map, nside2npix
import WignerFamilies: wigner3j_f!, WignerF, WignerSymbolVector, get_wigner_array
import FillArrays: Zeros
import OffsetArrays: OffsetArray, OffsetVector
using LinearAlgebra
# import Distributions: MvNormal
using Random
using CSV, DataFrames
# import LoopVectorization: @avx


include("util.jl")
include("spectralarray.jl")
include("workspace.jl")
include("modecoupling.jl")
include("covariance.jl")


export mcm, decouple_covmat!, mask!, map2cl, alm2cl
export compute_covmat_TTTT, compute_covmat_EEEE
export Field, SpectralWorkspace, SpectralVector, SpectralArray
export channelindex

export PolarizedField
export CovarianceWorkspace, coupled_covmat

export binning_matrix, read_commented_header
export synalm, synalm!


end
