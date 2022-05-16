module PowerSpectra

import ThreadPools: @qthreads
import Base.Threads: @threads
import UnsafeArrays: uview
import Combinatorics: permutations, combinations, with_replacement_combinations
import Healpix: HealpixMap, PolarizedHealpixMap, Alm, RingOrder, alm2cl, map2alm, numberOfAlms,
    RingInfo, getringinfo!, almIndex, alm2map, nside2npix, pix2ang, pix2vecRing,
    readMapFromFITS, nest2ring
import Pixell: Enmap, pixareamap, fullringnum, getwcs
import WignerFamilies: wigner3j_f!, WignerF, WignerSymbolVector, get_wigner_array
import FillArrays: Zeros, Ones
import OffsetArrays: OffsetArray, OffsetVector
import OffsetArrays
import StaticArrays: SA
import IdentityRanges: IdentityRange

using BandedMatrices
using LinearAlgebra
using Random
using CSV, DataFrames
using Lazy: @forward
using ReferenceImplementations
using LazyArtifacts
using FITSIO
# import Distributions: MvNormal
# import LoopVectorization: @avx


include("util.jl")
include("spectralarray.jl")
include("blockspectralmatrix.jl")
include("workspace.jl")
include("modecoupling.jl")
include("covariance.jl")
include("beam.jl")
include("exampledata.jl")


export IdentityRange
export mcm, decouple_covmat, mask!, scale!, master
export @spectra
export SpectralVector, SpectralArray, BlockSpectralMatrix
export spectralzeros, spectralones
export channelindex
export fitdipole, subtract_monopole_dipole!, nside2lmax
export CovField
export CovarianceWorkspace, coupledcov
export binning_matrix, read_commented_header
export synalm, synalm!
export SpectrumName

end
