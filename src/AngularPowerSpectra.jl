module AngularPowerSpectra

import ThreadPools: @qthreads
import Base.Threads: @threads
import OffsetArrays: OffsetArray
import UnsafeArrays: uview
import DataStructures: DefaultDict
import Combinatorics: permutations, combinations, with_replacement_combinations
import Healpix: Map, PolarizedMap, Alm, RingOrder, alm2cl, map2alm, numberOfAlms
import WignerFamilies: wigner3j_f!, WignerF, WignerSymbolVector, get_wigner_array, 
    swap_triangular
import FillArrays: Zeros

export cov, Field, CovarianceWorkspace, PowerSpectrum


const PowerSpectrum{T,AA<:AbstractArray} = OffsetArray{T,1,AA}
PowerSpectrum(A::AbstractVector) = OffsetArray(A, 0:(length(A)-1))
PowerSpectrum{T}(init::Union{UndefInitializer, Missing, Nothing}, 
                 arraysize::Int) where {T} = OffsetArray{T}(init, 0:(arraysize-1))


include("field.jl")
include("covariance.jl")


end
