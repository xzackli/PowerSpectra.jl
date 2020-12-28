using Test

@show Threads.nthreads()
include("test_util.jl")
include("test_mcm.jl")
include("test_covmat.jl")
