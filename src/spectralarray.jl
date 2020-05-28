"""
A SpectralArray is just a 0-based array.
"""
const SpectralVector{T,AA<:AbstractArray} = OffsetArray{T,1,AA}
const SpectralArray{T,N,AA<:AbstractArray} = OffsetArray{T,N,AA}

const ArrayInitializer = Union{UndefInitializer, Missing, Nothing}
SpectralVector(A::AbstractVector) = OffsetArray(A, 0:(length(A)-1))
SpectralVector{T}(init::ArrayInitializer, 
                  arraysize::Int) where {T} = OffsetArray{T}(init, 0:(arraysize-1))

function SpectralArray(A::AbstractArray{T,N}) where {T,N}
    SpectralArray{T,N,typeof(A)}(A, map(x->-1, size(A)))
end
SpectralArray{T,N}(init::ArrayInitializer, sizes::NTuple{N, Int}) where {T,N} =
    SpectralArray(Array{T,N}(init, sizes), map(x->-1, sizes))
SpectralArray{T}(init::ArrayInitializer, sizes::NTuple{N,Int}) where {T,N} = SpectralArray{T,N}(init, sizes)
SpectralArray{T,N}(init::ArrayInitializer, sizes::Vararg{Int,N}) where {T,N} = SpectralArray{T,N}(init, sizes)
SpectralArray{T}(init::ArrayInitializer, sizes::Vararg{Int,N}) where {T,N} = SpectralArray{T,N}(init, sizes)
