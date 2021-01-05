# a zero-based array for not having to keep track of off-by-one stuff
const SpectralVector{T,AA<:AbstractArray} = OffsetArray{T,1,AA}
const SpectralArray{T,N,AA<:AbstractArray} = OffsetArray{T,N,AA}

const ArrayInitializer = Union{UndefInitializer, Missing, Nothing}
SpectralVector(A::AbstractVector) = OffsetArray(A, 0:(length(A)-1))
SpectralVector{T}(init::ArrayInitializer,
                  arraysize::Int) where {T} = OffsetArray{T}(init, 0:(arraysize-1))

function SpectralArray(A::AbstractArray{T,N}) where {T,N}
    SpectralArray{T,N,typeof(A)}(A, map(x->-1, size(A)))
end

LinearAlgebra.lu(a::SpectralArray{T,2,AA}) where {T,AA} = LinearAlgebra.lu(a.parent::AA)
