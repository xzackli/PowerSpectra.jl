import LinearAlgebra: inv
import Base: parent, similar

# an OffsetArray that defaults to zero-based indexing.
struct SpectralArray{T,N,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::OffsetArray{T,N,AA}
end

@forward SpectralArray.parent (Base.getindex, Base.setindex, Base.setindex!,
    Base.size, Base.iterate, Base.axes, Base.show)

const SpectralVector{T,AA<:AbstractArray} = SpectralArray{T,1,AA}

# """Zero-based vector for representing spectra."""
SpectralVector(A::AbstractVector) = SpectralArray(OffsetArray(A, 0:(length(A)-1)))

"""Zero-based array for representing spectra matrices."""
function SpectralArray(A::AbstractArray{T,N}) where {T,N}
    SpectralArray{T,N,typeof(A)}(OffsetArray(A, map(x->-1, size(A))))
end

function SpectralArray(A::AbstractArray{T,N}, offsets) where {T,N}
    SpectralArray(OffsetArray(A, offsets))
end

function SpectralVector(A::AbstractArray{T,1}, offsets) where {T,N}
    SpectralVector(SpectralVector(A, offsets))
end

Base.parent(A::SpectralArray) = parent(A.parent)

Base.similar(A::SpectralArray, ::Type{T}, dims::Dims) where T =
    similar_SA(parent(A), T, dims)
function similar_SA(A::AbstractArray, ::Type{T}, inds::Tuple) where T
    B = similar(A, T, map(_indexlength, inds))
    return SpectralArray(OffsetArray(B, map(_offset, axes(B), inds)))
end
Base.BroadcastStyle(::Type{<:SpectralArray}) = Broadcast.ArrayStyle{SpectralArray}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SpectralArray}}, ::Type{ElType}) where ElType
    SpectralArray(similar(Array{ElType}, axes(bc)))
end


# function LinearAlgebra.lu(a::SpectralArray{T,2,AA}) where {T,AA}
#     return LinearAlgebra.lu(a.parent::AA)
# end
function LinearAlgebra.inv(a::SpectralArray{T,2,AA}) where {T,AA}
    return SpectralArray(LinearAlgebra.inv(a.parent.parent::AA),
        a.parent.offsets)
end
