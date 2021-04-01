import LinearAlgebra: inv, lu, lutype, copy_oftype, \
using Base: @propagate_inbounds


struct SpectralArray{T,N,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::OffsetArray{T,N,AA}
end

@forward SpectralArray.parent (Base.getindex, Base.setindex, Base.setindex!,
    Base.size, Base.iterate, Base.axes, Base.show, Base.in, Base.strides, Base.elsize,
    Base.dataids)

const SpectralVector{T,AA<:AbstractArray} = SpectralArray{T,1,AA}

SpectralVector(A::AbstractVector) = SpectralArray(OffsetArray(A, 0:(length(A)-1)))

# UnitRange slicing yields the parent array
@propagate_inbounds Base.getindex(x::SpectralArray, r::Vararg{UnitRange}) =
    (x.parent[r...])

# other slicing yields a SpectralArray
@propagate_inbounds Base.getindex(x::SpectralArray, r::Vararg{AbstractRange}) =
    SpectralArray(x.parent[r...])

"""
    SpectralArray(A::AbstractArray, [ranges])

A renamed OffsetArray. By default, it produces a 0-indexed array.
"""
function SpectralArray(A::AbstractArray{T,N}) where {T,N}
    SpectralArray{T,N,typeof(A)}(OffsetArray(A, map(x->-1, size(A))))
end

function SpectralArray(A::AA, offsets) where {T,N,AA<:AbstractArray{T,N}}
    SpectralArray{T,N,AA}(OffsetArray(A, offsets))
end

function SpectralVector(A::AA, offsets) where {T,AA<:AbstractArray{T,1}}
    SpectralArray{T,1,AA}(OffsetArray(A, offsets))
end

@inline SpectralArray(A::AbstractArray, inds::Vararg) = SpectralArray(OffsetArray(A, inds))

Base.parent(A::SpectralArray) = parent(A.parent)

Base.IndexStyle(::Type{OA}) where {OA<:SpectralArray} = IndexStyle(parenttype(OA))
parenttype(::Type{SpectralArray{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::SpectralArray) = parenttype(typeof(A))


function Base.similar(A::SpectralArray)
    return SpectralArray(similar(parent(A)), A.parent.offsets)
end
# function similar_SA(A::AbstractArray, ::Type{T}, inds::Tuple) where T
#     B = similar(A, T, map(OffsetArray._indexlength, inds))
#     return SpectralArray(OffsetArray(B, map(OffsetArray._offset, axes(B), inds)))
# end
Base.BroadcastStyle(::Type{<:SpectralArray}) = Broadcast.ArrayStyle{SpectralArray}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SpectralArray}}, ::Type{ElType}) where ElType
    SpectralArray(similar(Array{ElType}, axes(bc)))
end
Broadcast.broadcast_unalias(dest::SpectralArray, src::SpectralArray) =
    parent(dest) === parent(src) ? src : Broadcast.unalias(dest, src)

"""
    spectralzeros(size1, size2, ...)
    spectralzeros(range1, range2, ...)

Utility function for generating a SpectralArray by passing arguments of
ranges or sizes, just like zeros.
"""
spectralzeros(r::Vararg{Int}) = SpectralArray(zeros(r), map(x->-1, r))
spectralzeros(r::Vararg{AbstractRange}) = SpectralArray(zeros(r))


"""
    spectralones(size1::Int, size2::Int, ...)
    spectralones(range1::AbstractRange, range2::AbstractRange, ...)

Utility function for generating a SpectralArray by passing arguments of
ranges or sizes, just like ones.
"""
spectralones(r::Vararg{Int}) = SpectralArray(ones(r), map(x->-1, r))
spectralones(r::Vararg{AbstractRange}) = SpectralArray(ones(r))


function LinearAlgebra.inv(A::SpectralArray{T,2}) where T
    return SpectralArray(inv(parent(A)), A.parent.offsets)
end

# function LinearAlgebra.lu(A::SpectralArray{T,2}, pivot::Union{Val{false}, Val{true}}=Val(true);
#             check::Bool = true) where T
#     S = lutype(T)
#     pA = parent(A)
#     F = lu!(copy_oftype(pA, S), pivot; check = check)
#     return SpectralFactorization(F, A.parent.offsets)
# end
