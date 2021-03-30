import LinearAlgebra: inv

# an OffsetArray that defaults to zero-based indexing.
struct SpectralArray{T,N,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::OffsetArray{T,N,AA}
end

@forward SpectralArray.parent (Base.getindex, Base.setindex, Base.setindex!,
    Base.size, Base.iterate, Base.axes, Base.show, Base.in, Base.strides, Base.elsize,
    Base.dataids)

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

@inline SpectralArray(A::AbstractArray, inds::Vararg) = SpectralArray(OffsetArray(A, inds))

Base.parent(A::SpectralArray) = parent(A.parent)

Base.IndexStyle(::Type{OA}) where {OA<:SpectralArray} = IndexStyle(parenttype(OA))
parenttype(::Type{SpectralArray{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::SpectralArray) = parenttype(typeof(A))


# Base.similar(A::SpectralArray, ::Type{T}, dims::Dims) where T =
#     similar_SA(parent(A), T, dims)
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

# function LinearAlgebra.lu(a::SpectralArray{T,2,AA}) where {T,AA}
#     return LinearAlgebra.lu(a.parent::AA)
# end
function LinearAlgebra.inv(a::SpectralArray{T,2,AA}) where {T,AA}
    return SpectralArray(LinearAlgebra.inv(a.parent.parent::AA),
        a.parent.offsets)
end


# a contiguous matrix with some ell information slapped on for safety
struct BlockSpectralMatrix{T,M_BLOCKS,N_BLOCKS,AA} <: AbstractArray{T,2}
    parent::AA
    m_ells::NTuple{M_BLOCKS, UnitRange{Int64}}
    n_ells::NTuple{N_BLOCKS, UnitRange{Int64}}
end
@forward BlockSpectralMatrix.parent (Base.getindex, Base.setindex, Base.setindex!,
    Base.size, Base.iterate, Base.axes, Base.show, Base.in, Base.strides, Base.elsize,
    Base.dataids)


Base.parent(A::BlockSpectralMatrix) = A.parent

function Base.hvcat(rows::Tuple{Vararg{Int}}, values::SA...)  where {T, SA<:SpectralArray{T}}
    𝐀 = hvcat(rows, map(A -> parent(A), values)...)
    m_blocks = length(rows)
    n_blocks = first(rows)

    m_ells = map(A->axes(A,1), values[1:m_blocks])
    n_ells = map(A->axes(A,2), values[1:n_blocks:end])
    return BlockSpectralMatrix{T, m_blocks, n_blocks, parenttype(SA)}(𝐀, m_ells, n_ells)
end

function Base.vcat(values::SA...)  where {T, SA<:SpectralArray{T}}
    𝐀 = vcat(map(A -> parent(A), values)...)
    m_blocks = length(values)
    n_blocks = 1
    m_ells = map(A->axes(A,1), values)
    n_ells = (axes(first(values),2),)
    return BlockSpectralMatrix{T, m_blocks, n_blocks, parenttype(SA)}(𝐀, m_ells, n_ells)
end

function Base.hcat(values::SA...)  where {T, SA<:SpectralArray{T}}
    𝐀 = hcat(map(A -> parent(A), values)...)
    m_blocks = 1
    n_blocks = length(values)
    m_ells = (axes(first(values),1),)
    n_ells = map(A->axes(A,2), values)
    return BlockSpectralMatrix{T, m_blocks, n_blocks, parenttype(SA)}(𝐀, m_ells, n_ells)
end

function Base.show(io::IO, m::MIME"text/plain",
        ba::BlockSpectralMatrix{T,M_BLOCKS,N_BLOCKS}) where {T,M_BLOCKS,N_BLOCKS}
    print("BlockSpectralMatrix (column blocks=$(M_BLOCKS), row blocks=$(N_BLOCKS))\n")
    print("row ℓ:    ")
    Base.show(io, m, ba.m_ells)
    print("\ncolumn ℓ: ")
    Base.show(io, m, ba.n_ells)
    print("\nparent: ")
    Base.show(io, m, parent(ba))
end

# function BlockSpectralMatrix(A₁₁::SA, A₁₂::SA,
#         A₂₁::SA, A₂₂::SA) where {T, SA<:SpectralArray{T,2}}
#     𝐀 = [
#         parent(A₁₁)  parent(A₁₂);
#         parent(A₂₁)  parent(A₂₂)
#     ]
#     m_ells = (axes(A₁₁,1), axes(A₂₁,1))
#     n_ells = (axes(A₁₁,2), axes(A₁₂,2))
# end
