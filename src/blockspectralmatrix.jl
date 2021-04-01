


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
    return BlockSpectralMatrix{T, m_blocks, n_blocks, typeof(𝐀)}(𝐀, m_ells, n_ells)
end

function Base.vcat(values::SA...)  where {T, SA<:SpectralArray{T}}
    𝐀 = vcat(map(A -> parent(A), values)...)
    m_blocks = length(values)
    n_blocks = 1
    m_ells = map(A->axes(A,1), values)
    n_ells = (axes(first(values),2),)
    return BlockSpectralMatrix{T, m_blocks, n_blocks, typeof(𝐀)}(
        𝐀, m_ells, n_ells)
end

function Base.hcat(values::SA...)  where {T, SA<:SpectralArray{T}}
    𝐀 = hcat(map(A -> parent(A), values)...)
    m_blocks = 1
    n_blocks = length(values)
    m_ells = (axes(first(values),1),)
    n_ells = map(A->axes(A,2), values)
    return BlockSpectralMatrix{T, m_blocks, n_blocks, typeof(𝐀)}(
        𝐀, m_ells, n_ells)
end

function Base.show(io::IO, m::MIME"text/plain",
        ba::BlockSpectralMatrix{T,M_BLOCKS,N_BLOCKS}) where {T,M_BLOCKS,N_BLOCKS}
    print(io, "BlockSpectralMatrix (column blocks=$(M_BLOCKS), row blocks=$(N_BLOCKS))\n")
    print(io, "row ℓ:    ")
    Base.show(io, m, ba.m_ells)
    print(io, "\ncolumn ℓ: ")
    Base.show(io, m, ba.n_ells)

    print(io, "\nparent: ")

    nrows, ncols = displaysize(io)
    Base.show(
        IOContext(io, :compact => true, :displaysize => (min(10,nrows),ncols)),
        m, parent(ba))
end


struct BlockSpectralFactorization{T,M_BLOCKS,N_BLOCKS,AA,
        F<:Factorization{T}} <: Factorization{T}
    parent::F
    m_ells::NTuple{M_BLOCKS, UnitRange{Int64}}
    n_ells::NTuple{N_BLOCKS, UnitRange{Int64}}
end

function LinearAlgebra.inv(
        F::BlockSpectralFactorization{T,M_BLOCKS,N_BLOCKS,AA}) where {T,M_BLOCKS,N_BLOCKS,AA}
    return BlockSpectralMatrix{T,M_BLOCKS,N_BLOCKS,AA}(inv(F.parent), F.m_ells, F.n_ells)
end

Base.parent(F::BlockSpectralFactorization) = F.parent


function BlockSpectralMatrix(A::SpectralArray{T,2,AA}) where {T,AA}
    return  BlockSpectralMatrix{T,1,1,AA}(parent(A),
        (UnitRange(axes(A,1)),), (UnitRange(axes(A,2)),))
end

function BlockSpectralMatrix(A::SpectralArray{T,1,AA}) where {T,AA}
    return  BlockSpectralMatrix{T,1,1,AA}(parent(A),
        (UnitRange(axes(A,1)),), (UnitRange(axes(A,2)),))
end

function LinearAlgebra.lu(A::SpectralArray{T,2,AA}) where {T,AA}
    F = lu(parent(A))
    return BlockSpectralFactorization{T,1,1,AA,typeof(F)}(F,
        (UnitRange(axes(A,1)),), (UnitRange(axes(A,2)),))
end

function (\)(F::BlockSpectralFactorization, B::SpectralVector{T}) where T
    x = parent(F) \ parent(B)
    SpectralVector(x, B.parent.offsets)
end

function (\)(F::BlockSpectralFactorization,
        B::BlockSpectralMatrix{T,M_BLOCKS,N_BLOCKS,AA}) where {T,M_BLOCKS,N_BLOCKS,AA}
    x = parent(F) \ parent(B)
    BlockSpectralMatrix{T,M_BLOCKS,N_BLOCKS,AA}(x, B.m_ells, B.n_ells)
end

function (\)(A::SpectralArray{T,2}, B::SpectralVector{T}) where T
    F = lu(A)
    x = parent(F) \ parent(B)
    @show B.parent.offsets
    SpectralVector(x, B.parent.offsets)
end

# function (\)(F::SpectralFactorization, B::SpectralVector{T}) where T
#     @assert F.offsets[2] == B.parent.offsets
#     Y =
#     ldiv!()
# end

# function (\)(A::SpectralArray, B::SpectralVector{T}) where T
#     return lu(A) \ B
# end
