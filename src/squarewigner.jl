
"""
The default global thread-safe dictionaries used to cache results of calculations.
They map the symmetrized form to r,s pairs.
"""
const MAX_J = Ref(200)

const Wigner3j = WignerSymbols.wigner_dicts(Float64)[1]  # shared by all threads
const Wigner6j = WignerSymbols.wigner_dicts(Float64)[2]  # shared by all threads

# imitating Base.Random.THREAD_RNGs, initialize new caches on each thread
const wigner_caches = BoundedWignerCache[]
@inline get_local_cache() = get_local_cache(Threads.threadid())
@noinline function get_local_cache(tid::Int)
    @assert 0 < tid <= length(wigner_caches)
    if @inbounds isassigned(wigner_caches, tid)
        @inbounds cache = wigner_caches[tid]
    else
        # all caches share the global Wigner3j and Wigner6j ThreadSafeDicts
        cache = BoundedWignerCache(Wigner3j, Wigner6j, MAX_J[])
        @inbounds wigner_caches[tid] = cache
    end
    return cache
end


"""
    wigner3j²(T::Type{<:Real}=Rational{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁)

Square of the Wigner-3j symbol.
"""
function wigner3j²(cache::BoundedWignerCache{Tdict}, T::Type{<:Real}, 
                   j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂) where Tdict <: Number

    # check angular momenta
    for (jᵢ,mᵢ) in ((j₁, m₁), (j₂, m₂), (j₃, m₃))
        WignerSymbols.ϵ(jᵢ, mᵢ) || throw(DomainError((jᵢ, mᵢ), "invalid combination (jᵢ, mᵢ)"))
    end
    # check triangle condition and m₁+m₂+m₃ == 0
    if !δ(j₁, j₂, j₃) || !iszero(m₁+m₂+m₃)
        return zero(T)
    end

    new_max_j = Int(ceil(max(abs(j₁), abs(j₂), abs(j₃))))
    if(cache.max_j[] < new_max_j)
        WignerSymbols.grow!(2 * new_max_j, cache)
    end

    # we reorder such that j₁ >= j₂ >= j₃ and m₁ >= 0 or m₁ == 0 && m₂ >= 0
    j₁, j₂, j₃, m₁, m₂, m₃, sgn = WignerSymbols.reorder3j(j₁, j₂, j₃, m₁, m₂, m₃)
    # TODO: do we also want to use Regge symmetries?
    α₁ = convert(Int, j₂ - m₁ - j₃ ) # can be negative
    α₂ = convert(Int, j₁ + m₂ - j₃ ) # can be negative
    β₁ = convert(Int, j₁ + j₂ - j₃ )
    β₂ = convert(Int, j₁ - m₁ )
    β₃ = convert(Int, j₂ + m₂ )


    # dictionary lookup or compute
    if haskey(cache.Wigner3j, (β₁, β₂, β₃, α₁, α₂))
        rs = cache.Wigner3j[(β₁, β₂, β₃, α₁, α₂)]
    else
        # get buffered variables 
        s1n, s1d, s2n = cache.numbuf, cache.denbuf, cache.s2n
        snum, rnum, sden, rden = cache.snum, cache.rnum, cache.sden, cache.rden

        # mutating versions of the functions in the main WignerSymbols.jl file
        WignerSymbols.Δ²!(cache, s1n, s1d, j₁, j₂, j₃)
        WignerSymbols.splitsquare!(sden, rden, s1d)
        WignerSymbols.primefactorial!(cache, s2n, (β₂, β₁ - α₁, β₁ - α₂, β₃, β₃ - α₁, β₂ - α₂))
        s2n_mul_s1n = WignerSymbols._vadd!(s1n, s2n)  # multiply s2n and s1n
        WignerSymbols.splitsquare!(snum, rnum, s2n_mul_s1n)  # split square and store in snum and rnum

        s = WignerSymbols._convert!(cache, cache.snumint[], snum) // WignerSymbols._convert!(cache, cache.sdenint[], sden)
        r = WignerSymbols._convert!(cache, cache.rnumint[], rnum) // WignerSymbols._convert!(cache, cache.rdenint[], rden)

        series = WignerSymbols.compute3jseries(cache, β₁, β₂, β₃, α₁, α₂)

        # multiply s by series
        Base.GMP.MPZ.mul!(s.num, series.num)
        Base.GMP.MPZ.mul!(s.den, series.den)

        # store the r and s pair into the dictionary. 
        rs = convert(BigFloat, r) * convert(BigFloat, s)^2
        cache.Wigner3j[(β₁, β₂, β₃, α₁, α₂)] = convert(Tdict, rs)
    end
    return convert(T, rs)
end
function wigner3j²(T::Type{<:Real}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂) where Tdict <: Number
    return wigner3j²(get_local_cache(), T, j₁, j₂, j₃, m₁, m₂, m₃)
end
