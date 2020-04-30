using PSPlanck
using WignerSymbols
import Base.Threads:@threads, nthreads
import ThreadPools:@qthreads
using ThreadsX
function get_square_index!(T::Type{<:Integer}, R, j₁, j₂, j₃, m₁, m₂, m₃)
    # j₁, j₂, j₃, m₁, m₂, m₃, sgn = WignerSymbols.reorder3j(j₁, j₂, j₃, m₁, m₂, m₃)
    # return  Regge_variables!(R, j₁, j₂, j₃, m₁, m₂, m₃)
    return PSPlanck.Rasch_Yu_index!(T, R, j₁, j₂, j₃, m₁, m₂, m₃)
end

# const c1 = Channel{Int128}(1024)
# const ind_dict = Dict{Int128, Float64}()

# const ind_arr = Int128[]



function ind_writer()
    empty!(ind_arr)
    println("BORN")
    while true
        ind = take!(c1)
        if ind < 0
            break
        end
        # push!(ind_arr, ind)
    end
    println("DYING")
end




function test_TT(Tind, n, m₁, m₂, m₃)
    # Tind = NTuple{5,Int}
    # Threads.@spawn ind_writer()
    # RY_indices = [Int128[] for i in 1:Threads.nthreads()]
    Rbuffers = [zeros(Int, (3,3)) for i in 1:5*Threads.nthreads()]  # prevent false sharing
    # RY_indices = Array{Vector{Tind},1}(undef, Threads.nthreads())
    inds = Vector{Vector{Tuple{Int, Int, Int, Tind}}}(undef, Threads.nthreads())
    Threads.@threads for i in 1:length(inds)
        inds[i] = Tuple{Int, Int, Int, Tind}[]
    end

    @threads for j₁ in PSPlanck.swap_triangular(1:n)
        tid = Threads.threadid()
        R = Rbuffers[tid*5]
        for j₂ in 1:j₁
            j₃_start = abs(j₁ - j₂)
            for j₃ in j₃_start:j₂
                if( iseven(j₁ + j₂ + j₃) )  # special m1 = m2 = m3 = 0
                    RY_index = get_square_index!(Tind, R, j₁, j₂, j₃, m₁, m₂, m₃)
                    # push!(RY_indices[tid], ind)
                    push!(inds[tid], (j₁, j₂, j₃, RY_index))
                end
            end
        end
    end
    inds = vcat(inds...)
    ThreadsX.sort!(inds, by= x -> x[4])

    num_wigner = length(inds)
    wigner_results = Vector{Float64}(undef, length(inds))
    println("ENTERING WIGNER STAGE")
    
    @qthreads for i in 1:num_wigner
        j₁, j₂, j₃ = inds[i][1:3]
        wigner_results[i] = PSPlanck.wigner3j²(Float64, j₁, j₂, j₃, m₁, m₂, m₃)
    end


    # result = PSPlanck.wigner3j²(Float64, j₁, j₂, j₃, m₁, m₂, m₃)
    return inds

                #
                    #     result = PSPlanck.wigner3j²(Float64, j₁, j₂, j₃, m₁, m₂, m₃)
                    #     # put!(c1, (ind, result))

    # put!(c1,(-1))
    # RY_indices
    # @time result = vcat(RY_indices...)

end
##

@time begin
    a1 = test_TT(Int128, 400, 0, 0, 0)
end