using Distributed
##
addprocs(16)
##
@everywhere begin
    using WignerSymbols
    function bench!(arr, lmax)
        @sync @distributed for i in 1:lmax
            for j in 1:lmax
                arr[i,j] = convert(Float64,
                    wigner3j(i, j, i+j, 0, 0, 0).signedsquare
                )
            end
        end
    end
end

##
using SharedArrays
arr = SharedArray{Float64,2}((1000,1000), init = S -> 0)
@time bench!(arr, 1000)

##
function bench_serial!(arr, lmax)
    for i in 1:lmax
        for j in 1:lmax
            arr[i,j] = convert(Float64,
                wigner3j(i, j, i+j, 0, 0, 0).signedsquare
            )
        end
    end
end
##
arr = Array{Float64,2}(undef, (500,500))
@time bench_serial!(arr, 500)
