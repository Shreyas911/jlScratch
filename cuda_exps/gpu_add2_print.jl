using CUDA
using Test
using BenchmarkTools

function gpu_add2_print!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @cuprintln("thread $index, block $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function main()
	N = 2^20
	x_d = CUDA.fill(1.0f0, N)
	y_d = CUDA.fill(2.0f0, N)
	@cuda threads=16 gpu_add2_print!(y_d, x_d)
	# Apparently this is necessary to get the print output, similar to CUDA.@sync in essence
	synchronize()
end

main()
