using CUDA
using BenchmarkTools
using Test

function gpu_add3!(y, x)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = gridDim().x * blockDim().x
	for i = index:stride:length(y)
		@inbounds y[i] += x[i]
	end
	return nothing
end

function bench_gpu3!(y, x, numblocks)
	CUDA.@sync begin
		@cuda threads=256 blocks=numblocks gpu_add3!(y, x)
	end
end

function main()
	N = 2^20
	numblocks = ceil(Int, N/256)

	x_d = CUDA.fill(1.0f0, N)
	y_d = CUDA.fill(2.0f0, N)
	@btime bench_gpu3!($y_d, $x_d, $numblocks)

	# Testing after benchmarking never gives correct results
	fill!(y_d, 2)
	@cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d)
	@test all(Array(y_d) .== 3.0f0)
end

main()
