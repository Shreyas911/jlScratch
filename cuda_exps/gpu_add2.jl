using CUDA
using BenchmarkTools
using Test

function gpu_add2!(y, x)
	index = threadIdx().x
	stride = blockDim().x
	for i = index:stride:length(y)
		@inbounds y[i] += x[i]
	end
	return nothing
end

function bench_gpu2!(y, x)
	CUDA.@sync begin
		@cuda threads=256 gpu_add2!(y, x)
	end
end

function main()
	N = 2^20
	x_d = CUDA.fill(1.0f0, N)
	y_d = CUDA.fill(2.0f0, N)
	@btime bench_gpu2!($y_d, $x_d)

	# Testing after benchmarking never gives correct results
	fill!(y_d, 2)
	@cuda threads=256 gpu_add2!(y_d, x_d)
	@test all(Array(y_d) .== 3.0f0)
end

main()
