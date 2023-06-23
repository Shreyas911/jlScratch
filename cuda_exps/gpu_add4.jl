using CUDA
using BenchmarkTools
using Test

function gpu_add4!(y, x)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = gridDim().x * blockDim().x
	for i = index:stride:length(y)
		@inbounds y[i] += x[i]
	end
	return nothing
end

function bench_gpu4!(y, x)

	# Let launch configuration API figure out number of threads and blocks
	kernel = @cuda launch=false gpu_add4!(y, x)
	config = launch_configuration(kernel.fun)
	threads = min(length(y), config.threads)
	blocks = cld(length(y), threads)	

	CUDA.@sync begin
		kernel(y, x; threads, blocks)
	end
end

function main()
	N = 2^20
	x_d = CUDA.fill(1.0f0, N)
	y_d = CUDA.fill(2.0f0, N)

	# Let launch configuration API figure out number of threads and blocks
	kernel = @cuda launch=false gpu_add4!(y_d, x_d)
	config = launch_configuration(kernel.fun)
	threads = min(N, config.threads)
	blocks = cld(N, threads)	

	kernel(y_d, x_d; threads, blocks)
	@test all(Array(y_d) .== 3.0f0)

	# Benchmarking
	fill!(y_d, 2)
	@btime bench_gpu4!($y_d, $x_d)
end

main()
