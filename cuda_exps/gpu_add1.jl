using CUDA
using BenchmarkTools
using Test

function gpu_add1!(y, x)
	for i = 1:length(y)
		@inbounds y[i] += x[i]
		#@cuprintln(y[i])
	end
	return nothing
end

function bench_gpu1!(y, x)
	CUDA.@sync begin
		@cuda gpu_add1!(y, x)
	end
end

function main()
	N = 20
	x_d = CUDA.fill(1.0f0, N)
	y_d = CUDA.fill(2.0f0, N)
	@btime bench_gpu1!($y_d, $x_d)
end

main()
