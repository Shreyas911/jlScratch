using CUDA
using BenchmarkTools

function add_broadcast!(y, x)
	CUDA.@sync y .+= x
	return
end

function main()
	N = 2^20
	x_d = CUDA.fill(1.0f0, N)
	y_d = CUDA.fill(2.0f0, N)
	@btime add_broadcast!($y_d, $x_d)
end

main()
