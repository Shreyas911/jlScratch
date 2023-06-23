using Printf
using Enzyme
using MPI

function forward_problem(V_local::AbstractArray , xx::AbstractArray, i1::Int, i2::Int)

	nx_local = i2-i1
	h_capital = zeros((nx_local+1,2))

	xx_local = xx[i1:i2]
	h_capital[1:nx_local+1,2] .= 1.0 .+ xx_local[1:nx_local+1]

	if size > 1
		if rank == 0
			V_local[1] = sum(h_capital[1:nx_local,2]) + 0.5*h_capital[nx_local+1,2]
		elseif rank == size-1
			V_local[1] = 0.5*h_capital[1,2] + sum(h_capital[2:nx_local+1,2])
		else
			V_local[1] = 0.5*h_capital[1,2] + sum(h_capital[2:nx_local,2]) + 0.5*h_capital[nx_local+1,2]
		end
	else
		V_local[1] = sum(h_capital[1:nx_local+1,2])
	end

	return
end

function forward_problem_no_issue(V_local::AbstractArray , xx::AbstractArray, i1::Int, i2::Int)

	nx_local = i2-i1
	h_capital = zeros((nx_local+1))

	xx_local = xx[i1:i2]
	h_capital[1:nx_local+1] .= 1.0 .+ xx_local[1:nx_local+1]

	if size > 1
		if rank == 0
			V_local[1] = sum(h_capital[1:nx_local]) + 0.5*h_capital[nx_local+1]
		elseif rank == size-1
			V_local[1] = 0.5*h_capital[1] + sum(h_capital[2:nx_local+1])
		else
			V_local[1] = 0.5*h_capital[1] + sum(h_capital[2:nx_local]) + 0.5*h_capital[nx_local+1]
		end
	else
		V_local[1] = sum(h_capital[1:nx_local+1])
	end

	return
end

### To reproduce the bug, run with n = 2 processes
### mpiexecjl -n 2 julia mpi_mt_glacier_simple.jl

MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

i1 = Int(round(rank / size * (4+size))) - rank + 1
i2 = Int(round((rank + 1) / size * (4+size))) - rank

print("rank = $(rank), i1 = $(i1), i2 = $(i2)\n")

### WORKS INCORRECTLY

xx = zeros(5)
∂V_∂xx=zeros(5)
dV = [1.0]
V = [0.0]

autodiff(forward_problem, Duplicated(V, dV), Duplicated(xx, ∂V_∂xx), i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
MPI.Reduce!(∂V_∂xx, MPI.SUM, 0, MPI.COMM_WORLD)

if rank == 0
	print("V = $V, Incorrect ∂V_∂xx = $(∂V_∂xx)\n")
end

MPI.Barrier(MPI.COMM_WORLD)

### WORKS CORRECTLY

xx = zeros(5)
∂V_∂xx=zeros(5)
dV = [1.0]
V = [0.0]

autodiff(forward_problem_no_issue, Duplicated(V, dV), Duplicated(xx, ∂V_∂xx), i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
MPI.Reduce!(∂V_∂xx, MPI.SUM, 0, MPI.COMM_WORLD)

if rank == 0
	print("V = $V, Correct ∂V_∂xx = $(∂V_∂xx)\n")
end

MPI.Finalize()