using Printf
using Statistics
using Plots
using Enzyme
using MPI

@inline function update_h(h::Float64, b::Float64)
	if h < b
		h = b
	end
	return h
end

function forward_problem(V_local::AbstractArray , xx::AbstractArray, nx::Int, dx::Float64, xend::Float64, dt::Float64, tend::Float64, i1::Int, i2::Int)
	
	nt = Int(round(tend/dt))

	rank = MPI.Comm_rank(MPI.COMM_WORLD)
	size = MPI.Comm_size(MPI.COMM_WORLD)

	nx_local = i2-i1
	h_capital = zeros((nx_local+1, nt+1))

	xx_local = xx[i1:i2]
	M0 = .004
	M = M0 .+ xx_local

	h_capital[1:nx_local+1,nt+1] .= M[1:nx_local+1] .* dt 

	if size > 1
		if rank == 0
			V_local[1] = sum(h_capital[1:nx_local,nt+1].*dx) + 0.5*h_capital[nx_local+1,nt+1]*dx
		elseif rank == size-1
			V_local[1] = 0.5*h_capital[1,nt+1]*dx + sum(h_capital[2:nx_local+1,nt+1].*dx)
		else
			V_local[1] = 0.5*h_capital[1,nt+1]*dx + sum(h_capital[2:nx_local,nt+1].*dx) + 0.5*h_capital[nx_local+1,nt+1]*dx
		end
	else
		V_local[1] = sum(h_capital[1:nx_local+1,nt+1].*dx)
	end

	return
end

function forward_problem_no_issue(V_local::AbstractArray , xx::AbstractArray, nx::Int, dx::Float64, xend::Float64, dt::Float64, tend::Float64, i1::Int, i2::Int)
	
	nt = Int(round(tend/dt))

	rank = MPI.Comm_rank(MPI.COMM_WORLD)
	size = MPI.Comm_size(MPI.COMM_WORLD)

	nx_local = i2-i1
	h_capital = zeros((nx_local+1))

	xx_local = xx[i1:i2]
	M0 = .004
	M = M0 .+ xx_local

	h_capital[1:nx_local+1] .= M[1:nx_local+1] .* dt 

	if size > 1
		if rank == 0
			V_local[1] = sum(h_capital[1:nx_local].*dx) + 0.5*h_capital[nx_local+1]*dx
		elseif rank == size-1
			V_local[1] = 0.5*h_capital[1]*dx + sum(h_capital[2:nx_local+1].*dx)
		else
			V_local[1] = 0.5*h_capital[1]*dx + sum(h_capital[2:nx_local].*dx) + 0.5*h_capital[nx_local+1]*dx
		end
	else
		V_local[1] = sum(h_capital[1:nx_local+1].*dx)
	end

	return
end

### To reproduce the bug, run with n = 2 processes
### mpiexecjl -n 2 julia mpi_mt_glacier_simple.jl

MPI.Init()

### WORKS INCORRECTLY

dx = 3.0
xend = 30.0
dt = 1/12.0
tend = 1/12.0

nx = Int(round(xend/dx))
xx = zeros(nx+1)
∂V_∂xx=zeros(nx+1)
V = [0.0]

rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

i1 = Int(round(rank / size * (nx+size))) - rank + 1
i2 = Int(round((rank + 1) / size * (nx+size))) - rank

print("rank = $(rank), i1 = $(i1), i2 = $(i2)\n")

forward_problem(V, xx, nx, dx, xend, dt, tend, i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
if rank == 0
	print("V = $(V)\n")
end

dV = [1.0]
V = [0.0]
autodiff(forward_problem, Duplicated(V, dV), Duplicated(xx, ∂V_∂xx), nx, dx, xend, dt, tend, i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
MPI.Reduce!(∂V_∂xx, MPI.SUM, 0, MPI.COMM_WORLD)

if rank == 0
	print("V = $V, Incorrect ∂V_∂xx = $(∂V_∂xx)\n")
end

MPI.Barrier(MPI.COMM_WORLD)

### WORKS CORRECTLY

dx = 3.0
xend = 30.0
dt = 1/12.0
tend = 1/12.0

nx = Int(round(xend/dx))
xx = zeros(nx+1)
∂V_∂xx=zeros(nx+1)
V = [0.0]

rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

i1 = Int(round(rank / size * (nx+size))) - rank + 1
i2 = Int(round((rank + 1) / size * (nx+size))) - rank

print("rank = $(rank), i1 = $(i1), i2 = $(i2)\n")

forward_problem_no_issue(V, xx, nx, dx, xend, dt, tend, i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
if rank == 0
	print("V = $(V)\n")
end

dV = [1.0]
V = [0.0]
autodiff(forward_problem_no_issue, Duplicated(V, dV), Duplicated(xx, ∂V_∂xx), nx, dx, xend, dt, tend, i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
MPI.Reduce!(∂V_∂xx, MPI.SUM, 0, MPI.COMM_WORLD)

if rank == 0
	print("V = $V, Correct ∂V_∂xx = $(∂V_∂xx)\n")
end

MPI.Finalize()
