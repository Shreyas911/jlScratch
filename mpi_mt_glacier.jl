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

function forward_problem(xx::AbstractArray, nx::Int, dx::Float64, xend::Float64, nt::Int, dt::Float64, tend::Float64, AT)

	xarr = [(i-1)*dx for i in 1:nx+1]

	rho = 920.0
	g = 9.2
	n = 3
	A = 1.e-16
	C = 2*A/(n+2)*(rho*g)^n*(1.e3)^n
	bx = -0.0001
	M0 = .004
	M1 = 0.0002
	M = M0 .- xarr .* M1 .+ xx
	b = 1.0 .+ bx .* xarr

	D = AT(zeros(nx))
	phi = AT(zeros(nx))

	h = AT(zeros((nx+1, nt+1)))
	h[1,:] = AT(ones(size(h[1,:]))) .* b[1]
	h[:,1] = AT(ones(size(h[:,1]))) .* b[1]
	h[nx+1,:] = AT(ones(size(h[nx+1,:]))) .* b[nx+1]

	h_capital = AT(zeros((nx+1, nt+1)))
	h_capital[1,:] = h[1,:] .- b[1]
	h_capital[nx+1,:] = h[nx+1,:] .- b[nx+1]
	h_capital[:,1] = h[:,1] - b

	MPI.Init()
	comm = MPI.COMM_WORLD
	N = MPI.Comm_size(comm)
	rank = MPI.Comm_rank(comm)
	V_local = AT(zeros(1))
	V_total = 0.0
	# One step in time for one part of the domainor one thread 
	i1 = Int(round(rank / N * (nx+N))) - rank + 2
	i2 = Int(round((rank + 1) / N * (nx+N))) - rank

	if rank == 0
		i1 = 2
	end
	if rank == N - 1
		i2 = nx
	end

	print("Rank $(rank) of $(N) with domain limits $(i1), $(i2)\n")


	@inbounds for t in 1:nt

		D[i1-1:i2] .= C .* ((h_capital[i1-1:i2,t] .+ h_capital[i1:i2+1,t]) ./ 2.0).^(n+2) .* ((h[i1:i2+1,t] .- h[i1-1:i2,t]) ./ dx).^(n-1)

		phi[i1-1:i2] .= -D[i1-1:i2] .* (h[i1:i2+1,t] .- h[i1-1:i2,t]) ./ dx
		h[i1:i2,t+1] .= h[i1:i2,t] .+ M[i1:i2] .* dt .- dt/dx .* (phi[i1:i2] .- phi[i1-1:i2-1])

		h[i1:i2,t+1] .= update_h.(h[i1:i2,t+1], b[i1:i2])
		h_capital[i1:i2,t+1] .= h[i1:i2,t+1] .- b[i1:i2]

		MPI.Barrier(comm)
	end

	V_local[1] = sum(h_capital[:,nt+1].*dx)

	MPI.Barrier(comm)

	#Communications
	if rank > 0
		MPI.Send(V_local, 0, rank, comm)
		return
	else
		V_total = V_local[1]
		for i in 1:N-1
			MPI.Recv!(V_local, i, i, comm)
			V_total = V_total + V_local[1]
		end
		return V_total
	end

	MPI.Finalize()
end

dx = 1.0
xend = 30.0
dt = 1/12.0
tend = 10.0
nx = Int(round(xend/dx))
nt = Int(round(tend/dt))
xx = zeros(nx+1)


@show V_ice = forward_problem(xx, nx, dx, xend, nt, dt, tend, Array)

∂V_∂xx=Array(zero(xx))
autodiff(forward_problem, Active, Duplicated(xx, ∂V_∂xx), nx, dx, xend, nt, dt, tend, Array)
println(∂V_∂xx)

