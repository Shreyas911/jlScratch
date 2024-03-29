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

function forward_problem(V_local::AbstractArray , xx::AbstractArray, nx::Int, dx::Float64, xend::Float64, dt::Float64, tend::Float64, xarr::Vector{Float64}, i1::Int, i2::Int)
	rho = 920.0
	g = 9.2
	n = 3
	A = 1.e-16
	#dt = 1/12.0
	C = 2*A/(n+2)*(rho*g)^n*(1.e3)^n
	#tend = 1000.0
	bx = -0.0001
	M0 = .004
	M1 = 0.0002
	nt = Int(round(tend/dt))

	rank = MPI.Comm_rank(MPI.COMM_WORLD)
	size = MPI.Comm_size(MPI.COMM_WORLD)

	nx_local = i2-i1

	# print("Rank $(rank) of $(size) with domain limits $(i1), $(i2) and nx_local $(nx_local)\n")

	h = zeros((nx_local+1, nt+1))
	h_capital = zeros((nx_local+1, nt+1))

	D = zeros(nx_local)
	phi = zeros(nx_local)

	b = zeros(nx_local+1)
	x_local = xx[i1:i2]

	M = M0 .- xarr .* M1 .+ x_local
	b .= 1.0 .+ bx .* xarr

	if rank == 0
		h[1,:] .= ones(nt+1) .* b[1]
		h_capital[1,:] .= h[1,:] .- b[1]
	end
	if rank == size - 1
		h[nx_local+1,:] .= ones(nt+1) .* b[nx_local+1]
		h_capital[nx_local+1,:] .= h[nx_local+1,:] .- b[nx_local+1]
	end
	h[:,1] .= ones(nx_local+1) .* b
	h_capital[:,1] .= h[:,1] .- b

	for t in 1:nt

		D .= C .* ((h_capital[1:nx_local,t] .+ h_capital[2:nx_local+1,t]) ./ 2.0).^(n+2) .* ((h[2:nx_local+1,t] .- h[1:nx_local,t]) ./ dx).^(n-1)
		phi .= -D .* (h[2:nx_local+1,t] .- h[1:nx_local,t]) ./ dx
		h[2:nx_local,t+1] .= h[2:nx_local,t] .+ M[2:nx_local] .* dt .- dt/dx .* (phi[2:nx_local] .- phi[1:nx_local-1])

		h[2:nx_local,t+1] .= update_h.(h[2:nx_local,t+1], b[2:nx_local])
		h_capital[2:nx_local,t+1] .= h[2:nx_local,t+1] .- b[2:nx_local]

		send_mesg_left = [0.0]
		recv_mesg_left = [0.0]
		send_mesg_right = [0.0]
		recv_mesg_right = [0.0]

		if size > 1

			if rank == size-1

				fill!(send_mesg_left, phi[1])
				sreq_left = MPI.Send(send_mesg_left, rank-1, 100 + rank, MPI.COMM_WORLD)
				rreq_left = MPI.Recv!(recv_mesg_left, rank-1, 100 + rank - 1, MPI.COMM_WORLD)

				h[1,t+1] = h[1,t] + M[1] * dt - dt/dx * (phi[1] - recv_mesg_left[1])
				h[1,t+1] = update_h(h[1,t+1], b[1])
				h_capital[1,t+1] = h[1,t+1] - b[1]

			elseif rank == 0

				fill!(send_mesg_right, phi[nx_local])
				sreq_right = MPI.Send(send_mesg_right, rank+1, 100 + rank, MPI.COMM_WORLD)
				rreq_right = MPI.Recv!(recv_mesg_right, rank+1, 100 + rank + 1, MPI.COMM_WORLD)

				h[nx_local+1,t+1] = h[nx_local+1,t] + M[nx_local+1] * dt - dt/dx * (recv_mesg_right[1] - phi[nx_local])
				h[nx_local+1,t+1] = update_h(h[nx_local+1,t+1], b[nx_local+1])
				h_capital[nx_local+1,t+1] = h[nx_local+1,t+1] - b[nx_local+1]

			else

				fill!(send_mesg_right, phi[nx_local])
				sreq_right = MPI.Send(send_mesg_right, rank+1, 100 + rank, MPI.COMM_WORLD)
				rreq_right = MPI.Recv!(recv_mesg_right, rank+1, 100 + rank + 1, MPI.COMM_WORLD)

				h[nx_local+1,t+1] = h[nx_local+1,t] + M[nx_local+1] * dt - dt/dx * (recv_mesg_right[1] - phi[nx_local])
				h[nx_local+1,t+1] = update_h(h[nx_local+1,t+1], b[nx_local+1])
				h_capital[nx_local+1,t+1] = h[nx_local+1,t+1] - b[nx_local+1]

				fill!(send_mesg_left, phi[1])
				sreq_left = MPI.Send(send_mesg_left, rank-1, 100 + rank, MPI.COMM_WORLD)
				rreq_left = MPI.Recv!(recv_mesg_left, rank-1, 100 + rank - 1, MPI.COMM_WORLD)

				h[1,t+1] = h[1,t] + M[1] * dt - dt/dx * (phi[1] - recv_mesg_left[1])
				h[1,t+1] = update_h(h[1,t+1], b[1])
				h_capital[1,t+1] = h[1,t+1] - b[1]

			end
		end
	end

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

MPI.Init()

dx = 1.0
xend = 30.0
dt = 1/12.0
tend = 100.0

nx = Int(round(xend/dx))
xx = zeros(nx+1)
∂V_∂xx=zeros(nx+1)
V = [0.0]
rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

# One step in time for one part of the domainor one thread
i1 = Int(round(rank / size * (nx+size))) - rank + 1
i2 = Int(round((rank + 1) / size * (nx+size))) - rank
xarr = [(i-1)*dx for i in i1:i2]
forward_problem(V, xx, nx, dx, xend, dt, tend, xarr, i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
if rank == 0
	print("V = $(V)\n")
end

dV = [1.0]
V = [0.0]
autodiff(forward_problem, Duplicated(V, dV), Duplicated(xx, ∂V_∂xx), nx, dx, xend, dt, tend, xarr, i1, i2)
MPI.Reduce!(V, MPI.SUM, 0, MPI.COMM_WORLD)
MPI.Reduce!(∂V_∂xx, MPI.SUM, 0, MPI.COMM_WORLD)

if rank == 0
	print("V = $V, ∂V_∂xx = $(∂V_∂xx)\n")
end

MPI.Finalize()
