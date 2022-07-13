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

function forward_problem(xx::AbstractArray, nx::Int, dx::Float64, xend::Float64, dt::Float64, tend::Float64, AT)
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

	rank = MPI.Comm_rank(comm)
	size = MPI.Comm_size(comm)

	# One step in time for one part of the domainor one thread 
	i1 = Int(round(rank / size * (nx+size))) - rank + 1
	i2 = Int(round((rank + 1) / size * (nx+size))) - rank

	nx_local = i2-i1

	print("Rank $(rank) of $(size) with domain limits $(i1), $(i2) and nx_local $(nx_local)\n")

	h = AT(zeros((nx_local+1, nt+1)))
	h_capital = AT(zeros((nx_local+1, nt+1)))

	D = AT(zeros(nx_local))
	phi = AT(zeros(nx_local))
	xarr = AT([(i-1)*dx for i in i1:i2])
	M = AT(zeros(nx_local+1))
	b = AT(zeros(nx_local+1))

	M .= M0 .- xarr .* M1 .+ xx[i1:i2]
	b .= 1.0 .+ bx .* xarr

	if rank == 0
		h[1,:] .= AT(ones(nt+1)) .* b[1]
		h_capital[1,:] .= h[1,:] .- b[1]
	end
	if rank == size - 1
		h[nx_local+1,:] .= AT(ones(nt+1)) .* b[nx_local+1]
		h_capital[nx_local+1,:] .= h[nx_local+1,:] .- b[nx_local+1]
	end
	h[:,1] .= AT(ones(nx_local+1)) .* b
	h_capital[:,1] .= h[:,1] .- b

	@inbounds for t in 1:nt

		D .= C .* ((h_capital[1:nx_local,t] .+ h_capital[2:nx_local+1,t]) ./ 2.0).^(n+2) .* ((h[2:nx_local+1,t] .- h[1:nx_local,t]) ./ dx).^(n-1)
		phi .= -D .* (h[2:nx_local+1,t] .- h[1:nx_local,t]) ./ dx
		h[2:nx_local,t+1] .= h[2:nx_local,t] .+ M[2:nx_local] .* dt .- dt/dx .* (phi[2:nx_local] .- phi[1:nx_local-1])

		h[2:nx_local,t+1] .= update_h.(h[2:nx_local,t+1], b[2:nx_local])
		h_capital[2:nx_local,t+1] .= h[2:nx_local,t+1] .- b[2:nx_local]

		send_mesg_left = Array{Float64}(undef, 1)
		recv_mesg_left = Array{Float64}(undef, 1)
		send_mesg_right = Array{Float64}(undef, 1)
		recv_mesg_right = Array{Float64}(undef, 1)

		if size > 1

			if rank == size-1

				fill!(send_mesg_left, Float64(phi[0]))
				sreq_left = MPI.Send(send_mesg_left, rank-1, 100 + rank, comm)
				rreq_left = MPI.Recv!(recv_mesg_left, rank-1, 100 + rank - 1, comm)

				h[1,t+1] = h[1,t] + M[1] * dt - dt/dx * (phi[1] - recv_mesg_left[1])
				h[1,t+1] = update_h(h[1,t+1], b[1])
				h_capital[1,t+1] = h[1,t+1] - b[1]

			elseif rank == 0

				fill!(send_mesg_right, Float64(phi[nx_local]))
				sreq_right = MPI.Send(send_mesg_right, rank+1, 100 + rank, comm)
				rreq_right = MPI.Recv!(recv_mesg_right, rank+1, 100 + rank + 1, comm)

				h[nx_local+1,t+1] = h[nx_local+1,t] + M[nx_local+1] * dt - dt/dx * (recv_mesg_right[1] - phi[nx_local])
				h[nx_local+1,t+1] = update_h(h[nx_local+1,t+1], b[nx_local+1])
				h_capital[nx_local+1,t+1] = h[nx_local+1,t+1] - b[nx_local+1]

			else

				fill!(send_mesg_right, Float64(phi[nx_local]))
				sreq_right = MPI.Send(send_mesg_right, rank+1, 100 + rank, comm)
				rreq_right = MPI.Recv!(recv_mesg_right, rank+1, 100 + rank + 1, comm)

				h[nx_local+1,t+1] = h[nx_local+1,t] + M[nx_local+1] * dt - dt/dx * (recv_mesg_right[1] - phi[nx_local])
				h[nx_local+1,t+1] = update_h(h[nx_local+1,t+1], b[nx_local+1])
				h_capital[nx_local+1,t+1] = h[nx_local+1,t+1] - b[nx_local+1]

				fill!(send_mesg_left, Float64(phi[0]))
				sreq_left = MPI.Send(send_mesg_left, rank-1, 100 + rank, comm)
				rreq_left = MPI.Recv!(recv_mesg_left, rank-1, 100 + rank - 1, comm)

				h[1,t+1] = h[1,t] + M[1] * dt - dt/dx * (phi[1] - recv_mesg_left[1])
				h[1,t+1] = update_h(h[1,t+1], b[1])
				h_capital[1,t+1] = h[1,t+1] - b[1]	

			end		

		end

		MPI.Barrier(comm)

	end

	if size > 1

		if rank == 0
			V_local = sum(h_capital[1:nx_local,nt+1].*dx) + 0.5*h_capital[nx_local+1,nt+1]*dx
		elseif rank == size-1
			V_local = 0.5*h_capital[1,nt+1]*dx + sum(h_capital[2:nx_local+1,nt+1].*dx)
		else
			V_local = 0.5*h_capital[1,nt+1]*dx + sum(h_capital[2:nx_local,nt+1].*dx) + 0.5*h_capital[nx_local+1,nt+1]*dx
		end

		# send_mesg = Array{Float64}(undef, 1)
		# recv_mesg = Array{Float64}(undef, 1)

		# if rank > 0
		# 	fill!(send_mesg, Float64(V_local))
		# 	sreq = MPI.Send(send_mesg, 0, 200 + rank, comm)
		# else
		# 	for i in 1:size-1
		# 		rreq = MPI.Recv!(recv_mesg, i, 200 + i, comm)
		# 		V_local = V_local + recv_mesg[1]
		# 	end 
		# end 

	else

		V_local = sum(h_capital[1:nx_local+1,nt+1].*dx)

	end

	return V_local

end

MPI.Init()
comm = MPI.COMM_WORLD

dx = 1.0
xend = 30.0
dt = 1/12.0
tend = 100.0

nx = Int(round(xend/dx))
xx = zeros(nx+1)
∂V_∂xx=zero(xx)
V = forward_problem(xx,nx,dx,xend,dt,tend, Array)

print("$(MPI.Comm_rank(comm)), V = $(V)\n")

autodiff(forward_problem, Active, Duplicated(xx, ∂V_∂xx), nx, dx, xend, dt, tend, Array)

print("$(MPI.Comm_rank(comm)), ∂V_∂xx = $(∂V_∂xx)\n")

MPI.Finalize()
