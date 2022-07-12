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

	# 	# D[i1-1:i2] .= C .* ((h_capital[i1-1:i2,t] .+ h_capital[i1:i2+1,t]) ./ 2.0).^(n+2) .* ((h[i1:i2+1,t] .- h[i1-1:i2,t]) ./ dx).^(n-1)

	# 	# phi[i1-1:i2] .= -D[i1-1:i2] .* (h[i1:i2+1,t] .- h[i1-1:i2,t]) ./ dx
	# 	# h[i1:i2,t+1] .= h[i1:i2,t] .+ M[i1:i2] .* dt .- dt/dx .* (phi[i1:i2] .- phi[i1-1:i2-1])

	# 	# h[i1:i2,t+1] .= update_h.(h[i1:i2,t+1], b[i1:i2])
	# 	# h_capital[i1:i2,t+1] .= h[i1:i2,t+1] .- b[i1:i2]

		D .= C .* ((h_capital[1:nx_local,t] .+ h_capital[2:nx_local+1,t]) ./ 2.0).^(n+2) .* ((h[2:nx_local+1,t] .- h[1:nx_local,t]) ./ dx).^(n-1)
		phi .= -D .* (h[2:nx_local+1,t] .- h[1:nx_local,t]) ./ dx
		h[2:nx_local,t+1] .= h[2:nx_local,t] .+ M[2:nx_local] .* dt .- dt/dx .* (phi[2:nx_local] .- phi[1:nx_local-1])

		h[2:nx_local,t+1] .= update_h.(h[2:nx_local,t+1], b[2:nx_local])
		h_capital[2:nx_local,t+1] .= h[2:nx_local,t+1] .- b[2:nx_local]

		send_mesg = Array{Float64}(undef, 1)
		recv_mesg = Array{Float64}(undef, 1)
		if rank == 1
			fill!(send_mesg, Float64(phi[0]))
			sreq = MPI.Send(send_mesg, 0, 100 + rank, comm)
			rreq = MPI.Irecv!(recv_mesg, 0, 100 + rank - 1, comm)

			h[1,t+1] = h[1,t] + M[1] * dt - dt/dx * (phi[1] - recv_mesg[1])
			h[1,t+1] = update_h(h[1,t+1], b[1])
			h_capital[1,t+1] = h[1,t+1] - b[1]
			print("$(h_capital[nx_local+1,t+1]), $(h[nx_local+1,t]), $(M[nx_local+1]), $(phi[1] - recv_mesg[1])\n")
		end
		if rank == 0
			fill!(send_mesg, Float64(phi[nx_local]))
			sreq = MPI.Send(send_mesg, 1, 100 + rank, comm)
			rreq = MPI.Irecv!(recv_mesg, 1, 100 + rank + 1, comm)

			h[nx_local+1,t+1] = h[nx_local+1,t] + M[nx_local+1] * dt - dt/dx * (recv_mesg[1] - phi[nx_local])
			h[nx_local+1,t+1] = update_h(h[nx_local+1,t+1], b[nx_local+1])
			h_capital[nx_local+1,t+1] = h[nx_local+1,t+1] - b[nx_local+1]
			print("$(h_capital[nx_local+1,t+1]), $(h[nx_local+1,t]), $(M[nx_local+1]), $(recv_mesg[1] - phi[nx_local])\n")
		end
		MPI.Barrier(comm)


	end

	V_local = sum(h_capital[2:nx_local+1,nt+1].*dx)

	return V_local



	# MPI.Barrier(comm)

	#Communications
	# if rank > 0
	# 	MPI.Send(V_local, 0, rank, comm)
	# 	return V_local
	# else
	# 	V_total = V_local[1]
	# 	for i in 1:size-1
	# 		MPI.Recv!(V_local, i, i, comm)
	# 		V_total = V_total + V_local[1]
	# 	end
	# 	return V_total
	# end

end

MPI.Init()
comm = MPI.COMM_WORLD

dx = 1.0
xend = 30.0
dt = 1/12.0
tend = 1.0

nx = Int(round(xend/dx))
xx = zeros(nx+1)
∂V_∂xx=zero(xx)
@show V = forward_problem(xx,nx,dx,xend,dt,tend, Array)

# MPI.Barrier(comm)

# autodiff(forward_problem, Active, Duplicated(xx, ∂V_∂xx), nx, dx, xend, dt, tend, Array)

# MPI.Barrier(comm)

# for i in 0:MPI.Comm_size(comm)-1
# 	if MPI.Comm_rank(comm) == i
# 		println(∂V_∂xx)
# 	end
# end

MPI.Finalize()
