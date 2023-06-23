using Printf
using Statistics
using Plots
using Enzyme
# using CUDA

# a = 4.2
# b = [2.2, 3.3]; ∂f_∂b = zero(b)
# c = 55; d = 9

# f(a, b, c, d) = a * √(b[1]^2 + b[2]^2) + c^2 * d^2
# ∂f_∂a, ∂f_∂d = autodiff(f, Active, Active(a), Duplicated(b, ∂f_∂b), c, Active(d))

# println(∂f_∂a)
# println(∂f_∂b)
# println(∂f_∂d)
# # output
# # (3.966106403010388, [2.3297408241459623, 3.4946112362189434], 54450.0)
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
	h = AT(zeros((nx+1, nt+1)))
	h_capital = AT(zeros((nx+1, nt+1)))
	D = AT(zeros(nx))
	phi = AT(zeros(nx))
	xarr = AT([(i-1)*dx for i in 1:nx+1])
	M = AT(zeros(nx+1))
	b = AT(zeros(nx+1))
	M .= M0 .- xarr .* M1 .+ xx
	b .= 1.0 .+ bx .* xarr
	bfirst = 1.0
	bend = 1.0 + bx * nx *dx

	h[1,:] .= AT(ones(size(h[1,:]))) .* bfirst
	h[:,1] .= AT(ones(size(h[:,1]))) .* b
	h[nx+1,:] .= AT(ones(size(h[nx+1,:]))) .* bend
	h_capital[1,:] .= h[1,:] .- bfirst
	h_capital[nx+1,:] .= h[nx+1,:] .- bend
	h_capital[:,1] .= h[:,1] .- b
	@inbounds for t in 1:nt

		D .= C .* ((h_capital[1:nx,t] .+ h_capital[2:nx+1,t]) ./ 2.0).^(n+2) .* ((h[2:nx+1,t] .- h[1:nx,t]) ./ dx).^(n-1)
		phi .= -D .* (h[2:nx+1,t] .- h[1:nx,t]) ./ dx
		h[2:nx,t+1] .= h[2:nx,t] .+ M[2:nx] .* dt .- dt/dx .* (phi[2:nx] .- phi[1:nx-1])

		h[2:nx,t+1] .= update_h.(h[2:nx,t+1], b[2:nx])
		h_capital[:,t+1] .= h[:,t+1] .- b

	end
	V = sum(Array(h_capital[:,nt+1].*dx))
	return V

end

dx = 1.0
xend = 30.0
dt = 1/12.0
tend = 100.0

nx = Int(round(xend/dx))
xx = zeros(nx+1)
∂V_∂xx=zero(xx)
@show V = forward_problem(xx,nx,dx,xend,dt,tend, Array)
# autodiff(forward_problem, Active, Duplicated(xx, ∂V_∂xx), nx, dx, xend, dt, tend, Array)
# println(∂V_∂xx)
# if CUDA.has_cuda_gpu()
# 	cu_xx = CuArray(xx)
# 	@show cu_V = forward_problem(cu_xx,nx,dx,xend,dt,tend, CuArray)
# 	cu_xx = CuArray(xx)
# 	cu_∂V_∂xx = CuArray(∂V_∂xx)
# 	fill!(cu_∂V_∂xx,0)
# 	# autodiff(forward_problem, Active, Duplicated(cu_xx, cu_∂V_∂xx), nx, dx, xend, dt, tend, CuArray)
# 	println(cu_∂V_∂xx)
# end
