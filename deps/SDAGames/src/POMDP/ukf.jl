# function predict(ukf::UnscentedKalmanFilter, b, u, p=LLPF.parameters(ukf), t::Real = LLPF.index(ukf); R1 = LLPF.get_mat(ukf.R1, ukf.x, u, p, t))
#     (;dynamics,measurement,xs) = ukf
#     x,R = b.μ, b.Σ
#     R = get_mat(R)
#     ns = length(xs)
#     LLPF.sigmapoints!(xs,eltype(xs)(x),R) # TODO: these are calculated in the update step
#     for i in eachindex(xs)
#         xs[i] = dynamics(xs[i], u, p, t)
#     end
#     x = mean(xs)
#     R = LLPF.symmetrize(cov(xs)) + R1
#     ukf.t[] += 1
#     return MvNormal(x, R)
# end

# # TODO: lilkelihood we're looking for to update hypothesis is p(o|b,a) 
# # -- MAKE SURE THIS IS NOT BEING CALCULATED AS p(o|b′)
# function correct(ukf::UnscentedKalmanFilter, b, u, y, p=LLPF.parameters(ukf), t::Real = LLPF.index(ukf); R2 = LLPF.get_mat(ukf.R2, ukf.x, u, p, t))
#     (;measurement,xs,R,R1) = ukf
#     x,R = b.μ, b.Σ
#     R = get_mat(R)
#     n = size(R1,1)
#     m = size(R2,1)
#     ns = length(xs)
#     LLPF.sigmapoints!(xs,eltype(xs)(x),R) # Update sigmapoints here since untransformed points required
#     C = @SMatrix zeros(n,m)
#     ys = map(xs) do x
#         measurement(x, u, p, t)
#     end
#     ym = mean(ys)
#     @inbounds for i in eachindex(ys) # Cross cov between x and y
#         d   = ys[i]-ym
#         ca  = (xs[i]-x)*d'
#         C  += ca
#     end
#     e   = y .- ym
#     S   = LLPF.symmetrize(cov(ys)) + R2 # cov of y
#     Sᵪ  = cholesky(S)
#     K   = (C./ns)/Sᵪ # ns normalization to make it a covariance matrix
#     x = x .+ K*e
#     # mul!(x, K, e, 1, 1) # K and e will be SVectors if ukf correctly initialized
#     R = RmKSKT(R, K, S)
#     l = pdf(MvNormal(PDMat(S,Sᵪ)), e)
#     ll = log(l) #- 1/2*logdet(S) # logdet is included in logpdf
#     return MvNormal(x, R), (; ll, l, e, S, Sᵪ, K)
# end

# function update(kf::UnscentedKalmanFilter, b::MvNormal, a, o, p)
#     _bp = predict(kf, b, a, p)
#     return correct(kf, _bp, a, o, p)
# end

# function update(kf::UnscentedKalmanFilter, b::HybridBelief, a, o, p)
#     _bp = predict(kf, b, a, p)
#     return correct(kf, _bp, a, o, p)
# end

# ## utils -- move somewhere else
# Base.copy(d::SparseCat) = SparseCat(copy(d.vals), copy(d.probs))

# get_mat(x::AbstractArray) = x
# # get_mat(x::PDMats.PDiagMat) = diagm(x.diag)

# @inline RmKSKT(R, K, S) = LLPF.symmetrize(R .- K*S*K')
