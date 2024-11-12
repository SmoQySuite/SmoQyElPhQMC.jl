@doc raw"""
    ConjugateGradientSolver{T<:Number, E<:AbstractFloat}

Type to contain extra storage space in order to avoid performing dynamic memory
allocations while performing a Conjugate Gradient (CG) solve.

# FIELDS

- `maxiter::Int`: Max number of allowed CG iterations.
- `tol::E`: CG tolerance theshold.
- `N::Int`: Size of the linear system solver is intended to solve.
- `r::Vector{T}`: Vector to avoid temporary memory allocations.
- `p::Vector{T}`: Vector to avoid temporary memory allocations.
- `v::Vector{T}`: Vector to avoid temporary memory allocations.
"""
struct ConjugateGradientSolver{T<:Number, E<:AbstractFloat}
    
    maxiter::Int
    tol::E
    N::Int
    r::Vector{T}
    p::Vector{T}
    z::Vector{T}
end

@doc raw"""
    ConjugateGradientSolver(
        # ARGUMENTS
        v::AbstractArray{T};
        # KEYWORD ARGUMENTS
        maxiter::Int = length(z),
        tol::E = 1e-8
    ) where {T<:Number, E<:AbstractFloat}

Initialize and return an instance of the type [`ConjugateGradientSolver`].

# ARGUMENTS

- `v::AbstractVector{T}`: A sample vector that used to determine the data type and dimension of the linear system to be solve with CG.

# KEYWORD ARGUMENTS

- `maxiter::Int`: Max number of allowed CG iterations.
- `tol::E = 1e-8`: CG tolerance theshold.
"""
function ConjugateGradientSolver(
    # ARGUMENTS
    v::AbstractArray{T};
    # KEYWORD ARGUMENTS
    maxiter::Int = length(z),
    tol::E = 1e-8
) where {T<:Number, E<:AbstractFloat}

    v = vec(v)
    N = length(v)
    r = zero(v)
    p = zero(v)
    z = zero(v)
    return ConjugateGradientSolver{T,E}(maxiter, tol, N, r, p, z)
end


@doc raw"""
    cg_solve!(
        x::AbstractArray{T},
        A,
        b::AbstractArray{T},
        cg_solver::ConjugateGradientSolver{T,E},
        P::UniformScaling = I;
        # Keyword Arguments
        maxiter::Int = cg_solver.maxiter,
        tol::E = cg_solver.tol
    ) where {T<:Number, E<:AbstractFloat}

    cg_solve!(
        x::AbstractArray{T},
        A,
        b::AbstractArray{T},
        cg_solver::ConjugateGradientSolver{T,E},
        P;
        # Keyword Arguments
        maxiter::Int = cg_solver.maxiter,
        tol::E = cg_solver.tol
    ) where {T<:Number, E<:AbstractFloat}

Solve ``A \cdot x = b`` using the Conjugate Gradient method with
```math
P^{-1} \cdot A \cdot x = P^{-1} \cdot b.
```
with a left preconditioner ``P``. The vector ``x`` is modified in-place,
and the number of iterations and final error returned.
"""
function cg_solve!(
    x::AbstractArray{T},
    A,
    b::AbstractArray{T},
    cg_solver::ConjugateGradientSolver{T,E},
    P::UniformScaling = I;
    # Keyword Arguments
    maxiter::Int = cg_solver.maxiter,
    tol::E = cg_solver.tol
) where {T<:Number, E<:AbstractFloat}
    
    (; r, p, z) = cg_solver
    x = reshaped(x, length(x))
    b = reshaped(b, length(b))
    
    # |b|
    normb = norm(b)

    # if x and b refer to same array in memory
    if x === b
        # r₀ = b
        copyto!(r, b)
        # x = 0
        fill!(x, zero(T))
    else
        # r₀ = b - A⋅x₀
        mul!(r,A,x)
        axpby!(1.0,b,-1.0,r)
    end
    
    # p₀ = r₀
    copyto!(p,r)

    # r₀⋅r₀
    rdotr = dot(r,r)

    # calcualte initial tolerance
    ϵ = norm(r)/normb

    # check if linear system is already solved
    if ϵ < tol
        return 0, ϵ
    end
    
    @fastmath @inbounds for iter in 1:maxiter
        
        # αⱼ = (rⱼ⋅rⱼ)/(pⱼ⋅A⋅pⱼ)
        mul!(z,A,p)
        α = rdotr/dot(p,z)
        
        # xⱼ₊₁ = xⱼ + αⱼ⋅pⱼ
        axpy!(α,p,x)
        
        # rⱼ₊₁ = rⱼ - αⱼ⋅A⋅pⱼ
        axpy!(-α,z,r)

        # ϵ = |rⱼ₊₁|/|b| = |b-A⋅xⱼ₊₁|/|b|
        ϵ = norm(r)/normb
        
        # check stop criteria
        if ϵ < tol
            return iter, ϵ
        end
        
        # βⱼ = (rⱼ₊₁⋅rⱼ₊₁)/(rⱼ⋅rⱼ)
        new_rdotr = dot(r,r)
        β = new_rdotr/rdotr
        rdotr = new_rdotr
        
        # pⱼ₊₁ = rⱼ₊₁ + βⱼ⋅pⱼ
        axpby!(1.0,r,β,p)
    end
    
    return maxiter, ϵ
end

function cg_solve!(
    x::AbstractArray{T},
    A,
    b::AbstractArray{T},
    cg_solver::ConjugateGradientSolver{T,E},
    P;
    # Keyword Arguments
    maxiter::Int = cg_solver.maxiter,
    tol::E = cg_solver.tol
) where {T<:Number, E<:AbstractFloat}
    
    (; maxiter, tol, r, p, z) = cg_solver
    x = reshaped(x, length(x))
    b = reshaped(b, length(b))
    
    # |b|
    normb = norm(b)

    # if x and b refer to same array in memory
    if x === b
        # r₀ = b
        copyto!(r, b)
        # x = 0
        fill!(x, zero(T))
    else
        # r₀ = b - A⋅x₀
        mul!(r,A,x)
        axpby!(1.0,b,-1.0,r)
    end
    
    # z₀ = P⁻¹⋅r₀ = P⁻¹⋅(b - A⋅x₀)
    ldiv!(z,P,r)
    
    # p₀ = z₀
    copyto!(p,z)

    # r₀⋅z₀
    rdotz = dot(r,z)

    # calcualte initial tolerance
    ϵ = norm(r)/normb

    # check if linear system is already solved
    if ϵ < tol
        return 0, ϵ
    end
    
    @fastmath @inbounds for iter in 1:maxiter
        
        # αⱼ = (rⱼ⋅zⱼ)/(pⱼ⋅A⋅pⱼ)
        mul!(z,A,p)
        α = rdotz/dot(p,z)
        
        # xⱼ₊₁ = xⱼ + αⱼ⋅pⱼ
        axpy!(α,p,x)
        
        # rⱼ₊₁ = rⱼ - αⱼ⋅A⋅pⱼ
        axpy!(-α,z,r)

        # ϵ = |rⱼ₊₁|/|b|
        ϵ = norm(r)/normb
        
        # check stop criteria
        if ϵ < tol
            return iter, ϵ
        end
        
        # zⱼ₊₁ = P⁻¹⋅rⱼ₊₁
        ldiv!(z,P,r)
        
        # βⱼ = (rⱼ₊₁⋅zⱼ₊₁)/(rⱼ⋅zⱼ)
        new_rdotz = dot(r,z)
        β = new_rdotz/rdotz
        rdotz = new_rdotz
        
        # pⱼ₊₁ = zⱼ₊₁ + βⱼ⋅pⱼ
        axpby!(1.0,z,β,p)
    end
    
    return maxiter, ϵ
end