@doc raw"""
    GreensEstimator{T<:AbstractFloat, Dp1, Dp3, Tfft<:AbstractFFTs.Plan, Tifft<:AbstractFFTs.Plan}

This type is used to compute stochastic estimates of the Green's function and other correlation functions.
"""
struct GreensEstimator{T<:AbstractFloat, D, Dp1, Dp3, Tfft<:AbstractFFTs.Plan, Tifft<:AbstractFFTs.Plan}

    Nrv::Int
    V::Int
    Lτ::Int
    N::Int
    n::Int
    L::NTuple{D, Int}
    tmp::AbstractArray{Complex{T}, Dp1}
    CΔ0::AbstractArray{Complex{T}, Dp1}
    Rt::AbstractArray{Complex{T}, Dp3}
    GR::AbstractArray{Complex{T}, Dp3}
    MtR::Vector{Complex{T}}
    A::AbstractArray{Complex{T}, Dp1}
    B::AbstractArray{Complex{T}, Dp1}
    pfft!::Tfft
    pifft!::Tifft
end

@doc raw"""
    GreensEstimator(
        # Arguments
        fdm::FermionDetMatrix{T,E},
        unitcell::UnitCell{D,E},
        lattice::Lattice{D};
        # Keyword Arguments
        Nrv::Int = 10,
        preconditioner = I,
        rng::AbstractRNG = Random.default_rng(),
        maxiter::Int = fdm.cgs.maxiter,
        tol::E = fdm.cgs.tol
    ) where {D, T<:Number, E<:AbstractFloat}

Initialize an instance of the type [`GreensEstimator`](@ref).

# Arguments

- `fdm::FermionDetMatrix{T,E}`: Fermion determinant matrix.
- `unitcell::UnitCell{D,E}`: Defines unit cell of the lattice.
- `lattice::Lattice{D}`: Defines the lattice.

# Keyword Arguments

- `Nrv::Int = 10`: Number of random vectors used to approximate Green's function.
- `preconditioner = I`: Preconditioner used to solve linear system.
- `rng = Random.default_rng()`: Random number generator.
- `maxiter::Int = fdm.cgs.maxiter`: Maximum number of iterations for linear solver.
- `tol::E = fdm.cgs.tol`: Tolerance for linear solver.
"""
function GreensEstimator(
    # Arguments
    fdm::FermionDetMatrix{T,E},
    model_geometry::ModelGeometry{D,E};
    # Keyword Arguments
    Nrv::Int = 10,
    preconditioner = I,
    rng::AbstractRNG = Random.default_rng(),
    maxiter::Int = fdm.cgs.maxiter,
    tol::E = fdm.cgs.tol
) where {D, T<:Number, E<:AbstractFloat}

    (; unit_cell, lattice) = model_geometry
    # number of orbitals per unit cell
    n = unit_cell.n
    # dimensions of lattice unit cells
    L = tuple(lattice.L...)
    # number of unit cells
    N = lattice.N
    # dimension of fermion determinant matrix
    V = size(fdm, 1)
    # length of imaginary-time axis
    Lτ = V ÷ (n * N)

    # allocate for greens estimator
    Rt = zeros(Complex{T}, Lτ, n, L..., Nrv)
    GR = zeros(Complex{T}, Lτ, n, L..., Nrv)
    MtR = zeros(Complex{T}, V)
    tmp = zeros(Complex{T}, L..., Lτ+1)
    CΔ0 = zeros(Complex{T}, Lτ+1, L...)
    A = zeros(Complex{T}, 2Lτ, L...)
    B = zeros(Complex{T}, 2Lτ, L...)
    pfft! = plan_fft!(A, flags=FFTW.PATIENT)
    pifft! = plan_ifft!(A, flags=FFTW.PATIENT)

    # initilaize greens estimator
    Dp1 = D+1
    Dp3 = D+3
    Tfft = typeof(pfft!)
    Tifft = typeof(pifft!)
    greens_estimator = GreensEstimator{E,D,Dp1,Dp3,Tfft,Tifft}(Nrv, V, Lτ, N, n, L, tmp, CΔ0, Rt, GR, MtR, A, B, pfft!, pifft!)

    # update green's function estimator to reflect current fermion determinant matrix
    update_greens_estimator!(
        greens_estimator, fdm,
        preconditioner = preconditioner,
        rng = rng,
        maxiter = maxiter,
        tol = tol
    )

    return greens_estimator
end


# update the greens estimator to reflect the current fermion determinant matrix
function update_greens_estimator!(
    greens_estimator::GreensEstimator{T},
    fdm::FermionDetMatrix{T,E};
    preconditioner = I,
    rng = Random.default_rng(),
    maxiter::Int = fdm.cgs.maxiter,
    tol::E = max(fdm.cgs.tol^2, 1e-10)
) where {T<:Number, E<:AbstractFloat}

    (; Nrv, V, MtR) = greens_estimator

    # get appropriately shaped views into arrays
    R = reshape(greens_estimator.Rt, V, Nrv)
    GR = reshape(greens_estimator.GR, V, Nrv)
    
    # initialize R using complex numbers with unit amplitude with random phase
    randn!(rng, R)
    @. R = R / abs(R)

    # # initialize R using real random normal numbers
    # for i in eachindex(R)
    #     R[i] = randn(rng, E)
    # end

    # udpate preconditioner
    update_preconditioner!(preconditioner, fdm, rng)

    # iterate over random vectors
    for n in axes(R, 2)
        R′ = @view R[:, n]
        GR′ = @view GR[:, n]
        # Calculate Mᵀ⋅R
        mul_Mt!(MtR, fdm, R′)
        # solve [Mᵀ⋅M]⁻¹⋅x = Mᵀ⋅R ==> x = M⁻¹⋅R = G⋅R
        iters, ϵ = ldiv!(
            GR′, fdm, MtR,
            preconditioner = preconditioner,
            rng = rng,
            maxiter = maxiter,
            tol = tol
        )
    end

    # calculate conjugated random variables, Rt = conj(R)
    @. greens_estimator.Rt = conj(greens_estimator.Rt)

    return nothing
end


# measure Green's function G(Δ,0)
function measure_GΔ0!(
    correlation::AbstractArray{Complex{T}},
    greens_estimator::GreensEstimator{T},
    orbitals::NTuple{2, Int},
) where {T<:AbstractFloat}

    (; GR, Rt, A, B, pfft!, pifft!, Lτ) = greens_estimator
    GΔ0 = greens_estimator.CΔ0

    # get the number of random vectors
    Nrv = size(Rt, ndims(Rt))

    # get orbital species
    a, b = orbitals

    # initialize green's function to zero
    fill!(GΔ0, 0)

    # Get views for appropriate orbitals
    GR_a = selectdim(GR, 2, a)
    Rt_b = selectdim(Rt, 2, b)

    # iterate over all random vectors
    for i in 1:Nrv

        # approximate Green's function
        GR_a_i = selectdim(GR_a, ndims(GR_a), i)
        Rt_b_i = selectdim(Rt_b, ndims(Rt_b), i)

        # copy matrices aperiodically
        _aperiodic_copyto!(A, GR_a_i)
        _aperiodic_copyto!(B, Rt_b_i)

        # esimate Green's function
        _translational_average!(GΔ0, A, B, pfft!, pifft!)
    end

    # normalize Green's function
    @. GΔ0 = GΔ0 / Nrv

    # apply aperiodic boundary condition G(r,β) = δ(r) - G(r,0)
    # given that currently G(r,β) = G(r,0) instead.
    Gβ0 = selectdim(GΔ0, 1, Lτ+1)
    @. Gβ0 = -Gβ0
    if a == b
        Gβ0[1] += 1
    end

    # add contraction to correlation
    add_contraction_to_correlation!(correlation, GΔ0, 1.0)

    return nothing
end


# measure G(Δ,0)⋅G(Δ,0) = (1/N) sum_i G(a,i+r+r₁,τ|b,i+r₂,0)⋅G(c,i+r+r₃,τ|d,i+r₄,0)
#                       = (1/N) sum_i ⟨a(i+r+r₁,τ)⋅bᵀ(i+r₂,0)⟩⋅⟨c(i+r+r₃,τ)⋅dᵀ(i+r₄,0)⟩
#  for all Δ, where Δ = (r, τ) are displacements in space-time. The letters (a,b,c,d) denote
# the orbital species, and (r_a,r_b,r_c,r_d) are static displacements in unit cells.
# the sum over i runs over all N unit cells and averages over translation symmmetry.
function measure_GΔ0_GΔ0!(
    correlation::AbstractArray{Complex{T},Dp1},
    greens_estimator::GreensEstimator{T},
    orbitals::NTuple{4, Int},
    r1::Union{NTuple{D, Int}, SVector{D, Int}},
    r2::Union{NTuple{D, Int}, SVector{D, Int}},
    r3::Union{NTuple{D, Int}, SVector{D, Int}},
    r4::Union{NTuple{D, Int}, SVector{D, Int}},
    coef,
    tΔ::Union{Nothing, AbstractArray{E, Dp1}} = nothing,
    t0::Union{Nothing, AbstractArray{E, Dp1}} = nothing,
    conj_tΔ::Bool = false,
    conj_t0::Bool = false
) where {D, Dp1, T<:AbstractFloat, E<:Number}

    (; pfft!, pifft!, GR, Rt, L) = greens_estimator
    Gl = greens_estimator.A
    Gr = greens_estimator.B

    # get orbital species
    a, b, c, d = orbitals

    # get views based on appropriate orbitals species
    GR_a = selectdim(GR, 2, a)
    Rt_b = selectdim(Rt, 2, b)
    GR_c = selectdim(GR, 2, c)
    Rt_d = selectdim(Rt, 2, d)

    # get shifted views based on static displacements
    GR_a_r1 = ShiftedArrays.circshift(GR_a, (0, (-r for r in r1)..., 0))
    Rt_b_r2 = ShiftedArrays.circshift(Rt_b, (0, (-r for r in r2)..., 0))
    GR_c_r3 = ShiftedArrays.circshift(GR_c, (0, (-r for r in r3)..., 0))
    Rt_d_r4 = ShiftedArrays.circshift(Rt_d, (0, (-r for r in r4)..., 0))

    # get array to represent G(Δ,0)⋅G(Δ,0)
    GΔ0_GΔ0 = greens_estimator.CΔ0
    fill!(GΔ0_GΔ0, 0)

    # get number of random vectors
    Nrv = size(Rt, ndims(Rt))

    # get number of pairs of random vectros
    Npairs = binomial(Nrv, 2)

    # iterate over all pairs of random vectors
    for n in 1:(Nrv-1)
        for m in (n+1):Nrv

            # get views for appropriate random vectors
            GR_a_r1_n = selectdim(GR_a_r1, ndims(GR_a_r1), n)
            Rt_b_r2_n = selectdim(Rt_b_r2, ndims(Rt_b_r2), n)
            GR_c_r3_m = selectdim(GR_c_r3, ndims(GR_c_r3), m)
            Rt_d_r4_m = selectdim(Rt_d_r4, ndims(Rt_d_r4), m)

            # estimate G(Δ,0)⋅G(Δ,0) ≈ (GR_a_r1 ⊙ GR_c_r3) ⋆ (Rt_b_r2 ⊙ Rt_d_r4)
            _measure_CΔ0!(
                GΔ0_GΔ0, Gl, Gr,
                GR_a_r1_n, GR_c_r3_m, Rt_b_r2_n, Rt_d_r4_m,
                pfft!, pifft!,
                tΔ, t0, conj_tΔ, conj_t0
            )
        end
    end

    # normalize
    @. GΔ0_GΔ0 /= Npairs

    # Account For Boundary Terms At τ = β

    # G(Δ,0)⋅G(Δ,0) += -δ(a,b)⋅δ(r,r2-r1)⋅G(i-r1+r2+r3-r4,c|i,d)
    #               += -δ(a,b)⋅δ(r,r2-r1)⋅GR(i-r1+r2+r3-r4,c)⋅R(i,d)
    if a == b
        # iterate over all random vectors
        for nrv in 1:Nrv
            # get views based on random vector
            GR_c_nrv = selectdim(GR_c, ndims(GR_c), nrv)
            Rt_d_nrv = selectdim(Rt_d, ndims(Rt_d), nrv)
            # construct GR(i-r1+r2+r3-r4,c)
            GR_c_nrv_shifted = ShiftedArrays.circshift(
                GR_c_nrv,
                (0, (r1[n]-r2[n]-r3[n]+r4[n] for n in 1:D)...)
            )
            if isnothing(tΔ) && isnothing(t0)
                # evaluate G(Δ,0)⋅G(Δ,0) += -δ(a,b)⋅δ(r,r2-r1)⋅GR(i-r1+r2+r3-r4,c)⋅R(i,d)
                GΔ0_GΔ0[end, (mod1(1-r1[n]+r2[n], L[n]) for n in 1:D)...] -= sum(
                    GR_c_nrv_shifted[i] * Rt_d_nrv[i] for i in eachindex(GR_c_nrv_shifted)
                ) / (Nrv * length(GR_c_nrv_shifted))
            else
                tβ_shift = ShiftedArrays.circshift(tΔ, (0, (r1[n]-r2[n] for n in 1:D)...))
                # evaluate G(Δ,0)⋅G(Δ,0) += -δ(a,b)⋅δ(r,r2-r1)⋅conj(t[i+r])⋅t[i]⋅GR(i-r1+r2+r3-r4,c)⋅R(i,d)
                GΔ0_GΔ0[end, (mod1(1-r1[n]+r2[n], L[n]) for n in 1:D)...] -= sum(
                    bconj(tβ_shift[i], conj_tΔ) * bconj(t0[i], conj_t0) * GR_c_nrv_shifted[i] * Rt_d_nrv[i] for i in eachindex(GR_c_nrv_shifted)
                ) / (Nrv * length(GR_c_nrv_shifted))
            end
        end
    end

    # G(Δ,0)⋅G(Δ,0) += -δ(c,d)⋅δ(r, r4-r3)⋅G(i+r1-r2-r3+r4,a|i,b)
    #               += -δ(c,d)⋅δ(r, r4-r3)⋅GR(i+r1-r2-r3+r4,a)⋅R(i,b)
    if c == d
        # iterate over all random vectors
        for nrv in 1:Nrv
            # get views based on random vector
            GR_a_nrv = selectdim(GR_a, ndims(GR_a), nrv)
            Rt_b_nrv = selectdim(Rt_b, ndims(Rt_b), nrv)
            # construct GR(i+r1-r2-r3+r4,a)
            GR_a_nrv_shifted = ShiftedArrays.circshift(
                GR_a_nrv,
                (0, (-r1[n]+r2[n]+r3[n]-r4[n] for n in 1:D)...)
            )
            if isnothing(tΔ) && isnothing(t0)
                # evaluate G(Δ,0)⋅G(Δ,0) += -δ(c,d)⋅δ(r,r4-r3)⋅GR(i+r1-r2-r3+r4,a)⋅R(i,b)
                GΔ0_GΔ0[end, (mod1(1-r3[n]+r4[n], L[n]) for n in 1:D)...] -= sum(
                    GR_a_nrv_shifted[i] * Rt_b_nrv[i] for i in eachindex(GR_a_nrv_shifted)
                ) / (Nrv * length(GR_a_nrv_shifted))
            else
                tβ_shift = ShiftedArrays.circshift(tΔ, (0, (r3[n]-r4[n] for n in 1:D)...))
                # evaluate G(Δ,0)⋅G(Δ,0) += -δ(c,d)⋅δ(r,r4-r3)⋅conj(t[i+r])⋅t[i]⋅GR(i+r1-r2-r3+r4,a)⋅R(i,b)
                GΔ0_GΔ0[end, (mod1(1-r3[n]+r4[n], L[n]) for n in 1:D)...] -= sum(
                    bconj(tβ_shift[i], conj_tΔ) * bconj(t0[i], conj_t0) * GR_a_nrv_shifted[i] * Rt_b_nrv[i] for i in eachindex(GR_a_nrv_shifted)
                ) / (Nrv * length(GR_a_nrv_shifted))
            end
        end
    end

    # G(Δ,0)⋅G(Δ,0) += δ(a,b)⋅δ(c,d)⋅δ(r,r2-r1)⋅δ(r,r4-r3)
    if (
        (a == b)
        && (c == d)
        && all(mod(r2[n]-r1[n],L[n]) == mod(r4[n]-r3[n], L[n]) for n in 1:D)
    )
        if isnothing(tΔ) && isnothing(t0)
            # evaluate G(Δ,0)⋅G(Δ,0) += δ(a,b)⋅δ(c,d)⋅δ(r,r2-r1)⋅δ(r,r4-r3)
            GΔ0_GΔ0[end, (mod1(1+r2[n]-r1[n], L[n]) for n in 1:D)...] += 1
        else
            tβ_shift = ShiftedArrays.circshift(tΔ, (0, (r1[n]-r2[n] for n in 1:D)...))
            # evaluate G(Δ,0)⋅G(Δ,0) += δ(a,b)⋅δ(c,d)⋅δ(r,r2-r1)⋅δ(r,r4-r3)⋅conj(t[i+r])⋅t[i]
            GΔ0_GΔ0[end, (mod1(1+r2[n]-r1[n], L[n]) for n in 1:D)...] += sum(
                bconj(tβ_shift[i], conj_tΔ) * bonj(t0[i], conj_t0) for i in eachindex(tβ_shift)
            ) / length(tβ_shift)
        end
    end

    # add contraction to correlation
    add_contraction_to_correlation!(correlation, GΔ0_GΔ0, coef)

    return nothing
end


# measure G(Δ,Δ)⋅G(0,0) = (1/N) sum_i G(a,i+r+r₁,τ|b,i+r+r₂,τ)⋅G(c,i+r₃,0|d,i+r₄,0)
#                       = (1/N) sum_i ⟨a(i+r+r₁,τ)⋅bᵀ(i+r+r₂,τ)⟩⋅⟨c(i+r₃,0)⋅dᵀ(i+r₄,0)⟩
# for all Δ, where Δ = (r, τ) are displacements in space-time. The letters (a,b,c,d) denote
# the orbital species, and (r_a,r_b,r_c,r_d) are static displacements in unit cells.
# the sum over i runs over all N unit cells and averages over translation symmmetry.
function measure_GΔΔ_G00!(
    correlation::AbstractArray{Complex{T}, Dp1},
    greens_estimator::GreensEstimator{T},
    orbitals::NTuple{4, Int},
    r1::Union{NTuple{D, Int}, SVector{D, Int}},
    r2::Union{NTuple{D, Int}, SVector{D, Int}},
    r3::Union{NTuple{D, Int}, SVector{D, Int}},
    r4::Union{NTuple{D, Int}, SVector{D, Int}},
    coef,
    tΔ::Union{Nothing,AbstractArray{E, Dp1}} = nothing,
    t0::Union{Nothing,AbstractArray{E, Dp1}} = nothing,
    conj_tΔ::Bool = false,
    conj_t0::Bool = false
) where {D, Dp1, T<:AbstractFloat, E<:Number}

    (; pfft!, pifft!, GR, Rt, L) = greens_estimator
    Gl = greens_estimator.A
    Gr = greens_estimator.B

    # get orbital species
    a, b, c, d = orbitals

    # get views based on appropriate orbitals species
    GR_a = selectdim(GR, 2, a)
    Rt_b = selectdim(Rt, 2, b)
    GR_c = selectdim(GR, 2, c)
    Rt_d = selectdim(Rt, 2, d)

    # get shifted views based on static displacements
    GR_a_r1 = ShiftedArrays.circshift(GR_a, (0, (-r for r in r1)..., 0))
    Rt_b_r2 = ShiftedArrays.circshift(Rt_b, (0, (-r for r in r2)..., 0))
    GR_c_r3 = ShiftedArrays.circshift(GR_c, (0, (-r for r in r3)..., 0))
    Rt_d_r4 = ShiftedArrays.circshift(Rt_d, (0, (-r for r in r4)..., 0))

    # get array to represent G(Δ,Δ)⋅G(0,0)
    GΔΔ_G00 = greens_estimator.CΔ0
    fill!(GΔΔ_G00, 0)

    # get number of random vectors
    Nrv = size(Rt, ndims(Rt))

    # get number of pairs of random vectros
    Npairs = binomial(Nrv, 2)

    # iterate over all pairs of random vectors
    for n in 1:(Nrv-1)
        for m in (n+1):Nrv

            # get views for appropriate random vectors
            GR_a_r1_n = selectdim(GR_a_r1, ndims(GR_a_r1), n)
            Rt_b_r2_n = selectdim(Rt_b_r2, ndims(Rt_b_r2), n)
            GR_c_r3_m = selectdim(GR_c_r3, ndims(GR_c_r3), m)
            Rt_d_r4_m = selectdim(Rt_d_r4, ndims(Rt_d_r4), m)
    
            # estimate G(Δ,Δ)⋅G(0,0) ≈ (GR_a_r1 ⊙ Rt_b_r2) ⋆ (GR_c_r3 ⊙ Rt_d_r4)
            _measure_CΔ0!(
                GΔΔ_G00, Gl, Gr,
                GR_a_r1_n, Rt_b_r2_n, GR_c_r3_m, Rt_d_r4_m,
                pfft!, pifft!,
                tΔ, t0, conj_tΔ, conj_t0
            )
        end
    end

    # normalize
    @. GΔΔ_G00 /= Npairs

    # add contraction to correlation
    add_contraction_to_correlation!(correlation, GΔΔ_G00, coef)

    return nothing
end


# measure G(0,Δ)⋅G(Δ,0) = (1/N) sum_i G(a,i+r₁,0|b,i+r+r₂,τ)⋅G(c,i+r+r₃,τ|d,i+r₄,0)
#                       = (1/N) sum_i -⟨bᵀ(i+r+r₁,τ)⋅a(i+r₂,0)⟩⋅⟨c(i+r+r₃,τ)⋅dᵀ(i+r₄,0)⟩
# for all Δ, where Δ = (r, τ) are displacements in space-time. The letters (a,b,c,d) denote
# the orbital species, and (r_a,r_b,r_c,r_d) are static displacements in unit cells.
# the sum over i runs over all N unit cells and averages over translation symmmetry.
function measure_G0Δ_GΔ0!(
    correlation::AbstractArray{Complex{T}, Dp1},
    greens_estimator::GreensEstimator{T},
    orbitals::NTuple{4, Int},
    r1::Union{NTuple{D, Int}, SVector{D, Int}},
    r2::Union{NTuple{D, Int}, SVector{D, Int}},
    r3::Union{NTuple{D, Int}, SVector{D, Int}},
    r4::Union{NTuple{D, Int}, SVector{D, Int}},
    coef,
    tΔ::Union{Nothing,AbstractArray{E, Dp1}} = nothing,
    t0::Union{Nothing,AbstractArray{E, Dp1}} = nothing,
    conj_tΔ::Bool = false,
    conj_t0::Bool = false
) where {D, Dp1, T<:AbstractFloat, E<:Number}

    (; pfft!, pifft!, GR, Rt, L) = greens_estimator
    Gl = greens_estimator.A
    Gr = greens_estimator.B

    # get orbital species
    a, b, c, d = orbitals

    # get views based on appropriate orbitals species
    GR_a = selectdim(GR, 2, a)
    Rt_b = selectdim(Rt, 2, b)
    GR_c = selectdim(GR, 2, c)
    Rt_d = selectdim(Rt, 2, d)

    # get shifted views based on static displacements
    GR_a_r1 = ShiftedArrays.circshift(GR_a, (0, (-r for r in r1)..., 0))
    Rt_b_r2 = ShiftedArrays.circshift(Rt_b, (0, (-r for r in r2)..., 0))
    GR_c_r3 = ShiftedArrays.circshift(GR_c, (0, (-r for r in r3)..., 0))
    Rt_d_r4 = ShiftedArrays.circshift(Rt_d, (0, (-r for r in r4)..., 0))

    # get array to represent G(0,Δ)⋅G(Δ,0)
    G0Δ_GΔ0 = greens_estimator.CΔ0
    fill!(G0Δ_GΔ0, 0)

    # get number of random vectors
    Nrv = size(Rt, ndims(Rt))

    # get number of pairs of random vectros
    Npairs = binomial(Nrv, 2)

    # iterate over all pairs of random vectors
    for n in 1:(Nrv-1)
        for m in (n+1):Nrv

            # get views for appropriate random vectors
            GR_a_r1_n = selectdim(GR_a_r1, ndims(GR_a_r1), n)
            Rt_b_r2_n = selectdim(Rt_b_r2, ndims(Rt_b_r2), n)
            GR_c_r3_m = selectdim(GR_c_r3, ndims(GR_c_r3), m)
            Rt_d_r4_m = selectdim(Rt_d_r4, ndims(Rt_d_r4), m)

            # estimate G(0,Δ)⋅G(Δ,0) ≈ (Rt_b_r2 ⊙ GR_c_r3) ⋆ (GR_a_r1 ⊙ Rt_d_r4)
            _measure_CΔ0!(
                G0Δ_GΔ0, Gl, Gr,
                Rt_b_r2_n, GR_c_r3_m, GR_a_r1_n, Rt_d_r4_m,
                pfft!, pifft!,
                tΔ, t0, conj_tΔ, conj_t0
            )
        end
    end

    # normalize
    @. G0Δ_GΔ0 /= Npairs

    # Account For Boundary Terms at τ = 0

    # G(0,Δ)⋅G(Δ,0) += -δ(a,b)⋅δ(r,r1-r2)⋅G(i+r1-r2+r3-r4,c|i,d)
    #               += -δ(a,b)⋅δ(r,r1-r2)⋅GR(i+r1-r2+r3-r4,c)⋅R(i,d)
    if a == b
        # iterate over random vectors
        for nrv in 1:Nrv
            # get views based on random vector
            GR_c_nrv = selectdim(GR_c, ndims(GR_c), nrv)
            Rt_d_nrv = selectdim(Rt_d, ndims(Rt_d), nrv)
            # construct GR(i+r1-r2+r3-r4,c)
            GR_nrv_c_shifted = ShiftedArrays.circshift(
                GR_c_nrv,
                (0, (-r1[n]+r2[n]-r3[n]+r4[n] for n in 1:D)...)
            )
            if isnothing(tΔ) && isnothing(t0)
                # evaluate G(0,Δ)⋅G(Δ,0) += -δ(a,b)⋅δ(r,r1-r2)⋅GR(i+r1-r2+r3-r4,c)⋅R(i,d)
                G0Δ_GΔ0[1, (mod1(1+r1[n]-r2[n], L[n]) for n in 1:D)...] -= sum(
                    GR_nrv_c_shifted[i] * Rt_d_nrv[i] for i in eachindex(GR_nrv_c_shifted)
                ) / (Nrv * length(GR_nrv_c_shifted))
            else
                tβ_shift = ShiftedArrays.circshift(tΔ, (0, (-r1[n]+r2[n] for n in 1:D)...))
                # evaluate G(0,Δ)⋅G(Δ,0) += -δ(a,b)⋅δ(r,r1-r2)⋅conj(t[i+r1-r2])⋅t[i]⋅GR(i+r1-r2+r3-r4,c)⋅R(i,d)
                G0Δ_GΔ0[1, (mod1(1+r1[n]-r2[n], L[n]) for n in 1:D)...] -= sum(
                    bconj(tβ_shift[i], conj_tΔ) * bconj(t0[i], conj_t0) * GR_nrv_c_shifted[i] * Rt_d_nrv[i] for i in eachindex(GR_nrv_c_shifted)
                ) / (Nrv * length(GR_nrv_c_shifted))
            end
        end
    end

    # Account For Boundary Terms At τ = β

    # G(0,Δ)⋅G(Δ,0) += -δ(c,d)⋅δ(r,r4-r3)⋅G(i+r1-r2+r3-r4,a|i,b)
    #               += -δ(c,d)⋅δ(r,r4-r3)⋅GR(i+r1-r2+r3-r4,a)⋅R(i,b)
    if c == d
        # iterate over random vectors
        for nrv in 1:Nrv
            # get views based on random vector
            GR_a_nrv = selectdim(GR_a, ndims(GR_a), nrv)
            Rt_b_nrv = selectdim(Rt_b, ndims(Rt_b), nrv)
            # construct GR(i+r1-r2+r3-r4,a)
            GR_a_nrv_shifted = ShiftedArrays.circshift(
                GR_a_nrv,
                (0, (-r1[n]+r2[n]-r3[n]+r4[n] for n in 1:D)...)
            )
            if isnothing(tΔ) && isnothing(t0)
                # evaluate G(0,Δ)⋅G(Δ,0) += -δ(c,d)⋅δ(r,r4-r3)⋅GR(i+r1-r2+r3-r4,a)⋅R(i,b)
                G0Δ_GΔ0[end, (mod1(1+r4[n]-r3[n], L[n]) for n in 1:D)...] -= sum(
                    GR_a_nrv_shifted[i] * Rt_b_nrv[i] for i in eachindex(GR_a_nrv_shifted)
                ) / (Nrv * length(GR_a_nrv_shifted))
            else
                tβ_shift = ShiftedArrays.circshift(tΔ, (0, (-r4[n]+r3[n] for n in 1:D)...))
                # evaluate G(0,Δ)⋅G(Δ,0) += -δ(c,d)⋅δ(r,r4-r3)⋅conj(t[i+r4-r3])⋅t[i]⋅GR(i+r1-r2+r3-r4,a)⋅R(i,b)
                G0Δ_GΔ0[end, (mod1(1+r4[n]-r3[n], L[n]) for n in 1:D)...] -= sum(
                    bconj(tβ_shift[i], conj_tΔ) * bconj(t0[i], conj_t0) * GR_a_nrv_shifted[i] * Rt_b_nrv[i] for i in eachindex(GR_a_nrv_shifted)
                ) / (Nrv * length(GR_a_nrv_shifted))
            end
        end
    end

    # add contraction to correlation
    add_contraction_to_correlation!(correlation, G0Δ_GΔ0, coef)

    return nothing
end


# measure C[r] = (A[i+r]⋅B[i+r])⋆(C[i]⋅D[i])
function _measure_CΔ0!(
    CΔ0::AbstractArray{Complex{T}, Dp1},
    AΔBΔ::AbstractArray{Complex{T}, Dp1},
    C0D0::AbstractArray{Complex{T}, Dp1},
    AΔ::AbstractArray{Complex{T}, Dp1},
    BΔ::AbstractArray{Complex{T}, Dp1},
    C0::AbstractArray{Complex{T}, Dp1},
    D0::AbstractArray{Complex{T}, Dp1},
    pfft!::AbstractFFTs.Plan,
    pifft!::AbstractFFTs.Plan,
    tΔ::Union{Nothing,AbstractArray{E, Dp1}} = nothing,
    t0::Union{Nothing,AbstractArray{E, Dp1}} = nothing,
    conj_tΔ::Bool = false,
    conj_t0::Bool = false
) where {Dp1, T<:AbstractFloat, E<:Number}

    # periodic product in imaginary-time extending τ ∈ [0,β-Δτ]
    # to the range τ ∈ [0,2β-Δτ]
    _periodic_prod!(AΔBΔ, AΔ, BΔ, tΔ, conj_tΔ)
    _periodic_prod!(C0D0, C0, D0, t0, conj_t0)

    # estimate C[Δ] = ⟨A[Δ]⋅B[Δ]⋅C[0]⋅D[0]⟩
    _translational_average!(CΔ0, AΔBΔ, C0D0, pfft!, pifft!)

    return nothing
end


# aperiodic element-wise product
function _periodic_prod!(
    ab::AbstractArray{T, Dp1},
    a::AbstractArray{T, Dp1},
    b::AbstractArray{T, Dp1},
    t::Union{Nothing, AbstractArray{E, Dp1}} = nothing,
    conj_t::Bool = false
) where {Dp1, T<:Number, E<:Number}

    Lτ = size(a, 1)
    ab′ = selectdim(ab, 1, 1:Lτ)
    ab″ = selectdim(ab, 1, (Lτ+1):(2*Lτ))
    if isnothing(t)
        @. ab′ = a * b
    else
        @. ab′ = conj_t ? (conj(t) * a * b) : (t * a * b)
    end
    copyto!(ab″, ab′)

    return nothing
end


# aperiodic copyto! along the first dimension
function _aperiodic_copyto!(
    ap::AbstractArray{T, D},
    a::AbstractArray{T, D}
) where {D, T<:Number}

    # length of first dimension
    L = size(a, 1)
    ap′ = selectdim(ap, 1, 1:L)
    ap″ = selectdim(ap, 1, (L+1):(2*L))
    copyto!(ap′, a)
    @. ap″ = -ap′

    return nothing
end


# average over translational symmetry with circular cross correlation
# evaluated using FFTs. Specifically, this function calculates
# S[r] = S[r] + (1/N)sum_i(a[i+r]⋅b[i])
function _translational_average!(
    S::AbstractArray{Complex{T}, D},
    a::AbstractArray{Complex{T}, D},
    b::AbstractArray{Complex{T}, D},
    pfft!::AbstractFFTs.Plan,
    pifft!::AbstractFFTs.Plan
) where {D, T<:AbstractFloat}

    Lτ = size(S, 1) - 1
    mul!(a, pfft!, a)
    mul!(b, pifft!, b)
    ab = a
    @. ab = a * b
    mul!(ab, pifft!, ab)
    # record the result for τ ∈ [0,β-Δτ]
    S′ = selectdim(S, 1, 1:Lτ)
    ab′ = selectdim(ab, 1, 1:Lτ)
    @. S′ += ab′
    # deal with τ = β boundary condition where S[β] = S[0]
    S″ = selectdim(S, 1, Lτ+1)
    ab″ = selectdim(ab, 1, 1)
    @. S″ += ab″

    return nothing
end

# add contraction to correlation array
function add_contraction_to_correlation!(
    correlation::AbstractArray{T, Dp1},
    contraction::AbstractArray{T, Dp1},
    coef::E
) where {Dp1, T<:Number, E<:Number}

    # represent contraction so that last index instead of first index
    # corresponds to the imaginary-time axis
    permuted_contraction = PermutedDimsArray(contraction, (2:Dp1..., 1))

    # add contraction to correlation
    @. correlation += coef * permuted_contraction

    return nothing
end

# boolean conjugation operation
bconj(x::Number, b::Bool) = b ? conj(x) : x