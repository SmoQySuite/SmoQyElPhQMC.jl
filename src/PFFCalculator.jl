@doc raw"""
    PFFCalculator{T<:AbstractFloat}

The `PFFCalaculatr` type, short for pseudo-fermion field calcutor, is for facilitating the
sampling the pseudo-fermion fields ``\Phi``, evaluate the fermionic
action ``S_f`` and calculating it's partial derivatives ``\partial S_f/\partial x_{\tau,i}`` with
respect to each phonon field ``x_{\tau,i}.``
"""
struct PFFCalculator{T<:AbstractFloat}

    Φ::Matrix{Complex{T}}
    Λ::Matrix{T}
    u::Matrix{Complex{T}}
    u′::Matrix{Complex{T}}
    u″::Matrix{Complex{T}}
end

@doc raw"""
    PFFCalculator(
        # Arguments
        electron_phonon_parameters::ElectronPhononParameters{T},
        fermion_det_matrix::FermionDetMatrix{T};
    ) where {T<:Number, E<:AbstractFloat}

Initialize an instance of the [`PFFCalculator`](@ref) type used for calculating the
pseudo-fermion fields ``\Phi``. The `tol` and `maxiter` keywords specify the tolerance and maximum
number of iterations used when performing conjugate gradient solves to evaluate the
fermionic action given the current fields ``\Phi``.
"""
function PFFCalculator(
    electron_phonon_parameters::ElectronPhononParameters{T, E},
    fermion_det_matrix::FermionDetMatrix{T, E}
) where {T<:Number, E<:AbstractFloat}

    # length of imaginary-time axis
    Lτ = electron_phonon_parameters.Lτ

    # number of orbitals in lattice
    N = size(fermion_det_matrix, 1)
    Norbitals = N ÷ Lτ

    # allocate arrays
    Φ = zeros(Complex{E}, Lτ, Norbitals)
    Λ = zeros(E, Lτ, Norbitals)
    u = zeros(Complex{E}, Lτ, Norbitals)
    u′ = zeros(Complex{E}, Lτ, Norbitals)
    u″ = zeros(Complex{E}, Lτ, Norbitals)

    # allocate PFF calculate
    pff_calculator = PFFCalculator{E}(Φ, Λ, u, u′, u″)

    return pff_calculator
end

# sample Φ = Λᵀ⋅Mᵀ⋅R
function sample_pseudofermion_fields!(
    pff_calculator::PFFCalculator{E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    fermion_det_matrix::FermionDetMatrix{T,E},
    rng = default_rng()
) where {T<:Number, E<:AbstractFloat}

    (; Λ, Φ) = pff_calculator
    # initiliaze Λ
    update_Λ!(Λ, electron_phonon_parameters)
    # initialize R
    randn!(rng, Φ)
    # Sf = |R|²
    Sf = real(dot(Φ,Φ))
    # Mᵀ⋅R
    lmul_Mt!(fermion_det_matrix, Φ)
    # Φ = Λᵀ⋅Mᵀ⋅R
    mul_Λᵀ!(Φ, Λ, Φ)          
    
    return Sf
end

# calculate fermionic action Sf = Φᵀ⋅Ψ = Φᵀ⋅[Aᵀ⋅A]⁻¹⋅Φ = Φᵀ⋅Λ⁻¹⋅[Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ
function calculate_fermionic_action!(
    pff_calculator::PFFCalculator{E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    fermion_det_matrix::FermionDetMatrix{T,E},
    preconditioner,
    rng::AbstractRNG,
    tol::E = fermion_det_matrix.cg.tol,
    maxiter::Int = fermion_det_matrix.cg.maxiter
) where {T<:Number, E<:AbstractFloat}

    (; Φ, Λ, u) = pff_calculator
    Ψ = u
    MᵀM = fermion_det_matrix

    # update Λ
    update_Λ!(Λ, electron_phonon_parameters)
    # Ψ = Λ⁻ᵀ⋅Φ
    ldiv_Λᵀ!(Ψ, Λ, Φ)
    # Ψ = [Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ
    # EXPENSIVE PART, AS REQUIRES CONJUGATE GRADIENT SOLVE!!!
    iters, ϵ = ldiv!(
        Ψ, MᵀM, Ψ,
        preconditioner = preconditioner,
        rng = rng,
        maxiter = maxiter,
        tol = tol
    )
    # Ψ = Λ⁻¹⋅[Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ = [Aᵀ⋅A]⁻¹⋅Φ
    ldiv_Λ!(Ψ, Λ, Ψ)
    # Sf = Φᵀ⋅Ψ = Φᵀ⋅[Aᵀ⋅A]⁻¹⋅Φ
    Sf = dot(Φ,Ψ)
    @assert sqrt(tol) > abs(imag(Sf)/real(Sf)) "Complex Fermionic Action, Sf = $Sf"
    Sf = real(Sf)

    return Sf, iters, ϵ
end

# calculate the derivative of the fermionic action for a single spin species
function calculate_derivative_fermionic_action!(
    ∂Sf∂x::AbstractMatrix{E},
    pff_calculator::PFFCalculator{E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    fermion_det_matrix::FermionDetMatrix{T,E},
    preconditioner,
    rng::AbstractRNG,
    tol::E = fermion_det_matrix.cg.tol,
    maxiter::Int = fermion_det_matrix.cg.maxiter
) where {T<:Number, E<:AbstractFloat}

    (; Λ, u, u′, u″) = pff_calculator

    # Note: A = M⋅Λ <==> Aᵀ = Λᵀ⋅Mᵀ
    # Rename vectors for convenience
    Ψ, ΛΨ, AΨ, MᵀAΨ = u, u′, u″, u′

    # Calculate Ψ = Λ⁻¹⋅[Mᵀ⋅M]⁻¹⋅Λ⁻ᵀ⋅Φ = [Aᵀ⋅A]⁻¹⋅Φ
    Sf, iters, ϵ = calculate_fermionic_action!(
        pff_calculator,
        electron_phonon_parameters,
        fermion_det_matrix,
        preconditioner, rng,
        tol, maxiter
    )

    # Calculate Λ⋅Ψ
    mul_Λ!(ΛΨ, Λ, Ψ)
    # Calculate A⋅Ψ = M⋅Λ⋅Ψ
    mul_M!(AΨ, fermion_det_matrix, ΛΨ)
    # Calculate ∂Sf/∂x = -2⋅Re([A⋅Ψ]ᵀ⋅[∂M/∂x]⋅[Λ⋅Ψ])
    mul_νRe∂M∂x!(∂Sf∂x, -2.0, AΨ, ΛΨ, fermion_det_matrix, electron_phonon_parameters)
    
    # Calculate Mᵀ⋅A⋅Ψ = Mᵀ⋅M⋅Λ⋅Ψ
    mul_Mt!(MᵀAΨ, fermion_det_matrix, AΨ)
    # Calculate ∂Sf/∂x = -2⋅Re([A⋅Ψ]ᵀ⋅[∂M/∂x]⋅[Λ⋅Ψ]) - 2⋅Re([Mᵀ⋅A⋅Ψ]ᵀ⋅[∂Λ/∂x]⋅[Ψ]) = -2⋅Re([A⋅Ψ]ᵀ⋅[∂A/∂x]⋅[Ψ])
    mul_νRe∂Λ∂x!(∂Sf∂x, -2.0, MᵀAΨ, Ψ, Λ, electron_phonon_parameters)

    return Sf, iters, ϵ
end