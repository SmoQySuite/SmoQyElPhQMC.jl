# measure bond correlation
function measure_bond_correlation!(
    BB::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    b′::Bond{D}, b″::Bond{D}, coef::T = 1.0
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    b, a = b′.orbitals
    r′ = b′.displacement
    d, c = b″.orbitals
    r″ = b″.displacement
    z = @SVector zeros(Int, D)

    # BB(τ,r) = Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (a, b, c, d), r′, z, r″, z, 4*coef
    )

    # BB(τ,r) += Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (a, b, d, c), r′, z, z, r″, 4*coef
    )

    # BB(τ,r) += Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (b, a, c, d), z, r′, r″, z, 4*coef
    )

    # BB(τ,r) += Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (b, a, d, c), z, r′, z, r″, 4*coef
    )

    # BB(τ,r) += -δ(σ′,σ″)⋅Gσ′(c,i+r″,0|b,i+r,τ)⋅Gσ″(a,i+r+r′,τ|d,i,0)
    measure_G0Δ_GΔ0!(
        BB, greens_estimator,
        (c, b, a, d), r″, z, r′, z, -2*coef
    )

    # BB(τ, r) += -δ(σ′,σ″)⋅Gσ′(d,i,0|b,i+r,τ)⋅Gσ″(a,i+r+r′,τ|c,i+r″,0)
    measure_G0Δ_GΔ0!(
        BB, greens_estimator,
        (d, b, a, c), z, z, r′, r″, -2*coef
    )

    # BB(τ, r) += -δ(σ′,σ″)⋅Gσ′(c,i+r″,0|a,i+r+r′,τ)⋅Gσ″(b,i+r,τ|d,i,0)
    measure_G0Δ_GΔ0!(
        BB, greens_estimator,
        (c, a, b, d), r″, r′, z, z, -2*coef
    )

    # BB(τ, r) += -δ(σ′,σ″)⋅Gσ′(d,i,0|a,i+r+r′,τ)⋅Gσ″(b,i+r,τ|c,i+r″,0)
    measure_G0Δ_GΔ0!(
        BB, greens_estimator,
        (d, a, b, c), z, r′, z, r″, -2*coef
    )

    return nothing
end

# measure spin-resolved bond correlation
function measure_bond_correlation!(
    BB::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    b′::Bond{D}, b″::Bond{D}, σ′::Int, σ″::Int,
    coef::T = 1.0
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    b, a = b′.orbitals
    r′ = b′.displacement
    d, c = b″.orbitals
    r″ = b″.displacement
    z = @SVector zeros(Int, D)

    # BB(τ,r) = Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (a, b, c, d), r′, z, r″, z, coef
    )

    # BB(τ,r) += Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (a, b, d, c), r′, z, z, r″, coef
    )

    # BB(τ,r) += Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (b, a, c, d), z, r′, r″, z, coef
    )

    # BB(τ,r) += Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    measure_GΔΔ_G00!(
        BB, greens_estimator,
        (b, a, d, c), z, r′, z, r″, coef
    )

    # if equal spins
    if σ′ == σ″

        # BB(τ,r) += -δ(σ′,σ″)⋅Gσ′(c,i+r″,0|b,i+r,τ)⋅Gσ″(a,i+r+r′,τ|d,i,0)
        measure_G0Δ_GΔ0!(
            BB, greens_estimator,
            (c, b, a, d), r″, z, r′, z, -coef
        )

        # BB(τ, r) += -δ(σ′,σ″)⋅Gσ′(d,i,0|b,i+r,τ)⋅Gσ″(a,i+r+r′,τ|c,i+r″,0)
        measure_G0Δ_GΔ0!(
            BB, greens_estimator,
            (d, b, a, c), z, z, r′, r″, -coef
        )

        # BB(τ, r) += -δ(σ′,σ″)⋅Gσ′(c,i+r″,0|a,i+r+r′,τ)⋅Gσ″(b,i+r,τ|d,i,0)
        measure_G0Δ_GΔ0!(
            BB, greens_estimator,
            (c, a, b, d), r″, r′, z, z, -coef
        )

        # BB(τ, r) += -δ(σ′,σ″)⋅Gσ′(d,i,0|a,i+r+r′,τ)⋅Gσ″(b,i+r,τ|c,i+r″,0)
        measure_G0Δ_GΔ0!(
            BB, greens_estimator,
            (d, a, b, c), z, r′, z, r″, -coef
        )
    end

    return nothing
end