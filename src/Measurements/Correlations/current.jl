# measure current correlation
function measure_current_correlation!(
    CC::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    b′::Bond{D}, b″::Bond{D},
    t′::AbstractArray{T, Dp1}, t″::AbstractArray{T, Dp1},
    coef::T = 1.0
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    # measure equal-spin current correlation
    measure_current_correlation!(
        CC, greens_estimator,
        b′, b″, t′, t″, +1, +1, 2.0*coef
    )

    # measure unequal-spin current correlation
    measure_current_correlation!(
        CC, greens_estimator,
        b′, b″, t′, t″, +1, -1, 2.0*coef
    )

    return nothing
end

# measure spin-resolved current correlation
function measure_current_correlation!(
    CC::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    b′::Bond{D}, b″::Bond{D},
    t′::AbstractArray{T, Dp1}, t″::AbstractArray{T, Dp1},
    σ′::Int, σ″::Int,
    coef::T = one(eltype(t′))
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    b, a = b′.orbitals
    r′ = b′.displacement
    d, c = b″.orbitals
    r″ = b″.displacement
    z = @SVector zeros(Int, D)

    # CC(τ,r) = +t(b,i+r,τ|a,i+r+r′,τ)⋅t(c,i+r″,0|d,i,0)⋅Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    measure_GΔΔ_G00!(
        CC, greens_estimator,
        (a, b, d, c), r′, z, z, r″,
        +1.0*coef, t′, t″, true, false
    )

    # CC(τ,r) += -t(b,i+r,τ|a,i+r+r′,τ)⋅t(d,i,0|c,i+r″,0)⋅Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    measure_GΔΔ_G00!(
        CC, greens_estimator,
        (a, b, c, d), r′, z, r″, z,
        -1.0*coef, t′, t″, true, true
    )

    # CC(τ,r) += -t(a,i+r+r′,τ|b,i+r,τ)⋅t(c,i+r″,0|d,i,0)⋅Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    measure_GΔΔ_G00!(
        CC, greens_estimator,
        (b, a, d, c), z, r′, z, r″,
        -1.0*coef, t′, t″, false, false
    )

    # CC(τ,r) += +t(a,i+r+r′,τ|b,i+r,τ)⋅t(d,i,0|c,i+r″,0)⋅Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    measure_GΔΔ_G00!(
        CC, greens_estimator,
        (b, a, c, d), z, r′, r″, z,
        +1.0*coef, t′, t″, false, true
    )

    # if equal spins
    if σ′ == σ″

        # CC(τ,r) += -δ(σ′,σ″)⋅t(b,i+r,τ|a,i+r+r′,τ)⋅t(c,i+r″,0|d,i,0)⋅Gσ′(d,i,0|b,i+r,τ)⋅Gσ′(a,i+r+r′,τ|c+i+r″,0)
        measure_G0Δ_GΔ0!(
            CC, greens_estimator,
            (b, a, c, d), z, z, r′, r″,
            -1.0*coef, t′, t″, true, false
        )

        # CC(τ,r) += +δ(σ′,σ″)⋅t(b,i+r,τ|a,i+r+r′,τ)⋅t(d,i,0|c,i+r″,0)⋅Gσ′(c,i+r″,0|b,i+r,τ)⋅Gσ′(a,i+r+r′,τ|d,i,0)
        measure_G0Δ_GΔ0!(
            CC, greens_estimator,
            (b, a, d, c), r″, z, r′, z,
            +1.0*coef, t′, t″, true, true
        )

        # CC(τ,r) += +δ(σ′,σ″)⋅t(a,i+r+r′,τ|b,i+r,τ)⋅t(c,i+r″,0|d,i,0)⋅Gσ′(d,i,0|a,i+r+r′,τ)⋅Gσ′(b,i+r,τ|c,i+r″,0)
        measure_G0Δ_GΔ0!(
            CC, greens_estimator,
            (d, a, b, c), z, r′, z, r″,
            +1.0*coef, t′, t″, false, false
        )

        # CC(τ,r) += -δ(σ′,σ″)⋅t(a,i+r+r′,τ|b,i+r,τ)⋅t(d,i,0|c,i+r″,0)⋅Gσ′(c,i+r″,0|a,i+r+r′,τ)⋅Gσ′(b,i+r,τ|d,i,0)
        measure_G0Δ_GΔ0!(
            CC, greens_estimator,
            (c, a, b, d), r″, r′, z, z,
            -1.0*coef, t′, t″, false, true
        )
    end

    return nothing
end