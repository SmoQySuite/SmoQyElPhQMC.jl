# measure density-density correlation
function measure_density_correlation!(
    DD::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    a::Int, b::Int, coef::T = 1.0
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    # measure equal-spin density-density correlation
    measure_density_correlation!(
        DD, greens_estimator,
        a, b, +1, +1, 2.0*coef
    )

    # measure mixed-spin density-density correlation
    measure_density_correlation!(
        DD, greens_estimator,
        a, b, +1, -1, 2.0*coef
    )

    return nothing
end

# measure spin-resolve density-density correlation
function measure_density_correlation!(
    DD::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    a::Int, b::Int, σ::Int, σ′::Int,
    coef::T = 1.0
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    # DD(τ,r) = 1 - Gσ(a,i+r,τ|a,i+r,τ) - Gσ′(b,i,0|b,i,0)
    na = measure_n(greens_estimator, a)
    nb = measure_n(greens_estimator, b)
    # @. DD = DD + coef * (1 - (1-na) - (1-nb))
    @. DD = DD + coef * (na + nb - 1)

    # zero displacement
    z = @SVector zeros(Int, D)

    # DD(τ,r) = DD(τ,r) + 1/N sum_i Gσ(a,i+r,τ|a,i+r,τ)⋅Gσ'(b,i,0|b,i,0)
    measure_GΔΔ_G00!(
        DD, greens_estimator,
        (a, a, b, b), z, z, z, z, coef
    )

    # if equal spin
    if σ == σ′

        # DD(τ,r) = DD(τ,r) - 1/N sum_i δ(σ,σ′)⋅Gσ(b,i,0|a,i+r,τ)⋅Gσ(a,i+r,τ|b,i,0)
        measure_G0Δ_GΔ0!(
            DD, greens_estimator,
            (b, a, a, b), z, z, z, z, -coef
        )
    end

    return nothing
end