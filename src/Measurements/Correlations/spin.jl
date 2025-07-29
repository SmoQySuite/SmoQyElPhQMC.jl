# measure spin-x/y/z correlation
function measure_spin_correlation!(
    SzSz::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    a::Int, b::Int, coef::T = 1.0
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    # SzSz(τ,r) = -2⋅G(b,i,0|a,i+r,τ)⋅G(a,i+r,τ|b,i,0)/4
    z = @SVector zeros(Int, D)
    measure_G0Δ_GΔ0!(
        SzSz, greens_estimator,
        (b, a, a, b), z, z, z, z, -0.5*coef
    )

    return nothing
end