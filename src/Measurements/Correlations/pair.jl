# measure pair correlation
function measure_pair_correlation!(
    PP::AbstractArray{Complex{E}, Dp1},
    greens_estimator::GreensEstimator{E, D},
    b′::Bond{D}, b″::Bond{D}, coef::T = 1.0
) where {Dp1, D, E<:AbstractFloat, T<:Number}

    # PP(τ,r) = G₊(a,i+r+r′,τ|c,i+r″,0)⋅G₋(b,i+r,τ|d,i,0)
    b, a = b′.orbitals
    r′ = b′.displacement
    d, c = b″.orbitals
    r″ = b″.displacement
    z = @SVector zeros(Int, D)
    measure_GΔ0_GΔ0!(
        PP, greens_estimator,
        (a, c, b, d),
        r′, r″, z, z,
        coef
    )

    return nothing
end