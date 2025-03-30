# measure density for single-spin species and specified orbital species
function measure_n(
    greens_estimator::GreensEstimator{T},
    orbital::Int
) where {T}

    Rt′ = selectdim(greens_estimator.Rt, 2, orbital)
    GR′ = selectdim(greens_estimator.GR, 2, orbital)
    n = measure_n(greens_estimator, Rt′, GR′)

    return n
end

# measure density for single spin-species
function measure_n(
    greens_estimator::GreensEstimator{T},
    Rt::AbstractArray{Complex{T}} = greens_estimator.Rt,
    GR::AbstractArray{Complex{T}} = greens_estimator.GR
) where {T<:AbstractFloat}
    
    R = Rt
    @. R = conj(Rt)
    n = 1 - dot(R, GR)/length(R)
    @. Rt = conj(R)

    return n
end


# measure ⟨N²⟩ globally
function measure_Nsqrd(
    greens_estimator::GreensEstimator
)

    (; V, N, n, Lτ, Nrv, Rt, GR) = greens_estimator
    R = Rt
    @. R = conj(Rt)

    # N is number of unit cells
    # Lτ is length of imaginary time axis
    # Nrv is the number of random vectors

    # approximate ⟨nₛ⟩
    ns = measure_n(greens_estimator)

    # calculate ⟨Nₛ⟩ = N⋅n⋅⟨nₛ⟩
    Ns = (N*n)*ns

    # approximate ⟨N⟩² = ⟨(N₊+N₋)⟩² = ⟨(2⋅Nₛ)⟩² = 4⟨Nₛ⟩²
    N2 = 4*Ns^2

    # approximate Tr[G] ≈ ⟨R|GR⟩ / (Nrv*Lτ)
    TrG = dot(R, GR)/(Nrv*Lτ)

    # approximate Tr[G²] ≈ ⟨GR|GR⟩ / (Nrv*Lτ)
    TrGsqrd = dot(GR, GR)/(Nrv*Lτ)

    # approximate ⟨N²⟩ = ⟨N⟩² + 2⋅Tr[G] - 2⋅Tr[G²]
    Nsqrd = N2 + 2*TrG - 2*TrGsqrd

    # restore conjugated random vectors
    @. Rt = conj(R)

    return Nsqrd
end


# measure the double-occupancy for specific spin species
function measure_double_occ(
    greens_estimator::GreensEstimator{T},
    orbital::Int
) where {T}

    Rt′ = selectdim(greens_estimator.Rt, 2, orbital)
    GR′ = selectdim(greens_estimator.GR, 2, orbital)
    d = measure_double_occ(greens_estimator, Rt′, GR′)

    return d
end

# measure the globally averaged double-occupancy
function measure_double_occ(
    greens_estimator::GreensEstimator{T},
    Rt::AbstractArray{Complex{T}} = greens_estimator.Rt,
    GR::AbstractArray{Complex{T}} = greens_estimator.GR
) where {T<:AbstractFloat}

    (; V) = greens_estimator

    # number of random vectors
    Nrv = size(Rt, ndims(Rt))

    # number of random vector pairs
    Npairs = binomial(Nrv, 2)

    # initialize double-occupancy to zero
    d = zero(Complex{T})

    # iterate over all pairs of random vectors
    for i in 1:(Nrv-1)
        GRup = selectdim(GR, ndims(GR), i)
        Rtup = selectdim(Rt, ndims(Rt), i)
        for j in i:Nrv
            GRdn = selectdim(GR, ndims(GR), j)
            Rtdn = selectdim(Rt, ndims(Rt), j)
            # ⟨n₊n₋⟩ = ⟨(1-G₊(0))⋅(1-G₋(0))⟩
            d += sum(
                r -> (1-GRup[r]*Rtup[r]) * (1-GRdn[r]*Rtdn[r]),
                eachindex(Rtup)
            ) / V
        end
    end

    # normalize double occupancy measurement
    d /= Npairs

    return d
end