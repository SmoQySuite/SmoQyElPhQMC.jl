# measure density for single-spin species and specified orbital species
function measure_n(
    greens_estimator::GreensEstimator{T},
    orbital::Int
) where {T}

    Rc′ = selectdim(greens_estimator.Rc, 2, orbital)
    GR′ = selectdim(greens_estimator.GR, 2, orbital)
    measure_n(greens_estimator, Rc′, GR′)

    return n
end

# measure density for single spin-species
function measure_n(
    greens_estimator::GreensEstimator{T},
    Rc::AbstractArray{Complex{T}} = greens_estimator.Rc,
    GR::AbstractArray{Complex{T}} = greens_estimator.GR
) where {T<:AbstractFloat}
    
    R = Rc
    @. R = conj(Rc)
    n = 1 - dot(R, GR)/length(R)
    @. Rc = conj(R)

    return n
end


# measure ⟨N²⟩ globally
function measure_Nsqrd(
    greens_estimator::GreensEstimator
)

    (; V, N, n, Lτ, Nrv) = greens_estimator
    R = Rc
    @. R = conj(Rc)

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

    # approximate ⟨N²⟩ = ⟨N⟩² + 2Tr[G] - 2Tr[G²]
    Nsqrd = N2 + 2*TrG - 2*TrGsqrd

    # restore conjugated random vectors
    @. Rc = conj(Rc)

    return Nsqrd
end


# measure the double-occupancy for specific spin species
function measure_double_occ(
    greens_estimator::GreensEstimator{T},
    orbital::Int
) where {T}

    Rc′ = selectdim(greens_estimator.Rc, 2, orbital)
    GR′ = selectdim(greens_estimator.GR, 2, orbital)
    measure_double_occ(greens_estimator, Rc′, GR′)

    return d
end

# measure the globally averaged double-occupancy
function measure_double_occ(
    greens_estimator::GreensEstimator{T},
    Rc::AbstractArray{Complex{T}} = greens_estimator.Rc,
    GR::AbstractArray{Complex{T}} = greens_estimator.GR
) where {T<:AbstractFloat}

    (; V) = greens_estimator

    # number of random vectors
    Nrv = size(Rc, ndims(Rc))

    # number of random vector pairs
    Npairs = binomial(Nrv, 2)

    # initialize double-occupancy to zero
    d = zero(Complex{T})

    # iterate over all pairs of random vectors
    for i in 1:(Nrv-1)
        GRup = selectdim(GR, ndims(GR), i)
        Rup = selectdim(Rc, ndims(Rc), i)
        for j in i:Nrv
            GRdn = selectdim(GR, ndims(GR), j)
            Rdn = selectdim(Rc, ndims(Rc), j)
            # ⟨n₊n₋⟩ = ⟨(1-G₊(0))⋅(1-G₋(0))⟩
            d += sum(
                r -> (1-Rup[r]*GRup[r]) * (1-Rdn[r]*GRdn[r]),
                eachindex(Rup)
            ) / V
        end
    end

    # normalize double occupancy measurement
    d /= Npairs

    return d
end