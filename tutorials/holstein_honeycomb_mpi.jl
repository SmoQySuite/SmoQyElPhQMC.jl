# # 2b) Honeycomb Holstein Model with MPI Parallelization

# ## Import packages
# We now need to import the [MPI.jl](https://github.com/JuliaParallel/MPI.jl.git) package as well.

using SmoQyElPhQMC

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf
using MPI

# ## Specify simulation parameters
# Here we have introduced the `comm` argument to the `run_simulation` function, which is a type exported by the
# [MPI.jl](https://github.com/JuliaParallel/MPI.jl.git) package to facilitate communication and synchronization
# between the different MPI processes.

## Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    ## KEYWORD ARGUMENTS
    sID, # Simulation ID.
    Ω, # Phonon energy.
    α, # Electron-phonon coupling.
    μ, # Chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_updates, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

# ## Initialize simulation
# Now when initializing the [`SimulationInfo`](@ref) type, we also need to include the
# MPI process ID `pID`, which can be retrieved using the
# [`MPI.Comm_rank`](https://juliaparallel.org/MPI.jl/stable/reference/comm/#MPI.Comm_rank)
# function.

# We also the [`initialize_datafolder`](@ref) function such that it takes the `comm` as the
# first argument. This ensures that all the MPI processes remained synchronized, and none
# try proceeding beyond this point until the data folder has been initialized.

    ## Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_honeycomb_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

    ## Get MPI process ID.
    pID = MPI.Comm_rank(comm)

    ## Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,                     
        datafolder_prefix = datafolder_prefix,
        sID = sID,
        pID = pID
    )

    ## Initialize the directory the data will be written to.
    initialize_datafolder(comm, simulation_info)

# ## Initialize simulation metadata
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Initialize random number generator
    rng = Xoshiro(seed)

    ## Initialize additiona_info dictionary
    metadata = Dict()

    ## Record simulation parameters.
    metadata["N_therm"] = N_therm
    metadata["N_updates"] = N_updates
    metadata["N_bins"] = N_bins
    metadata["n_stab"] = n_stab
    metadata["dG_max"] = δG_max
    metadata["symmetric"] = symmetric
    metadata["checkerboard"] = checkerboard
    metadata["seed"] = seed
    metadata["hmc_acceptance_rate"] = 0.0
    metadata["reflection_acceptance_rate"] = 0.0
    metadata["swap_acceptance_rate"] = 0.0

# ## Initialize model
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Define the unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[3/2,√3/2],
                        [3/2,-√3/2]],
        basis_vecs   = [[0.,0.],
                        [1.,0.]]
    )

    ## Define finite lattice with periodic boundary conditions.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    ## Initialize model geometry.
    model_geometry = ModelGeometry(unit_cell, lattice)

    ## Define the first nearest-neighbor bond in a honeycomb lattice.
    bond_1 = lu.Bond(orbitals = (1,2), displacement = [0,0])

    ## Add the first nearest-neighbor bond in a honeycomb lattice to the model.
    bond_1_id = add_bond!(model_geometry, bond_1)

    ## Define the second nearest-neighbor bond in a honeycomb lattice.
    bond_2 = lu.Bond(orbitals = (1,2), displacement = [-1,0])

    ## Add the second nearest-neighbor bond in a honeycomb lattice to the model.
    bond_2_id = add_bond!(model_geometry, bond_2)

    ## Define the third nearest-neighbor bond in a honeycomb lattice.
    bond_3 = lu.Bond(orbitals = (1,2), displacement = [0,-1])

    ## Add the third nearest-neighbor bond in a honeycomb lattice to the model.
    bond_3_id = add_bond!(model_geometry, bond_3)

    ## Set neartest-neighbor hopping amplitude to unity,
    ## setting the energy scale in the model.
    t = 1.0

    ## Define the honeycomb tight-binding model.
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds        = [bond_1, bond_2, bond_3], # defines hopping
        t_mean         = [t, t, t], # defines corresponding hopping amplitude
        μ              = μ, # set chemical potential
        ϵ_mean         = [0.0, 0.0] # set the (mean) on-site energy
    )

    ## Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    ## Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon_1 = PhononMode(orbital = 1, Ω_mean = Ω)

    ## Add the phonon mode definition to the electron-phonon model.
    phonon_1_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_1
    )

    ## Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon_2 = PhononMode(orbital = 2, Ω_mean = Ω)

    ## Add the phonon mode definition to the electron-phonon model.
    phonon_2_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_2
    )

    ## Define first local Holstein coupling for first phonon mode.
    holstein_coupling_1 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_mode = phonon_1_id,
        ## Couple the first phonon mode to first orbital in the unit cell.
        bond = lu.Bond(orbitals = (1,1), displacement = [0, 0]),
        α_mean = α
    )

    ## Add the first local Holstein coupling definition to the model.
    holstein_coupling_1_id = add_holstein_coupling!(
        electron_phonon_model = electron_phonon_model,
        holstein_coupling = holstein_coupling_1,
        model_geometry = model_geometry
    )

    ## Define first local Holstein coupling for first phonon mode.
    holstein_coupling_2 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_mode = phonon_2_id,
        ## Couple the second phonon mode to second orbital in the unit cell.
        bond = lu.Bond(orbitals = (2,2), displacement = [0, 0]),
        α_mean = α
    )

    ## Add the first local Holstein coupling definition to the model.
    holstein_coupling_2_id = add_holstein_coupling!(
        electron_phonon_model = electron_phonon_model,
        holstein_coupling = holstein_coupling_2,
        model_geometry = model_geometry
    )

    ## Write model summary TOML file specifying Hamiltonian that will be simulated.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (electron_phonon_model,)
    )

# ## Initialize model parameters
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Initialize tight-binding parameters.
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    ## Initialize electron-phonon parameters.
    electron_phonon_parameters = ElectronPhononParameters(
        β = β, Δτ = Δτ,
        electron_phonon_model = electron_phonon_model,
        tight_binding_parameters = tight_binding_parameters,
        model_geometry = model_geometry,
        rng = rng
    )

# ## Initialize meuasurements
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Initialize the container that measurements will be accumulated into.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    ## Initialize the tight-binding model related measurements, like the hopping energy.
    initialize_measurements!(measurement_container, tight_binding_model)

    ## Initialize the electron-phonon interaction related measurements.
    initialize_measurements!(measurement_container, electron_phonon_model)

    ## Initialize the single-particle electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens",
        time_displaced = true,
        pairs = [
            ## Measure green's functions for all pairs or orbitals.
            (1, 1), (2, 2), (1, 2)
        ]
    )

    ## Initialize the single-particle electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "phonon_greens",
        time_displaced = true,
        pairs = [
            ## Measure green's functions for all pairs of modes.
            (1, 1), (2, 2), (1, 2)
        ]
    )

    ## Initialize density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = true,
        pairs = [
            (1, 1), (2, 2),
        ]
    )

    ## Initialize the pair correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [
            ## Measure local s-wave pair susceptibility associated with
            ## each orbital in the unit cell.
            (1, 1), (2, 2)
        ]
    )

    ## Initialize the spin-z correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = true,
        pairs = [
            (1, 1), (2, 2)
        ]
    )

    ## Initialize CDW correlation measurement.
    initialize_composite_correlation_measurement!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        name = "cdw",
        correlation = "density",
        ids = [1, 2],
        coefficients = [1.0, -1.0],
        time_displaced = false,
        integrated = true
    )

    ## Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(simulation_info, measurement_container)

# ## Setup DQMC simulation
# In this section of the code we only need to make one very minor change in adding a call to the
# [`MPI.Barrier`](https://juliaparallel.org/MPI.jl/stable/reference/comm/#MPI.Barrier) function
# to synchronize all the MPI processes.
# This ensures that the proper directory structure for the simulation is in place before the simulation begins.

    ## Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # Allocate a single FermionPathIntegral for both spin-up and down electrons.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize FermionPathIntegral type to account for electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Initialize fermion determinant matrix. Also set the default tolerance and max iteration count
    # used in conjugate gradient (CG) solves of linear systems involving this matrix.
    fermion_det_matrix = AsymFermionDetMatrix(
        fermion_path_integral,
        maxiter = maxiter, tol = tol
    )

    # Initialize pseudofermion field calculator.
    pff_calculator = PFFCalculator(electron_phonon_parameters, fermion_det_matrix)

    # Initialize KPM preconditioner.
    kpm_preconditioner = KPMPreconditioner(fermion_det_matrix, rng = rng)

    # Initialize Green's function estimator for making measurements.
    greens_estimator = GreensEstimator(fermion_det_matrix, model_geometry)

    # Integrated trajectory time; one quarter the period of the bare phonon mode.
    Tt = π/(2Ω)

    # Fermionic time-step used in HMC update.
    Δt = Tt/Nt

    hmc_updater = EFAPFFHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        Nt = Nt, Δt = Δt,
        η = 0.0, # Regularization parameter for exact fourier acceleration (EFA)
        δ = 0.05 # Fractional max amplitude of noise added to time-step Δt before each HMC update.
    )

# ## Setup EFA-HMC Updates
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.


    ## Number of fermionic time-steps in HMC update.
    Nt = 10

    ## Fermionic time-step used in HMC update.
    Δt = π/(2*Ω*Nt)

    ## Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt
    )

# ## Thermalize system
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Iterate over number of thermalization updates to perform.
    for update in 1:N_therm

        ## Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        ## Perform a swap update.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        ## Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        ## Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted
    end

# ## Make measurements
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Reset diagonostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetG)
    δθ = zero(sgndetG)

    ## Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    ## Iterate over updates and measurements.
    for update in 1:N_updates

        ## Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        ## Perform a swap update.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        ## Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        ## Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

        ## Make measurements.
        (logdetG, sgndetG, δG, δθ) = make_measurements!(
            measurement_container,
            logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ,
            model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
            coupling_parameters = (electron_phonon_parameters,)
        )

        ## Check if bin averaged measurements need to be written to file.
        if update % bin_size == 0

            ## Write the bin-averaged measurements to file.
            write_measurements!(
                measurement_container = measurement_container,
                simulation_info = simulation_info,
                model_geometry = model_geometry,
                bin = update ÷ bin_size,
                bin_size = bin_size,
                Δτ = Δτ
            )
        end
    end

# ## Record simulation metadata
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_updates + N_therm)
    metadata["reflection_acceptance_rate"] /= (N_updates + N_therm)
    metadata["swap_acceptance_rate"] /= (N_updates + N_therm)

    ## Record largest numerical error encountered during simulation.
    metadata["dG"] = δG

    ## Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

# ## Post-process results
# We need start with section with a call to the [`MPI.Barrier`](https://juliaparallel.org/MPI.jl/stable/reference/comm/#MPI.Barrier)
# function to ensure that we don't begin processing the results until all the simulations running in parallel have finished.
# Additionally, we need to make sure to call the 
# the [`process_measurements`](@ref), [`compute_correlation_ratio`](@ref) and [`compress_jld2_bins`](@ref) function
# such that the first argument is the `comm` object, thereby ensuring a parallelized version of each method is called.

    ## Process the simulation results, calculating final error bars for all measurements,
    ## writing final statisitics to CSV files.
    process_measurements(comm, simulation_info.datafolder, N_bins, time_displaced = true)

    ## Merge binary files containing binned data into a single file.
    compress_jld2_bins(comm, folder = simulation_info.datafolder)

    return nothing
end # end of run_simulation function

# ## Execute script
# Here we first need to initialize MPI using the
# [`MPI.Init`](https://juliaparallel.org/MPI.jl/stable/reference/environment/#MPI.Init) command.
# Then, we need to make sure to pass the `comm = MPI.COMM_WORLD` to the `run_simulation` function.
# At the very end of simulation it is good practice to run the `MPI.Finalize()` function even though
# it is typically not strictly required.

## Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    ## Initialize MPI
    MPI.Init()

    ## Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    ## Run the simulation.
    run_simulation(
        comm;
        sID       = parse(Int,     ARGS[1]), # Simulation ID.
        Ω         = parse(Float64, ARGS[2]), # Phonon energy.
        α         = parse(Float64, ARGS[3]), # Electron-phonon coupling.
        μ         = parse(Float64, ARGS[4]), # Chemical potential.
        L         = parse(Int,     ARGS[5]), # System size.
        β         = parse(Float64, ARGS[6]), # Inverse temperature.
        N_therm   = parse(Int,     ARGS[7]), # Number of thermalization updates.
        N_updates = parse(Int,     ARGS[8]), # Total number of measurements and measurement updates.
        N_bins    = parse(Int,     ARGS[9])  # Number of times bin-averaged measurements are written to file.
    )

    ## Finalize MPI.
    MPI.Finalize()
end

# Here is an example of what the command to run this script might look like:
# ```bash
# mpiexecjl -n 16 julia holstein_honeycomb_mpi.jl 1 1.0 1.5 0.0 3 4.0 5000 10000 100
# ```
# This will 16 MPI processes, each running and independent simulation using a different random seed
# the the final results arrived at by averaging over all 16 walkers.
# Here `mpiexecjl` is the MPI exectuable that can be easily install using the directions
# found [here](https://juliaparallel.org/MPI.jl/stable/usage/#Julia-wrapper-for-mpiexec) in the
# [MPI.jl](https://github.com/JuliaParallel/MPI.jl) documentation. However, you can substitute a
# different MPI executable here if one is already configured on your system.

# Also, when submitting jobs via [SLURM](https://slurm.schedmd.com/documentation.html)
# on a High-Performance Computing (HPC) cluster, if a default MPI exectuable
# is already configured on the system, as is frequently the case, then the script can likely be run inside the
# `*.sh` job file using the [`srun`](https://slurm.schedmd.com/srun.html) command:
# ```bash
# srun julia holstein_honeycomb_mpi.jl 1 1.0 1.5 0.0 3 4.0 5000 10000 100
# ```
# The `srun` command should automatically detect the number of available cores requested by the job and run
# the script using the MPI executable with the appropriate number of processes.