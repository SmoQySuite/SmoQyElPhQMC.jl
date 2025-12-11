
# ## Import packages
# We now need to import the [MPI.jl](https://github.com/JuliaParallel/MPI.jl.git) package as well.

using SmoQyElPhQMC
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu

using LinearAlgebra
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
    Nt = 25, # Number of time-steps in HMC update.
    Nrv = 10, # Number of random vectors used to estimate fermionic correlation functions.
    tol = 1e-10, # CG iterations tolerance.
    maxiter = 10_000, # Maximum number of CG iterations.
    write_bins_concurrent = true, # Whether to write HDF5 bins during the simulation.
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
        write_bins_concurrent = write_bins_concurrent,
        sID = sID,
        pID = pID
    )

    ## Initialize the directory the data will be written to.
    initialize_datafolder(comm, simulation_info)

# ## Initialize simulation metadata
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Initialize random number generator
    rng = Xoshiro(seed)

    ## Initialize metadata dictionary
    metadata = Dict()

    ## Record simulation parameters.
    metadata["N_therm"] = N_therm  # Number of thermalization updates
    metadata["N_updates"] = N_updates  # Total number of measurements and measurement updates
    metadata["N_bins"] = N_bins # Number of times bin-averaged measurements are written to file
    metadata["maxiter"] = maxiter # Maximum number of conjugate gradient iterations
    metadata["tol"] = tol # Tolerance used for conjugate gradient solves
    metadata["Nt"] = Nt # Number of time-steps in HMC update
    metadata["Nrv"] = Nrv # Number of random vectors used to estimate fermionic correlation functions
    metadata["seed"] = seed  # Random seed used to initialize random number generator in simulation
    metadata["hmc_acceptance_rate"] = 0.0 # HMC acceptance rate
    metadata["reflection_acceptance_rate"] = 0.0 # Reflection update acceptance rate
    metadata["swap_acceptance_rate"] = 0.0 # Swap update acceptance rate
    metadata["hmc_iters"] = 0.0 # Avg number of CG iterations per solve in HMC update.
    metadata["reflection_iters"] = 0.0 # Avg number of CG iterations per solve in reflection update.
    metadata["swap_iters"] = 0.0 # Avg number of CG iterations per solve in swap update.
    metadata["measurement_iters"] = 0.0 # Avg number of CG iterations per solve while making measurements.

# ## Initialize model
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Define lattice vectors.
    a1 = [+3/2, +√3/2]
    a2 = [+3/2, -√3/2]

    ## Define basis vectors for two orbitals in the honeycomb unit cell.
    r1 = [0.0, 0.0] # Location of first orbital in unit cell.
    r2 = [1.0, 0.0] # Location of second orbital in unit cell.

    ## Define the unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [a1, a2],
        basis_vecs   = [r1, r2]
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

    ## Set nearest-neighbor hopping amplitude to unity,
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
    phonon_1 = PhononMode(
        basis_vec = r1,
        Ω_mean = Ω
    )

    ## Add the phonon mode definition to the electron-phonon model.
    phonon_1_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_1
    )

    ## Define a dispersionless electron-phonon mode to live on the second sublattice.
    phonon_2 = PhononMode(
        basis_vec = r2,
        Ω_mean = Ω
    )

    ## Add the phonon mode definition to the electron-phonon model.
    phonon_2_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_2
    )

    ## Define first local Holstein coupling for first phonon mode.
    holstein_coupling_1 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_id = phonon_1_id,
        orbital_id = 1,
        displacement = [0, 0],
        α_mean = α,
        ph_sym_form = true,
    )

    ## Add the first local Holstein coupling definition to the model.
    holstein_coupling_1_id = add_holstein_coupling!(
        electron_phonon_model = electron_phonon_model,
        holstein_coupling = holstein_coupling_1,
        model_geometry = model_geometry
    )

    ## Define second local Holstein coupling for second phonon mode.
    holstein_coupling_2 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_id = phonon_2_id,
        orbital_id = 2,
        displacement = [0, 0],
        α_mean = α,
        ph_sym_form = true,
    )

    ## Add the second local Holstein coupling definition to the model.
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

# ## Initialize measurements
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

    ## Initialize measurement of electron Green's function traced
    ## over both orbitals in the unit cell.
    initialize_composite_correlation_measurement!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        name = "tr_greens",
        correlation = "greens",
        id_pairs = [(1,1), (2,2)],
        coefficients = [1.0, 1.0],
        time_displaced = true,
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

# ## Setup QMC simulation
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Allocate a single FermionPathIntegral for both spin-up and down electrons.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    ## Initialize FermionPathIntegral type to account for electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    ## Initialize fermion determinant matrix. Also set the default tolerance and max iteration count
    ## used in conjugate gradient (CG) solves of linear systems involving this matrix.
    fermion_det_matrix = SymFermionDetMatrix(
        fermion_path_integral,
        maxiter = maxiter, tol = tol
    )

    ## Initialize pseudofermion field calculator.
    pff_calculator = PFFCalculator(electron_phonon_parameters, fermion_det_matrix)

    ## Initialize KPM preconditioner.
    preconditioner = KPMPreconditioner(fermion_det_matrix, rng = rng)

    ## Initialize Green's function estimator for making measurements.
    greens_estimator = GreensEstimator(fermion_det_matrix, model_geometry)

# ## Setup EFA-PFF-HMC Updates
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Initialize Hamiltonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAPFFHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        Nt = Nt, Δt = π/(2*Nt)
    )

# ## Thermalize system
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Iterate over number of thermalization updates to perform.
    for n in 1:N_therm

        ## Perform a reflection update.
        (accepted, iters) = reflection_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        ## Record the number of CG iterations performed for the reflection update.
        metadata["reflection_iters"] += iters

        ## Perform a swap update.
        (accepted, iters) = swap_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        ## Record the number of CG iterations performed for the reflection update.
        metadata["swap_iters"] += iters

        ## Perform an HMC update.
        (accepted, iters) = hmc_update!(
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            pff_calculator = pff_calculator,
            preconditioner = preconditioner,
            tol_action = tol, tol_force = sqrt(tol), maxiter = maxiter,
            rng = rng,
        )

        ## Record the average number of iterations per CG solve for hmc update.
        metadata["hmc_acceptance_rate"] += accepted

        ## Record the number of CG iterations performed for the reflection update.
        metadata["hmc_iters"] += iters
    end

# ## Make measurements
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    ## Iterate over bins.
    for update in 1:N_updates

        ## Perform a reflection update.
        (accepted, iters) = reflection_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        ## Record the number of CG iterations performed for the reflection update.
        metadata["reflection_iters"] += iters

        ## Perform a swap update.
        (accepted, iters) = swap_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        ## Record the number of CG iterations performed for the reflection update.
        metadata["swap_iters"] += iters

        ## Perform an HMC update.
        (accepted, iters) = hmc_update!(
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            pff_calculator = pff_calculator,
            preconditioner = preconditioner,
            tol_action = tol, tol_force = sqrt(tol), maxiter = maxiter,
            rng = rng,
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

        ## Record the average number of iterations per CG solve for hmc update.
        metadata["hmc_iters"] += iters

        ## Make measurements.
        iters = make_measurements!(
            measurement_container, fermion_det_matrix, greens_estimator,
            model_geometry = model_geometry,
            fermion_path_integral = fermion_path_integral,
            tight_binding_parameters = tight_binding_parameters,
            electron_phonon_parameters = electron_phonon_parameters,
            preconditioner = preconditioner,
            tol = tol, maxiter = maxiter,
            rng = rng
        )

        ## Record the average number of iterations per CG solve for measurements.
        metadata["measurement_iters"] += iters

        ## Write the bin-averaged measurements to file if update ÷ bin_size == 0.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            measurement = update,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

# ## Merge binned data
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

# ## Record simulation metadata
# No changes need to made to this section of the code from the previous [2a) Honeycomb Holstein Model](@ref) tutorial.

    ## Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_updates + N_therm)
    metadata["reflection_acceptance_rate"] /= (N_updates + N_therm)
    metadata["swap_acceptance_rate"] /= (N_updates + N_therm)

    ## Calculate average number of CG iterations.
    metadata["hmc_iters"] /= (N_updates + N_therm)
    metadata["reflection_iters"] /= (N_updates + N_therm)
    metadata["swap_iters"] /= (N_updates + N_therm)
    metadata["measurement_iters"] /= N_updates

    ## Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

# ## Post-process results
# The main change we need to make from the previous [2a) Honeycomb Holstein Model](@ref) tutorial is to call
# the [`process_measurements`](@ref) and [`compute_composite_correlation_ratio`](@ref) functions
# such that the first argument is the `comm` object, thereby ensuring a parallelized version of each method is called.

    ## Process the simulation results, calculating final error bars for all measurements.
    ## writing final statistics to CSV files.
    process_measurements(
        comm,
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 7,
        delimiter = " "
    )

    ## Calculate CDW correlation ratio.
    Rcdw, ΔRcdw = compute_composite_correlation_ratio(
        datafolder = simulation_info.datafolder,
        name = "cdw",
        type = "equal-time",
        q_point = (0, 0),
        q_neighbors = [
            (1,0),   (0,1),   (1,1),
            (L-1,0), (0,L-1), (L-1,L-1)
        ]
    )

    ## Record the AFM correlation ratio mean and standard deviation.
    metadata["Rcdw_mean_real"] = real(Rcdw)
    metadata["Rcdw_mean_imag"] = imag(Rcdw)
    metadata["Rcdw_std"]       = ΔRcdw

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    return nothing
end # end of run_simulation function

# ## Execute script
# Here we first need to initialize MPI using the
# [`MPI.Init`](https://juliaparallel.org/MPI.jl/stable/reference/environment/#MPI.Init) command.
# Then, we need to make sure to pass the `comm = MPI.COMM_WORLD` to the `run_simulation` function.
# At the very end of simulation it is good practice to run the `MPI.Finalize()` function even though
# it is typically not strictly required.

## Only execute if the script is run directly from the command line.
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
# Here `mpiexecjl` is the MPI executable that can be easily install using the directions
# found [here](https://juliaparallel.org/MPI.jl/stable/usage/#Julia-wrapper-for-mpiexec) in the
# [MPI.jl](https://github.com/JuliaParallel/MPI.jl) documentation. However, you can substitute a
# different MPI executable here if one is already configured on your system.

# Also, when submitting jobs via [SLURM](https://slurm.schedmd.com/documentation.html)
# on a High-Performance Computing (HPC) cluster, if a default MPI executable
# is already configured on the system, as is frequently the case, then the script can likely be run inside the
# `*.sh` job file using the [`srun`](https://slurm.schedmd.com/srun.html) command:
# ```bash
# srun julia holstein_honeycomb_mpi.jl 1 1.0 1.5 0.0 3 4.0 5000 10000 100
# ```
# The `srun` command should automatically detect the number of available cores requested by the job and run
# the script using the MPI executable with the appropriate number of processes.