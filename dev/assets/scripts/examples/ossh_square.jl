using SmoQyElPhQMC
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu

using Random
using Printf
using MPI

# Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    Ω, # Phonon energy.
    α, # Electron-phonon coupling.
    μ, # Chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_updates, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
    checkpoint_freq, # Frequency with which checkpoint files are written in hours.
    runtime_limit = Inf, # Simulation runtime limit in hours.
    Δτ = 0.05, # Discretization in imaginary time.
    Nt = 25, # Number of time-steps in HMC update.
    Nrv = 10, # Number of random vectors used to estimate fermionic correlation functions.
    tol = 1e-10, # CG iterations tolerance.
    maxiter = 10_000, # Maximum number of CG iterations.
    write_bins_concurrent = true, # Whether to write the HDF5 bins files during the simulation.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

    # Record when the simulation began.
    start_timestamp = time()

    # Convert runtime limit from hours to seconds.
    runtime_limit = runtime_limit * 60.0^2

    # Convert checkpoint frequency from hours to seconds.
    checkpoint_freq = checkpoint_freq * 60.0^2

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "ossh_square_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

    # Get MPI process ID.
    pID = MPI.Comm_rank(comm)

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = write_bins_concurrent,
        sID = sID,
        pID = pID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(comm, simulation_info)

    # If starting a new simulation i.e. not resuming a previous simulation.
    if !simulation_info.resuming

        # Begin thermalization updates from start.
        n_therm = 1

        # Begin measurement updates from start.
        n_updates = 1

        # Initialize random number generator
        rng = Xoshiro(seed)

        # Initialize metadata dictionary
        metadata = Dict()

        # Record simulation parameters.
        metadata["Nt"] = Nt
        metadata["N_therm"] = N_therm
        metadata["N_updates"] = N_updates
        metadata["N_bins"] = N_bins
        metadata["Nrv"] = Nrv
        metadata["maxiter"] = maxiter
        metadata["tol"] = tol
        metadata["seed"] = seed
        metadata["hmc_acceptance_rate"] = 0.0
        metadata["reflection_acceptance_rate"] = 0.0
        metadata["swap_acceptance_rate"] = 0.0
        metadata["hmc_iters"] = 0.0
        metadata["reflection_iters"] = 0.0
        metadata["swap_iters"] = 0.0
        metadata["measurement_iters"] = 0.0

        # Initialize an instance of the type UnitCell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [[1.0, 0.0],
                            [0.0, 1.0]],
            basis_vecs   = [[0.0, 0.0]]
        )

        # Initialize an instance of the type Lattice.
        lattice = lu.Lattice(
            L = [L,L],
            periodic = [true,true]
        )

        # Get the number of sites in the lattice.
        N = lu.nsites(unit_cell, lattice)

        # Initialize an instance of the ModelGeometry type.
        model_geometry = ModelGeometry(unit_cell, lattice)

        # Define the nearest-neighbor bond in the x-direction.
        bond_px = lu.Bond(orbitals = (1,1), displacement = [1,0])

        # Add this bond in x-direction to the model geometry.
        bond_px_id = add_bond!(model_geometry, bond_px)

        # Define the nearest-neighbor bond in the y-direction.
        bond_py = lu.Bond(orbitals = (1,1), displacement = [0,1])

        # Add this bond in y-direction to the model geometry.
        bond_py_id = add_bond!(model_geometry, bond_py)

        # Define the nearest-neighbor bond in the -x-direction.
        bond_nx = lu.Bond(orbitals = (1,1), displacement = [-1,0])

        # Add this bond in +x-direction to the model geometry.
        bond_nx_id = add_bond!(model_geometry, bond_nx)

        # Define the nearest-neighbor bond in the -y-direction.
        bond_ny = lu.Bond(orbitals = (1,1), displacement = [0,-1])

        # Add this bond in +y-direction to the model geometry.
        bond_ny_id = add_bond!(model_geometry, bond_ny)

        # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
        t = 1.0

        # Define the tight-binding model
        tight_binding_model = TightBindingModel(
            model_geometry = model_geometry,
            t_bonds = [bond_px, bond_py], # defines hopping
            t_mean = [t, t], # defines corresponding hopping amplitude
            μ = μ, # set chemical potential
            ϵ_mean = [0.] # set the (mean) on-site energy
        )

        # Initialize a null electron-phonon model.
        electron_phonon_model = ElectronPhononModel(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model
        )

        # Define a dispersionless phonon mode to represent vibrations in the x-direction.
        phonon_x = PhononMode(
            basis_vec = [0.0,0.0],
            Ω_mean = Ω
        )

        # Add x-direction optical ssh phonon to electron-phonon model.
        phonon_x_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_x
        )

        # Define a dispersionless phonon mode to represent vibrations in the y-direction.
        phonon_y = PhononMode(
            basis_vec = [0.0,0.0],
            Ω_mean = Ω
        )

        # Add y-direction optical ssh phonon to electron-phonon model.
        phonon_y_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_y
        )

        # Defines ssh e-ph coupling such that total effective hopping.
        ossh_x_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_x_id, phonon_x_id),
            bond = bond_px,
            α_mean = α
        )

        # Add x-direction optical SSH coupling to the electron-phonon model.
        ossh_x_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_x_coupling,
            tight_binding_model = tight_binding_model
        )

        # Defines ssh e-ph coupling such that total effective hopping.
        ossh_y_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_y_id, phonon_y_id),
            bond = bond_py,
            α_mean = α
        )

        # Add y-direction optical SSH coupling to the electron-phonon model.
        ossh_y_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_y_coupling,
            tight_binding_model = tight_binding_model
        )

        # Write a model summary to file.
        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (electron_phonon_model,)
        )

        # Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize electron-phonon parameters.
        electron_phonon_parameters = ElectronPhononParameters(
            β = β, Δτ = Δτ,
            electron_phonon_model = electron_phonon_model,
            tight_binding_parameters = tight_binding_parameters,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        # Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        # Initialize the electron-phonon interaction related measurements.
        initialize_measurements!(measurement_container, electron_phonon_model)

        # Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens",
            time_displaced = true,
            pairs = [
                # Measure green's functions for all pairs or orbitals.
                (1, 1),
            ]
        )

        # Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "phonon_greens",
            time_displaced = true,
            pairs = [
                # Measure green's functions for all pairs of modes.
                (phonon_x_id, phonon_x_id), (phonon_y_id, phonon_y_id),
            ]
        )

        # Initialize density correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "density",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1),
            ]
        )

        # Initialize the pair correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "pair",
            time_displaced = false,
            integrated = true,
            pairs = [
                # Measure local s-wave pair susceptibility associated with
                # each orbital in the unit cell.
                (1, 1),
            ]
        )

        # Initialize the spin-z correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "spin_z",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1),
            ]
        )

        # Initialize the bond correlation measurement
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "bond",
            time_displaced = false,
            integrated = true,
            pairs = [
                (bond_px_id, bond_px_id),
                (bond_py_id, bond_py_id),
                (bond_px_id, bond_py_id),
            ]
        )

        # Measure composite bond correlation for detecting a bond ordered wave (BOW)
        # that breaks a C4 rotation symmetry.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "BOW_C4",
            correlation = "bond",
            ids = [bond_px_id, bond_py_id, bond_nx_id, bond_ny_id],
            coefficients = [+1.0, +1.0im, -1.0, -1.0im],
            displacement_vecs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            time_displaced = false,
            integrated = true
        )

        # Measure composite bond correlation for detecting a bond ordered wave (BOW)
        # that breaks a C2 rotation symmetry.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "BOW_C2",
            correlation = "bond",
            ids = [bond_px_id, bond_py_id, bond_nx_id, bond_ny_id],
            coefficients = [+1.0, -1.0, +1.0, -1.0],
            displacement_vecs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            time_displaced = false,
            integrated = true
        )

        # Write initial checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm, n_updates,
            tight_binding_parameters, electron_phonon_parameters,
            measurement_container, model_geometry, metadata, rng
        )

    # If resuming a previous simulation.
    else

        # Load the checkpoint file.
        checkpoint, checkpoint_timestamp = read_jld2_checkpoint(simulation_info)

        # Unpack contents of checkpoint dictionary.
        tight_binding_parameters    = checkpoint["tight_binding_parameters"]
        electron_phonon_parameters  = checkpoint["electron_phonon_parameters"]
        measurement_container       = checkpoint["measurement_container"]
        model_geometry              = checkpoint["model_geometry"]
        metadata                    = checkpoint["metadata"]
        rng                         = checkpoint["rng"]
        n_therm                     = checkpoint["n_therm"]
        n_updates                   = checkpoint["n_updates"]
    end

    # Allocate a single FermionPathIntegral for both spin-up and down electrons.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize FermionPathIntegral type to account for electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Initialize fermion determinant matrix. Also set the default tolerance and max iteration count
    # used in conjugate gradient (CG) solves of linear systems involving this matrix.
    fermion_det_matrix = SymFermionDetMatrix(
        fermion_path_integral,
        maxiter = maxiter, tol = tol
    )

    # Initialize pseudofermion field calculator.
    pff_calculator = PFFCalculator(electron_phonon_parameters, fermion_det_matrix)

    # Initialize KPM preconditioner.
    preconditioner = KPMPreconditioner(fermion_det_matrix, rng = rng)

    # Initialize Green's function estimator for making measurements.
    greens_estimator = GreensEstimator(fermion_det_matrix, model_geometry)

    # Initialize Hamiltonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAPFFHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        Nt = Nt, Δt = π/(2*Nt)
    )

    # Iterate over number of thermalization updates to perform.
    for update in n_therm:N_therm

        # Perform a reflection update.
        (accepted, iters) = reflection_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        metadata["reflection_iters"] += iters

        # Perform a swap update.
        (accepted, iters) = swap_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        metadata["swap_iters"] += iters

        # Perform an HMC update.
        (accepted, iters) = hmc_update!(
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            pff_calculator = pff_calculator,
            preconditioner = preconditioner,
            tol_action = tol, tol_force = sqrt(tol), maxiter = maxiter,
            rng = rng,
        )

        # Record the average number of iterations per CG solve for hmc update.
        metadata["hmc_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        metadata["hmc_iters"] += iters

        # Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm  = update + 1,
            n_updates = 1,
            tight_binding_parameters, electron_phonon_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    # Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    # Iterate over updates and measurements.
    for update in n_updates:N_updates

        # Perform a reflection update.
        (accepted, iters) = reflection_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        metadata["reflection_iters"] += iters

        # Perform a swap update.
        (accepted, iters) = swap_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        metadata["swap_iters"] += iters

        # Perform an HMC update.
        (accepted, iters) = hmc_update!(
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            pff_calculator = pff_calculator,
            preconditioner = preconditioner,
            tol_action = tol, tol_force = sqrt(tol), maxiter = maxiter,
            rng = rng,
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

        # Record the average number of iterations per CG solve for hmc update.
        metadata["hmc_iters"] += iters

        # Make measurements.
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

        # Record the average number of iterations per CG solve for measurements.
        metadata["measurement_iters"] += iters

        # Write the bin-averaged measurements to file if update ÷ bin_size == 0.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            measurement = update,
            bin_size = bin_size,
            Δτ = Δτ
        )

        # Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm  = N_therm + 1,
            n_updates = update + 1,
            tight_binding_parameters, electron_phonon_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    # Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

    # Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_updates + N_therm)
    metadata["reflection_acceptance_rate"] /= (N_updates + N_therm)
    metadata["swap_acceptance_rate"] /= (N_updates + N_therm)

    # Calculate average number of CG iterations.
    metadata["hmc_iters"] /= (N_updates + N_therm)
    metadata["reflection_iters"] /= (N_updates + N_therm)
    metadata["swap_iters"] /= (N_updates + N_therm)
    metadata["measurement_iters"] /= N_updates

    # Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

    # Process the simulation results, calculating final error bars for all measurements.
    # writing final statistics to CSV files.
    process_measurements(
        comm;
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 9,
        delimiter = ", "
    )

    # Calculate C4 BOW q=(π,π) correlation ratio.
    Rbow, ΔRbow = compute_composite_correlation_ratio(
        comm;
        datafolder = simulation_info.datafolder,
        name = "BOW_C4",
        type = "equal-time",
        q_point = (L÷2, L÷2),
        q_neighbors = [
            (L÷2+1, L÷2), (L÷2, L÷2+1),
            (L÷2-1, L÷2), (L÷2, L÷2-1)
        ]
    )

    # Record the correlation ratio.
    metadata["Rbow_mean_real"] = real(Rbow)
    metadata["Rbow_mean_imag"] = imag(Rbow)
    metadata["Rbow_std"] = ΔRbow

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    # Rename the data folder to indicate the simulation is complete.
    simulation_info = rename_complete_simulation(
        comm, simulation_info,
        delete_jld2_checkpoints = true
    )

    return nothing
end # end of run_simulation function

# Only execute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Initialize MPI
    MPI.Init()

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Run the simulation.
    run_simulation(
        comm;
        sID             = parse(Int,     ARGS[1]),  # Simulation ID.
        Ω               = parse(Float64, ARGS[2]),  # Phonon energy.
        α               = parse(Float64, ARGS[3]),  # Electron-phonon coupling.
        μ               = parse(Float64, ARGS[4]),  # Chemical potential.
        L               = parse(Int,     ARGS[5]),  # System size.
        β               = parse(Float64, ARGS[6]),  # Inverse temperature.
        N_therm         = parse(Int,     ARGS[7]),  # Number of thermalization updates.
        N_updates       = parse(Int,     ARGS[8]),  # Total number of measurements and measurement updates.
        N_bins          = parse(Int,     ARGS[9]),  # Number of times bin-averaged measurements are written to file.
        checkpoint_freq = parse(Float64, ARGS[10]), # Frequency with which checkpoint files are written in hours.
    )

    # Finalize MPI.
    MPI.Finalize()
end
