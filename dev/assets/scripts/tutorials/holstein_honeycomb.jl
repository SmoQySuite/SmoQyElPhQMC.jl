using SmoQyElPhQMC

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu

using Random
using Printf

# Top-level function to run simulation.
function run_simulation(;
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
    Δτ = 0.05, # Discretization in imaginary time.
    Nt = 100, # Numer of time-steps in HMC update.
    Nrv = 10, # Number of random vectors used to estimate fermionic correlation functions.
    tol = 1e-10, # CG iterations tolerance.
    maxiter = 1000, # Maximum number of CG iterations.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_honeycomb_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

    # Initialize random number generator
    rng = Xoshiro(seed)

    # Initialize additiona_info dictionary
    additional_info = Dict()

    # Record simulation parameters.
    additional_info["N_therm"]   = N_therm    # Number of thermalization updates
    additional_info["N_updates"] = N_updates  # Total number of measurements and measurement updates
    additional_info["N_bins"]    = N_bins     # Number of times bin-averaged measurements are written to file
    additional_info["maxiter"]   = maxiter    # Maximum number of conjugate gradient iterations
    additional_info["tol"]       = tol        # Tolerance used for conjugate gradient solves
    additional_info["Nt"]        = Nt         # Number of time-steps in HMC update
    additional_info["Nrv"]       = Nrv        # Number of random vectors used to estimate fermionic correlation functions
    additional_info["seed"]      = seed       # Random seed used to initialize random number generator in simulation

    # Define the unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[3/2,√3/2],
                        [3/2,-√3/2]],
        basis_vecs   = [[0.,0.],
                        [1.,0.]]
    )

    # Define finite lattice with periodic boundary conditions.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    # Initialize model geometry.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Define the first nearest-neighbor bond in a honeycomb lattice.
    bond_1 = lu.Bond(orbitals = (1,2), displacement = [0,0])

    # Add the first nearest-neighbor bond in a honeycomb lattice to the model.
    bond_1_id = add_bond!(model_geometry, bond_1)

    # Define the second nearest-neighbor bond in a honeycomb lattice.
    bond_2 = lu.Bond(orbitals = (1,2), displacement = [-1,0])

    # Add the second nearest-neighbor bond in a honeycomb lattice to the model.
    bond_2_id = add_bond!(model_geometry, bond_2)

    # Define the third nearest-neighbor bond in a honeycomb lattice.
    bond_3 = lu.Bond(orbitals = (1,2), displacement = [0,-1])

    # Add the third nearest-neighbor bond in a honeycomb lattice to the model.
    bond_3_id = add_bond!(model_geometry, bond_3)

    # Set neartest-neighbor hopping amplitude to unity,
    # setting the energy scale in the model.
    t = 1.0

    # Define the honeycomb tight-binding model.
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds        = [bond_1, bond_2, bond_3], # defines hopping
        t_mean         = [t, t, t], # defines corresponding hopping amplitude
        μ              = μ, # set chemical potential
        ϵ_mean         = [0.0, 0.0] # set the (mean) on-site energy
    )

    # Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon_1 = PhononMode(orbital = 1, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_1_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_1
    )

    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon_2 = PhononMode(orbital = 2, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_2_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_2
    )

    # Define first local Holstein coupling for first phonon mode.
    holstein_coupling_1 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_mode = phonon_1_id,
        # Couple the first phonon mode to first orbital in the unit cell.
        bond = lu.Bond(orbitals = (1,1), displacement = [0, 0]),
        α_mean = α
    )

    # Add the first local Holstein coupling definition to the model.
    holstein_coupling_1_id = add_holstein_coupling!(
        electron_phonon_model = electron_phonon_model,
        holstein_coupling = holstein_coupling_1,
        model_geometry = model_geometry
    )

    # Define first local Holstein coupling for first phonon mode.
    holstein_coupling_2 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_mode = phonon_2_id,
        # Couple the second phonon mode to second orbital in the unit cell.
        bond = lu.Bond(orbitals = (2,2), displacement = [0, 0]),
        α_mean = α
    )

    # Add the first local Holstein coupling definition to the model.
    holstein_coupling_2_id = add_holstein_coupling!(
        electron_phonon_model = electron_phonon_model,
        holstein_coupling = holstein_coupling_2,
        model_geometry = model_geometry
    )

    # Write model summary TOML file specifying Hamiltonian that will be simulated.
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
            (1, 1), (2, 2), (1, 2)
        ]
    )

    # Initialize the single-particle electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "phonon_greens",
        time_displaced = true,
        pairs = [
            # Measure green's functions for all pairs or orbitals.
            (1, 1), (2, 2), (1, 2)
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
            (1, 1), (2, 2),
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
            (1, 1), (2, 2)
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
            (1, 1), (2, 2)
        ]
    )

    # Initialize CDW correlation measurement.
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

    # Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(simulation_info, measurement_container)

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

    # Initialize variables to record acceptance rates for various udpates.
    additional_info["hmc_acceptance_rate"] = 0.0
    additional_info["reflection_acceptance_rate"] = 0.0
    additional_info["swap_acceptance_rate"] = 0.0

    # Initialize variables to record the average number of CG iterations
    # for each type of update and measurements.
    additional_info["hmc_iters"] = 0.0
    additional_info["reflection_iters"] = 0.0
    additional_info["swap_iters"] = 0.0
    additional_info["measurement_iters"] = 0.0

    # Iterate over number of thermalization updates to perform.
    for n in 1:N_therm

        # Perform a reflection update.
        (accepted, iters) = reflection_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = kpm_preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        # Record whether the reflection update was accepted or rejected.
        additional_info["reflection_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        additional_info["reflection_iters"] += iters

        # Perform a swap update.
        (accepted, iters) = swap_update!(
            electron_phonon_parameters, pff_calculator,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            preconditioner = kpm_preconditioner,
            rng = rng, tol = tol, maxiter = maxiter
        )

        # Record whether the reflection update was accepted or rejected.
        additional_info["swap_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        additional_info["swap_iters"] += iters

        # Perform an HMC update.
        (accepted, iters) = hmc_update!(
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            pff_calculator = pff_calculator,
            preconditioner = kpm_preconditioner,
            tol_action = tol, tol_force = sqrt(tol), maxiter = maxiter,
            rng = rng,
        )

        # Record the average number of iterations per CG solve for hmc update.
        additional_info["hmc_acceptance_rate"] += accepted

        # Record the number of CG iterations performed for the reflection update.
        additional_info["hmc_iters"] += iters
    end

    # Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    # Iterate over bins.
    for bin in 1:N_bins

        # Iterate over update sweeps and measurements in bin.
        for n in 1:bin_size

            # Perform a reflection update.
            (accepted, iters) = reflection_update!(
                electron_phonon_parameters, pff_calculator,
                fermion_path_integral = fermion_path_integral,
                fermion_det_matrix = fermion_det_matrix,
                preconditioner = kpm_preconditioner,
                rng = rng, tol = tol, maxiter = maxiter
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["reflection_acceptance_rate"] += accepted

            # Record the number of CG iterations performed for the reflection update.
            additional_info["reflection_iters"] += iters

            # Perform a swap update.
            (accepted, iters) = swap_update!(
                electron_phonon_parameters, pff_calculator,
                fermion_path_integral = fermion_path_integral,
                fermion_det_matrix = fermion_det_matrix,
                preconditioner = kpm_preconditioner,
                rng = rng, tol = tol, maxiter = maxiter
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["swap_acceptance_rate"] += accepted

            # Record the number of CG iterations performed for the reflection update.
            additional_info["swap_iters"] += iters

            # Perform an HMC update.
            (accepted, iters) = hmc_update!(
                electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_det_matrix = fermion_det_matrix,
                pff_calculator = pff_calculator,
                preconditioner = kpm_preconditioner,
                tol_action = tol, tol_force = sqrt(tol), maxiter = maxiter,
                rng = rng,
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

            # Record the average number of iterations per CG solve for hmc update.
            additional_info["hmc_iters"] += iters

            # Make measurements.
            iters = make_measurements!(
                measurement_container, fermion_det_matrix, greens_estimator,
                model_geometry = model_geometry,
                fermion_path_integral = fermion_path_integral,
                tight_binding_parameters = tight_binding_parameters,
                electron_phonon_parameters = electron_phonon_parameters,
                preconditioner = kpm_preconditioner,
                tol = tol, maxiter = maxiter,
                rng = rng
            )

            # Record the average number of iterations per CG solve for measurements.
            additional_info["measurement_iters"] += iters
        end

        # Write the bin-averaged measurements to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

    # Calculate acceptance rates.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_therm)
    additional_info["reflection_acceptance_rate"] /= (N_updates + N_therm)
    additional_info["swap_acceptance_rate"] /= (N_updates + N_therm)

    # Calculate average number of CG iterations.
    additional_info["hmc_iters"] /= (N_updates + N_therm)
    additional_info["reflection_iters"] /= (N_updates + N_therm)
    additional_info["swap_iters"] /= (N_updates + N_therm)
    additional_info["measurement_iters"] /= N_updates

    # Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, additional_info)

    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(simulation_info.datafolder, N_bins, time_displaced = false)

    # Merge binary files containing binned data into a single file.
    compress_jld2_bins(folder = simulation_info.datafolder)

    return nothing
end # end of run_simulation function

# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Run the simulation.
    run_simulation(
        sID       = parse(Int,     ARGS[1]),
        Ω         = parse(Float64, ARGS[2]),
        α         = parse(Float64, ARGS[3]),
        μ         = parse(Float64, ARGS[4]),
        L         = parse(Int,     ARGS[5]),
        β         = parse(Float64, ARGS[6]),
        N_therm   = parse(Int,     ARGS[7]),
        N_updates = parse(Int,     ARGS[8]),
        N_bins    = parse(Int,     ARGS[9]),
    )
end
