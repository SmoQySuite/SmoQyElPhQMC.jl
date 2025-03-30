using LinearAlgebra
using Random
using Printf

using  SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
import SmoQyDQMC.MuTuner           as mt
using  SmoQyElPhQMC

# Define top-level function for running the DQMC simulation.
function run_holstein_chain_simulation(sID, Ω, α, μ, β, L, N_burnin, N_updates, N_bins; filepath = ".")

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_chain_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

    # Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

    # Initialize a random number generator that will be used throughout the simulation.
    seed = abs(rand(Int))
    rng = Xoshiro(seed)

    # Set the discretization in imaginary time for the DQMC simulation.
    Δτ = 0.05

    # Tolerance for CG solves.
    tol = 1e-7

    # Max iterations for CG solve.
    maxiter = 10_000

    # Calculate the bin size.
    bin_size = div(N_updates, N_bins)

    # Number of fermionic time-steps in HMC update.
    Nt = 100

    # Fermionic time-step used in HMC update.
    Δt = π/(2*Ω*Nt)

    # Initialize a dictionary to store additional information about the simulation.
    additional_info = Dict(
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "hmc_acceptance_rate" => 0.0,
        "cg_iters" => 0.0,
        "Nt" => Nt,
        "dt" => Δt,
        "tol" => tol,
        "maxiter" => maxiter,
        "seed" => seed,
    )

    #######################
    ### DEFINE THE MODEL ##
    #######################

    # Initialize an instance of the type UnitCell.
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0]],
                            basis_vecs   = [[0.0]])

    # Initialize an instance of the type Lattice.
    lattice = lu.Lattice(
        L = [L],
        periodic = [true]
    )

    # Get the number of sites in the lattice.
    N = lu.nsites(unit_cell, lattice)

    # Initialize an instance of the ModelGeometry type.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Define the nearest-neighbor bond for a 1D chain.
    bond = lu.Bond(orbitals = (1,1), displacement = [1])

    # Add this bond to the model, by adding it to the ModelGeometry type.
    bond_id = add_bond!(model_geometry, bond)

    # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0

    # Define the tight-binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond], # defines hopping
        t_mean = [t],     # defines corresponding hopping amplitude
        μ = μ,            # set chemical potential
        ϵ_mean = [0.]     # set the (mean) on-site energy
    )

    # Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon = PhononMode(orbital = 1, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon
    )

    # Define a on-site Holstein coupling between the electron and the local dispersionless phonon mode.
    holstein_coupling = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0]),
    	α_mean = α,
        shifted = false
    )

    # Add the Holstein coupling definition to the model.
    holstein_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_coupling,
    	model_geometry = model_geometry
    )

    # Write a model summary to file.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (electron_phonon_model,)
    )

    #################################################
    ### INITIALIZE FINITE LATTICE MODEL PARAMETERS ##
    #################################################

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

    ##############################
    ### INITIALIZE MEASUREMENTS ##
    ##############################

    ## Initialize the container that measurements will be accumulated into.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    ## Initialize the tight-binding model related measurements, like the hopping energy.
    initialize_measurements!(measurement_container, tight_binding_model)

    ## Initialize the electron-phonon interaction related measurements.
    initialize_measurements!(measurement_container, electron_phonon_model)

    # Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    ############################
    ### SET-UP QMC SIMULATION ##
    ############################

    # Allocate fermion path integral type.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the fermion path integral type with respect to electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Initialize fermion determinant matrix.
    fermion_det_matrix = AsymFermionDetMatrix(fermion_path_integral)

    # Initilialize KPM Preconditioner.
    kpm_preconditioner = KPMPreconditioner(
        fermion_det_matrix,
        rng = rng,
        rbuf = 0.10,
        n = 20
    )

    # Initialize the Green's function estimator.
    greens_estimator = GreensEstimator(
        fermion_det_matrix, model_geometry,
        Nrv = 10,
        preconditioner = kpm_preconditioner,
        rng = rng
    )

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAPFFHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        fermion_det_matrix = fermion_det_matrix,
        Nt = Nt,
        Δt = Δt
    )

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization updates.
    for n in 1:N_burnin

        # Perform an HMC update.
        accepted, iters_avg = hmc_update!(
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_det_matrix = fermion_det_matrix,
            rng = rng,
            preconditioner = kpm_preconditioner
        )

        # Record whether the HMC update was accepted or rejected.
        additional_info["hmc_acceptance_rate"] += accepted

        # Record average number of CG iterations.
        additional_info["cg_iters"] += iters_avg
    end

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in 1:N_bins

        # Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            # Perform an HMC update.
            accepted, iters_avg = hmc_update!(
                electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_det_matrix = fermion_det_matrix,
                rng = rng,
                preconditioner = kpm_preconditioner
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

            # Record average number of CG iterations.
            additional_info["cg_iters"] += iters_avg

            # Make measurements.
            make_measurements!(
                measurement_container,
                fermion_det_matrix,
                greens_estimator,
                model_geometry = model_geometry,
                fermion_path_integral = fermion_path_integral,
                tight_binding_parameters = tight_binding_parameters,
                electron_phonon_parameters = electron_phonon_parameters,
                preconditioner = kpm_preconditioner,
                rng = rng
            )
        end

        ## Write the average measurements for the current bin to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

    # Calculate HMC acceptance rate.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)

    # Calculate average number of CG iterations.
    additional_info["cg_iters"] /= (N_updates + N_burnin)

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

    ## Process the simulation results, calculating final error bars for all measurements,
    ## writing final statisitics to CSV files.
    process_measurements(simulation_info.datafolder, N_bins)

    return nothing
end

# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Read in the command line arguments.
    sID = parse(Int, ARGS[1])
    Ω = parse(Float64, ARGS[2])
    α = parse(Float64, ARGS[3])
    μ = parse(Float64, ARGS[4])
    β = parse(Float64, ARGS[5])
    L = parse(Int, ARGS[6])
    N_burnin = parse(Int, ARGS[7])
    N_updates = parse(Int, ARGS[8])
    N_bins = parse(Int, ARGS[9])

    # Run the simulation.
    run_holstein_chain_simulation(sID, Ω, α, μ, β, L, N_burnin, N_updates, N_bins)
end
