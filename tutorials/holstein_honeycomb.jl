# # 1a) Honeycomb Holstein Model

# In this example we reimplement the
# [SmoQyDQMC tuturial](https://smoqysuite.github.io/SmoQyDQMC.jl/stable/tutorials/holstein_honeycomb/)
# on simulating the Holstein model on a Honeycomb lattice using [SmoQyElPhQMC](https://github.com/SmoQySuite/SmoQyElPhQMC.jl.git).
# The Holstein Hamiltonian is given by
# ```math
# \begin{align*}
# \hat{H} = & -t \sum_{\langle i, j \rangle, \sigma} (\hat{c}^{\dagger}_{\sigma,i}, \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.})
# - \mu \sum_{i,\sigma} \hat{n}_{\sigma,i} \\
# & + \frac{1}{2} M \Omega^2 \sum_{i} \hat{X}_i^2 + \sum_i \frac{1}{2M} \hat{P}_i^2
# + \alpha \sum_i \hat{X}_i (\hat{n}_{\uparrow,i} + \hat{n}_{\downarrow,i} - 1)
# \end{align*}
# ```
# where ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creates (annihilates) a spin ``\sigma``
# electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
# is the spin-``\sigma`` electron number operator for site ``i``. Here ``\mu`` is the chemical potential and  ``t`` is the nearest-neighbor
# hopping amplitude, with the sum over ``\langle i,j \rangle`` denoting a sum over all nearest-neighbor pairs of sites.
# A local dispersionless phonon mode is then placed on each site in the lattice, with ``\hat{X}_i`` and ``\hat{P}_i`` the corresponding
# phonon position and momentum operator on site ``i`` in the lattice. The phonon mass and energy are denoted ``M`` and ``\Omega`` respectively.
# Lastly, the phonon displacement ``\hat{X}_i`` couples to the total local density ``\hat{n}_{\uparrow,i} + \hat{n}_{\downarrow,i},`` with the
# parameter ``\alpha`` controlling the strength of this coupling.

# ## Import packages
# First, we begin by importing the necessary packages.
# The [SmoQyElPhQMC](https://github.com/SmoQySuite/SmoQyElPhQMC.jl.git) package is built as an extension pacakge on top of
# [SmoQyDQMC](https://github.com/SmoQySuite/SmoQyDQMC.jl.git), enabling the simulution of strictly spin-symmetric electron-phonon models.
# Therefore, in addition to importing [SmoQyElPhQMC](https://github.com/SmoQySuite/SmoQyElPhQMC.jl.git),
# we also need to import [SmoQyDQMC](https://github.com/SmoQySuite/SmoQyDQMC.jl.git).
# The [SmoQyDQMC](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) package also then rexports 
# the [LatticeUtilities](https://github.com/SmoQySuite/LatticeUtilities.jl.git) package,
# which we will use to define the lattice geometry for our model.

# Lastly, we use the Standard Library packages [Random](https://docs.julialang.org/en/v1/stdlib/Random/)
# and [Printf](https://docs.julialang.org/en/v1/stdlib/Printf/) for random number generation and C-style string
# formatting, respectively.

using SmoQyElPhQMC
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu

using LinearAlgebra
using Random
using Printf

# ## Specify simulation parameters
# The entire main body of the simulation we will wrapped in a top-level function named `run_simulation`
# that will take as keyword arguments various model and simulation parameters that we may want to change.
# The function arguments with default values are ones that are typically left unchanged between simulations.
# The specific meaning of each argument will be discussed in later sections of the tutorial.

## Top-level function to run simulation.
function run_simulation(;
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
    Nt = 25, # Numer of time-steps in HMC update.
    Nrv = 10, # Number of random vectors used to estimate fermionic correlation functions.
    tol = 1e-10, # CG iterations tolerance.
    maxiter = 10_000, # Maximum number of CG iterations.
    write_bins_concurrent = true, # Whether to write HDF5 bins during the simulation.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

# ## Initialize simulation
# In this first part of the script we name and initialize our simulation, creating the data folder our simulation results will be written to.
# This is done by initializing an instances of the [`SmoQyDQMC.SimulationInfo`](@extref) type, as well as an `metadata` dictionary where we will store useful metadata about the simulation.
# Finally, the integer `seed` is used to initialize the random number generator `rng` that will be used to generate random numbers throughout the rest of the simulation.

# Next we record relevant simulation parameters to the `metadata` dictionary.
# Think of the `metadata` dictionary as a place to record any additional information during the simulation that will not otherwise be automatically recorded and written to file.

    ## Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_honeycomb_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

    ## Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,                     
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = write_bins_concurrent,
        sID = sID
    )

    ## Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

# ## Initialize simulation metadata
# In this section of the code we record important metadata about the simulation, including initializing the random number
# generator that will be used throughout the simulation.
# The important metadata within the simulation will be recorded in the `metadata` dictionary.

    ## Initialize random number generator
    rng = Xoshiro(seed)

    ## Initialize additiona_info dictionary
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
# Here we also update variables to keep track of the acceptance rates for the various types of Monte Carlo updates
# that will be performed during the simulation. This will be discussed in more detail in later sections of the tutorial.
    metadata["hmc_acceptance_rate"] = 0.0 # HMC acceptance rate
    metadata["reflection_acceptance_rate"] = 0.0 # Reflection update acceptance rate
    metadata["swap_acceptance_rate"] = 0.0 # Swap update acceptance rate
# Initialize variables to record the average number of CG iterations for each type of update and measurements.
    metadata["hmc_iters"] = 0.0 # Avg number of CG iteractions per solve in HMC update.
    metadata["reflection_iters"] = 0.0 # Avg number of CG iterations per solve in reflection update.
    metadata["swap_iters"] = 0.0 # Avg number of CG iterations per solve in swap update.
    metadata["measurement_iters"] = 0.0 # Avg number of CG iterations per solve while making measurements.

# ## Initialize model
# The next step is define the model we wish to simulate.
# In this example the relevant model parameters the phonon energy ``\Omega`` (`Ω`), electron-phonon coupling ``\alpha`` (`α`),
# chemical potential ``\mu`` (`μ`), and lattice size ``L`` (`L`).
# The neasrest-neighbor hopping amplitude and phonon mass are normalized to unity, ``t = M = 1``.

# First we define the lattice geometry for our model, relying on the
# [LatticeUtilities](https://github.com/SmoQySuite/LatticeUtilities.jl.git) package to do so.
# We define a the unit cell and size of our finite lattice using the [`LatticeUtilities.UnitCell`](@extref)
# and [`LatticeUtilities.Lattice`](@extref) types, respectively.
# Lastly, we define various instances of the [`LatticeUtilities.Bond`](@extref) type to represent the
# the nearest-neighbor and next-nearest-neighbor bonds.
# All of this information regarding the lattice geometry is then stored in an instance of the [`SmoQyDQMC.ModelGeometry`](@extref) type.

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

# Next we specify the Honeycomb tight-binding term in our Hamiltonian with the [`SmoQyDQMC.TightBindingModel`](@extref) type.

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

# Now we need to initialize the electron-phonon part of the Hamiltonian with the `ElectronPhononModel` type.

    ## Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

# Then we need to define and add two types phonon modes to the model, one for each orbital in the Honeycomb unit cell,
# using the [`SmoQyDQMC.PhononMode`](@extref) type and [`SmoQyDQMC.add_phonon_mode!`](@extref) function.

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

# Now we need to define and add a local Holstein couplings to our model for each of the two phonon modes
# in each unit cell using the [`SmoQyDQMC.HolsteinCoupling`](@extref) type and [`SmoQyDQMC.add_holstein_coupling!`](@extref) function.

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

# Lastly, the [`SmoQyDQMC.model_summary`](@extref) function is used to write a `model_summary.toml` file,
# completely specifying the Hamiltonian that will be simulated.

    ## Write model summary TOML file specifying Hamiltonian that will be simulated.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (electron_phonon_model,)
    )

# ## Initialize model parameters
# The next step is to initialize our model parameters given the size of our finite lattice.
# To clarify, both the [`SmoQyDQMC.TightBindingModel`](@extref)
# and 
# [`SmoQyDQMC.ElectronPhononModel`](@extref)
# types are agnostic to the size of the lattice being simulated,
# defining the model in a translationally invariant way. As [SmoQyDQMC](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) and
# [SmoQyElPhQMC](https://github.com/SmoQySuite/SmoQyElPhQMC.jl.git) supports
# random disorder in the terms appearing in the Hamiltonian, it is necessary to initialize seperate parameter values for each unit cell in the lattice.
# For instance, we need to initialize a seperate number to represent the on-site energy for each orbital in our finite lattice.

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
# Having initialized both our model and the corresponding model parameters,
# the next step is to initialize the various measurements we want to make during our DQMC simulation.

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
            ## Measure green's functions for all pairs or orbitals.
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

# It is also useful to initialize more specialized composite correlation function measurements.

# First, it can be useful to measure the time-displaced single-particle electron Green's function traced over both orbitals in the unit cell.
# We can easily implement this measurement using the [`initialize_composite_correlation_measurement!`](@ref) function, as shown below.

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

# Additionally, to detect the formation of charge-density wave order where the electrons preferentially
# localize on one of the two sub-lattices of the honeycomb lattice, it is useful to measure the correlation function
# ```math
# C_\text{cdw}(\mathbf{r},\tau) = \frac{1}{L^2}\sum_{\mathbf{i}} \langle \hat{\Phi}^{\dagger}_{\mathbf{i}+\mathbf{r}}(\tau) \hat{\Phi}^{\phantom\dagger}_{\mathbf{i}}(0) \rangle,
# ```
# where
# ```math
# \hat{\Phi}_{\mathbf{i}}(\tau) = \hat{n}_{\mathbf{i},A}(\tau) - \hat{n}_{\mathbf{i},B}(\tau)
# ```
# and ``\hat{n}_{\mathbf{i},\gamma} = (\hat{n}_{\uparrow,\mathbf{i},o} + \hat{n}_{\downarrow,\mathbf{i},o})`` is the total electron number
# operator for orbital ``\gamma \in \{A,B\}`` in unit cell ``\mathbf{i}``.
# It is then also useful to calculate the corresponding structure factor ``S_\text{cdw}(\mathbf{q},\tau)`` and susceptibility ``\chi_\text{cdw}(\mathbf{q}).``
# Again, this can all be easily calculated using the [`initialize_composite_correlation_measurement!`](@ref) function, as shown below.

    ## Initialize CDW correlation measurement.
    initialize_composite_correlation_measurement!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        name = "cdw",
        correlation = "density",
        ids = [1, 2],
        coefficients = [1.0, -1.0],
        displacement_vecs = [[0.0, 0.0], [0.0, 0.0]],
        time_displaced = false,
        integrated = true
    )

# ## Setup QMC Simulation
# This section of the code sets up the QMC simulation by allocating the initializing the relevant types and arrays we will need in the simulation.

# This section of code is perhaps the most opaque and difficult to understand, and will be discussed in more detail once written.
# That said, you do not need to fully comprehend everything that goes on in this section as most of it is fairly boilerplate,
# and will not need to be changed much once written.
# This is true even if you want to modify this script to perform a QMC simulation for a different Hamiltonian.

    ## Allocate a single FermionPathIntegral for both spin-up and down electrons.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    ## Initialize FermionPathIntegral type to account for electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

# At the start of this section, an instance of the
# [`FermionPathIntegral`](https://smoqysuite.github.io/SmoQyDQMC.jl/stable/api/#SmoQyDQMC.FermionPathIntegral)
# type was allocated and then initialized.
# Recall that after discretizing the imaginary-time axis and applying the Suszuki-Trotter approximation, the resulting
# Hamiltonian is quadratic in fermion creation and annihilation operators, but fluctuates in imaginary-time as a result of the phonon fields.
# Therefore, this Hamiltonian may be expressed as
# ```math
# \hat{H}_l = \sum_\sigma \hat{\mathbf{c}}_\sigma^\dagger \left[ H_{\sigma,l} \right] \hat{\mathbf{c}}_\sigma
# = \sum_\sigma \hat{\mathbf{c}}_\sigma^\dagger \left[ K_{\sigma,l} + V_{\sigma,l} \right] \hat{\mathbf{c}}_\sigma,
# ```
# at imaginary-time ``\tau = \Delta\tau \cdot l``,
# where ``\hat{\mathbf{c}}_\sigma \ (\hat{\mathbf{c}}_\sigma^\dagger)`` is a column (row) vector of spin-``\sigma`` electron annihilation (creation) operators for each orbital in the lattice.
# Here ``H_{\sigma,l}`` is the spin-``\sigma`` Hamiltonian matrix for imaginary-time ``\tau``, which can be expressed as the sum of the
# electron kinetic and potential energy matrices ``K_{\sigma,l}`` and ``V_{\sigma,l}``, respectively.

# The purpose of the
# [`SmoQyDQMC.FermionPathIntegral`](@extref)
# type is to contain the minimal information required to reconstruct each ``K_{\sigma,l}`` and ``V_{\sigma,l}`` matrices.
# Here we only need to allocate a single instance of the [`SmoQyDQMC.FermionPathIntegral`](@extref)
# type as we assume spin symmetry.
# The [`SmoQyDQMC.FermionPathIntegral`](@extref) instance is first allocated and initialized to reflect just the non-interacting component of the Hamiltonian.
# Then the subsequent
# [`SmoQyDQMC.initialize!`](https://smoqysuite.github.io/SmoQyDQMC.jl/stable/api/#SmoQyDQMC.initialize!-Union{Tuple{E},%20Tuple{T},%20Tuple{FermionPathIntegral{T,%20E},%20FermionPathIntegral{T,%20E},%20HubbardParameters{E}}}%20where%20{T,%20E})
# call modifies the
# [`SmoQyDQMC.FermionPathIntegral`](@extref) to reflect the contribution from the initial phonon field configuration.

# Next we initialize an instance of the [`SymFermionDetMatrix`](@ref) type of represent the Fermion determinant matrix,
# where is an inherited type from the abstracy [`FermionDetMatrix`](@ref) type.
# We could have used an instance of the [`AsymFermionDetMatrix`](@ref) here instead if we wanted to.

    ## Initialize fermion determinant matrix. Also set the default tolerance and max iteration count
    ## used in conjugate gradient (CG) solves of linear systems involving this matrix.
    fermion_det_matrix = SymFermionDetMatrix(
        fermion_path_integral,
        maxiter = maxiter, tol = tol
    )

# Now we can initialize an instance of the [`PFFCalculator`](@ref) type,
# which is used to sample and store the complex pseudofermion fields ``\Phi``
# and evaluate the fermionic action
# ```math
# S_F(x,\Phi) = \Phi^\dagger \left[\Lambda^\dagger(x) M^\dagger(x) M^{\phantom\dagger}(x) \Lambda^{\phantom\dagger}(x)\right]^{-1} \Phi^{\phantom\dagger};
# ```
# where ``M(x)`` is the fermion determinant matrix and ``\Lambda(x)`` is a unitary transformation specially chosen to improve sampling.
# These auxialary fields result from replacing the fermion determinants by a complex multivariate Gaussian integral
# ```math
# |\det M(x)|^2 \propto \int d\Phi e^{-S_F(x,\Phi)}.
# ```

    ## Initialize pseudofermion field calculator.
    pff_calculator = PFFCalculator(electron_phonon_parameters, fermion_det_matrix)

# Evaluating the fermionic action ``S(F,x)``, and its partial derivatives with respect to the phonon fields, requires solving linear system of the form
# ```math
# \left[ M^\dagger(x) M^{\phantom\dagger}(x) \right] v = b,
# ```
# which is done using the conjugate gradient (CG) method.
# This is the most expensive operation in the QMC simulation.
# We use the [`KPMPreconditioner`](@ref) type to accelerate the convergence of the CG calculations, thereby accelerating the simulations.

    ## Initialize KPM preconditioner.
    preconditioner = KPMPreconditioner(fermion_det_matrix, rng = rng)

# Finally, we initialize an instance of the [`GreensEstimator`](@ref) type, which is for
# estimating fermionic correlation functions when making measurements.

    ## Initialize Green's function estimator for making measurements.
    greens_estimator = GreensEstimator(fermion_det_matrix, model_geometry)

# ## [Setup EFA-PFF-HMC Updates](@id holstein_square_efa-hmc_updates)
# Before we begin the simulation, we also want to initialize an instance of the [`EFAPFFHMCUpdater`](@ref) type,
# which will be used to perform hybrid Monte Carlo (HMC) updates to the phonon fields that use
# exact fourier acceleration (EFA) to further reduce autocorrelation times.

# The two main parameters that need to be specified are the time-step size ``\Delta t`` and number of time-steps ``N_t``
# performed in the HMC update, with the corresponding integrated trajectory time then equalling ``T_t = N_t \cdot \Delta t.``
# Note that the computational cost of an HMC update is linearly proportional to ``N_t,`` while the acceptance rate is
# approximately proportional to ``1/(\Delta t)^2.``

# [Previous studies](https://arxiv.org/abs/2404.09723) have shown that a good place to start
# with the integrated trajectory time ``T_t`` is a quarter the period associated with the bare phonon frequency,
# ``T_t \approx \frac{1}{4} \left( \frac{2\pi}{\Omega} \right) = \pi/(2\Omega).`` However, in our implementation we effectively normalize all of the
# bare phonon frequencies to unity in the dynamics. Therefore, a good choice for the trajectory time in our implementation is simply ``T_t = \pi/2``.
# Therefore, in most cases you simply need to select a value for ``N_t`` and then use the default assigned time-step ``\Delta t = \pi / (2 N_t)``,
# such that the trajectory length is held fixed at ``T_t = \pi/2``.
# With this convention the computational cost of performing updates still increases linearly with ``N_t``, but the acceptance rate also increases with ``N_t``.
# Note that it can be important to keep the acceptance rate for the HMC updates above ``\sim 90\%`` to avoid numerical instabilities from occuring.

# Based on user experience, a good (conservative) starting place is to set the number of time-steps to ``N_t \approx 10.``
# Then, if the acceptance rate is too low you increase ``N_t,`` which implicitly results in a reduction of ``\Delta t.``
# Conversely, if the acceptance rate is very high ``(\gtrsim 99 \% )`` it may be useful to decrease ``N_t``,
# thereby increasing ``\Delta t,`` as this will reduce the computational cost of performing an EFA-HMC update.

    ## Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAPFFHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        Nt = Nt, Δt = π/(2*Nt)
    )

# ## Thermalize system
# The next section of code performs updates to thermalize the system prior to beginning measurements.
# In addition to EFA-HMC updates that will be performed using the [`EFAPFFHMCUpdater`](@ref) type initialized above and
# the [`hmc_update!`](@ref) function below, we will also perform reflection and swap updates using the
# [`reflection_update!`](@ref) and [`swap_update!`](@ref) functions respectively.

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
# In this next section of code we continue to sample the phonon fields as above,
# but will also begin making measurements as well. For more discussion on the overall
# structure of this part of the code, refer to here.

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
            update = update,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

# ## Merge binned data
# At this point the simulation is essentially complete, with all updates and measurements having been performed.
# However, the binned measurement data resides in many seperate HDF5 files currently.
# Here we will merge these seperate HDF5 files into a single file containing all the binned data
# using the [`merge_bins`](@extref) function.

    ## Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

# ## Record simulation metadata
# At this point we are done sampling and taking measurements.
# Next, we want to calculate the final acceptance rate for the various types of
# udpates we performed, as well as write the simulation metadata to file,
# including the contents of the `metadata` dictionary.

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
# In this final section of code we post-process the binned data.
# This includes calculating the final estimates for the mean and error of all measured observables,
# which will be written to an HDF5 file using the [`process_measurements`](@ref) function.
# Inside this function the binned data gets further rebinned into `n_bins`,
# where `n_bins` is any positive integer satisfying the constraints `(N_bins ≥ n_bin)` and `(N_bins % n_bins == 0)`.
# Note that the [`process_measurements`](@ref) function has many additional keyword arguments that can be used to control the output.
# For instance, in this example in addition to writing the statistics to an HDF5 file, we also export the statistics to CSV files
# by setting `export_to_csv = true`, with additional keyword arguments controlling the formatting of the CSV files.
# Again, for more information on how to interpret the output refer the [Simulation Output Overview](@ref) page.

    ## Process the simulation results, calculating final error bars for all measurements.
    ## writing final statisitics to CSV files.
    process_measurements(
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 7,
        delimiter = " "
    )

# A common measurement that needs to be computed at the end of a DQMC simulation is something called the correlation
# ratio with respect to the ordering wave-vector for a specified type of structure factor measured during the simulation.
# In the case of the honeycomb Holstein model, we are interested in measureing the correlation ratio
# ```math
# R_\text{cdw}(0) = 1 - \frac{1}{4} \sum_{\delta\mathbf{q}} \frac{S_\text{cdw}(0 + \delta\mathbf{q})}{S_\text{cdw}(0)}
# ```
# with respect to the equal-time charge density wave (CDW) structure factor ``S_\text{cdw}(0)``, where ````S_\text{cdw}(q)``` is
# equal-time structure factor corresponding to the composite correlation function ``C_\text{cdw}(\mathbf{r},\tau)`` defined earlier in this tutorial.
# Note that the CDW ordering wave-vector is ``\mathbf{Q}_\text{cdw} = 0`` in this case, which describes the electrons preferentially
# localizing on one of the two sub-lattices of the honeycomb lattice.
# The sum over ``\delta\mathbf{q}`` runs over the four wave-vectors that neigbor ``\mathbf{Q}_\text{cdw} = 0.``

# Here we use the [`compute_composite_correlation_ratio`](@ref) function to compute to compute this correlation ratio.
# Note that the ``\mathbf{Q}_\text{cdw} = 0`` is specified using the `q_point` keyword argument, and the four neighboring wave-vectors
# ``\delta\mathbf{q}`` are specified using the `q_neighbors` keyword argument.
# These wave-vectors are specified using the convention described [here](@ref vector_reporting_conventions) in the [Simulation Output Overview](@ref) page.
# Note that because the honeycomb lattice has a ``C_6`` rotation symmetry, each wave-vector in momentum-space has six nearest-neighbor wave-vectors.
# Below we specify all six wave-vectors that neighbor the ``\mathbf{Q}_\text{cdw} = 0`` wave-vector ordering wave-vector, accounting for the fact
# that the Brilliouin zone is periodic in the reciprocal lattice vectors.

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

# Next, we record the measurement in the `metadata` dictionary, and then write a new version of the simulation summary TOML file that
# contains this new information using the [`save_simulation_info`](@ref) function.

    ## Record the AFM correlation ratio mean and standard deviation.
    metadata["Rcdw_mean_real"] = real(Rcdw)
    metadata["Rcdw_mean_imag"] = imag(Rcdw)
    metadata["Rcdw_std"]       = ΔRcdw

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

# Note that as long as the binned data persists the [`process_measurements`](@ref) and [`compute_correlation_ratio`](@ref)
# functions can be rerun to recompute the final statistics for the measurements without needing to rerun the simulation.

    return nothing
end # end of run_simulation function

# ## Execute script
# DQMC simulations are typically run from the command line as jobs on a computing cluster.
# With this in mind, the following block of code only executes if the Julia script is run from the command line,
# also reading in additional command line arguments.

## Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    ## Run the simulation.
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

# For instance, the command
# ```
# > julia holstein_honeycomb.jl 1 1.0 1.5 0.0 3 4.0 5000 10000 100
# ```
# runs a DQMC simulation of a Holstein model on a ``3 \times 3`` unit cell (`N = 2 \times 3^2 = 18` site) honeycomb lattice
# at half-filling ``(\mu = 0)`` and inverse temperature ``\beta = 4.0``.
# The phonon energy is set to ``\Omega = 1.0`` and the electron-phonon coupling is set to ``\alpha = 1.5.``
# In the DQMC simulation, 5,000 EFA-HMC, reflection and swap updates are performed to thermalize the system.
# Then an additional 10,000 such udpates are performed, after each of set of which measurements are made.
# During the simulation, bin-averaged measurements are written to file 100 times,
# with each bin of data containing the average of 10,000/100 = 100 sequential measurements.