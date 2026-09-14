# Tips & Tricks

## Accessing the Phonon Fields Directly

In certain situations it is advantageous to directly access the phonon field configurations in a DQMC simulation.
For instance, if you know the ground state is ordered in a certain way — such as the charge density wave (CDW) phase in the half-filled Holstein model — it can be helpful to initialize the phonon configuration in a simulation to begin in a staggered pattern that reflects the expected CDW phase, as this can significantly reduce the number of updates required to thermalize the system.

Instructions on how to do so can be found [here](https://smoqysuite.github.io/SmoQyDQMC.jl/stable/tips_and_tricks/#Accessing-the-Phonon-Fields-Directly)
in the [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) documentation.