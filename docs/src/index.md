```@meta
CurrentModule = SmoQyElPhQMC
```

# SmoQyElPhQMC

Documentation for [SmoQyElPhQMC](https://github.com/SmoQySuite/SmoQyElPhQMC.jl).

The SmoQyElPhQMC package is part of [SmoQySuite](https://github.com/SmoQySuite), and is a package that extends the
functionality of the [SmoQyDQMC](https://github.com/SmoQySuite/SmoQyDQMC.jl) package. Specifically, it allows for near
linear scaling quantum Monte Carlo simulations of spin-symmetric electron-phonon models, absent any Hubbard interactions.
This package implements a modified version of the algorithm introduced in this article:

```bibtex
@article{PhysRevE.105.065302,
  title = {Fast and scalable quantum Monte Carlo simulations of electron-phonon models},
  author = {Cohen-Stead, Benjamin and Bradley, Owen and Miles, Cole and Batrouni, George and Scalettar, Richard and Barros, Kipton},
  journal = {Phys. Rev. E},
  volume = {105},
  issue = {6},
  pages = {065302},
  numpages = {22},
  year = {2022},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.105.065302},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.105.065302}
}
```

A more detailed description of the algorithms used in this package will appear in a future publication.

## Funding

The development of this package was supported by the National Science Foundation under Award number OAC-2410280 and the Simons Foundation.

## Contact Us

For question and comments regarding this package, please email either Dr. Benjamin Cohen-Stead at [bcohenst@utk.edu](mailto:bcohenst@utk.edu) or Professor Steven Johnston at [sjohn145@utk.edu](mailto:sjohn145@utk.edu).