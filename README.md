# HEG screening models
**Repo**: heg_screening_models
**Author**: Aaron D. Kaplan
**contact**: aaron.kaplan.physics[at]gmail.com

### Description:
Code used to generate figures and analysis in A.D. Kaplan and A. Ruzsinszky, "Revealing quasi-excitations in the low-density homogeneous electron gas with model exchange-correlation kernels", *submitted to J. Chem. Phys.* (2023).
This code uses many standard exchange-correlation (xc) kernels in time-dependent density functional theory (TD-DFT) to analyze both true excitations (plasmons) and quasi-excitations (ghost excitons, collective pseudo-modes arising from density fluctuations) in the 2D and 3D homogeneous electron gas (HEG).

### Contents:
*(alphabetical order)*
- constants.py : ensures consistent values of shared constants across the library
- DFT:
    - Sub-library for TD-DFT xc kernels and response functions
    - asymptotics_2D.py : low- and high-q limits of the exact xc kernel in the 2D HEG
    - asymptotics.py : low- and high-q limits of the exact xc kernel in the 3D HEG
    - chi.py : non-interacting and interacting linear response functions $\chi(q,\omega)$, dielectric functions $\epsilon(q,\omega)$, and screened interactions $W([r_\mathrm{s}];r,q,\omega)$, all in wave-vector and frequency space. For 2D and 3D HEGs.
    - common.py : defines HEG class to store attributes of HEG at given rs, as well as constants/functions only used in DFT lib. For 2D and 3D HEGs.
    - fxc.py : implements `get_fxc` function used to switch between XC kernels in 2D and 3D HEGs
    - fxc_\<ID\>.py : various xc kernels:
        - AKCK : [Kaplan2023]
        - ALDA_2D : adiabatic local density approximation (ALDA) for a 2D HEG due to [Attaccalite2003]
        - ALDA : ALDA due to Perdew and Wang [Perdew1992].
        Users can also specify the Perdew-Zunger [Perdew1981] parameterization via kwarg
        - CDOP : [Corradini1998]
        - DPGT : static kernel for the 2D HEG [Davoudi2001]
        - GKI : [Gross1986,Iwamoto1987]
        - MCP07 : [Ruzsinszky2020]
        - QV: [Qian2002]
    - figs : directory to store figures generated by main.py
    - frequency_moments.py : generates arbitrary moments $M_k(q)$ of the spectral function, 
    $S = - \mathrm{Im} ~ \chi(q,\omega)/(\pi n)$
    $M_k(q) \equiv \int_0^\infty d\omega ~ \omega^k S(q,\omega)$
    - frequency_moments: stores data files generated by frequency_moments.py
    - main.py : generates all figures in the manuscript
    - pseudo_plasmon_dispersion.py : generates pseudo-plasmonic dispersion $\omega_\mathrm{c}(q)$ defined by
    $\mathrm{Re} ~ \epsilon^{-1}(q,\omega_\mathrm{c}(q)) = 0$. 
    - pseudo_plasmon_disp and pseudo_plasmon_disp_for_taylor : collects data generated by plasmon_dispersion.py
    - plot_cartoons.py : generates Fig. 1 of the manuscript (rough visualization of different possible kinds of electronic screening)
    - plot_spectral_mcp07.py : generates Appendix Figs. 11 and 12 in the manuscript (position of particle-hole continuum frequencies relative to half-maximum frequencies on MCP07 spectral function, and estimated lifetimes, respectively)
    - real_space_transform.py : generates 3D Fourier transforms, with particular application to the screened interaction in real-space $W([r_\mathrm{s}];r)$
    - real_space : stores data generated by real_space_transform.py
    - utilities : library for shared utility functions, adapted from https://github.com/esoteric-ephemera/tc21 

### References:
*(alphabetical, chronological order)*
- [[Attaccalite2003](https://doi.org/10.1103/PhysRevLett.88.256601)] C. Attaccalite, S. Moroni, P. Gori-Giorgi, and G. B. Bachelet, Phys. Rev. Lett. **88**, 256601 (2002)DOI: 10.1103/PhysRevLett.88.256601. Erratum, *ibid.* **91**, 109902 (2003). 
- [[Corradini1998](https://doi.org/10.1103/PhysRevB.57.14569)]M. Corradini, R. Del Sole, G. Onida, and M. Palummo, Phys. Rev. B **57**, 14569 (1998). DOI: 10.1103/PhysRevB.57.14569
- [[Davoudi2001](https://doi.org/10.1103/PhysRevB.64.153101)] B. Davoudi, M. Polini, G. F. Giuliani, and M. P. Tosi, Phys. Rev. B **64**, 153101 (2001). DOI: 10.1103/PhysRevB.64.153101
- [[Gross1986](https://doi.org/10.1103/PhysRevLett.55.2850)] E. K. U. Gross and W. Kohn, Phys. Rev. Lett. **55**, 2850 (1985). DOI: 10.1103/PhysRevLett.55.2850. Erratum, *ibid.* 57, 923 1986.
- [[Iwamoto1987](https://doi.org/10.1103/PhysRevB.35.3003)] N. Iwamoto and E. K. U. Gross, Phys. Rev. B 35, 3003 (1987). DOI: 10.1103/PhysRevB.35.3003
- [[Kaplan2023](https://doi.org/10.1103/PhysRevB.107.L201120)] A. D. Kaplan and C. A. Kukkonen, Phys. Rev. B **107**, L201120 (2023). DOI: 10.1103/PhysRevB.107.L201120
- [[Perdew1981](https://doi.org/10.1103/PhysRevB.23.5048)] J. P. Perdew and A. Zunger, Phys. Rev. B **23**, 5048 (1981). DOI: 10.1103/PhysRevB.23.5048
- [[Perdew1992](https://doi.org/10.1103/PhysRevB.45.13244)] J. P. Perdew and Y. Wang, Phys. Rev. B **45**, 13244 (1992). DOI: 10.1103/PhysRevB.45.13244
- [[Qian2002](https://doi.org/10.1103/PhysRevB.65.235121)] Z. Qian and G. Vignale, Phys. Rev. B 65, 235121 (2002). DOI: 10.1103/PhysRevB.65.235121
- [[Ruzsinszky2020](https://doi.org/10.1103/PhysRevB.101.245135)] A. Ruzsinszky, N. K. Nepal, J. M. Pitarke, and J. P. Perdew, Phys. Rev. B 101, 245135 (2020). DOI: 10.1103/PhysRevB.101.245135
