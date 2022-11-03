Description
-----------
A collection of command-line Python-based scripts for voxel-wise computation of:
* Joint R1f & Macromolecular Proton Fraction (MPF) maps using a Variable Flip Angle (VFA) & a single {MTw/MT<sub>0</sub>} protocol [1] (*fit_JSPqMT_CLI.py*); a single-point extension of the proposed method in [2].
* T<sub>1</sub> maps using the Variable Flip Angle method [3] (*fit_VFA_CLI.py*).
* MPF maps using the original Single-Point MPF method [4] (*fit_SPqMT_CLI.py*).
* MTsat maps as described in [5] (*fit_SPqMT_CLI.py*).

MRI Sequences
-------------
* **Siemens users**: two prototype sequences (__greMT__ [VB17/VE11C/VE11E/XA20] & __vibeMT__ [XA20/XA30/XA31]) are made available on the Siemens C<sup>2</sup>P Exchange platform (teamplay).
* **Bruker users**: a prototype SPGR sequence is made available upon request [@LucasSoustelle](https://twitter.com/LucasSoustelle).

The user interfaces for both constructor match the specific parameters to be passed in the *fit_JSPqMT_CLI.py* and *fit_SPqMT_CLI.py* scripts.

Of common note, the *fit_JSPqMT_CLI.py* script is intended to be used on MTw data prepared with sine-modulated off-resonance pulses [1].

Additional features
-------------------
* As in Ref. [6], a synthetic MT reference can be computed from a 2-points VFA protocol (*synt_SPGR_CLI.py*).
* For original SP-MPF computation (*fit_SPqMT_CLI.py*), two presets of constraint qMT parameters are proposed for adult human brain at 3T [3] and adult mouse brain at 7T [7].

Dependencies
------------
* The package is intended to have NIfTI files as inputs; please consider conversion package such as [DICOMIFIER](https://github.com/lamyj/dicomifier) or [dcm2niix](https://github.com/rordenlab/dcm2niix).
* Specific Python packages: numpy , scipy and nibabel.
    - Tested on Python 3.8.2 (Windows 10, Ubuntu 20.04 & 18.04), with numpy-1.19.4, scipy-1.5.4 and nibabel-3.2.0
    - For computational efficiency, it is strongly advised to have a numpy package compiled with MKL (native in conda default channels)

Usage
-----
#### Example for joint Single-Point qMT:
```
python3 fit_JSPqMT_CLI.py  \
            MT0.nii.gz,MTw.nii.gz \
            VFA.nii.gz \
            MPF_JSPqMT.nii.gz \
            R1f_JSPqMT.nii.gz \
            --MTw_TIMINGS 12.0,2.1,1.0,30.0 \
            --VFA_TIMINGS 1.0,30.0 \
            --VFA_PARX 6,10,25,50,1 \
            --MTw_PARX 560,4000.0,0.0,1,10,1 \
            --B1 B1_map.nii.gz \
            --mask mask.nii.gz \
            --nworkers 20
```

#### Example for original Single-Point MPF:
```
python3 fit_VFA_CLI.py  \
            *FA{4,10,25}*.nii* \
            T1_map.nii.gz \
            --B1 B1_map.nii.gz \
            --mask mask.nii.gz \
            --nworkers 6 \
            --TR 18.0 --FA 4,10,25
          
python3 fit_SPqMT_CLI.py  \
            MT0.nii,MTw.nii \
            T1_map.nii.gz \
            MPF.nii.gz \
            12.0,2.1,0.2,31.0 \
            560,4e3,0,1,10 \
            --RecoTypePreset 1 \
            --B1 B1_map.nii.gz \
            --mask mask.nii.gz \
            --nworkers 6 
```
See the `--help` option from each script for detailed information.

References
----------
[1] L. Soustelle et al., Quantitative Magnetization Transfer parametric mapping unbiased by on-resonance saturation and dipolar order contributions, ISMRM 2022

[2] P. Mossahebi et al., Analysis and correction of biases in cross-relaxation MRI due to biexponential longitudinal relaxation, MRM 2014;71:830-838

[3] L. Chang et al., Linear least-squares method for unbiased estimation of T1 from SPGR signals, MRM 2008;60:496-501

[4] V. Yarnykh, Fast macromolecular proton fraction mapping from a single off-resonance magnetization transfer measurement, MRM 2012;68:166-178

[5] G. Helms et al., High-resolution maps of magnetization transfer with inherent correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI, MRM 2008;60:1396-1407

[6] V. Yarnykh, Time-efficient, high-resolution, whole brain three-dimensional macromolecular proton fraction mapping, MRM 2016;75:2100-2106 

[7] L. Soustelle et al., Determination of optimal parameters for 3D single‐point macromolecular proton fraction mapping at 7T in healthy and demyelinated mouse brain, MRM 2021;85:369-379 

