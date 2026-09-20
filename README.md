## Description
A collection of command-line Python-based scripts for voxel-wise computation of:
* Joint T<sub>1,f</sub> & Macromolecular Proton Fraction (MPF) maps using a Variable Flip Angle (VFA) & a single {MTw/MT<sub>0</sub>} protocol [1] (*fit-JSPqMT*; "Joint Single-Point qMT"). On-resonance macromolecular saturation can be modeled by the Generalized Bloch model (linear approximation R<sub>2</sub><sup>s,l</sup>) [2] or a super-Lorentzian with a residual broadening term [1,3] (default).
* T<sub>1</sub> maps using the Variable Flip Angle method [4] (*fit-VFA*).
* MPF maps using the original Single-Point MPF method [5] (*fit-SPqMT*).
* MTsat maps as described in [6] (*fit-MTsat*).

## Installation
```bash
git clone https://github.com/lsoustelle/qMT.git && cd qMT
conda env create -f environment.yml
conda activate qMT
pip install .
```

* Note that the package is intended to work with NIfTI files; please consider conversion packages such as [dcm2niix](https://github.com/rordenlab/dcm2niix) or [DICOMIFIER](https://github.com/lamyj/dicomifier).

## MRI Sequences
* **Siemens users**: a prototype sequence (__vibeMT__) is available on Siemens' C<sup>2</sup>P Exchange platform (teamplay).
* **Bruker users**: a prototype MT-prepared SPGR sequence is available upon [request](https://crmbm.univ-amu.fr/resources/mt-prepared-spgr-sequence/).

The user interfaces for both constructors are designed to match the parameters to be passed in the *fit-JSPqMT* script. 
The latter is intended to be used with MT-weighted data prepared with sine-modulated off-resonance pulses [1].

## Usage
#### Data pre-processing (human brain only)
Efficient pre-processing pipeline: (i) [MP-PCA denoising](https://github.com/lsoustelle/tMPPCA)/SoS (optional) [8], (ii) realign MT/VFA data [9], and (iii) register to common anatomical space & reslice B<sub>1</sub> map). 
Outputs are data-fitting-ready: MT0/MTw (`MT.nii.gz`), VFA (`VFA.nii.gz`), B<sub>1</sub> map (`B1_MAP.nii.gz`) in relative unit and brain mask (`MASK_ANAT.nii.gz`).
```bash
preproc-qMT --anat ${FLD_DATA}/ANAT.nii \
            --VFA ${FLD_DATA}/PDw.nii,${FLD_DATA}/T1w.nii \
            --MT ${FLD_DATA}/MT0.nii,${FLD_DATA}/MTw.nii \
            --B1 B1map_raw.nii --B1_fac 800 \
            --mppca --n_sos 1 \
            --output_dir ${TMP_FLD}/ \
            --nworkers 12
```
Set `--B1_fac` to `1` if no normalization is required.
See the `--help` option for detailed information.

#### Joint Single-Point qMT
```bash
fit-JSPqMT  ${TMP_FLD}/MT.nii.gz ${TMP_FLD}/VFA.nii.gz \
            ${RES_FLD}/MPF.nii.gz ${RES_FLD}/T1f.nii.gz \
            --MTw_TIMINGS 12.0,2.1,0.25,30.0 \
            --VFA_TIMINGS 0.25,30.0 \
            --VFA_PARX 6.0,25.0,BP \
            --MTw_PARX 10.0,BP,560.0,4000.0,Hann-Sine \
            --B1 ${TMP_FLD}/B1_MAP.nii.gz \
            --mask ${TMP_FLD}/MASK_ANAT.nii.gz \
            --use_GBM --cpp_opt \
            --nworkers 12
```
See the `--help` option for detailed information.


## Other features
For original SP-MPF computation
* As in Ref. [7], a synthetic MT reference can be computed from a 2-points VFA protocol (*synt-SPGR*).
* In *fit-SPqMT*, two presets of constraint qMT parameters are proposed for adult human brain at 3T [5] and adult mouse brain at 7T [10].

#### Example for original Single-Point MPF (deprecated)
```bash
fit-VFA  \
    *FA{4,10,25}*.nii* \
    T1_map.nii.gz \
    --B1 B1_map.nii.gz \
    --mask mask.nii.gz \
    --nworkers 6 \
    --TR 18.0 --FA 4,10,25
          
fit-SPqMT  \
    MT0.nii,MTw.nii \
    T1_map.nii.gz \
    MPF_map.nii.gz \
    12.0,2.1,0.2,31.0 \
    560,4e3,0,1,10 \
    --RecoTypePreset 1 \
    --B1 B1_map.nii.gz \
    --mask mask.nii.gz \
    --nworkers 6 
```


## References
1. L. Soustelle et al., Quantitative magnetization transfer MRI unbiased by on‐resonance saturation and dipolar order contributions, MRM 2023
2. J. Assländer et al., Generalized Bloch model: A theory for pulsed magnetization transfer, MRM 2022;87:2003-2017
3. A. Pampel et al., Orientation dependence of magnetization transfer parameters in human white matter, NeuroImage 2015;114:136-146
4. L. Chang et al., Linear least-squares method for unbiased estimation of T1 from SPGR signals, MRM 2008;60:496-501
5. V. Yarnykh, Fast macromolecular proton fraction mapping from a single off-resonance magnetization transfer measurement, MRM 2012;68:166-178
6. G. Helms et al., High-resolution maps of magnetization transfer with inherent correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI, MRM 2008;60:1396-1407
7. V. Yarnykh, Time-efficient, high-resolution, whole brain three-dimensional macromolecular proton fraction mapping, MRM 2016;75:2100-2106 
8. J. Olesen et al., Tensor denoising of multidimensional MRI data, MRM 2022;89:1-13
9. B. Avants et al., A reproducible evaluation of ANTs similarity metric performance in brain image registration, NeuroImage 2011;2033-2044
10. L. Soustelle et al., Determination of optimal parameters for 3D single‐point macromolecular proton fraction mapping at 7T in healthy and demyelinated mouse brain, MRM 2021;85:369-379 

