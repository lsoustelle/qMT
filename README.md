## Description
A collection of command-line Python-based scripts for voxel-wise computation of:
* T1 maps using the Variable Flip Angle method [1] (*fit_VFA_CLI.py*).
* Macromolecular Proton Fraction (MPF) maps using the Single-Point quantitative Magnetization Transfer method [2] (*fit_SPqMT_CLI.py*).

## Additional features
* As in Ref. [3], a synthetic MT reference can be computed from a 2-points VFA protocol (*synt_SPGR_CLI.py*).
* For SP-MPF computation, two preset parameters are proposed for adult human brain at 3T [3] and adult mouse brain at 7T [4].

## Usage
Example:
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
See the `--help` option from each scripts for further information.

## Dependencies
* The package is intended to have NIfTI files as inputs; please consider conversion package such as [DICOMIFIER](https://github.com/lamyj/dicomifier) or [dcm2niix](https://github.com/rordenlab/dcm2niix).
* Specific Python packages: numpy , scipy and nibabel.
	- Tested on Python 3.8.2 (Windows 10, Ubuntu 20.04 & 18.04), with numpy-1.19.4, scipy-1.5.4 and nibabel-3.2.0

## References
[1] Chang et al., Linear least-squares method for unbiased estimation of T1 from SPGR signals, MRM 2008;60:496-501

[2] V. Yarnykh, Fast macromolecular proton fraction mapping from a single off-resonance magnetization transfer measurement, MRM 2012;68:166-178

[3] V. Yarnykh, Time-efficient, high-resolution, whole brain three-dimensional macromolecular proton fraction mapping, MRM 2016;75:2100-2106 

[4] L. Soustelle et al., Determination of optimal parameters for 3D single‐point macromolecular proton fraction mapping at 7T in healthy and demyelinated mouse brain, MRM 2021;85:369-379 
                        