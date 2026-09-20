#!/usr/bin/env python3
"""
preproc_qMT.py - Preprocessing pipeline for quantitative Magnetization Transfer (qMT) MRI data.

Performs:
    1) Anatomical preprocessing: TE1 extraction, denoising (ANTs), N4 bias field correction,
       skull-stripping (nipreps-synthstrip) and brain masking
    2) MP-PCA denoising + sum-of-squares (SoS) echo combination (optional),
       followed by brain masking of each VFA and MT (MT0/MTw) volume
       N4 bias field estimation and application to all volumes
    3) Rigid motion correction of every volumes onto the reference volume
    4) Rigid registration of the reference volume onto the (denoised, N4, skull-stripped) anatomical volume
    5) Composition of the MoCo + "to-ANAT" transforms, applied to the
       *original* (non-N4, non-extracted) volumes, followed by masking
    6) Reassembly of the corrected VFA and MT volumes into two separate 4D
       NIfTI stacks
    7) B1 map preprocessing: intensity normalization, reslice onto the anatomical grid, Gaussian
       smoothing and masking
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import numpy
import nibabel
from tmppca import tmppca_cpp
from argparse import RawTextHelpFormatter
from pathlib import Path
from datetime import datetime
import urllib.request

# Globals (populated after argument parsing)
tmp_fld: Path | None = None
flag_keep_tmp: bool = False

# Cleanup helpers
def cleanup() -> None:
    global tmp_fld
    if tmp_fld is not None and tmp_fld.is_dir():
        shutil.rmtree(tmp_fld, ignore_errors=True)

def _signal_handler(sig, frame) -> None:  # noqa: ANN001
    if not flag_keep_tmp:
        cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


###################################################################
############## Argument parsing
###################################################################
DESCRIPTION = """\
Preprocessing pipeline for quantitative Magnetization Transfer (qMT) MRI data.
Inputs (VFA, MT, anatomical) can be either 3D or 4D (multi-gradient-echo).

Usage:
  preproc-qMT --anat path/to/anat.nii.gz \\
              --VFA path/to/PDw.nii.gz,path/to/T1w.nii.gz \\
              --MT path/to/MT0.nii.gz,path/to/MTw.nii.gz \\
              --B1 path/to/B1raw.nii.gz --B1_fac 800 \\
              --output_dir path/to/dir/ --nworkers 20
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    # Required
    parser.add_argument("--VFA",        "-V", required=True, help="Comma-separated Variable Flip-Angle NIfTI paths (e.g., VFA_0.nii,VFA_1.nii,...,VFA_N-1.nii).\n"
                                                                  "Internally labelled vfa0,vfa1,vfa2,etc.; see --refvfa_reg option if appropriate (i.e., N>2)")
    parser.add_argument("--MT",         "-M", required=True, help="Comma-separated MT0,MTw pair NIfTI paths.")
    parser.add_argument("--anat",       "-A", required=True, help="Anatomical NIfTI path.")
    parser.add_argument("--B1",         "-B", required=True, help="Raw B1 map NIfTI path.")
    parser.add_argument("--B1_fac",type=float,required=True, help="Scaling factor applied to the raw B1 map to restore a B1=1.0 <-> no inhomogeneity (B1map = B1raw/B1_fac).")
    parser.add_argument("--output_dir", "-o", required=True, help="Output folder for pre-processed VFA & MT0/MTw stacks and B1 map.\n" 
                                                                  "All volumes are resliced in anatomical space.\n"
                                                                  "Outputs are:\n  - VFA.nii.gz\n  - MT.nii.gz\n  - B1map.nii.gz\n  - MASK_ANAT.nii.gz")

    # Optional
    parser.add_argument("--mppca",       "-d", action="store_true", help="Perform MP-PCA denoising of raw ihMT images.")
    parser.add_argument("--mppca_window","-e", help="MP-PCA kernel extent as comma-separated integers (default: 5,5,5).")
    parser.add_argument("--n_sos",       "-s", type=int, default=1, help="Number of first multi-TE to consider for Sum-of-Square (default: 1 = no SoS).")
    parser.add_argument("--refvfa_reg",  "-r", default=None, help="Volume's label used as the motion-correction reference among the VFA stack (vfa0,vfa1,...,vfaN).\n"
                                                                  "(default: the last --VFA entry, typically the T1w volume in the usual 2-FA PDw+T1w case -- 'vfa1').")
    parser.add_argument("--nworkers",   "-n", type=int, default=1, help="Number of threads for operations (default: 1).")
    parser.add_argument("--keep_tmp",   "-k", action="store_true", help="Keep temporary files.")
    parser.add_argument("--verbose",    "-v", action="store_true", help="High verbosity mode.")

    return parser.parse_args()


###################################################################
############## Validation helpers
###################################################################
def _parse_path_list(s: str, name: str, parser: argparse.ArgumentParser, expected_n: int | None = None) -> list[Path]:
    paths = [Path(p.strip()) for p in s.split(",")]
    if expected_n is not None and len(paths) != expected_n:
        parser.error(f"--{name}: expected exactly {expected_n} comma-separated paths (found {len(paths)}).")
    if expected_n is None and len(paths) < 1:
        parser.error(f"--{name}: expected at least 1 path.")
    for p in paths:
        if not p.is_file():
            parser.error(f"--{name}: file not found: {p}")
    return paths


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    v: dict = {}

    # anat / B1
    anat_path = Path(args.anat)
    if not anat_path.is_file():
        parser.error(f"--anat: file not found: {anat_path}")
    v["anat_path"] = anat_path

    b1_path = Path(args.B1)
    if not b1_path.is_file():
        parser.error(f"--B1: file not found: {b1_path}")
    v["b1_path"] = b1_path
    v["b1_fac"]  = args.B1_fac

    # VFA (N>=2 flip angles) / MT (exactly MT0,MTw)
    vfa_paths   = _parse_path_list(args.VFA, "vfa", parser, expected_n=None)
    mt_paths    = _parse_path_list(args.MT, "mt", parser, expected_n=2)

    # Build entries with explicit labels, used for regrouping at the end.
    # VFA labels are generated from the order --VFA was given (vfa0,vfa1,...)    
    v["entries"] = (
        [{"path": p, "modality": "vfa", "role": f"vfa{i}"} for i, p in enumerate(vfa_paths)]
        + [
            {"path": mt_paths[0], "modality": "mt", "role": "mt0"},
            {"path": mt_paths[1], "modality": "mt", "role": "mtw"},
        ]
    )

    valid_roles = [e["role"] for e in v["entries"]]
    if args.refvfa_reg is None:
        refvfa_reg = f"vfa{len(vfa_paths) - 1}" # default: last VFA entry
    else:
        refvfa_reg = args.refvfa_reg
        if refvfa_reg not in valid_roles:
            parser.error(f"--refvfa_reg: '{refvfa_reg}' is not one of the volumes provided ({', '.join(valid_roles)}).")
    v["refvfa_reg"] = refvfa_reg

    # determine control SoS parx (if any)
    if args.n_sos is not None:
        n_sos = args.n_sos 
    else:
        n_sos = 1
    if args.n_sos <= 0:
        parser.error('--n_sos should be >= 1')
    v["n_sos"] = n_sos
    v["mppca"] = args.mppca

    # MP-PCA kernel extent
    if args.mppca_window is not None:
        try:
            mppca_window = [int(x) for x in args.mppca_window.split(",")]
        except ValueError:
            parser.error("--mppca_window: expected comma-separated integers, e.g. 5,5,5")
        if len(mppca_window) != 3:
            parser.error("--mppca_window: expected exactly 3 comma-separated integers.")
    else:
        mppca_window = [5, 5, 5]
    v["mppca_window"] = mppca_window

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    v["output_dir"] = output_dir

    # Threads
    if args.nworkers is None:
        args.nworkers = 1
    if args.nworkers <= 0:
        parser.error("--nworkers: expected > 0")
    v["nthreads"] = min(get_physCPU_number(), args.nworkers)

    # Skull-stripping weights
    v["synthstrip_weights"] = get_synthstrip_weights()

    # Keep tmp
    global flag_keep_tmp
    flag_keep_tmp = True if args.keep_tmp else False

    # Verbosity
    v["verbose"] = True if args.verbose else False

    return v


###################################################################
############## ANTs / synthstrip subprocess wrappers
###################################################################
def _run(cmd: list, verbose: bool = False) -> None:
    subprocess.run(
        [str(c) for c in cmd],
        check=True,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )

def _imagemath(dim: int, out_path: Path, op: str, *op_args) -> Path:
    _run(["ImageMath", dim, out_path, op, *op_args])
    return Path(out_path)

def _denoise(in_path: Path, out_path: Path, verbose: bool = False) -> Path:
    _run(["DenoiseImage", "-d", "3", "-i", in_path, "-o", out_path, "-v", "1"], verbose=verbose)
    return Path(out_path)

def _n4(in_path: Path, out_path: Path, bias_output_path: Path | None = None,
        convergence: str = "[50x50x50x50x50,0.0000001]", verbose: bool = False) -> Path:
    out_arg = f"[{out_path},{bias_output_path}]" if bias_output_path is not None else str(out_path)
    _run(["N4BiasFieldCorrection", "-d", "3", "-i", in_path, "-o", out_arg,
          "-c", convergence, "-v", "1"], verbose=verbose)
    return Path(out_path)

def _synthstrip(in_path: Path, mask_out_path: Path, weights_path: Path, nthreads: int, verbose: bool = False) -> Path:
    _run(["nipreps-synthstrip", "-i", in_path, "-m", mask_out_path,
          "--model", weights_path, "--num-threads", nthreads], verbose=verbose)
    return Path(mask_out_path)

def _ants_rigid_register(fixed: Path, moving: Path, out_prefix: Path,
                          convergence: str, shrink_factors: str, smoothing_sigmas: str,
                          verbose: bool = False) -> Path:
    warped_path = Path(f"{out_prefix}Warped.nii.gz")
    cmd = [
        "antsRegistration", "-d", "3", "-v", "1", "--float", "0",
        "--output", f"[{out_prefix},{warped_path}]",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
        "--convergence", f"[{convergence}]",
        "--shrink-factors", shrink_factors,
        "--smoothing-sigmas", smoothing_sigmas, # "--random-seed", "12345",
    ]
    _run(cmd, verbose=verbose)
    return Path(f"{out_prefix}0GenericAffine.mat")

def _apply_transforms(in_path: Path, ref_path: Path, out_path: Path,
                       transforms: list[Path] | None = None, verbose: bool = False) -> Path:
    # transforms=None (or []) => identity resample onto ref_path's grid, no -t flag.
    cmd = ["antsApplyTransforms", "-d", "3", "-v", "1",
           "-i", str(in_path), "-o", str(out_path), "-r", str(ref_path)]
    for t in (transforms or []):
        cmd += ["-t", str(t)]
    _run(cmd, verbose=verbose)
    return Path(out_path)

def _load_4d(path: Path):
    nii = nibabel.load(path)
    data = nii.get_fdata()
    if data.ndim == 3:
        data = data[..., numpy.newaxis]
    return nii, data

def _mppca_sos_denoise(entries: list[dict], tmp_fld: Path, do_mppca: bool, mppca_window: list[int],
                        n_sos: int, nthreads: int, parser: argparse.ArgumentParser) -> None:
    # Concatenate every VFA/MT (possibly multi-echo) volume along the 4th dimension, optionally run MP-PCA denoising jointly
    # over the whole stack, then sum-of-squares (SoS) combine the first n_sos echoes of each individual volume back into a single 3D volume. 
    niis, datas = zip(*(_load_4d(e["path"]) for e in entries))

    img_cat = datas[0]
    for d in datas[1:]:
        img_cat = numpy.concatenate([img_cat, d], axis=3)

    if do_mppca:
        print("Running MP-PCA over concatenated volumes...")
        img_cat, *_ = tmppca_cpp.denoise_tmppca(img_cat, window=mppca_window, num_threads=nthreads)
    else:
        print("Skipping MP-PCA...")

    img_idx = 0
    for e, nii, data in zip(entries, niis, datas):
        n_mge = data.shape[3]
        if n_mge < n_sos:
            parser.error(f"--n_sos: number of echoes in '{e['path']}' ({n_mge}) < --n_sos ({n_sos}).")

        img_sos = numpy.square(img_cat[:, :, :, img_idx:img_idx + n_sos])
        img_sos = numpy.sqrt(numpy.sum(img_sos, axis=3))
        new_nii = nibabel.Nifti1Image(img_sos, nii.affine, nii.header)

        out_path = tmp_fld / f"{e['role']}_denMPPCA_SOS.nii.gz"
        nibabel.save(new_nii, out_path)
        e["denoised"] = out_path

        img_idx += n_mge


###################################################################
############## Get mri_synthstrip weights path
###################################################################
def get_synthstrip_weights() -> Path:
    # Make sure synthstrip.1.pt is present next to the nipreps-synthstrip binary.
    # Try downloading it on first use if it isn't there yet.
    # Returns the path to the weights file.
    SYNTHSTRIP_WEIGHTS_URL = ("https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt")
    SYNTHSTRIP_WEIGHTS_NAME = "synthstrip.1.pt"
    bin_path = shutil.which("nipreps-synthstrip")
    if bin_path is None:
        raise RuntimeError("nipreps-synthstrip not found in PATH; check installation.")

    dest = Path(bin_path).parent / SYNTHSTRIP_WEIGHTS_NAME
    if dest.is_file():
        return dest

    print(
        "\n[preproc-qMT] -------------------------------------------------------\n"
        "[preproc-qMT] synthstrip.1.pt (niprep-synthstrip requirement) weight not found...\n"
        "[preproc-qMT] synthstrip.1.pt is part of the FreeSurfer software.\n"
        "[preproc-qMT] By downloading this file, you acknowledge the FreeSurfer\n"
        "[preproc-qMT] licence (MIT): https://choosealicense.com/licenses/mit/\n"
        "[preproc-qMT] More information at:\n"
        "[preproc-qMT]   https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/\n"
        "[preproc-qMT] -------------------------------------------------------\n"
    )
    print(f"[preproc-qMT] Downloading synthstrip weights -> {dest}...")
    try:
        urllib.request.urlretrieve(SYNTHSTRIP_WEIGHTS_URL, dest)
        print(f"[preproc-qMT] synthstrip weights downloaded: {dest}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download synthstrip weights: {exc}\n"
            f"  Download manually with:\n"
            f"    wget {SYNTHSTRIP_WEIGHTS_URL} -O {dest}"
        ) from exc

    return dest


###################################################################
############## Get CPU info
###################################################################
def get_physCPU_number():
    # from joblib source code (commit d5c8274)
    # https://github.com/joblib/joblib/blob/master/joblib/externals/loky/backend/context.py#L220-L246
    if sys.platform == "linux":
        cpu_info = subprocess.run(
            "lscpu --parse=core".split(" "), capture_output=True)
        cpu_info = cpu_info.stdout.decode("utf-8").splitlines()
        cpu_info = {line for line in cpu_info if not line.startswith("#")}
        cpu_count_physical = len(cpu_info)
    elif sys.platform == "win32":
        cpu_info = subprocess.run(
            "wmic CPU Get NumberOfCores /Format:csv".split(" "),
            capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8').splitlines()
        cpu_info = [l.split(",")[1] for l in cpu_info
                    if (l and l != "Node,NumberOfCores")]
        cpu_count_physical = sum(map(int, cpu_info))
    elif sys.platform == "darwin":
        cpu_info = subprocess.run(
            "sysctl -n hw.physicalcpu".split(" "), capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8')
        cpu_count_physical = int(cpu_info)
    else:
        raise NotImplementedError(
            "unsupported platform: {}".format(sys.platform))
    if cpu_count_physical < 1:
        raise ValueError(
            "found {} physical cores < 1".format(cpu_count_physical))
    return cpu_count_physical


###################################################################
############## main
###################################################################
def main():
    global tmp_fld, flag_keep_tmp

    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    args = parse_args()
    v = validate_args(args, parser)

    print("preproc-qMT...")
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(v["nthreads"])
    print(f"Processing with {v['nthreads']} thread(s)")

    entries: list[dict] = v["entries"]
    refvfa_reg: str     = v["refvfa_reg"]
    print(f"Reference volume for Motion Correction: '{refvfa_reg}'")
    output_dir: Path    = v["output_dir"]
    weights_path: Path  = v["synthstrip_weights"]
    nthreads: int       = v["nthreads"]
    verbose: bool       = v["verbose"]

    # Temporary directory
    tmp_fld = Path(f"/tmp/tmp_qMT_{datetime.now().strftime('%y%m%d-%H%M%S-%f')[:-3]}")
    tmp_fld.mkdir(parents=True, exist_ok=True)
    print(f"Temporary folder: {tmp_fld}")

    # Output paths
    anat_den_path       = tmp_fld / "ANAT_den.nii"
    anat_denn4_path     = tmp_fld / "ANAT_denN4.nii"
    anat_masked_path    = tmp_fld / "ANAT_denN4_masked.nii"
    b1_in_anat_path     = output_dir / "B1_MAP.nii.gz"
    mask_anat_path      = output_dir / "MASK_ANAT.nii.gz"
    mt_in_anat_path     = output_dir / "MT.nii.gz"
    vfa_in_anat_path    = output_dir / "VFA.nii.gz"

    # 1. ANAT: TE1 extraction, denoise, N4, mask
    print("\n--- preproc-qMT - Step 1: anatomical preprocessing (extract TE1, denoise, N4, mask)")
    te1_prefix = tmp_fld / "ANAT_TE1_"
    _imagemath(4, Path(f"{te1_prefix}.nii.gz"), "TimeSeriesSubset", v["anat_path"], 1)
    te1_path = Path(f"{te1_prefix}100.nii.gz")
    _denoise(te1_path, anat_den_path, verbose=verbose)
    _n4(anat_den_path, anat_denn4_path, verbose=verbose)
    _synthstrip(anat_denn4_path, mask_anat_path, weights_path, nthreads, verbose=verbose)
    _imagemath(3, anat_masked_path, "m", mask_anat_path, anat_denn4_path)
    print("--- preproc-qMT - Step 1: done\n")

    # 2. VFA/MT: MPPCA/SoS, brain masks, N4+brain extraction
    print("--- preproc-qMT - Step 2: MP-PCA/SoS + brain masking + bias correction of VFA/MT volumes")

    _mppca_sos_denoise(entries, tmp_fld, do_mppca=v["mppca"], mppca_window=v["mppca_window"],
                        n_sos=v["n_sos"], nthreads=nthreads, parser=parser)

    for e in entries:
        mask_path = tmp_fld / f"{e['role']}_mask.nii.gz"
        _synthstrip(e["denoised"], mask_path, weights_path, nthreads, verbose=verbose)
        e["mask"] = mask_path

    ref_entry = next(e for e in entries if e["role"] == refvfa_reg)
    bias_field_path = tmp_fld / f"BiasField_{refvfa_reg}.nii.gz"
    n4_ref_path = tmp_fld / f"{refvfa_reg}_N4.nii.gz"
    _n4(ref_entry["denoised"], n4_ref_path, bias_output_path=bias_field_path, verbose=verbose)

    for e in entries:
        N4be_path = tmp_fld / f"{e['role']}_N4be.nii.gz"
        _imagemath(3, N4be_path, "/", e["denoised"], bias_field_path)
        _imagemath(3, N4be_path, "m", N4be_path, e["mask"])
        e["N4be"] = N4be_path
    print("--- preproc-qMT - Step 2: done\n")

    # 3. MoCo: register every N4be volume onto the N4be reference
    print(f"--- preproc-qMT - Step 3: rigid MoCo onto reference contrast '{refvfa_reg}'")
    for e in entries:
        out_prefix = tmp_fld / f"{e['role']}_MoCo_"
        warped_path = Path(f"{out_prefix}Warped.nii.gz")
        if e['role'] != refvfa_reg:
            print(f"  Registering '{e['role']}' -> '{refvfa_reg}'...")
        _ants_rigid_register(
            fixed=ref_entry["N4be"], moving=e["N4be"], out_prefix=out_prefix,
            convergence="250x100,1e-6,10",
            shrink_factors="2x1", smoothing_sigmas="1x0vox", verbose=verbose)
        e["moco_transform"] = Path(f"{out_prefix}0GenericAffine.mat")
    print("--- preproc-qMT - Step 3: done\n")

    # 4. Register reference contrasts onto ANAT
    print(f"--- preproc-qMT - Step 4: register reference contrast '{refvfa_reg}' onto ANAT")
    to_anat_prefix = tmp_fld / f"{refvfa_reg}_toAnat_"
    to_anat_warped_path = Path(f"{to_anat_prefix}Warped.nii.gz")
    _ants_rigid_register(
        fixed=anat_masked_path, moving=ref_entry["N4be"], out_prefix=to_anat_prefix,
        convergence="100,1e-6,10",
        shrink_factors="1", smoothing_sigmas="0vox", verbose=verbose,
    )
    to_anat_transform = Path(f"{to_anat_prefix}0GenericAffine.mat")
    print("--- preproc-qMT - Step 4: done\n")

    # 5. Apply composed transforms to the original volumes + mask
    print("--- preproc-qMT - Step 5: apply MoCo + to-ANAT transforms to original volumes")
    for e in entries:
        out_path = tmp_fld / f"{e['role']}_inANAT.nii.gz"
        # transform order matches the original script: per-volume MoCo transform
        # first, then the reference-to-ANAT transform.
        _apply_transforms(
            e["denoised"], anat_masked_path, out_path,
            transforms=[e["moco_transform"], to_anat_transform], verbose=verbose,
        )
        _imagemath(3, out_path, "m", out_path, mask_anat_path)
        e["in_anat"] = out_path
    print("--- preproc-qMT - Step 5: done\n")

    # 6. Reassemble VFA and MT (MT0/MTw) 4D stacks
    print("--- preproc-qMT - Step 6: reassembling 4D VFA/MTw stacks")
    # order within each group follows the entries' order, matching the original --VFA/--MT argument order.
    vfa_group = [e["in_anat"] for e in entries if e["modality"] == "vfa"]
    mt_group = [e["in_anat"] for e in entries if e["modality"] == "mt"]

    _imagemath(4, vfa_in_anat_path, "TimeSeriesAssemble", 1, 0, *vfa_group)
    _imagemath(4, mt_in_anat_path, "TimeSeriesAssemble", 1, 0, *mt_group)
    print("--- preproc-qMT - Step 6: done\n")

    # 7. B1 map: scale, resample onto ANAT grid, smooth, mask
    print("--- preproc-qMT - Step 7: B1 map preprocessing")
    _imagemath(3, b1_in_anat_path, "/", v["b1_path"], v["b1_fac"])
    _apply_transforms(b1_in_anat_path, anat_denn4_path, b1_in_anat_path, transforms=None, verbose=verbose)
    _imagemath(3, b1_in_anat_path, "G", b1_in_anat_path, 3)
    _imagemath(3, b1_in_anat_path, "m", b1_in_anat_path, mask_anat_path)
    print("--- preproc-qMT - Step 7: done\n")

    print(f"  VFA stack : {vfa_in_anat_path}")
    print(f"  MTw stack : {mt_in_anat_path}")
    print(f"  B1 map    : {b1_in_anat_path}")
    print(f"  ANAT mask : {mask_anat_path}")

    # Cleanup
    if not flag_keep_tmp:
        cleanup()
        print("\nTemporary files removed.")
    else:
        tmp_dest = v["output_dir"] / tmp_fld.name
        shutil.move(tmp_fld, tmp_dest)
        print(f"\nTemporary files kept at: {tmp_dest}")

    print("\npreproc-qMT: done")


if __name__ == "__main__":
    sys.exit(main())