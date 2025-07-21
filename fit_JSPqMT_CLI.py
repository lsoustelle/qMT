# ///////////////////////////////////////////////////////////////////////////////////////////////
# // L. SOUSTELLE, PhD, Aix Marseille Univ, CNRS, CRMBM, Marseille, France
# // Contact: lucas.soustelle@univ-amu.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////

import sys
import os
# prevent multi-threading trigger of numpy (see https://numpy.org/devdocs/reference/global_state.html)
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy
import nibabel
import scipy.optimize
import scipy.integrate
import scipy.linalg
import time
import multiprocessing
import argparse; from argparse import RawTextHelpFormatter
import subprocess
import collections
import warnings

def main():
    global gamma; gamma = 267.513 * 1e6 # rad/s/T
    global VFA_parx; global MTw_parx; global qMTcontrainst_parx
    global NWORKERS;

    ## parse arguments
    text_description = "Fast quantitative MT joint fitting for R1f and MPF mapping from a VFA & {MTw;MT0} protocol.\
                        \nNotes: \
                        \n\t1) The model used for joint fitting is described in Ref. [1], an expansion of the Single‐Point qMT introduced in [2].\
                        \n\t2) Parameters are straightforwardly the same as i) in the Special Card interface from the Siemens' vibeMT C2P sequences, and ii) in the MT preparation card from the MT_SPGR Bruker's sequence.\
                        \n\t3) The conventions T1f = T1b = T1 (=1/R1) and M0f + M0b = 1.0 are adopted.\
                        \n\t4) B1 correction is strongly advised [3].\
                        \n\t5) MT0 & MTw images are considered to share the same readout flip angle.\
                        \n\t6) The Generalized Bloch Model described in Ref. [4] can be used for the modeling of on-resonance pulses macromolecular saturation.\
                        \n\t7) B0 correction considerations:\
                        \n\t  - Readout pulses: dB0 is assumed to have no effect regarding on-resonance saturation or resulting flip angle.\
                        \n\t  - Saturation pulses: the script is intended to have a dual-offset saturated (i.e. using a sine-modulated preparation pulse) MTw volume as input.\
                        \nReferences:\
                        \n\t [1] L. Soustelle et al., Quantitative Magnetization Transfer parametric mapping unbiased by on-resonance saturation and dipolar order contributions, ISMRM 2022 \
                        \n\t [2] V. Yarnykh, Fast macromolecular proton fraction mapping from a single off-resonance magnetization transfer measurement, MRM 2012;68:166-178 \
                        \n\t [3] V. Yarnykh et al., Scan–Rescan Repeatability and Impact of B0 and B1 Field Nonuniformity Corrections in Single‐Point Whole‐Brain Macromolecular Proton Fraction Mapping, JMRI 2020;51:1789-1798 \
                        \n\t [4] J. Assländer et al., Generalized Bloch model: A theory for pulsed magnetization transfer, MRM 2022;87:2003-2017 \
                        " 
    parser = argparse.ArgumentParser(description=text_description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('MT',           nargs="+",help="Input couple MT0/MTw NIfTI path(s) (comma-separated for 3D, single path for 4D)")
    parser.add_argument('VFA',          nargs="+",help="Input VFA NIfTI path(s) (comma-separated for 3D, single path for 4D)")
    parser.add_argument('MPF',          help="Output MPF NIfTI path")
    parser.add_argument('T1f',          help="Output T1f NIfTI path")
    parser.add_argument('--R1f',        help="Output R1f NIfTI path (optional)")
    parser.add_argument('--MTw_TIMINGS',required=True, help="Sequence timings in ms (comma-separated), in this order:   \n"
                                                                "\t 1) Saturation pulse duration (ms) \n"
                                                                "\t 2) Interdelay between Saturation pulse and Readout pulse (ms) \n"
                                                                "\t 3) Readout pulse duration (ms) \n"
                                                                "\t 4) Sequence Time-to-Repetition (TR; ms) \n"
                                                                "e.g. --MTw_TIMINGS 12.0,2.1,1.0,30.0")
    parser.add_argument('--VFA_TIMINGS',required=True, help="Sequence timings in ms (comma-separated), in this order:   \n"
                                                                "\t 1) Readout pulse duration (ms) \n"
                                                                "\t 2) Sequence Time-to-Repetition (TR; ms) \n"
                                                                "e.g. --VFA_TIMINGS 1.0,30.0")
    parser.add_argument('--MTw_PARX',   required=True,  help="Saturation parameters (comma-separated), in this order:   \n"
                                                                "\t 1) Readout flip angle of MT0/MTw (deg; single common value) \n"
                                                                "\t 2) Readout pulse shape (Hann, BP) \n" 
                                                                "\t 3) Saturation pulse flip angle (deg) \n"
                                                                "\t 4) Saturation pulse off-resonance frequency (Hz) \n"
                                                                "\t 5) Saturation pulse shape (Hann-Sine, GaussHann-Sine, Gauss-Sine) \n" 
                                                                "\t 6) Gaussian saturation pulse FWHM (Hz; not used if purely Hann-Sine-shaped) \n"
                                                                "e.g. --MTw_PARX 10.0,BP,560.0,4000.0,Hann-Sine")
    parser.add_argument('--VFA_PARX',   required=True,  help="Readout pulse parameters of experiments (comma-separated), in this order:   \n"
                                                                "\t 1) Readout flip angles [VFA1, VFA2, ..., VFAn] (deg; same order as in provided VFA volume(s)) \n"
                                                                "\t 2) Readout pulse shape (Hann, BP) \n" 
                                                                "e.g. --VFA_PARX 6,10,25,BP")
    parser.add_argument('--B1',                 nargs="?",help="Input normalized B1 map NIfTI path (strongly advised)")
    parser.add_argument('--B0',                 nargs="?",help="Input B0 map NIfTI path (in Hz; computation time is much longer)")
    parser.add_argument('--mask',               nargs="?",help="Input Mask binary NIfTI path")
    parser.add_argument('--nworkers',           nargs="?",type=int, default=1, help="Use this for multi-threading acceleration (default: 1)")
    parser.add_argument('--qMTconstraint_PARX', help="Constained parameters for SP-qMT estimation (comma-separated) in this order:\n"  
                                                                "\t 1) R1fT2f (default: 0.0158) \n"
                                                                "\t 2) T2b (s; default: 10.0e-6 s) \n"
                                                                "\t 3) R (s-1; default: 21.1 s-1) \n"
                                                                "e.g. --qMTconstraint_PARX 0.0158,10.0e-6,21.1")
    parser.add_argument('--useGBM', action='store_true', help="Use the Generalized Bloch Model")

    args                = parser.parse_args()
    MT_in_niipaths      = [','.join(args.MT)] # ensure it's a comma-separated list
    VFA_in_niipaths     = [','.join(args.VFA)] # ensure it's a comma-separated list
    MPF_out_niipaths    = args.MPF # ensure it's a comma-separated list
    T1f_out_niipath     = args.T1f
    R1f_out_niipath     = args.R1f
    B1_in_niipath       = args.B1
    B0_in_niipath       = args.B0
    mask_in_niipath     = args.mask
    NWORKERS            = args.nworkers if args.nworkers <= get_physCPU_number() else get_physCPU_number()
    print('\nWorking with {} cores'.format(NWORKERS))
    
    #### Check inputs
    print('')
    print('--------------------------------------------------')
    print('---- Checking entries for JSP-qMT processing -----')
    print('--------------------------------------------------')
    print('')
    
    MTw_NT                  = collections.namedtuple('MTw_NT','FAsat delta_f ROFA ROshape MTshape FWHM ROdur Tm Ts TR')
    args.MTw_TIMINGS        = args.MTw_TIMINGS.split(',')
    args.MTw_PARX           = args.MTw_PARX.split(',')
    if len(args.MTw_TIMINGS) != 4: 
        parser.error('Wrong amount of Sequence Parameters (--MTw_TIMINGS \
                         --- expected 4, found {})'.format(len(args.MTw_TIMINGS)))
    if len(args.MTw_PARX) < 5: 
        parser.error('Wrong amount of Saturation/Readout Parameters (--MTw_PARX \
                         --- expected 5 or 6, found {})'.format(len(args.MTw_PARX)))
    if args.MTw_PARX[4] in ("GaussHann-Sine", "Gauss-Sine") and len(args.MTw_PARX) < 6:
        parser.error('GaussHann-Sine or Gauss-Sine saturation pulse set, but no FWHM found')
    MTw_parx = MTw_NT(  ROFA       = float(args.MTw_PARX[0]),
                        ROshape    = str(args.MTw_PARX[1]),
                        FAsat      = float(args.MTw_PARX[2]), 
                        delta_f    = float(args.MTw_PARX[3]),
                        MTshape    = str(args.MTw_PARX[4]),
                        FWHM       = float(args.MTw_PARX[5]) if len(args.MTw_PARX) > 5 else None,
                        Tm         = float(args.MTw_TIMINGS[0])*1e-3, # convert to sec
                        Ts         = float(args.MTw_TIMINGS[1])*1e-3,
                        ROdur      = float(args.MTw_TIMINGS[2])*1e-3,
                        TR         = float(args.MTw_TIMINGS[3])*1e-3)
    
    VFA_NT                  = collections.namedtuple('VFA_NT','ROFA ROshape ROdur TR')
    args.VFA_TIMINGS        = args.VFA_TIMINGS.split(',')
    args.VFA_PARX           = args.VFA_PARX.split(',')
    if len(args.VFA_TIMINGS) != 2: 
        parser.error('Wrong amount of Sequence Parameters (--VFA_TIMINGS \
                         --- expected 2, found {})'.format(len(args.VFA_TIMINGS)))
    VFA_parx = VFA_NT(  ROFA       = numpy.array(args.VFA_PARX[:-1]).astype(numpy.float64), 
                        ROshape    = str(args.VFA_PARX[-1]),
                        ROdur      = float(args.VFA_TIMINGS[0])*1e-3, # convert to sec
                        TR         = float(args.VFA_TIMINGS[1])*1e-3)

    qMTcontraint_NT = collections.namedtuple('qMTcontraint_NT', 'R1fT2f T2b R')
    if args.qMTconstraint_PARX is not None:   
        args.qMTconstraint_PARX  = args.qMTconstraint_PARX.split(',')
        if len(args.qMTconstraint_PARX) != 3: 
            parser.error('Wrong amount of constraint qMT parameters (qMTconstraint_PARX \
                             --- expected 3, found {})'.format(len(args.qMTconstraint_PARX)))
        qMTcontrainst_parx = qMTcontraint_NT(   R1fT2f = float(args.qMTconstraint_PARX[0]), 
                                                T2b    = float(args.qMTconstraint_PARX[1]),
                                                R      = float(args.qMTconstraint_PARX[2]))         
    else:
        print('--qMTconstraint_PARX not set, setting to default values \n')
        qMTcontrainst_parx = qMTcontraint_NT(   R1fT2f = 0.0158, 
                                                T2b    = 10.0e-6,
                                                R      = 21.1)  

    FLAG_useGBM = True if args.useGBM else False

    print('Summary of input MTw/MT0 sequence parameters:')
    print('\t Saturation flip angle: {:.1f} deg'.format(MTw_parx.FAsat))
    print('\t Saturation pulse off-resonance frenquency: {:.1f} Hz'.format(MTw_parx.delta_f))
    print('\t Saturation pulse shape: {}'.format(MTw_parx.MTshape))
    if len(args.MTw_PARX) > 5:
        print('\t Gaussian pulse FWHM: {:.1f} Hz'.format(MTw_parx.FWHM))
    print('\t Readout flip angle: {:.1f} deg'.format(MTw_parx.ROFA))
    print('\t Readout pulse duration: {:.1f} ms'.format(MTw_parx.ROdur*1e3))
    print('\t Readout pulse shape: {}'.format(MTw_parx.ROshape))
    print('\t Saturation pulse duration: {:.1f} ms'.format(MTw_parx.Tm*1e3))
    print('\t Interdelay saturation pulse <--> Readout pulse: {:.2f} ms'.format(MTw_parx.Ts*1e3))
    print('\t Sequence Time-to-Repetition: {:.1f} ms'.format(MTw_parx.TR*1e3))
    print('')
    print('Summary of input VFA sequence parameters:')
    print('\t Readout flip angles: [' + ', '.join('{:.1f}'.format(v) for v in VFA_parx.ROFA) + '] deg')
    print('\t Readout pulse duration: {:.1f} ms'.format(VFA_parx.ROdur*1e3))
    print('\t Readout pulse shape: {}'.format(VFA_parx.ROshape))
    print('\t Sequence Time-to-Repetition: {:.1f} ms'.format(VFA_parx.TR*1e3))
    print('')
    print('Summary of constraint qMT parameters:')
    print('\t R1fT2f: {:.4f}'.format(qMTcontrainst_parx.R1fT2f))
    print('\t T2b:\t {:.1f} us'.format(qMTcontrainst_parx.T2b*1e6))
    print('\t R:\t {:.1f} s-1'.format(qMTcontrainst_parx.R))
    print('')
    if args.useGBM:
        print('Using Generalized Bloch Model (readout pulses)')
    print('')

    # last check before going in
    for field in qMTcontrainst_parx._fields:
        if getattr(qMTcontrainst_parx, field) < 0:
            parser.error('All qMTcontrainst_parx values should be positive')
    for field in VFA_parx._fields:
        # skip the pulse shape parameter (string)
        if isinstance(getattr(VFA_parx, field), str): continue
        # special treatment for element (numpy array of FA [ROFA])
        if isinstance(getattr(VFA_parx, field), numpy.ndarray): 
            if any(X <= 0 for X in getattr(VFA_parx, field)):
                parser.error('All VFA provided numerical values should be strictly positive')
        elif getattr(VFA_parx, field) <= 0:
            parser.error('All VFA provided numerical values should be strictly positive')
    for field in MTw_parx._fields:
        # skip the pulse shape parameter (string) & potentially undefined FWHM (= None)
        if isinstance(getattr(MTw_parx, field), str) or getattr(MTw_parx, field) is None: continue
        if getattr(MTw_parx, field) <= 0:
            parser.error('All MTw provided numerical values should be strictly positive')
    if MTw_parx.ROshape not in ("Hann", "BP"):
        parser.error('Unrecognized MTw readout pulse shape (allowed: "Hann" or "BP")')
    if MTw_parx.MTshape not in ("Hann-Sine", "GaussHann-Sine", "Gauss-Sine"):
        parser.error('Unrecognized MT saturation pulse shape (allowed: "Hann-Sine", "GaussHann-Sine" or "Gauss-Sine")')
    if VFA_parx.ROshape not in ("Hann", "BP"):
        parser.error('Unrecognized VFA readout pulse shape (allowed: "Hann" or "BP")')

    #### check input data
    # check MT0/MTw data
    if os.path.isfile(MT_in_niipaths[0]) and len(nibabel.load(MT_in_niipaths[0]).shape) == 4:
        FLAG_isqMT4D = 1
        print('MT0/MTw provided volume (4D) exist')
    else:
        FLAG_isqMT4D = 0
        MT_in_niipaths = MT_in_niipaths[0].split(',')
        for vol_niipath in MT_in_niipaths:
            if not os.path.isfile(vol_niipath):
                parser.error('Volume {} does not exist'.format(vol_niipath))
        print('MT0/MTw provided volumes (3D) exist')
        
    # check VFA volumes
    if os.path.isfile(VFA_in_niipaths[0]) and len(nibabel.load(VFA_in_niipaths[0]).shape) == 4:
        FLAG_isVFA4D    = 1
        N_VFA_VOL       = nibabel.load(VFA_in_niipaths[0]).shape[3]
        print('VFA provided volume (4D) exist')
    else:
        FLAG_isVFA4D    = 0
        VFA_in_niipaths = VFA_in_niipaths[0].split(',')
        N_VFA_VOL       = len(VFA_in_niipaths)
        for vol_niipath in VFA_in_niipaths:
            if not os.path.isfile(vol_niipath):
                parser.error('Volume {} does not exist'.format(vol_niipath))
        print('VFA provided volumes (3D) exist')
    
    if N_VFA_VOL != len(VFA_parx.ROFA):
        parser.error('Mismatch in number of provided VFA flip angles (found {}) vs. number of provided VFA volumes (found {})'.format(len(VFA_parx.ROFA),N_VFA_VOL))
    
    # check B1 map
    if args.B1 is None:
        print('No B1 map provided (this is highly not recommended)')
    elif args.B1 is not None and not os.path.isfile(B1_in_niipath):
        parser.error('B1 map volume {} does not exist'.format(B1_in_niipath))
    else:
        print('B1 map provided volume exist')
        
    # check B0 map
    if args.B0 is None:
        print('No B0 map provided')
    elif args.B0 is not None and not os.path.isfile(B0_in_niipath):
        parser.error('B0 map volume {} does not exist'.format(B0_in_niipath))
    else:
        print('B0 map provided volume exist')
        
    # check mask
    if args.mask is None:
        print('No mask provided')
    elif args.mask is not None and not os.path.isfile(mask_in_niipath):
        parser.error('Mask map volume {} does not exist'.format(mask_in_niipath))
    else:
        print('Mask provided volume exist')
        
    #### load data
    # get MT0/MTw data
    MT_data    = list()
    if FLAG_isqMT4D: # case 4D
        MT_nii = nibabel.load(MT_in_niipaths[0]).get_fdata()
        for ii in range(MT_nii.shape[3]):
            MT_data.append(MT_nii[:,:,:,ii])
    else:
        for ii in range(len(MT_in_niipaths)):
            MT_data.append(nibabel.load(MT_in_niipaths[ii]).get_fdata())

    # get VFA data
    VFA_data    = list()
    if FLAG_isVFA4D: # case 4D
        VFA_nii = nibabel.load(VFA_in_niipaths[0]).get_fdata()
        for ii in range(VFA_nii.shape[3]):
            VFA_data.append(VFA_nii[:,:,:,ii])
    else:
        for ii in range(len(VFA_in_niipaths)):
            VFA_data.append(nibabel.load(VFA_in_niipaths[ii]).get_fdata())

    # get B1 data
    if args.B1 is not None:
        B1_data = nibabel.load(B1_in_niipath).get_fdata()
    else:
        B1_data = numpy.ones(nibabel.load(MT_in_niipaths[0]).shape[0:3])
        
    # get B0 data
    if args.B0 is not None:
        B0_data = nibabel.load(B0_in_niipath).get_fdata()
    else:
        B0_data = numpy.zeros(nibabel.load(MT_in_niipaths[0]).shape[0:3])
    
    # get indices to process from mask
    if args.mask is not None:
        mask_data   = nibabel.load(mask_in_niipath).get_fdata()
    else:
        mask_data   = numpy.ones(nibabel.load(MT_in_niipaths[0]).shape[0:3])
    mask_idx = numpy.asarray(numpy.where(mask_data == 1))

    
    #### build xData (prepare qMT parx)
    B1_data = B1_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    B0_data = B0_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    xData   = func_prepare_qMTparx(B1_data,B0_data,FLAG_useGBM)
    
    #### build yData
    MT0_data = MT_data[0][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    MTw_data = MT_data[1][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    yData    = numpy.zeros((MTw_data.shape[0],len(VFA_data)))
    for ii in range(len(VFA_data)):
        yData[:,ii] = VFA_data[ii][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T.ravel()
    yData = numpy.divide(numpy.concatenate((yData,MTw_data),axis=1), MT0_data, where=MT0_data>0)
    list_iterable = [*zip(xData,yData)]


    #### run
    print('')
    print('--------------------------------------------------')
    print('------ Proceeding to R1f & MPF estimations -------')
    print('--------------------------------------------------')
    print('')
    
    start_time = time.time()
    with multiprocessing.Pool(NWORKERS) as pool:
        res     = pool.starmap(fit_JSPqMT_lsq,list_iterable)
    delay = time.time()
    print("---- Done in {:.3f} seconds ----".format(delay - start_time))
    
    # store MPF & T1f/R1f + save nifti
    ref_nii = nibabel.load(MT_in_niipaths[0])

    R1f_map     = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    R1f_map[mask_idx[0],mask_idx[1],mask_idx[2]] = [a_tup[0] for a_tup in res] # get specific array elements from tuple in a tuple list
    ones_map    = numpy.full(ref_nii.shape[0:3],1,dtype=float)
    T1f_map     = numpy.divide(ones_map, R1f_map, where=R1f_map>0)
    new_img     = nibabel.Nifti1Image(T1f_map, ref_nii.affine, ref_nii.header)
    nibabel.save(new_img, T1f_out_niipath)
    if args.R1f is not None:
        new_img = nibabel.Nifti1Image(R1f_map, ref_nii.affine, ref_nii.header)
        nibabel.save(new_img, R1f_out_niipath)

    MPF_map = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    MPF_map[mask_idx[0],mask_idx[1],mask_idx[2]] = [a_tup[1] for a_tup in res] 
    new_img = nibabel.Nifti1Image(MPF_map, ref_nii.affine, ref_nii.header)
    nibabel.save(new_img, MPF_out_niipaths)
    
    
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
############## Preparation-related functions
################################################################### 
def func_prepare_qMTparx(B1_data,B0_data,FLAG_useGBM):
    
    print('Preparing qMT quantities ...')
    ### MTw/MT0 
    # SAT Pulse AI/PI, w1RMS nominal & actual Wb arrays
    MTw_SAT_AI,MTw_SAT_PI   = func_computeAIPI_SatPulse(MTw_parx.Tm,MTw_parx.MTshape,MTw_parx.FWHM)
    MTw_SAT_B1peak_nom      = MTw_parx.FAsat*(numpy.pi/180) / (gamma*MTw_SAT_AI*MTw_parx.Tm)
    MTw_SAT_w1RMS_nom       = gamma*MTw_SAT_B1peak_nom*numpy.sqrt(MTw_SAT_PI)
    if any(B0_data != 0.0): # compute G for each voxel
        print('--- Computing G(delta_f) for all voxels (B0 corrected) ...')
        T2b_array       = numpy.full(B0_data.shape[0],qMTcontrainst_parx.T2b)[numpy.newaxis,:].T
        delta_f_array   = numpy.full(B0_data.shape[0],MTw_parx.delta_f)[numpy.newaxis,:].T
        list_iterable   = numpy.hstack((T2b_array,delta_f_array,B0_data))
        start_time = time.time()
        with multiprocessing.Pool(NWORKERS) as pool:
            MTw_SAT_G  = pool.starmap(func_computeG_SuperLorentzian,list_iterable)
        delay = time.time()
        print("--- ... Done in {:.3f} seconds".format(delay - start_time))
        MTw_SAT_G  = numpy.array(MTw_SAT_G)[numpy.newaxis,:].T
    else: # same G for all voxels
        MTw_SAT_G = func_computeG_SuperLorentzian(qMTcontrainst_parx.T2b,MTw_parx.delta_f,0)
        MTw_SAT_G = numpy.tile(MTw_SAT_G,B1_data.shape[0])[numpy.newaxis,:].T
    MTw_SAT_w1RMS_array = MTw_SAT_w1RMS_nom * B1_data
    MTw_WbSAT_array     = (numpy.pi * MTw_SAT_w1RMS_nom**2  * MTw_SAT_G) * B1_data**2

    ### AI/PI and shapes of VFA/MTw RO pulses
    if VFA_parx.ROshape == "Hann":
        VFA_RO_AI,VFA_RO_PI   = 0.5,0.375 # Hann-shaped
        VFA_RO_shape  = lambda t: ( 0.5*(1 - numpy.cos((2*numpy.pi*t)/VFA_parx.ROdur)) )
        VFA_RO_t_grid = numpy.linspace(0.0,VFA_parx.ROdur,301)
    elif VFA_parx.ROshape == "BP":
        VFA_RO_AI,VFA_RO_PI   = 1.0,1.0 # Rectangular-shaped
        VFA_RO_shape  = lambda t: numpy.ones_like(t) # BP
        VFA_RO_t_grid = numpy.linspace(0.0,VFA_parx.ROdur,3)
    if MTw_parx.ROshape == "Hann":
        MTw_RO_AI,MTw_RO_PI   = 0.5,0.375 # Hann-shaped
        MTw_RO_shape  = lambda t: ( 0.5*(1 - numpy.cos((2*numpy.pi*t)/MTw_parx.ROdur)) )
        MTw_RO_t_grid = numpy.linspace(0.0,MTw_parx.ROdur,301)
    elif MTw_parx.ROshape == "BP":
        MTw_RO_AI,MTw_RO_PI   = 1.0,1.0 # Rectangular-shaped
        MTw_RO_shape  = lambda t: numpy.ones_like(t) # BP
        MTw_RO_t_grid = numpy.linspace(0.0,MTw_parx.ROdur,3)

    if FLAG_useGBM:
        # Build LUT of R2sl associated to readout pulses
        # T2b is fixed, so the only variables are rB1, pulse width & flip angles
        print("--- Pre-computing R2s,l quantities ...")
        start_time      = time.time()
        rB1_grid        = numpy.linspace(0.30,1.60,131) # raster rB1=0.01; fixed range suitable for 7T
        B1_data_clip    = numpy.clip(B1_data, 0.30, 1.60) # just for R2sl considerations, avoid inconsistencies
        idx_B1_data     = numpy.abs(rB1_grid[:, None] - numpy.round(B1_data_clip.ravel()[None, :], 2)).argmin(axis=0) # for mapping with same raster as rB1_grid

        # MT0/MTw experiments
        MTw_R2slRO_array= numpy.zeros((len(B1_data),1))
        R2slRO_list     = numpy.zeros((len(rB1_grid),1))
        LUT_Gval        = compute_greens_LUT(MTw_parx.ROdur, 1/qMTcontrainst_parx.T2b, N=100)
        LUT_Gval_iter   = [LUT_Gval] * len(rB1_grid)
        PW_iter         = [MTw_parx.ROdur] * len(rB1_grid)
        omega_y_iter    = MTw_parx.ROFA*numpy.pi/180.0/MTw_parx.ROdur/MTw_RO_AI*MTw_RO_shape(MTw_RO_t_grid) # rad/s
        omega_y_iter    = [omega_y_iter] * len(rB1_grid)
        R2sl_iter       = [*zip(omega_y_iter,PW_iter,LUT_Gval_iter,rB1_grid)]
        with multiprocessing.Pool(NWORKERS) as pool: # loop over rB1_grid basically
            R2slRO_list = numpy.array( pool.starmap(func_precompute_R2sl,R2sl_iter) )
        MTw_R2slRO_array = R2slRO_list[idx_B1_data][numpy.newaxis,:].T

        # VFA experiments
        VFA_R2slRO_array= numpy.zeros((len(B1_data),len(VFA_parx.ROFA)))
        R2slRO_list     = numpy.zeros((len(rB1_grid),len(VFA_parx.ROFA)))
        LUT_Gval        = compute_greens_LUT(VFA_parx.ROdur, 1/qMTcontrainst_parx.T2b, N=100)
        LUT_Gval_iter   = [LUT_Gval] * len(rB1_grid)
        PW_iter         = [MTw_parx.ROdur] * len(rB1_grid)
        for ii in range(len(VFA_parx.ROFA)):
            omega_y_iter= VFA_parx.ROFA[ii]*numpy.pi/180.0/VFA_parx.ROdur/VFA_RO_AI*VFA_RO_shape(VFA_RO_t_grid) # rad/s
            omega_y_iter= [omega_y_iter] * len(rB1_grid)
            R2sl_iter   = [*zip(omega_y_iter,PW_iter,LUT_Gval_iter,rB1_grid)]
            with multiprocessing.Pool(NWORKERS) as pool: # loop over rB1_grid basically
                R2slRO_list[:,ii] = numpy.array( pool.starmap(func_precompute_R2sl,R2sl_iter) )
            VFA_R2slRO_array[:,ii] = R2slRO_list[idx_B1_data,ii]
        print("--- ... Done in {:.3f} seconds".format(time.time() - start_time))  
    else:
        # RO Pulse AI/PI, w1RMS nominal & actual Wb arrays
        VFA_RO_B1peak_nom   = VFA_parx.ROFA*(numpy.pi/180) / (gamma*VFA_RO_AI*VFA_parx.ROdur)
        VFA_RO_w1RMS_nom    = gamma*VFA_RO_B1peak_nom*numpy.sqrt(VFA_RO_PI)    
        VFA_RO_G            = func_computeG_SphericalLineshape(qMTcontrainst_parx.T2b,0) # assume no dB0 impact
        VFA_RO_G            = numpy.tile(VFA_RO_G,B1_data.shape[0])[numpy.newaxis,:].T
        MTw_RO_B1peak_nom   = MTw_parx.ROFA*(numpy.pi/180) / (gamma*MTw_RO_AI*MTw_parx.ROdur)
        MTw_RO_w1RMS_nom    = gamma*MTw_RO_B1peak_nom*numpy.sqrt(MTw_RO_PI)   
        MTw_RO_G            = func_computeG_SphericalLineshape(qMTcontrainst_parx.T2b,0.0) # assume no dB0 impact
        MTw_RO_G            = numpy.tile(MTw_RO_G,B1_data.shape[0])[numpy.newaxis,:].T    
        MTw_WbRO_array      = (numpy.pi * MTw_RO_w1RMS_nom**2   * MTw_RO_G)  * B1_data**2
        VFA_WbRO_array      = (numpy.pi * VFA_RO_w1RMS_nom**2   * VFA_RO_G)  * B1_data**2
     
    # common
    MTw_ROFA_array      = MTw_parx.ROFA*numpy.pi/180 * B1_data # rad
    VFA_ROFA_array      = VFA_parx.ROFA*numpy.pi/180 * B1_data # rad

    
    ### build xData
    FLAG_useGBM_array   = numpy.full((len(B1_data), 1), FLAG_useGBM)
    if FLAG_useGBM:
        xData   = numpy.hstack((MTw_SAT_w1RMS_array,MTw_WbSAT_array,MTw_R2slRO_array,MTw_ROFA_array, \
                                VFA_R2slRO_array,VFA_ROFA_array, \
                                B0_data,FLAG_useGBM_array))
    else:
        xData   = numpy.hstack((MTw_SAT_w1RMS_array,MTw_WbSAT_array,MTw_WbRO_array,MTw_ROFA_array, \
                                VFA_WbRO_array,VFA_ROFA_array, \
                                B0_data,FLAG_useGBM_array))
    print('... Done')
    return xData
    
def func_computeAIPI_SatPulse(tau,shape,BW):
    if shape == "Hann-Sine": # pure Hann
        satPulse   = lambda t: ( 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )
        satPulseSq = lambda t: ( 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )**2
    elif shape == "GaussHann-Sine": # Gauss-Hann
        sigma      = numpy.sqrt(2*numpy.log(2) / (numpy.pi*BW)**2)
        satPulse   = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) * 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )
        satPulseSq = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) * 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )**2
    elif shape == "Gauss-Sine": # pure Gauss
        sigma      = numpy.sqrt(2*numpy.log(2) / (numpy.pi*BW)**2)
        satPulse   = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) )
        satPulseSq = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) )**2
    integ   = scipy.integrate.quad(satPulse,0,tau)
    integSq = scipy.integrate.quad(satPulseSq,0,tau)
    return integ[0]/tau, integSq[0]/tau ## normalized AI, PI
    
def func_computeG_SuperLorentzian(T2b,delta_f,dB0): # super-lorentzian lineshape
    if dB0 == 0.0:
        F = lambda x: (T2b/numpy.abs(3*x**2-1)) * \
                        numpy.exp(-2*((2*numpy.pi * delta_f * T2b)/(3*x**2-1))**2)
        INTEG_RES = scipy.integrate.quad(F,0,1)
        return numpy.sqrt(2/numpy.pi)*INTEG_RES[0]
    else:
        Fp = lambda x: (T2b/numpy.abs(3*x**2-1)) * \
                        numpy.exp(-2*((2*numpy.pi * (delta_f+dB0) * T2b)/(3*x**2-1))**2)
        Fm = lambda x: (T2b/numpy.abs(3*x**2-1)) * \
                        numpy.exp(-2*((2*numpy.pi * (-delta_f+dB0) * T2b)/(3*x**2-1))**2)
        INTEG_RESp = scipy.integrate.quad(Fp,0,1)
        INTEG_RESm = scipy.integrate.quad(Fm,0,1)
        return numpy.sqrt(2/numpy.pi)*(INTEG_RESp[0]+INTEG_RESm[0])/2 # average of G(df+dB0) & G(-df+dB0)
    
def func_computeG_SphericalLineshape(T2b,delta_f):
    # Function for Spherical lineshape integration
    # include neighboors contribution to remove singularity at the magic angle
    # see Pampel et al. NeuroImage 114 (2015) 136–146
    T2neighboors    = 1/31.4  # T2neighboors -> +Inf virtually mean no effect of T2neighboors
    def SphericalLineshape(theta):
        T2bSph  = 2*T2b/numpy.abs(3*numpy.cos(theta)**2-1)
        T2eff   = 1/numpy.sqrt(1/(T2bSph)**2+1/(T2neighboors)**2)
        return numpy.sin(theta)*T2eff*numpy.exp(-1/2*(2*numpy.pi*delta_f*T2eff)**2)
    INTEG_RES = scipy.integrate.quad(SphericalLineshape,0,numpy.pi/2)
    return numpy.sqrt(1./(2*numpy.pi))*INTEG_RES[0] 

def compute_greens_LUT(PW, R2b, N):
    tau_arr = numpy.linspace(0, PW, N)
    def SL(zeta, tau): return numpy.exp(- (R2b**2) * tau**2 * (3*zeta**2 - 1)**2 / 8)
    LUT_Gval = numpy.array([
        scipy.integrate.quad(lambda zeta: SL(zeta, tau), 0.0, 1.0, epsabs=1e-8, epsrel=1e-8)[0]
        for tau in tau_arr
    ])
    return LUT_Gval

# implementation of eq. 9 for ode solving
def func_GenBlochZs(omega_y, PW, LUT_Gval):
    t_history = []
    M_history = []

    def func(t, M):
        nonlocal t_history, M_history

        if t == 0.0:
            t_history = [0.0]
            M_history = [M.item()]
        else:
            t_history.append(t)
            M_history.append(M.item())

        # Ensure unique and sorted
        t_hist_np   = numpy.array(t_history)
        M_hist_np   = numpy.array(M_history)
        t_hist_unique, idx_unique = numpy.unique(t_hist_np, return_index=True)
        t_hist_np   = t_hist_unique
        M_hist_np   = M_hist_np[idx_unique]

        # Interpolators
        PW_grid     = numpy.linspace(0, PW, len(omega_y))
        G_grid      = numpy.linspace(0, PW, len(LUT_Gval))
        def Mzb_t(tau):     return numpy.interp(tau, t_hist_np, M_hist_np)
        def omega_y_t(tau): return numpy.interp(tau, PW_grid, omega_y)
        def G_LUT_t(tau):   return numpy.interp(tau, G_grid, LUT_Gval)
        if t == 0.0: # initial tspan value, avoid having impossible interpolation with Mzb(tau)
            integ_y = 0.0
        else:
            def integrand_y_t(tau): return G_LUT_t(t - tau) * omega_y_t(tau) * Mzb_t(tau)
            # adapted num. of samples for discrete Simpson's integration (raster 0.5 us)
            tau_grid    = numpy.linspace(0.0, t, int(numpy.ceil(t/0.05e-6))) 
            eval_y      = integrand_y_t(tau_grid)
            integ_y     = scipy.integrate.simpson(eval_y, tau_grid)

        dM = -omega_y_t(t) * integ_y
        return dM
    return func

def func_GenBloch_R2sl(R2sl, xData, yData): 
    # solving eq. 23 with piece-wise integration over a discretized pulse with arbitrary shape
    # the pulse should be symmetric and have an odd-number of samples to avoid extensive compution
    omega_y, PW = xData
    Mzb_Zs      = yData
    N           = len(omega_y) - 1  # num. of intervals
    Dt          = PW/N
    Xt_PULSE    = numpy.zeros((2, 2, N))

    # First half of the pulse
    for ii in range(N // 2):
        MAT = numpy.array([ [-R2sl,        omega_y[ii]],
                         [-omega_y[ii],       0    ] ])
        Xt_PULSE[:, :, ii] = scipy.linalg.expm(MAT * Dt)
    # Second half: symmetric fill
    for ii in range(N // 2, N):
        Xt_PULSE[:, :, ii] = Xt_PULSE[:, :, N-ii-1]
    Xt = numpy.eye(2)
    for ii in range(N):
        Xt = Xt @ Xt_PULSE[:, :, ii]
        
    M = Xt @ numpy.array([0.0, 1.0]) # Mzb0 = 1.0
    Mzb_R2sl = M[1]
    return Mzb_Zs - Mzb_R2sl

def func_GenBlochBP_R2sl(R2sl, xData, yData):
    # Solve the eq. 23 from 0 to PW (analytical solution to eq. R2sl); works for BP pulse only
    omega_y, PW = xData
    Mzb_Zs      = yData
    s           = numpy.sqrt(complex(R2sl**2 - 4 * omega_y[0]**2))
    lambda1     = (-R2sl + s) / 2
    lambda2     = (-R2sl - s) / 2
    Mzb_R2sl    = numpy.real((R2sl + lambda1) / s * numpy.exp(lambda1 * PW) - 
                          (R2sl + lambda2) / s * numpy.exp(lambda2 * PW)) * 1.0 # Mzb0 = 1.0
    return Mzb_Zs - Mzb_R2sl

def func_precompute_R2sl(omega_y,PW,LUT_Gval,rB1):
    # compute target M_zs at the end of the pulse (using simplified eq. 9)
    rhs_func = func_GenBlochZs(omega_y*rB1, PW, LUT_Gval)
    sol = scipy.integrate.solve_ivp(rhs_func, t_span=(0, PW), y0=[1.0], method='RK23', rtol=1e-8, atol=1e-8)

    # root-finding with Brent's method on R2sl to meet Mzb_Zs
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq
    yData = sol.y[0][-1] # target Mzb_Zs 
    xData = (omega_y*rB1,PW)
    try:
        if numpy.all(omega_y == omega_y[0]): # BP
            print('... Rectangular pulse')
            R2sl = scipy.optimize.brentq(func_GenBlochBP_R2sl, 0, 1e6, (xData,yData),xtol=1e-4)
        else: # Hann
            print('... Shaped pulse')
            R2sl = scipy.optimize.brentq(func_GenBloch_R2sl, 0, 1e6, (xData,yData),xtol=1e-4)
        return R2sl
    except:
        return 0


###################################################################
############## Fitting-related functions
###################################################################        
def func_JSPqMT(xData,R1f,M0b):
    ### parse xData
    MTw_SAT_w1RMS   = xData[0]
    MTw_WbSAT       = xData[1]
    MTw_ROFA        = xData[3]
    VFA_ROFA        = xData[5+len(VFA_parx.ROFA)-1:5+2*len(VFA_parx.ROFA)-1]
    B0              = xData[-2]
    FLAG_useGBM     = xData[-1]

    if FLAG_useGBM:
        MTw_R2slRO  = xData[2]
        VFA_R2slRO  = xData[4:4+len(VFA_parx.ROFA)]
    else:
        MTw_WbRO    = xData[2]
        VFA_WbRO    = xData[4:4+len(VFA_parx.ROFA)]

    MTw_Ts          = MTw_parx.Ts
    MTw_Tm          = MTw_parx.Tm
    MTw_ROdur       = MTw_parx.ROdur
    MTw_Tr          = MTw_parx.TR - MTw_parx.ROdur - MTw_parx.Tm - MTw_parx.Ts
    VFA_Tr          = VFA_parx.TR - VFA_parx.ROdur
    VFA_ROdur       = VFA_parx.ROdur

    ### qMT parx
    R1b         = R1f
    M0f         = 1-M0b
    R           = qMTcontrainst_parx.R
    R2f         = 1/(qMTcontrainst_parx.R1fT2f/R1f)
    MTw_WfSAT   = ( (MTw_SAT_w1RMS/(2*numpy.pi*(MTw_parx.delta_f+B0)))**2 + 
                    (MTw_SAT_w1RMS/(2*numpy.pi*(-MTw_parx.delta_f+B0)))**2)/(2*(1/R2f)) # average
    
    ### build matrices
    if FLAG_useGBM:
        # common
        REX         = numpy.array([ [-R2f,  0,       0,             0,    0],
                                    [0,     -R2f,    0,             0,    0],
                                    [0,     0,       -(R1f+R*M0b),  0,    R*M0f], 
                                    [0,     0,       0,             0,    0],       
                                    [0,     0,       R*M0b,         0,    -(R1b+R*M0f)] ])
        C           = numpy.array([[0, 0, R1f*M0f, 0, R1b*M0b]]).T

        At_REX      = numpy.hstack( (REX, C) )
        At_REX      = numpy.vstack( (At_REX,numpy.zeros((1,6), dtype=float)) ) # 3 Mx/My/Mz Free Pool + 1 Mz Bound Pool + 1 additionnal (C)

        # MTw
        At_MTw_SAT  = numpy.diag([0, 0, -MTw_WfSAT, 0, -MTw_WbSAT])+REX
        At_MTw_SAT  = numpy.hstack( (At_MTw_SAT, C) )
        At_MTw_SAT  = numpy.vstack( (At_MTw_SAT,numpy.zeros((1,6), dtype=float)) )

        omega       = MTw_ROFA/MTw_ROdur # RO is equivalent to a BP in R2sl formalism
        At_MTw_RO   = numpy.array([ [0, 0,      0,     0,           0],
                                    [0, 0,      omega, 0,           0],
                                    [0, -omega, 0,     0,           0],
                                    [0, 0,      0,     -MTw_R2slRO, omega],
                                    [0, 0,      0,     -omega,      0] ])+REX
        At_MTw_RO   = numpy.hstack( (At_MTw_RO, C) )
        At_MTw_RO   = numpy.vstack( (At_MTw_RO,numpy.zeros((1,6), dtype=float)) )

        # VFA
        At_VFA_RO   = numpy.zeros((6,6,len(VFA_ROFA)))
        for ii in range(len(VFA_ROFA)):
            omega = VFA_ROFA[ii]/VFA_ROdur # RO is equivalent to a BP in R2sl formalism
            TMP                 = numpy.array([ [0, 0,      0,     0,               0],
                                                [0, 0,      omega, 0,               0],
                                                [0, -omega, 0,     0,               0],
                                                [0, 0,      0,     -VFA_R2slRO[ii], omega],
                                                [0, 0,      0,     -omega,          0] ])+REX
            TMP                 = numpy.hstack( (TMP, C) )
            At_VFA_RO[:,:,ii]   = numpy.vstack( (TMP,numpy.zeros((1,6), dtype=float)) )

        ### Xtilde spoil
        Xt_PHI_SPOIL = numpy.diag([0, 0, 1, 0, 1, 1])

    else: # standard "Graham"
        # common
        REX         = numpy.array([ [-R2f,    0,        0,               0],
                                    [0,       -R2f,     0,               0],
                                    [0,       0,        -(R1f+R*M0b),    R*M0f], 
                                    [0,       0,        R*M0b,           -(R1b+R*M0f)] ])
        C           = numpy.array([[0, 0, R1f*M0f, R1b*M0b]]).T

        At_REX      = numpy.hstack( (REX, C) )
        At_REX      = numpy.vstack( (At_REX,numpy.zeros((1,5), dtype=float)) ) # 3 Mx/My/Mz Free Pool + 1 Mz Bound Pool + 1 additionnal (C)

        # MTw
        At_MTw_SAT  = numpy.diag([0, 0, -MTw_WfSAT, -MTw_WbSAT])+REX
        At_MTw_SAT  = numpy.hstack( (At_MTw_SAT, C) )
        At_MTw_SAT  = numpy.vstack( (At_MTw_SAT,numpy.zeros((1,5), dtype=float)) )
        
        omega = MTw_ROFA/MTw_ROdur # assume rectangular pulse
        At_MTw_RO   = numpy.array([ [0, 0,      0,      0],
                                    [0, 0,      omega,  0],
                                    [0, -omega, 0,      0],
                                    [0, 0,      0,      -MTw_WbRO] ])+REX
        At_MTw_RO   = numpy.hstack( (At_MTw_RO, C) )
        At_MTw_RO   = numpy.vstack( (At_MTw_RO,numpy.zeros((1,5), dtype=float)) )

        # VFA
        At_VFA_RO   = numpy.zeros((5,5,len(VFA_ROFA)))
        for ii in range(len(VFA_ROFA)):
            omega = VFA_ROFA[ii]/VFA_ROdur # assume rectangular pulse
            TMP                 = numpy.array([ [0, 0,        0,     0],
                                                [0, 0,        omega, 0],
                                                [0, -omega,   0,     0],
                                                [0, 0,        0,     -VFA_WbRO[ii]] ])+REX
            TMP                 = numpy.hstack( (TMP, C) )
            At_VFA_RO[:,:,ii]   = numpy.vstack( (TMP,numpy.zeros((1,5), dtype=float)) )
        
        ### Xtilde spoil
        Xt_PHI_SPOIL = numpy.diag([0, 0, 1, 1, 1])    

    ### Xtilde operators: MTw
    Xt_MTw_RD   = scipy.linalg.expm(At_REX*MTw_Ts) # resting delay SAT pulse to RO pulse
    Xt_MTw_TR   = scipy.linalg.expm(At_REX*MTw_Tr) # resting delay RO pulse to TR
    Xt_MTw_SAT  = scipy.linalg.expm(At_MTw_SAT*MTw_Tm)
    Xt_MT0_RD   = scipy.linalg.expm(At_REX*(MTw_Tm+MTw_Ts+MTw_Tr))
    Xt_MTw_RO   = scipy.linalg.expm(At_MTw_RO*MTw_ROdur)

    ### Xtilde operators: VFA
    Xt_VFA_TR   = scipy.linalg.expm(At_REX*VFA_Tr) # resting delay RO pulse to TR
    Xt_VFA_RO   = numpy.zeros((At_REX.shape[0],At_REX.shape[1],len(VFA_ROFA)))
    for ii in range(len(VFA_ROFA)):
        Xt_VFA_RO[:,:,ii] = scipy.linalg.expm(At_VFA_RO[:,:,ii]*VFA_ROdur) # resting delay SAT pulse to RO pulse

    ### compute Mxys in steady-state
    # MT0
    w,v = numpy.linalg.eig(Xt_MT0_RD @ Xt_PHI_SPOIL @ Xt_MTw_RO)
    Mss = v[:, numpy.argmax(w)]
    Mss = Mss/Mss[-1]
    Mxy_MT0 = Mss[2]*numpy.sin(MTw_ROFA)

    # MTw
    w,v = numpy.linalg.eig(Xt_MTw_RD @ Xt_MTw_SAT @ Xt_MTw_TR @ Xt_PHI_SPOIL @ Xt_MTw_RO)
    Mss = v[:, numpy.argmax(w)]
    Mss = Mss/Mss[-1]
    Mxy_MTw = Mss[2]*numpy.sin(MTw_ROFA)

    # VFA
    Mxy_VFA = numpy.zeros((len(VFA_ROFA),))
    for ii in range(len(VFA_ROFA)):
        w,v = numpy.linalg.eig(Xt_VFA_TR @ Xt_PHI_SPOIL @ Xt_VFA_RO[:,:,ii])
        Mss = v[:, numpy.argmax(w)]
        Mss = Mss/Mss[-1]
        Mxy_VFA[ii] = Mss[2]*numpy.sin(VFA_ROFA[ii])

    return numpy.divide(numpy.concatenate((Mxy_VFA,[Mxy_MTw]),axis=0), Mxy_MT0, where=Mxy_MT0>0) # return Mxy/MT0 array
    
def fit_JSPqMT_lsq(xData,yData):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            popt, pcov = scipy.optimize.curve_fit(func_JSPqMT,xData,yData,
                                         p0=[1.0, 0.1], bounds=([0.05, 0],[3.0, 0.5]), 
                                         method='trf', maxfev=400,
                                         xtol=1e-6,ftol=1e-6)
            return popt
        except RuntimeError:
            return numpy.array([0, 0])
        except RuntimeWarning:
            return numpy.array([0, 0])
        except Exception:
            return numpy.array([0, 0])
    
#### main
if __name__ == "__main__":
    sys.exit(main()) 