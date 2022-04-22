# ///////////////////////////////////////////////////////////////////////////////////////////////
# // L. SOUSTELLE, PhD, Aix Marseille Univ, CNRS, CRMBM, Marseille, France
# // Contact: lucas.soustelle@univ-amu.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////

import sys
import os
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

def main():
    global gamma; gamma = 267.513 * 1e6 # rad/s/T
    global VFA_parx; global MTw_parx; global qMTcontrainst_parx
    global NWORKERS;

    ## parse arguments
    text_description = "Fast quantitative MT joint fitting for R1f and MPF mapping from a VFA & {MTw;MT0} protocol.\
                        \nNotes: \
                        \n\t1) The model used for joint fitting is described in Ref. [1].\
                        \n\t2) The implemented saturation pulse is gaussian shaped with a user-defined FWHM.\
                        \n\tA Hann apodization is made possible, and setting the FWHM to 0.0 Hz yields a pure Hann-shaped pulse.\
                        \n\t  - Siemens' users: parameters are straightforwardly the same as in the Special Card interface from the greMT/vibeMT C2P sequences.\
                        \n\t  - Bruker's users: \"gauss\" pulse is a pure Gauss pulse with an FWHM of 218 Hz (differs from ParaVision's UI value).\
                        \n\t3) As in Yarnykh's original paper about SP-qMT [2], it is considered that R1f = R1b = R1.\
                        \n\t4) B1 correction is strongly advised [3].\
                        \n\t5) B0 correction considerations:\
                        \n\t  - Readout pulses: dB0 is assumed to have no effect regarding on-resonance saturation or resulting flip angle.\
                        \n\t  - Saturation pulses: the script is intended to have a dual-offset saturated (i.e. using a sine-modulated preparation pulse) MTw volume as input.\
                        \nReferences:\
                        \n\t [1] L. Soustelle et al., Quantitative Magnetization Transfer parametric mapping unbiased by on-resonance saturation and dipolar order contributions, ISMRM 2022 \
                        \n\t [2] V. Yarnykh, Fast macromolecular proton fraction mapping from a single off-resonance magnetization transfer measurement, MRM 2012;68:166-178 \
                        \n\t [3] V. Yarnykh et al., Scan–Rescan Repeatability and Impact of B0 and B1 Field Nonuniformity Corrections in Single‐Point Whole‐Brain Macromolecular Proton Fraction Mapping, JMRI 2020;51:1789-1798 \
                        " 
    parser = argparse.ArgumentParser(description=text_description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('MT',           nargs="+",help="Input couple MT0/MTw NIfTI path(s) (comma-separated for 3D, single path for 4D)")
    parser.add_argument('VFA',          nargs="+",help="Input VFA NIfTI path(s) (comma-separated for 3D, single path for 4D)")
    parser.add_argument('MPF',          help="Output MPF NIfTI path")
    parser.add_argument('R1f',          help="Output R1f NIfTI path")
    parser.add_argument('--MTw_TIMINGS', required=True, help="Sequence timings in ms (comma-separated), in this order:   \n"
                                                                "\t 1) Saturation pulse duration (ms) \n"
                                                                "\t 2) Interdelay between Saturation pulse and Readout pulse (ms) \n"
                                                                "\t 3) Readout pulse duration (ms) \n"
                                                                "\t 4) Sequence Time-to-Repetition (TR; ms) \n"
                                                                "e.g. --MTw_TIMINGS 12.0,2.1,1.0,30.0")
    parser.add_argument('--VFA_TIMINGS', required=True, help="Sequence timings in ms (comma-separated), in this order:   \n"
                                                                "\t 1) Readout pulse duration (ms) \n"
                                                                "\t 2) Sequence Time-to-Repetition (TR; ms) \n"
                                                                "e.g. --VFA_TIMINGS 1.0,30.0")
    parser.add_argument('--MTw_PARX', required=True,  help="Saturation parameters (comma-separated), in this order:   \n"
                                                                "\t 1) Saturation pulse flip angle (deg) \n"
                                                                "\t 2) Saturation pulse off-resonance frequency (Hz) \n"
                                                                "\t 3) Gaussian saturation pulse FWHM (Hz) \n"
                                                                "\t 4) Hann apodization (boolean; default: 1) \n" 
                                                                "\t 5) Readout flip angle of MTw/MT0 (deg; single common value) \n"
                                                                "\t 6) Readout pulse shape (Hann: 1, Rect.: 2; default: 1) \n" 
                                                                "e.g. --MTw_PARX 560,4000.0,100.0,1,10,1")
    parser.add_argument('--VFA_PARX', required=True,  help="Readout pulse parameters of experiments (comma-separated), in this order:   \n"
                                                                "\t 1) Readout flip angles [VFA1, VFA2, ..., VFAn] (deg; same order as in provided VFA volume(s)) \n"
                                                                "\t 2) Readout pulse shape (Hann: 1, Rect.: 2; default: 1) \n" 
                                                                "e.g. --VFA_PARX 6,10,25,1")
    parser.add_argument('--B1',                 nargs="?",help="Input normalized B1 map NIfTI path (strongly advised)")
    parser.add_argument('--B0',                 nargs="?",help="Input B0 map NIfTI path (in Hz; computation time is much longer)")
    parser.add_argument('--mask',               nargs="?",help="Input Mask binary NIfTI path")
    parser.add_argument('--nworkers',           nargs="?",type=int, default=1, help="Use this for multi-threading acceleration (default: 1)")
    parser.add_argument('--qMTconstraint_PARX',  help="Constained parameters for SP-qMT estimation (comma-separated) in this order:\n"  
                                                                "\t 1) R1fT2f \n"
                                                                "\t 2) T2r (s) \n"
                                                                "\t 3) R (s-1) \n"
                                                                "e.g. --qMTconstraint_PARX 0.022,10.0e-6,19")
    args                = parser.parse_args()
    MT_in_niipaths      = [','.join(args.MT)] # ensure it's a comma-separated list
    VFA_in_niipaths     = [','.join(args.VFA)] # ensure it's a comma-separated list
    MPF_out_niipaths    = args.MPF # ensure it's a comma-separated list
    R1f_out_niipaths    = args.R1f # ensure it's a comma-separated list
    B1_in_niipath       = args.B1
    B0_in_niipath       = args.B0
    mask_in_niipath     = args.mask
    NWORKERS            = args.nworkers if args.nworkers <= get_physCPU_number() else get_physCPU_number()
    print('Working with {} cores'.format(NWORKERS))
    
    #### Check inputs
    print('')
    print('--------------------------------------------------')
    print('---- Checking entries for JSP-qMT processing -----')
    print('--------------------------------------------------')
    print('')
    
    MTw_NT                  = collections.namedtuple('MTw_NT','FAsat delta_f FWHM bHannApo ROFA ROdur ROshape Tm Ts TR')
    args.MTw_TIMINGS        = args.MTw_TIMINGS.split(',')
    args.MTw_PARX           = args.MTw_PARX.split(',')
    if len(args.MTw_TIMINGS) != 4: 
        parser.error('Wrong amount of Sequence Parameters (--MTw_TIMINGS \
                         --- expected 4, found {})'.format(len(args.MTw_TIMINGS)))
    if len(args.MTw_PARX) != 6: 
        parser.error('Wrong amount of Saturation/Readout Parameters (--MTw_PARX \
                         --- expected 6, found {})'.format(len(args.MTw_PARX)))
    MTw_parx = MTw_NT(  FAsat      = float(args.MTw_PARX[0]), 
                        delta_f    = float(args.MTw_PARX[1]),
                        FWHM       = float(args.MTw_PARX[2]),
                        bHannApo   = bool(int(args.MTw_PARX[3])),
                        ROFA       = float(args.MTw_PARX[4]),
                        ROshape    = int(args.MTw_PARX[5]),
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
                        ROshape    = int(args.VFA_PARX[-1]),
                        ROdur      = float(args.VFA_TIMINGS[0])*1e-3, # convert to sec
                        TR         = float(args.VFA_TIMINGS[1])*1e-3)

    qMTcontraint_NT = collections.namedtuple('qMTcontraint_NT', 'R1fT2f T2r R')
    if args.qMTconstraint_PARX is not None:   
        args.qMTconstraint_PARX  = args.qMTconstraint_PARX.split(',')
        if len(args.qMTconstraint_PARX) != 3: 
            parser.error('Wrong amount of constraint qMT parameters (qMTconstraint_PARX \
                             --- expected 3, found {})'.format(len(args.qMTconstraint_PARX)))
        qMTcontrainst_parx = qMTcontraint_NT(   R1fT2f = float(args.qMTconstraint_PARX[0]), 
                                                T2r    = float(args.qMTconstraint_PARX[1]),
                                                R      = float(args.qMTconstraint_PARX[2]))         
    else:
        print('--qMTconstraint_PARX not set, setting to default values \n')
        qMTcontrainst_parx = qMTcontraint_NT(   R1fT2f = 0.0152, 
                                                T2r    = 10.1e-6,
                                                R      = 19.4)  
    print('Summary of input MTw/MT0 sequence parameters:')
    print('\t Saturation flip angle: {:.1f} deg'.format(MTw_parx.FAsat))
    print('\t Saturation pulse off-resonance frenquency: {:.1f} Hz'.format(MTw_parx.delta_f))
    print('\t Gaussian pulse FWHM: {:.1f} Hz'.format(MTw_parx.FWHM))
    print('\t Hann apodization: {}'.format(MTw_parx.bHannApo))
    print('\t Readout flip angle: {:.1f} deg'.format(MTw_parx.ROFA))
    print('\t Readout pulse duration: {:.1f} ms'.format(MTw_parx.ROdur*1e3))
    if MTw_parx.ROshape == 1:
        print('\t Readout pulse shape: Hann')
    else:
        print('\t Readout pulse shape: Rectangular')
    print('\t Saturation pulse duration: {:.1f} ms'.format(MTw_parx.Tm*1e3))
    print('\t Interdelay saturation pulse <--> Readout pulse: {:.2f} ms'.format(MTw_parx.Ts*1e3))
    print('\t Sequence Time-to-Repetition: {:.1f} ms'.format(MTw_parx.TR*1e3))
    print('')
    print('Summary of input VFA sequence parameters:')
    print('\t Readout flip angles: [' + ', '.join('{:.1f}'.format(v) for v in VFA_parx.ROFA) + '] deg')
    print('\t Readout pulse duration: {:.1f} ms'.format(VFA_parx.ROdur*1e3))
    if VFA_parx.ROshape == 1:
        print('\t Readout pulse shape: Hann')
    else:
        print('\t Readout pulse shape: Rectangular')
    print('\t Sequence Time-to-Repetition: {:.1f} ms'.format(VFA_parx.TR*1e3))
    print('')
    print('Summary of constraint qMT parameters:')
    print('\t R1fT2f: {:.4f}'.format(qMTcontrainst_parx.R1fT2f))
    print('\t T2r:\t {:.1f} us'.format(qMTcontrainst_parx.T2r*1e6))
    print('\t R:\t {:.1f} s-1'.format(qMTcontrainst_parx.R))
    print('')
    
    # last check before continuing
    for field in qMTcontrainst_parx._fields:
        if(getattr(qMTcontrainst_parx, field) < 0):
            parser.error('All qMTcontrainst_parx values should be positive')
    for field in VFA_parx._fields:
        if isinstance(getattr(VFA_parx, field), numpy.ndarray): # special treatment for first element (numpy array)
            if any(X < 0 for X in getattr(VFA_parx, field)):
                parser.error('All VFA parameter values should be positive')
        elif (getattr(VFA_parx, field) < 0):
            parser.error('All VFA parameter values should be positive')
    for field in MTw_parx._fields:
        if(getattr(MTw_parx, field) < 0):
            parser.error('All MTw parameter values should be positive')
    if MTw_parx.ROshape < 1 or MTw_parx.ROshape > 2:
        parser.error('Unrecognized MTw Readout pulse shape (should be 1 or 2)')
    if VFA_parx.ROshape < 1 or VFA_parx.ROshape > 2:
        parser.error('Unrecognized VFA Readout pulse shape (should be 1 or 2)')

    #### check input data
    # check MT0/MTw data
    if os.path.isfile(MT_in_niipaths[0]) and len(nibabel.load(MT_in_niipaths[0]).shape) == 4:
        FLAG_isqMT4D = 1
        print('MT0/MTw provided volume (4D) exist')
    else:
        FLAG_isqMT4D    = 0
        MT_in_niipaths = MT_in_niipaths[0].split(',')
        for vol_niipath in MT_in_niipaths:
            if not os.path.isfile(vol_niipath):
                parser.error('Volume {} does not exist'.format(vol_niipath))
        print('MT0/MTw provided volumes (3D) exist')
        
    # check T1 map
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
    xData   = func_prepare_qMTparx(B1_data,B0_data)
    
    #### build yData
    MT0_data = MT_data[0][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    MTw_data = MT_data[1][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    yData    = numpy.zeros((MTw_data.shape[0],len(VFA_data)))
    for ii in range(len(VFA_data)):
        yData[:,ii] = VFA_data[ii][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T.ravel()
    yData   = numpy.concatenate((yData,MTw_data),axis=1) / MT0_data
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
    print("---- Done in {} seconds ----".format(delay - start_time))
    
    # store MPF & R1f + save nifti
    ref_nii = nibabel.load(MT_in_niipaths[0])

    R1f_map = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    R1f_map[mask_idx[0],mask_idx[1],mask_idx[2]] = [a_tup[0] for a_tup in res] # get specific array elements from tuple in a tuple list
    new_img = nibabel.Nifti1Image(R1f_map, ref_nii.affine, ref_nii.header)
    nibabel.save(new_img, R1f_out_niipaths)

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
def func_prepare_qMTparx(B1_data,B0_data):
    
    print('Preparing qMT quantities ...')
    ### VFA
    if VFA_parx.ROshape == 1:
        VFA_RO_AI,VFA_RO_PI   = 0.5,0.375 # Hann-shaped
    elif VFA_parx.ROshape == 2:
        VFA_RO_AI,VFA_RO_PI   = 1.0,1.0 # Rectangular-shaped
    VFA_RO_B1peak_nom  = VFA_parx.ROFA*(numpy.pi/180) / (gamma*VFA_RO_AI*VFA_parx.ROdur)
    VFA_RO_w1RMS_nom   = gamma*VFA_RO_B1peak_nom*numpy.sqrt(VFA_RO_PI)    
    VFA_RO_G           = func_computeG_SphericalLineshape(qMTcontrainst_parx.T2r,0) # assume no dB0 impact
    VFA_RO_G           = numpy.tile(VFA_RO_G,B1_data.shape[0])[numpy.newaxis,:].T


    ### MTw/MT0 
    # SAT Pulse AI/PI & w1RMS nominal
    MTw_SAT_AI,MTw_SAT_PI   = func_AI_PI_ShapedGauss(MTw_parx.Tm,MTw_parx.FWHM,MTw_parx.bHannApo)
    MTw_SAT_B1peak_nom      = MTw_parx.FAsat*(numpy.pi/180) / (gamma*MTw_SAT_AI*MTw_parx.Tm)
    MTw_SAT_w1RMS_nom       = gamma*MTw_SAT_B1peak_nom*numpy.sqrt(MTw_SAT_PI)
    if any(B0_data != 0.0): # compute G for each voxel
        print('--- Computing G(delta_f) for all voxels (B0 corrected) ...')
        T2r_array       = numpy.full(B0_data.shape[0],qMTcontrainst_parx.T2r)[numpy.newaxis,:].T
        delta_f_array   = numpy.full(B0_data.shape[0],MTw_parx.delta_f)[numpy.newaxis,:].T
        list_iterable   = numpy.hstack((T2r_array,delta_f_corr,B0_data))
        start_time = time.time()
        with multiprocessing.Pool(NWORKERS) as pool:
            MTw_SAT_G  = pool.starmap(func_computeG_SuperLorentzian,list_iterable)
        delay = time.time()
        print("--- ... Done in {} seconds".format(delay - start_time))
        MTw_SAT_G  = numpy.array(MTw_SAT_G)[numpy.newaxis,:].T
    else: # same G for all voxels
        MTw_SAT_G = func_computeG_SuperLorentzian(qMTcontrainst_parx.T2r,MTw_parx.delta_f,0)
        MTw_SAT_G = numpy.tile(MTw_SAT_G,B1_data.shape[0])[numpy.newaxis,:].T

    # RO Pulse AI/PI & w1RMS nominal
    if MTw_parx.ROshape == 1:
        MTw_RO_AI,MTw_RO_PI   = 0.5,0.375 # Hann-shaped
    elif MTw_parx.ROshape == 2:
        MTw_RO_AI,MTw_RO_PI   = 1.0,1.0 # Rectangular-shaped
    MTw_RO_B1peak_nom  = MTw_parx.ROFA*(numpy.pi/180) / (gamma*MTw_RO_AI*MTw_parx.ROdur)
    MTw_RO_w1RMS_nom   = gamma*MTw_RO_B1peak_nom*numpy.sqrt(MTw_RO_PI)   
    MTw_RO_G           = func_computeG_SphericalLineshape(qMTcontrainst_parx.T2r,0.0) # assume no dB0 impact
    MTw_RO_G           = numpy.tile(MTw_RO_G,B1_data.shape[0])[numpy.newaxis,:].T    


    ### Wb/FA arrays
    MTw_SAT_w1RMS_array = MTw_SAT_w1RMS_nom * B1_data
    MTw_WbSAT_array     = (numpy.pi * MTw_SAT_w1RMS_nom**2  * MTw_SAT_G) * B1_data**2
    MTw_WbRO_array      = (numpy.pi * MTw_RO_w1RMS_nom**2   * MTw_RO_G)  * B1_data**2
    VFA_WbRO_array      = (numpy.pi * VFA_RO_w1RMS_nom**2   * VFA_RO_G)  * B1_data**2
    MTw_ROFA_array      = MTw_parx.ROFA*numpy.pi/180 * B1_data
    VFA_ROFA_array      = VFA_parx.ROFA*numpy.pi/180 * B1_data

    ### build xData
    MTw_SAT_w1RMS   = numpy.full(MTw_WbSAT_array.shape[0],qMTcontrainst_parx.R)[numpy.newaxis,:].T

    xData           = numpy.hstack((MTw_SAT_w1RMS_array,MTw_WbSAT_array,MTw_WbRO_array,MTw_ROFA_array, \
                                    VFA_WbRO_array,VFA_ROFA_array, \
                                    B0_data))
    print('... Done')
    return xData
    

def func_AI_PI_ShapedGauss(tau,BW,boolHannApo):
    if boolHannApo and BW != 0: # Gauss-Hann
        sigma      = numpy.sqrt(2*numpy.log(2) / (numpy.pi*BW)**2)
        satPulse   = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) * 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )
        satPulseSq = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) * 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )**2
    if boolHannApo and BW == 0: # pure Hann
        satPulse   = lambda t: ( 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )
        satPulseSq = lambda t: ( 0.5*(1 - numpy.cos((2*numpy.pi*t)/tau)) )**2
    else: # pure Gauss
        sigma      = numpy.sqrt(2*numpy.log(2) / (numpy.pi*BW)**2)
        satPulse   = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) )
        satPulseSq = lambda t: ( numpy.exp(-((t-(tau/2))**2)/(2*sigma**2)) )**2
    integ   = scipy.integrate.quad(satPulse,0,tau)
    integSq = scipy.integrate.quad(satPulseSq,0,tau)
    
    return integ[0]/tau, integSq[0]/tau ## normalized AI, PI
    
def func_computeG_SuperLorentzian(T2r,delta_f,dB0): # super-lorentzian lineshape
    if dB0 == 0.0:
        F = lambda x: (T2r/numpy.abs(3*x**2-1)) * \
                        numpy.exp(-2*((2*numpy.pi * delta_f * T2r)/(3*x**2-1))**2)
        INTEG_RES = scipy.integrate.quad(F,0,1)
        return numpy.sqrt(2/numpy.pi)*INTEG_RES[0]
    else:
        Fp = lambda x: (T2r/numpy.abs(3*x**2-1)) * \
                        numpy.exp(-2*((2*numpy.pi * (delta_f+dB0) * T2r)/(3*x**2-1))**2)
        Fm = lambda x: (T2r/numpy.abs(3*x**2-1)) * \
                        numpy.exp(-2*((2*numpy.pi * (-delta_f+dB0) * T2r)/(3*x**2-1))**2)
        INTEG_RESp = scipy.integrate.quad(Fp,0,1)
        INTEG_RESm = scipy.integrate.quad(Fm,0,1)
        return numpy.sqrt(2/numpy.pi)*(INTEG_RESp[0]+INTEG_RESm[0])/2 # average of G(df+dB0) & G(-df+dB0)
    

def func_computeG_SphericalLineshape(T2r,delta_f):
    # Function for Spherical lineshape integration
    # include neighboors contribution to remove singularity at the magic angle
    # see Pampel et al. NeuroImage 114 (2015) 136–146
    T2neighboors    = 1/31.4  # T2neighboors -> +Inf virtually mean no effect of T2neighboors
    def SphericalLineshape(theta):
        T2rSph  = 2*T2r/numpy.abs(3*numpy.cos(theta)**2-1)
        T2eff   = 1/numpy.sqrt(1/(T2rSph)**2+1/(T2neighboors)**2)
        return numpy.sin(theta)*T2eff*numpy.exp(-1/2*(2*numpy.pi*delta_f*T2eff)**2)
    INTEG_RES = scipy.integrate.quad(SphericalLineshape,0,numpy.pi/2)
    return numpy.sqrt(1./(2*numpy.pi))*INTEG_RES[0] 


###################################################################
############## Fitting-related functions
###################################################################        
def func_JSPqMT(xData,R1f,M0b):

    ### parse xData
    MTw_SAT_w1RMS   = xData[0]
    MTw_WbSAT       = xData[1]
    MTw_WbRO        = xData[2]
    MTw_ROFA        = xData[3]
    VFA_WbRO        = xData[4:4+len(VFA_parx.ROFA)]
    VFA_ROFA        = xData[5+len(VFA_parx.ROFA)-1:5+2*len(VFA_parx.ROFA)-1]
    B0              = xData[-1]


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
    T2f         = qMTcontrainst_parx.R1fT2f/R1f
    MTw_WfSAT   = ( (MTw_SAT_w1RMS/(2*numpy.pi*(MTw_parx.delta_f+B0)))**2 + 
                    (MTw_SAT_w1RMS/(2*numpy.pi*(-MTw_parx.delta_f+B0)))**2)/(2*T2f) # average

    ### build matrices
    REX         = numpy.array([ [-1/T2f,    0,         0,               0],
                                [0,         -1/T2f,    0,               0],
                                [0,         0,         -(R1f+R*M0b),    R*M0f], 
                                [0,         0,         R*M0b,           -(R1b+R*M0f)] ])
    C           = numpy.array([[0, 0, R1f*M0f, R1b*M0b]]).T

    REX_At      = numpy.hstack( (REX, C) )
    REX_At      = numpy.vstack( (REX_At,numpy.zeros((1,5), dtype=float)) ) # 3 Mx/My/Mz Free Pool + 1 Mz Bound Pool + 1 additionnal (C)

    MTw_SAT_At  = numpy.diag([0, 0, -MTw_WfSAT, -MTw_WbSAT])+REX
    MTw_SAT_At  = numpy.hstack( (MTw_SAT_At, C) )
    MTw_SAT_At  = numpy.vstack( (MTw_SAT_At,numpy.zeros((1,5), dtype=float)) )

    MTw_RO_At   = numpy.array([ [0, 0,                      0,                  0],
                                [0, 0,                      MTw_ROFA/MTw_ROdur, 0],
                                [0, -MTw_ROFA/MTw_ROdur,    0,                  0],
                                [0, 0,                      0,                  -MTw_WbRO] ])+REX
    MTw_RO_At   = numpy.hstack( (MTw_RO_At, C) )
    MTw_RO_At   = numpy.vstack( (MTw_RO_At,numpy.zeros((1,5), dtype=float)) )


    ### Xtilde operators: MTw
    Xt_MTw_RD   = scipy.linalg.expm(REX_At*MTw_Ts) # resting delay SAT pulse to RO pulse
    Xt_MTw_TR   = scipy.linalg.expm(REX_At*MTw_Tr) # resting delay RO pulse to TR
    Xt_MTw_SAT  = scipy.linalg.expm(MTw_SAT_At*MTw_Tm)
    Xt_MT0_RD   = scipy.linalg.expm(REX_At*(MTw_Tm+MTw_Ts+MTw_Tr))
    Xt_MTw_RO   = scipy.linalg.expm(MTw_RO_At*MTw_ROdur)

    ### Xtilde operators: VFA
    Xt_VFA_TR   = scipy.linalg.expm(REX_At*VFA_Tr) # resting delay RO pulse to TR
    Xt_VFA_RO   = numpy.zeros((5,5,len(VFA_ROFA)))
    for ii in range(len(VFA_ROFA)):
        TMP     = numpy.array([ [0, 0,                      0,                      0],
                                [0, 0,                      VFA_ROFA[ii]/VFA_ROdur, 0],
                                [0, -VFA_ROFA[ii]/VFA_ROdur,0,                      0],
                                [0, 0,                      0,                      -VFA_WbRO[ii]] ])+REX
        TMP     = numpy.hstack( (TMP, C) )
        TMP     = numpy.vstack( (TMP,numpy.zeros((1,5), dtype=float)) )
        numpy.set_printoptions(precision=4)
        Xt_VFA_RO[:,:,ii] = scipy.linalg.expm(TMP*VFA_ROdur) # resting delay SAT pulse to RO pulse

    ### Xtilde spoil
    Xt_PHI_SPOIL = numpy.diag([0, 0, 1, 1, 1])


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

    return numpy.concatenate((Mxy_VFA,[Mxy_MTw]),axis=0)/Mxy_MT0 # return Mxy/MT0 array
    
def fit_JSPqMT_lsq(xData,yData):
    try:
        popt, pcov = scipy.optimize.curve_fit(func_JSPqMT,xData,yData,
                                     p0=[1.0, 0.1], bounds=([0.05, 0],[3.0, 0.5]), 
                                     method='trf', maxfev=400,
                                     xtol=1e-6,ftol=1e-6)
        return popt
    except:
        return 0
    
    
#### main
if __name__ == "__main__":
    sys.exit(main()) 