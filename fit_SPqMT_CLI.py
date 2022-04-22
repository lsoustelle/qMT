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
    global SEQparx; global qMTparx
    global NWORKERS;

    ## parse arguments
    text_description = "Fit a Macromolecular Proton Fraction (MPF) map from a Single-Point qMT protocol.\
                        \nNotes: \
                        \n\t1) The implemented saturation pulse is gaussian shaped with a user-defined FWHM.\
                        \n\tA Hann apodization is made possible, and a FWHM of 0.0 Hz yields a pure Hann-shaped pulse.\
                        \n\t  - Siemens' users: parameters are straightforwardly the same as in the Special Card interface from the greMT C2P sequence.\
                        \n\t  - Bruker's users: \"gauss\" pulse is a pure Gauss pulse with an FWHM of 218 Hz (differ from ParaVision's UI value).\
                        \n\t2) As in Yarnykh's original paper about SP-qMT [1], an assumption is made such that R1f = R1b = R1.\
                        \n\t3) The MT0 image to be provided can be computed from a 2-points VFA protocol [2], and synthetized via the synt_MT0_SPqMT.py script.\
                        \n\t4) B0 correction is not essential for SP-MPF mapping, but B1 is [3].\
                        \nReferences:\
                        \n\t [1] V. Yarnykh, Fast macromolecular proton fraction mapping from a single off-resonance magnetization transfer measurement, MRM 2012;68:166-178 \
                        \n\t [2] V. Yarnykh, Time-efficient, high-resolution, whole brain three-dimensional macromolecular proton fraction mapping, MRM 2016;75:2100-2106 \
                        \n\t [3] V. Yarnykh et al., Scan–Rescan Repeatability and Impact of B0 and B1 Field Nonuniformity Corrections in Single‐Point Whole‐Brain Macromolecular Proton Fraction Mapping, JMRI 2020;51:1789-1798 \
                        \n\t [4] L. Soustelle et al., Determination of optimal parameters for 3D single‐point macromolecular proton fraction mapping at 7T in healthy and demyelinated mouse brain, MRM 2021;85:369-379 \
                        " 
    parser = argparse.ArgumentParser(description=text_description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('MT',       nargs="+",help="Input couple MT0/MTw NIfTI path(s) (comma-separated for 3D, single path for 4D)")
    parser.add_argument('T1',       help="Input T1 NIfTI path")
    parser.add_argument('MPF',      help="Output MPF NIfTI path")
    parser.add_argument('SEQtimings', help="Sequence timings in ms (comma-separated), in this order:   \n"
                                                                "\t 1) Saturation pulse duration \n"
                                                                "\t 2) Interdelay between Saturation pulse and Readout pulse \n"
                                                                "\t 3) Readout pulse duration \n"
                                                                "\t 4) Sequence Time to Repetition (TR) \n"
                                                                "e.g. 12.0,2.0,0.2,30.0")
    parser.add_argument('SATROparx',  help="Saturation and Readout pulse parameters (comma-separated), in this order:   \n"
                                                                "\t 1) Saturation pulse Flip Angle (deg) \n"
                                                                "\t 2) Saturation pulse off-resonance frequency (Hz) \n"
                                                                "\t 3) Gaussian saturation pulse FWHM (Hz) \n"
                                                                "\t 4) Hann apodization (boolean; default: 1) \n"                                                               
                                                                "\t 5) Readout Flip Angle (deg) \n"
                                                                "e.g. 560.0,4000.0,200.0,1,10.0")
    parser.add_argument('--B1',        nargs="?",help="Input B1 map NIfTI path")
    parser.add_argument('--B0',        nargs="?",help="Input B0 map NIfTI path")
    parser.add_argument('--mask',      nargs="?",help="Input Mask binary NIfTI path")
    parser.add_argument('--nworkers',  nargs="?",type=int, default=1, help="Use this for multi-threading computation (default: 1)")
    parser.add_argument('--RecoTypePreset',  nargs="?",type=int, default=1,help="SP-qMT reconstruction type (integer): \n"  
                                                                "\t 1: Adult human brain 3T [1,2] (default) \n"
                                                                "\t 2: Adult mouse brain 7T [4]")
    parser.add_argument('--qMTconstraintParx',  help="Constained parameters for SP-qMT estimation (comma-separated; overrules --RecoTypePreset) in this order:\n"  
                                                                "\t 1) R1fT2f \n"
                                                                "\t 2) T2r (s) \n"
                                                                "\t 3) R (s-1) \n"
                                                                "e.g. 0.022,10.0e-6,19")
    args                = parser.parse_args()
    MT_in_niipaths      = [','.join(args.MT)] # ensure it's a comma-separated list
    MPF_out_niipaths    = args.MPF # ensure it's a comma-separated list
    T1map_in_niipath    = args.T1
    B1_in_niipath       = args.B1
    B0_in_niipath       = args.B0
    mask_in_niipath     = args.mask
    NWORKERS            = args.nworkers if args.nworkers <= get_physCPU_number() else get_physCPU_number()

    
    #### SP-qMT parx 
    print('')
    print('--------------------------------------------------')
    print('----- Checking entries for SP-qMT processing -----')
    print('--------------------------------------------------')
    print('')
    
    SEQparx_NT              = collections.namedtuple('SEQparx_NT','FAsat delta_f FWHM HannApo FAro Tm Ts Tp TR')
    args.SEQtimings         = args.SEQtimings.split(',')
    args.SATROparx          = args.SATROparx.split(',')
    if len(args.SEQtimings) != 4: 
        parser.error('Wrong amount of Sequence Parameters (SEQtimings \
                         --- expected 4, found {})' .format(len(args.SEQtimings)))
    if len(args.SATROparx) != 5: 
        parser.error('Wrong amount of Saturation/Readout Parameters (SATROparx \
                         --- expected 5, found {})' .format(len(args.SATROparx)))
    SEQparx = SEQparx_NT(FAsat      = float(args.SATROparx[0]), 
                         delta_f    = float(args.SATROparx[1]),
                         FWHM       = float(args.SATROparx[2]),
                         HannApo    = bool(int(args.SATROparx[3])),
                         FAro       = float(args.SATROparx[4]),
                         Tm         = float(args.SEQtimings[0])*1e-3,
                         Ts         = float(args.SEQtimings[1])*1e-3,
                         Tp         = float(args.SEQtimings[2])*1e-3,
                         TR         = float(args.SEQtimings[3])*1e-3)

    qMTparx_NT = collections.namedtuple('qMTparx_NT', 'R1fT2f T2r R')
    if args.qMTconstraintParx is None:
        if int(args.RecoTypePreset) == 1 and args.qMTconstraintParx is None: ## Yarkykh MRM 2012, 10.1002/mrm.23224
            RecoType = '3T adult human brain'
            qMTparx = qMTparx_NT(R1fT2f = 0.022,
                                 T2r    = 10.0e-6,
                                 R      = 19.0)
        elif int(args.RecoTypePreset) == 2 and args.qMTconstraintParx is None: ## Soustelle et al. MRM 2021, 10.1002/mrm.28397
            RecoType = '7T adult mouse brain'
            qMTparx = qMTparx_NT(R1fT2f = 0.0129, 
                                 T2r    = 9.1e-6,
                                 R      = 26.5)
        elif int(args.RecoTypePreset) != 1 or int(args.RecoTypePreset) != 2:
            parser.error('Cannot recognized --RecoType option value')
    else:
        args.qMTconstraintParx  = args.qMTconstraintParx.split(',')
        RecoType = 'User-defined'
        if len(args.qMTconstraintParx) != 3: 
            parser.error('Wrong amount of constraint qMT parameters (qMTconstraintParx \
                             --- expected 3, found {})' .format(len(args.qMTconstraintParx)))
        qMTparx = qMTparx_NT(   R1fT2f = float(args.qMTconstraintParx[0]), 
                                T2r    = float(args.qMTconstraintParx[1]),
                                R      = float(args.qMTconstraintParx[2]))
            
    print('Summary of input sequence parameters:')
    print('\t Saturation Flip Angle: {:.1f} deg'.format(SEQparx.FAsat))
    print('\t Saturation pulse off-resonance frenquency: {:.1f} Hz'.format(SEQparx.delta_f))
    print('\t Gaussian pulse FWHM: {:.1f} Hz'.format(SEQparx.FWHM))
    print('\t Hann apodization: {}'.format(SEQparx.HannApo))
    print('\t Readout Flip Angle: {:.1f} deg'.format(SEQparx.FAro))
    print('\t Saturation pulse duration: {:.2f} ms'.format(SEQparx.Tm*1e3))
    print('\t Interdelay Saturation pulse <--> Readout pulse: {:.2f} ms'.format(SEQparx.Ts*1e3))
    print('\t Readout pulse duration: {:.2f} ms'.format(SEQparx.Tp*1e3))
    print('\t Sequence Time to Repetition: {:.1f} ms'.format(SEQparx.TR*1e3))
    print('')
    print('Summary of constraint qMT parameters ({}):' .format(RecoType))
    print('\t R1fT2f: {:.4f}'.format(qMTparx.R1fT2f))
    print('\t T2r:\t {:.1f} us'.format(qMTparx.T2r*1e6))
    print('\t R:\t {:.1f} s-1'.format(qMTparx.R))
    print('')
    
    # last check before continuing
    for field in qMTparx._fields:
        if(getattr(qMTparx, field) < 0):
            parser.error('All qMTparx values should be positive')
    for field in SEQparx._fields:
        if(getattr(SEQparx, field) < 0):
            parser.error('All SEQparx values should be positive')
    
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
                parser.error('Volume {} does not exist' .format(vol_niipath))
        print('MT0/MTw provided volumes (3D) exist')
        
    # check T1 map
    if not os.path.isfile(T1map_in_niipath):
        parser.error('T1 map volume {} does not exist' .format(T1map_in_niipath))
    print('T1 map provided volume exist')
    
    # check B1 map
    if args.B1 is None:
        print('No B1 map provided (this is highly not recommended)')
    elif args.B1 is not None and not os.path.isfile(B1_in_niipath):
        parser.error('B1 map volume {} does not exist' .format(B1_in_niipath))
    else:
        print('B1 map provided volume exist')
        
    # check B0 map
    if args.B0 is None:
        print('No B0 map provided')
    elif args.B0 is not None and not os.path.isfile(B0_in_niipath):
        parser.error('B0 map volume {} does not exist' .format(B0_in_niipath))
    else:
        print('B0 map provided volume exist')
        
    # check mask
    if args.mask is None:
        print('No mask provided')
    elif args.mask is not None and not os.path.isfile(mask_in_niipath):
        parser.error('Mask map volume {} does not exist' .format(mask_in_niipath))
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
    
    # get T1 data
    T1_data = nibabel.load(T1map_in_niipath).get_fdata()

    # get indices to process from mask
    if args.mask is not None:
        mask_data   = nibabel.load(mask_in_niipath).get_fdata()
    else:
        mask_data   = numpy.ones(nibabel.load(MT_in_niipaths[0]).shape[0:3])
    mask_idx = numpy.asarray(numpy.where(mask_data == 1))

    
    #### build xData (prepare qMT parx)
    T1_data = T1_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    B1_data = B1_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    B0_data = B0_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    xData   = func_prepare_qMTparx(B1_data,T1_data,B0_data)
    
    ### build yData
    MT0_data = MT_data[0][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    MTw_data = MT_data[1][mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    yData    = MTw_data / MT0_data
    list_iterable = [*zip(xData,yData)]
    
    ### run
    print('')
    print('--------------------------------------------------')
    print('---------- Proceeding to MPF estimation ----------')
    print('--------------------------------------------------')
    print('')
    
    start_time = time.time()
    with multiprocessing.Pool(NWORKERS) as pool:
        res     = pool.starmap(fit_SPqMT_brentq,list_iterable)
    delay = time.time()
    print("---- Done in {} seconds ----" .format(delay - start_time))
    
    # store MPF & save nifti
    ref_nii = nibabel.load(MT_in_niipaths[0])
    MPF_map = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    MPF_map[mask_idx[0],mask_idx[1],mask_idx[2]] = res # get specific array elements from tuple in a tuple list
    MPF_map = MPF_map / (1 + MPF_map)
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
def func_prepare_qMTparx(B1_data,T1_data,B0_data):
    
    print('Preparing qMT quantities ...')
        
    # Pulse AI/PI & w1RMS nominal
    AI,PI       = func_AI_PI_ShapedGauss(SEQparx.Tm,SEQparx.FWHM,SEQparx.HannApo)
    B1peak_nom  = SEQparx.FAsat*(numpy.pi/180) / (gamma*AI*SEQparx.Tm)
    w1RMS_nom   = gamma*B1peak_nom * numpy.sqrt( PI )

    # G(delta_f)
    if any(B0_data != 0.0): # different G for voxels
        print('--- Computing G(delta_f) for all voxels (B0 corrected) ...')
        delta_f_corr    = SEQparx.delta_f + B0_data
        T2r             = numpy.full(B0_data.shape[0],qMTparx.T2r)[numpy.newaxis,:].T
        list_iterable   = numpy.hstack((T2r,delta_f_corr))
        start_time = time.time()
        with multiprocessing.Pool(NWORKERS) as pool:
            G_res  = pool.starmap(func_G,list_iterable)
        delay = time.time()
        print("--- ... Done in {} seconds" .format(delay - start_time))
        G  = numpy.array(G_res)[numpy.newaxis,:].T
    else: # same G for all voxels
        G = func_G(qMTparx.T2r,SEQparx.delta_f)
        G = numpy.tile(G,B1_data.shape[0])[numpy.newaxis,:].T
        
    
    ### Wb array
    Wb_array    =  (numpy.pi * w1RMS_nom**2 * G) * B1_data**2

    #### Wf array 
    R1_data     = numpy.reciprocal(T1_data)
    Wf_array    = (w1RMS_nom/(2*numpy.pi))**2 / qMTparx.R1fT2f / (SEQparx.delta_f+B0_data)**2 \
                    * numpy.multiply(R1_data,B1_data**2)
    
    ### FAro array
    FAro_array  = B1_data * SEQparx.FAro
    
    ### build xData
    R_array     = numpy.full(Wb_array.shape[0],qMTparx.R)[numpy.newaxis,:].T
    Ts_array    = numpy.full(Wb_array.shape[0],SEQparx.Ts)[numpy.newaxis,:].T
    Tm_array    = numpy.full(Wb_array.shape[0],SEQparx.Tm)[numpy.newaxis,:].T
    Tp_array    = numpy.full(Wb_array.shape[0],SEQparx.Tp)[numpy.newaxis,:].T
    Tr_array    = numpy.full(Wb_array.shape[0],SEQparx.TR-SEQparx.Ts-SEQparx.Tm-SEQparx.Tp)[numpy.newaxis,:].T
    
    xData = numpy.hstack(( Wb_array,Wf_array,FAro_array,R1_data,R_array, \
                              Ts_array,Tm_array,Tp_array,Tr_array ))
    
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
    
def func_G(T2r,delta_f): # super-lorentzian lineshape
    F = lambda x: numpy.sqrt(2/numpy.pi) * (T2r/numpy.abs(3*x**2-1)) * \
                    numpy.exp(-2*((2*numpy.pi *delta_f * T2r)/(3*x**2-1))**2)
    G = scipy.integrate.quad(F,0,1)
    
    return G[0]
 

###################################################################
############## Fitting-related functions
###################################################################        
def func_SPqMT_root(F,xData,yData):
    
    Wb,Wf,FAro,R1,R,Ts,Tm,Tp,Tr = xData
    R1r = R1f = R1
    f   = F/(1+F)
    
    # non-variable
    Rl  = numpy.array([[-R1f-R*F, R],[R*F, -R1r-R]])
    Meq = numpy.array([1-f, f])
    A   = R1f*R1r + R1f*R + R1r*R*F
    D   = A + (R1f+R*F)*Wb + (R1r+R)*Wf + Wb*Wf
    Es  = scipy.linalg.expm(Rl*Ts)
    Er  = scipy.linalg.expm(Rl*Tr)
    C   = numpy.diag([numpy.cos(FAro*numpy.pi/180),1.0])
    I   = numpy.eye(2)
    W   = numpy.array([[-Wf, 0],[0, -Wb]])
    
    # MTw
    Mss =  1/D*numpy.array([(1-f)*(A+R1f*Wb), f*(A+R1r*Wf)])   
    Em  =  scipy.linalg.expm( (Rl+W)*Tm )
    Mz  =  scipy.linalg.inv(I - Es @ Em @ Er @ C) @ \
            ( (Es @ Em @ (I-Er) + (I-Es)) @ Meq + Es @ (I-Em) @ Mss )
    
    # MT0
    MssN = 1/A * numpy.array([(1-f)*A, f*A])
    EmN  = scipy.linalg.expm( Rl*Tm )
    MzN  = scipy.linalg.inv(I - Es @ EmN @ Er @ C) @ \
            ( (Es @ EmN @ (I - Er) + (I-Es)) @ Meq + Es @ ( I-EmN ) @ MssN )
        
    return Mz[0]/MzN[0] - yData

    
def fit_SPqMT_brentq(xData,yData):
    # root-finding with Brent's method
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq
    try:
        x0 = scipy.optimize.brentq(func_SPqMT_root, 0, 0.3,
                                        (xData,yData),xtol=1e-5)
        return x0
    except:
        return 0
    
    
#### main
if __name__ == "__main__":
    sys.exit(main()) 

   
    
    
    
    
    
    
    
    
    
    
    
    
    