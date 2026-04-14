# ///////////////////////////////////////////////////////////////////////////////////////////////
# // L. SOUSTELLE, PhD, Aix Marseille Univ, CNRS, CRMBM, Marseille, France
# // 2021/02/06
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
    ## parse arguments
    text_description = "Compute MT saturation map [1,2] from an MT-prepared SPGR experiment. Outputs are in percentage unit.\
                        \nReferences:\
                        \n\t [1] G. Helms et al., High-resolution maps of magnetization transfer with inherent correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI, MRM 2008;60:1396-1407 \
                        \n\t [2] G. Helms et al., Modeling the influence of TR and excitation flip angle on the magnetization transfer ratio (MTR) in human brain obtained from 3D spoiled gradient echo MRI. MRM 2010;64:177-185 \
                        " 
    parser = argparse.ArgumentParser(description=text_description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('MT',           help="Input 4D (MT0/MTw) NIfTI path")
    parser.add_argument('T1',           help="Input T1 (in sec) NIfTI path")
    parser.add_argument('MTsat',        help="Output MTsat NIfTI path")
    parser.add_argument('SEQparx',      nargs="?",help="Sequence parameters (comma-separated), in this order:   \n"
                                                                "\t 1) MT preparation module duration (ms) \n"
                                                                "\t 2) Sequence TR (ms) \n"
                                                                "\t 3) Readout flip angle (deg) \n"
                                                                "\t e.g. 10.0,43.0,10.0")
    parser.add_argument('--MTsatB1sq',  nargs="?",help="Output MTsat image normalized by squared B1 NIfTI path")
    parser.add_argument('--B1',         nargs="?",help="Input B1 map (in absolute unit) NIfTI path")
    parser.add_argument('--mask',       nargs="?",help="Input binary mask NIfTI path")
    parser.add_argument('--xtol',       nargs="?",type=float, default=1e-6, help="x tolerance for root finding (default: 1e-6)")
    parser.add_argument('--nworkers',   nargs="?",type=int, default=1, help="Use this for multi-threading computation (default: 1)")
    
    args                = parser.parse_args()
    MT_in_niipath       = args.MT
    MTsat_out_niipath   = args.MTsat 
    T1map_in_niipath    = args.T1
    B1_in_niipath       = args.B1
    mask_in_niipath     = args.mask
    xtolVal             = args.xtol
    NWORKERS            = args.nworkers if args.nworkers <= get_physCPU_number() else get_physCPU_number()

    #### Sequence parx 
    print('')
    print('--------------------------------------------------')
    print('----- Checking entries for MTsat processing ----')
    print('--------------------------------------------------')
    print('')        
    
    args.SEQparx = args.SEQparx.split(',')        
    if len(args.SEQparx) == 3: 
        SEQparx_NT                      = collections.namedtuple('SEQparx_NT','TR1 TR FA')
        SEQparx = SEQparx_NT(  TR1      = float(args.SEQparx[0])*1e-3, 
                               TR       = float(args.SEQparx[1])*1e-3,
                               FA       = float(args.SEQparx[2]))
        print('Summary of input sequence parameters:')
        print('\t MT preparation module duration: {:.1f} ms'.format(SEQparx.TR1*1e3))
        print('\t Sequence TR: {:.1f} ms'.format(SEQparx.TR*1e3))
        print('\t Readout flip angle: {:.1f} deg'.format(SEQparx.FA))
        print('')  
    else: 
        parser.error('Wrong amount of sequence parameters (SEQparx \
                             --- expected 3, found {})' .format(len(args.SEQparx)))
    
    # last check 
    for field in SEQparx._fields:
        if(getattr(SEQparx, field) < 0):
            parser.error('All SEQparx values should be positive')
    
    
    #### check input data
    # check MT data
    if not os.path.isfile(MT_in_niipath) or len(nibabel.load(MT_in_niipath).shape) != 4:
        parser.error('Volume {} does not exist or is not 4D' .format(MT_in_niipath))
    print('MT provided volume exist')
        
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
        
        
    # check mask
    if args.mask is None:
        print('No mask provided')
    elif args.mask is not None and not os.path.isfile(mask_in_niipath):
        parser.error('Mask map volume {} does not exist' .format(mask_in_niipath))
    else:
        print('Mask provided volume exist')
        

    #### load data
    # get MT data
    MT_data = nibabel.load(MT_in_niipath).get_fdata()

    # get B1 data
    if args.B1 is not None:
        B1_map = nibabel.load(B1_in_niipath).get_fdata()
    else:
        B1_map = numpy.ones(MT_data.shape[0:3])
        
    # get T1 data
    T1_map = nibabel.load(T1map_in_niipath).get_fdata()

    # get indices to process from mask
    if args.mask is not None:
        mask_data   = nibabel.load(mask_in_niipath).get_fdata()
    else:
        mask_data   = numpy.ones(MT_data.shape[0:3])
    mask_idx = numpy.asarray(numpy.where(mask_data == 1))
    

    ################ Estimation
    #### build xData
    T1_data     = T1_map[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    B1_data     = B1_map[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    
    cosFA_RO    = numpy.cos(SEQparx.FA * B1_data *numpy.pi/180)
    E1          = numpy.exp(numpy.divide(-SEQparx.TR1,T1_data, \
                                         out=numpy.ones(T1_data.shape, dtype=float), where=T1_data!=0))
    E2          = numpy.exp(numpy.divide(-(SEQparx.TR-SEQparx.TR1),T1_data, \
                                         out=numpy.ones(T1_data.shape, dtype=float), where=T1_data!=0))
    Mz_MT0      = numpy.divide(1-E1*E2,(1-E1*E2*cosFA_RO), \
                                         out=numpy.ones(T1_data.shape, dtype=float), where=T1_data!=0)    
    xData       = numpy.hstack((E1,E2,cosFA_RO,Mz_MT0))
    
    #### build yData
    MT0_data    = MT_data[:,:,:,0]
    MTw_data    = MT_data[:,:,:,1]
    MT0_ydata   = MT0_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T
    MTw_yData   = numpy.divide(MTw_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:].T,MT0_ydata, \
                                         out=numpy.ones(T1_data.shape, dtype=float), where=MT0_ydata!=0)
    
    #### iterable lists
    xtolVal     = [xtolVal] * xData.shape[0]
    MT_iterable = [*zip(xData,MTw_yData,xtolVal)]
        
    #### run
    print('')
    print('--------------------------------------------------')
    print('--------- Proceeding to MTsat estimation ---------')
    print('--------------------------------------------------')
    print('')
    
    start_time = time.time()
    with multiprocessing.Pool(NWORKERS) as pool:
        MTsat = pool.starmap(fit_MTsat_brentq,MT_iterable)
    delay = time.time()
    print("---- Done in {} seconds ----" .format(delay - start_time))
    MTsat = numpy.array(MTsat,dtype=float)
    
    
    ################ store & save NIfTI(s)
    ref_nii = nibabel.load(MT_in_niipath)
    MTsat_map = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    MTsat_map[mask_idx[0],mask_idx[1],mask_idx[2]] = MTsat*100
    # MTsat_map[(MTsat_map < 0) | (MTsat_map > 1000)] = 0
    new_img = nibabel.Nifti1Image(MTsat_map, ref_nii.affine, ref_nii.header)
    nibabel.save(new_img, MTsat_out_niipath)
    
    if args.MTsatB1sq is not None and args.B1 is not None:
        MTsatB1sq_map = numpy.full(ref_nii.shape[0:3],0,dtype=float)
        MTsatB1sq_map[mask_idx[0],mask_idx[1],mask_idx[2]] = MTsat*100
        MTsatB1sq_map = numpy.divide(MTsat_map, B1_map**2, out=numpy.zeros(MTsat_map.shape, dtype=float), where=B1_map!=0)
        new_img = nibabel.Nifti1Image(MTsatB1sq_map, ref_nii.affine, ref_nii.header)
        nibabel.save(new_img, args.MTsatB1sq)

   
###################################################################
############## Fitting-related functions
###################################################################
def func_MTsat_GRE_root(delta,xData,yData):
    E1,E2,cosFA_RO,Mz_MT0 = xData
    Mz_MTw = ((1-E1) + E1*(1-delta)*(1-E2)) / (1-E1*E2*cosFA_RO*(1-delta))
    return Mz_MTw/Mz_MT0 - yData

def fit_MTsat_brentq(xData,yData,xtolVal):
    # root-finding with Brent's method
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq
    try:
        x0 = scipy.optimize.brentq(func_MTsat_GRE_root, 0, 0.3,
                                    (xData,yData),xtol=xtolVal)
        return x0
    except:
        return 0
    
    
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
if __name__ == "__main__":
    sys.exit(main()) 

   
    
    
    
    
    
    
    
    
    
    
    
    
    