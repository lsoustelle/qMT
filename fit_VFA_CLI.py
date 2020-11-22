# ///////////////////////////////////////////////////////////////////////////////////////////////
# // L. SOUSTELLE, PhD, Aix Marseille Univ, CNRS, CRMBM, Marseille, France
# // Contact: lucas.soustelle@univ-amu.fr
# // Acknowledgement: J. LAMY, PhD, Université de Strasbourg, CNRS, ICube, Strasbourg, France
# ///////////////////////////////////////////////////////////////////////////////////////////////

import sys
import os
import numpy
import nibabel
import scipy.optimize
import scipy.linalg
import time
import multiprocessing
import argparse; from argparse import RawTextHelpFormatter
import json
import re
import subprocess

def main():  
    #### parse arguments
    text_description="Fit T1 from a Variable Flip Angle (VFA) protocol. \
        \nReturns maps of T1 (s) and S0 from: S=S0*(1-E1)*sin(FA)/(1-E1*cos(FA))\
        \nNotes: \
        \n   - If a series of 3D volumes is passed and no --FA or --TR arguments are passed, \
        \n     *.json paired files will be sought and parsed to get appropriate FA/TR values (specify otherwise);\
        \n   - If a 4D VFA volume is passed, --FA and --TR must be provided.\
        \nReferences:\
        \n\t [1] Chang et al., Linear least-squares method for unbiased estimation of T1 from SPGR signals, MRM 2008;60:496-501"
                     
    parser = argparse.ArgumentParser(description=text_description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('VFA',         nargs="+",help="Input VFA NIfTI path(s).\n"
                                                        "Comma-separated for 3D (same order as FA arguments if --FA is passed), "\
                                                        "single path for 4D.")
    parser.add_argument('T1',          help="Output T1 NIfTI path.")
    parser.add_argument('--TR',        type=float, help="Sequence Time to Repetition (ms).")
    parser.add_argument('--FA',        help="Comma-separated Flip Angles (degrees; e.g. 6,10,25).")
    parser.add_argument('--nworkers',  type=int, default=1, help="Use this for multi-threading computation (default: 1).")
    parser.add_argument('--B1',        nargs="?",help="Input B1 map NIfTI path.")
    parser.add_argument('--mask',      nargs="?",help="Input Mask binary NIfTI path.")
    parser.add_argument('--S0',        nargs="?", help="Output S0 NIfTI path.")
    parser.add_argument('--FitType',   nargs="?", default='NLS', help="Fitting type \n"
                                                                    "\t NLS: Nonlinear Least-Square (default)\n"
                                                                    "\t LLS: Linear Least-Square\n" 
                                                                    "\tNLS fit is more appropriate for noisy data [1], although slower.\n"
                                                                    "\tFor 2-points VFA, LLS will be used (faster and equivalent to NLS).")
    args = parser.parse_args()
    NWORKERS            = args.nworkers if args.nworkers <= get_physCPU_number() else get_physCPU_number()
    VFA_in_niipaths     = [','.join(args.VFA)] # ensure it's a comma-separated list
    T1map_out_niipath   = args.T1
    FA_inputs           = args.FA
    TR_input            = args.TR
    B1_in_niipath       = args.B1
    mask_in_niipath     = args.mask
    S0_out_niipath      = args.S0
    
    
    #### parse input paths, check existence and set flag
    print('')
    print('--------------------------------------------------')
    print('------- Checking entries for T1 estimation -------')
    print('--------------------------------------------------')
    print('')
    
    # check VFA
    if os.path.isfile(VFA_in_niipaths[0]) and len(nibabel.load(VFA_in_niipaths[0]).shape) == 4:
        FLAG_isVFA4D = 1
        print('VFA provided volume (4D) exist')
    else:
        FLAG_isVFA4D    = 0
        VFA_in_niipaths = VFA_in_niipaths[0].split(',')
        for vol_niipath in VFA_in_niipaths:
            if not os.path.isfile(vol_niipath):
                parser.error('Volume {} does not exist' .format(vol_niipath))
        print('VFA provided volumes (3D) exist')
    
    # get JSONs paths 
    meta_data_paths = list()
    for VFA_niipath in VFA_in_niipaths:
        meta_data_path = re.sub(r"\.nii(\.gz)?$", ".json", str(VFA_niipath))
        if not os.path.isfile(meta_data_path):
            meta_data_path = None
        meta_data_paths.append(meta_data_path)
       
    # parse FA from JSONs
    if args.FA is None and FLAG_isVFA4D == 0:
        FA_inputs   = list()
        print('No FA provided; seeking in JSONs ...')
        for meta_data_path in meta_data_paths:
            if meta_data_path is None or not os.path.isfile(meta_data_path):
                parser.error('Missing JSON file')
            FA_inputs.append(func_parse_JSON(meta_data_path,'FlipAngle'))
        if None in FA_inputs:
            parser.error('Missing FA field/value in one of the JSONs')
        elif len(FA_inputs) < len(VFA_in_niipaths):
            parser.error('Not enough FA values found (found: {}, expected: {})' \
                         .format(len(FA_inputs),len(VFA_in_niipaths)))
        print('... Found FA = {} deg' .format(FA_inputs))
    elif args.FA is None and FLAG_isVFA4D == 1: 
        parser.error('No FA provided and VFA volume is 4D; please provide FA values (see --FA option)')
    
    # parse TR from JSONs  
    if args.TR is None and FLAG_isVFA4D == 0:
        TR_input = None
        print('No TR provided; seeking in JSONs ...')
        for meta_data_path in meta_data_paths:
            if meta_data_path is None: continue
            else: 
                TR_input = func_parse_JSON(meta_data_path,'RepetitionTime') 
                if TR_input is None: continue
                else: break
        if TR_input is None:
            parser.error('No JSON file found, \
                     please provide TR value (see --TR option)')
        print('... Found TR = {} ms' .format(TR_input))    
    if args.TR is None and FLAG_isVFA4D == 1:
        parser.error('Inconsistency: VFA provided volume is 4D; please provide TR values (see --TR option)')
            
    # check B1
    if args.B1 is None:
        print('No B1 map provided (this is highly not recommended)')
    elif not os.path.isfile(B1_in_niipath):
        parser.error('B1 map volume {} does not exist' .format(B1_in_niipath))
    else:
        print('B1 map provided volume exist')
    
    # check mask
    if args.mask is None:
        print('No mask provided')
    elif not os.path.isfile(mask_in_niipath):
        parser.error('Mask map volume {} does not exist' .format(mask_in_niipath))
    else:
        print('Mask provided volume exist')
    
    #### settle TR & FAs parx
    global TR
    TR              = TR_input*1e-3
    if isinstance(FA_inputs, str):
        FA_array    = numpy.array([float(FA) for FA in FA_inputs.split(',')])[numpy.newaxis,:] *numpy.pi/180
    else:
        FA_array    = numpy.array(FA_inputs)[numpy.newaxis,:] *numpy.pi/180
          
    
    #### load data
    # get VFA data
    VFA_data    = list()
    if FLAG_isVFA4D: # case 4D
        VFA_nii = nibabel.load(VFA_in_niipaths[0]).get_fdata()
        for ii in range(VFA_nii.shape[3]):
            VFA_data.append(VFA_nii[:,:,:,ii])
    else:
        for ii in range(len(VFA_in_niipaths)):
            VFA_data.append(nibabel.load(VFA_in_niipaths[ii]).get_fdata())
    if len(VFA_data) == 2 and args.FitType == 'NLS':
        args.FitType == 'LLS'
        print('2-points VFA case: LLS fitting')
    
    # get B1 data
    if args.B1 is not None:
        B1_data = nibabel.load(B1_in_niipath).get_fdata()
    else:
        B1_data = numpy.ones(nibabel.load(VFA_in_niipaths[0]).shape[0:3])
            
    # get indices to process from mask
    if args.mask is not None:
        mask_data   = nibabel.load(mask_in_niipath).get_fdata()
    else:
        mask_data   = numpy.ones(nibabel.load(VFA_in_niipaths[0]).shape[0:3])
    mask_idx = numpy.asarray(numpy.where(mask_data == 1))
    
    
    #### build arrays and iterable for fitting & starmap()
    # build xdata array (experimental alpha, B1 corrected) 
    B1_data_extract = B1_data[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:]
    xdata_array    = numpy.squeeze(numpy.dot(FA_array.T,B1_data_extract).T)
    
    # build ydata array (experimental points) 
    ydata_array = list()
    for ii in range(len(VFA_data)):
        ydata_array.append(VFA_data[ii][mask_idx[0],mask_idx[1],mask_idx[2]])
    ydata_array = numpy.array(ydata_array).T
    
    # concatenate into list of tuples
    list_iterable   = [*zip(xdata_array,ydata_array)]


    #### compute T1 & S0 maps
    print('')
    print('-------------------------------------------')
    print('------- Proceeding to T1 estimation -------')
    print('-------------------------------------------')
    print('')
    
    ref_nii = nibabel.load(VFA_in_niipaths[0])
    if args.FitType == 'NLS':
        T1_map, S0_map = compute_T1_S0_map_NonLin(list_iterable,NWORKERS,mask_idx,ref_nii)
    elif args.FitType == 'LLS':    
        T1_map, S0_map = compute_T1_S0_map_Lin(list_iterable,NWORKERS,mask_idx,ref_nii)
    else:
       parser.error('Unknown --FitType input') 


    #### save NIfTI(s)
    # T1 map
    new_img = nibabel.Nifti1Image(T1_map, ref_nii.affine, ref_nii.header)
    nibabel.save(new_img, T1map_out_niipath)
    
    # S0 map
    if S0_out_niipath is not None:
        new_img = nibabel.Nifti1Image(S0_map, ref_nii.affine, ref_nii.header)
        nibabel.save(new_img, S0_out_niipath)


###################################################################
############## Parsing JSON & get CPU info
###################################################################
def func_parse_JSON(meta_data_path,key_JSON):
    meta_data = list()
    with open(meta_data_path) as fd:
        meta_data.append(json.load(fd)) 
        parx = meta_data[0].get(key_JSON)
        return parx[0]
    
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
############## Non-linear T1-VFA estimation
###################################################################    
def compute_T1_S0_map_NonLin(list_iterable,NWORKERS,mask_idx,ref_nii):
    start_time = time.time()
    with multiprocessing.Pool(NWORKERS) as pool:
        res     = pool.starmap(fit_VFA_NonLin,list_iterable)
    delay = time.time()
    print("---- Done in {} seconds ----" .format(delay - start_time))
    
    #### build & fill T1/S0 3D arrays
    T1_map  = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    T1_map[mask_idx[0],mask_idx[1],mask_idx[2]] = [a_tup[0][1] for a_tup in res] # get specific array elements from tuple in a tuple list
    S0_map  = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    S0_map[mask_idx[0],mask_idx[1],mask_idx[2]] = [a_tup[0][0] for a_tup in res]
    
    return T1_map,S0_map

def func_VFA_NonLin(alpha, S0, T1):
    return S0*(1-numpy.exp(-TR/T1))/(1-numpy.exp(-TR/T1)*numpy.cos(alpha))*numpy.sin(alpha)

def fit_VFA_NonLin(alpha,yData):
    try:
        popt = scipy.optimize.curve_fit(func_VFA_NonLin,
                                        alpha, 
                                        yData, 
                                        p0=[5*yData[0],1.0], 
                                        bounds=([0,0], [100*yData[0],10.0]), 
                                        maxfev=1000)
        return popt
    except RuntimeError:
        return (numpy.array([0,0]),numpy.array([0,0,0,0]))
    except ValueError:
        return (numpy.array([0,0]),numpy.array([0,0,0,0]))
    
    
###################################################################
############## Linear T1-VFA estimation
###################################################################
def compute_T1_S0_map_Lin(list_iterable,NWORKERS,mask_idx,ref_nii):
    start_time = time.time()
    with multiprocessing.Pool(NWORKERS) as pool:
        res     = pool.starmap(fit_VFA_Lin,list_iterable)
    delay = time.time()
    print("--- Done in {} seconds ---" .format(delay - start_time))
    
    #### build & fill T1/S0 3D arrays
    T1_map  = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    T1_map[mask_idx[0],mask_idx[1],mask_idx[2]] = [a_tup[1] for a_tup in res] # get specific array elements from tuple in a tuple list
    S0_map  = numpy.full(ref_nii.shape[0:3],0,dtype=float)
    S0_map[mask_idx[0],mask_idx[1],mask_idx[2]] = [a_tup[0] for a_tup in res]
    
    # delete outliers 
    idx_discard = numpy.hstack([numpy.asarray(numpy.where(T1_map < 0)), \
                                numpy.asarray(numpy.where(T1_map > 10)),\
                                numpy.asarray(numpy.where(S0_map < 0))])
    T1_map[idx_discard[0],idx_discard[1],idx_discard[2]] = 0
    S0_map[idx_discard[0],idx_discard[1],idx_discard[2]] = 0
    
    return T1_map,S0_map

def fit_VFA_Lin(alpha,yData):
    X = yData/numpy.tan(alpha)
    Y = yData/numpy.sin(alpha)
    try:        
        popt    = numpy.polyfit(X,Y,1)
        T1      = 1/(-numpy.log(popt[0])/TR)
        S0      = popt[1]/(1-popt[0])
        return (S0,T1)
    except:
        return (0,0)
    
    
###################################################################
############## Run main
###################################################################
if __name__ == "__main__":
    sys.exit(main()) 
    
