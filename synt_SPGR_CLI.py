# ///////////////////////////////////////////////////////////////////////////////////////////////
# // L. SOUSTELLE, PhD, Aix Marseille Univ, CNRS, CRMBM, Marseille, France
# // Contact: lucas.soustelle@univ-amu.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////

import sys
import os
import numpy
import nibabel
import time
import argparse; from argparse import RawTextHelpFormatter

def main():  
    #### parse arguments
    text_description="Synthetizes an SPGR volume from an S0/T1 map couples at a specified TR/FA from: \
        \n\t S=S0*(1-E1)*sin(FA)/(1-E1*cos(FA))\
        \nNotes: \
        \n   - The synthesis of an SPGR volume can be used to generate an MT0 reference image in SP-qMT protocols [1]. \
        \nReferences:\
        \n\t [1] V. Yarnykh, Time-efficient, high-resolution, whole brain three-dimensional macromolecular proton fraction mapping, MRM 2016;75:2100-2106\
        "
                     
    parser = argparse.ArgumentParser(description=text_description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('S0',          help="Input S0 map NIfTI path.")
    parser.add_argument('T1',          help="Input T1 map NIfTI path (s).")
    parser.add_argument('TR',          type=float, help="Sequence Time to Repetition (ms).")
    parser.add_argument('FA',          type=float,help="Flip Angle (degrees).")
    parser.add_argument('SPGR',        help="Output synthetized SPGR NIfTI path.")
    parser.add_argument('--B1',        help="Input B1 map NIfTI path.")

    args = parser.parse_args()
    SPGR_out_niipath    = args.SPGR
    T1map_in_niipath    = args.T1
    S0map_in_niipath    = args.S0
    FA_input            = args.FA
    TR_input            = args.TR
    B1map_in_niipath    = args.B1
    
    
    #### parse input paths, check existence and set flag
    print('')
    print('--------------------------------------------------')
    print('------- Checking entries for SPGR synthesis ------')
    print('--------------------------------------------------')
    print('')
    
    # check T1/S0
    if os.path.isfile(S0map_in_niipath):
        print('S0 map volume exist')
    else:
        parser.error('S0 map volume {} does not exist' .format(S0map_in_niipath))
    if os.path.isfile(T1map_in_niipath):
        print('T1 map volume exist')
    else:
        parser.error('T1 map volume {} does not exist' .format(T1map_in_niipath))

    # check FA/TR
    if args.FA < 0:
        parser.error('FA should be positive')
    if args.TR < 0:
        parser.error('TR should be positive')
            
    # check B1
    if args.B1 is None:
        print('No B1 map provided (this is highly not recommended)')
    elif not os.path.isfile(B1map_in_niipath):
        parser.error('B1 map volume {} does not exist' .format(B1map_in_niipath))
    else:
        print('B1 map volume exist')
        
    
    #### load data
    # get S0/T1 data
    S0_img = nibabel.load(S0map_in_niipath).get_fdata()
    T1_img = nibabel.load(T1map_in_niipath).get_fdata()
    
    # get B1 data
    if args.B1 is not None:
        B1_img = nibabel.load(B1map_in_niipath).get_fdata()
    else:
        B1_img = numpy.ones(nibabel.load(S0map_in_niipath).shape[0:3])
            
        
    #### compute SPGR volume
    print('')
    print('-------------------------------------------')
    print('------- Proceeding to SPGR synthesis ------')
    print('-------------------------------------------')
    print('')
    mask_idx    = numpy.asarray(numpy.where(S0_img > 0))
    T1_data     = T1_img[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:]
    S0_data     = S0_img[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:]
    E1_data     = numpy.exp(-TR_input*1e-3/T1_data)
    FA_input    = FA_input*(numpy.pi/180) *B1_img[mask_idx[0],mask_idx[1],mask_idx[2]][numpy.newaxis,:]
    
    start_time = time.time()
    SPGR_data   = compute_SPGR(S0_data,E1_data,FA_input)
    SPGR_vol    = numpy.zeros(T1_img.shape)
    SPGR_vol[mask_idx[0],mask_idx[1],mask_idx[2]] = SPGR_data
    delay = time.time()
    print("--- Done in {} seconds ---" .format(delay - start_time))
    
    #### save NIfTI
    ref_nii = nibabel.load(T1map_in_niipath)
    new_img = nibabel.Nifti1Image(SPGR_vol, ref_nii.affine, ref_nii.header)
    nibabel.save(new_img, SPGR_out_niipath)
    


def compute_SPGR(S0,E1,FA):
    return S0*(1-E1)/(1-E1*numpy.cos(FA))*numpy.sin(FA)
    
     

    
    
###################################################################
############## Run main
###################################################################
if __name__ == "__main__":
    sys.exit(main()) 
    
