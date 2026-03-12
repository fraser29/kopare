# kopare
Mask and inversion of head MRI

Kōpare is Māori for "facial mask". 

## Main role

Mask external and internal air and invert ZTE MRI to generate pseudoCT DICOM images. 

## Installation

Clone this repository: 
```bash
git clone https://github.com/fraser29/kopare.git
```

Activate your virtual envirmonment and then install the project and requirements
```bash
cd kopare
pip install -e .
```

## Run 

To see help information run:
```bash
kopare -h
```

## Parameters file

The parameters file is a JSON file that contains the parameters for the kopare pipeline. It is located in the kopare directory and is called kopare_parameters.json.

Possible parameters:

- Median_filter_size: 
- Write_intermediate_files: bool - set true to write out intermediary working files 
- Output_format: format of the output files. Currently supported formats are: "mha", "vti", "nii", "nii.gz".
- Write_true_orientation_image_data: Only use for intermediate output files - will not write DICOMS
- Scale: Use with Write_intermediate_files to scale files (1=mm, 1000=m)

**TODO**
    "BC_Number_of_fitting_levels": bias correction number of fitting levels,
    "BC_Maximum_number_of_iterations": bias correction number of iterations,
    "BC_Shrink_factor": bias correction shrink factor,
    "Denoising_alpha": denoising (non-local means) alpha (0 = no denoising),
    "Denoising_patch_size": 9,
    "Denoising_patch_distance": 5,
    "n_shrink_wrap_iterations": number of shrink wrap iterations for face mask (keep below 8),
    "EdgeSmoothing_nIterations": 7 - not used currently

### Default parameters
    "Write_intermediate_files": true,
    "Write_true_orientation_image_data": true,
    "Scale": 1000, 
    "Median_filter_size": 5,
    "Output_format": "vti",
    "BC_Number_of_fitting_levels": 0,
    "BC_Maximum_number_of_iterations": 50,
    "BC_Shrink_factor": 3,
    "Denoising_alpha": 0,
    "Denoising_patch_size": 9,
    "Denoising_patch_distance": 5,
    "n_shrink_wrap_iterations": 5,
    "EdgeSmoothing_nIterations": 7

## Status

Beta. 

Current status is a basic implementation with simple image filtering and thresholding. 

*.MHA intermediary files are output - the full pipeline of producing masked, inverted, pseudoCT like DICOMS is not implemented. 

### Know issues: 

- [ ] 3D volume data output with incorrect format - probably spydcmtk issue. 


## Roadmap: 

- [ ] Bias correction
- [ ] Advanced masking
    - [ ] Per slice external air masking ? e.g. this may be an alternative to bias correction to accomodate inter-slice intensity variation
    - [ ] Denoising - e.g. non-local means (in sci-kit image)
- [ ] Inversion
- [ ] Boundary smoothing
- [ ] Write out DICOMS



