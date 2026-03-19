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

- "Write_intermediate_files": bool - set true to write out intermediary working files ,
- "Write_true_orientation_image_data": Useful for registration. Only use for intermediate output files - will not write DICOMS
- "Scale": Scale the input image data by this factor. Useful for converting from mm to m. [default=1000.0 ]
- "Median_filter_size": Size of the median filter to apply to the volume data. [default=0]
- "Output_format": format of the output files. Currently supported formats are: "mha", "vti", "nii", "nii.gz".
- "BC_Number_of_fitting_levels": Number of fitting levels for the bias field correction. [default=6]
- "BC_Maximum_number_of_iterations": Maximum number of iterations for the bias field correction. [default=150]
- "BC_Shrink_factor": Shrink factor for the bias field correction. [default=4]
- "Denoising_alpha": Alpha parameter for the non-local means denoising. [default=5]
- "Denoising_patch_size": Patch size for the non-local means denoising. [default=9]
- "Denoising_patch_distance": Patch distance for the non-local means denoising. [default=5]
- "Gaussian_smoothing_sigma": Sigma parameter for the Gaussian smoothing. [default=2.5]
- "Gaussian_smoothing_radius_factor": Radius factor for the Gaussian smoothing. [default=1.5]
- "EdgeSmoothing_nIterations": Number of iterations for the edge smoothing. [default=7]
- "n_shrink_wrap_iterations": Number of shrink wrap iterations for external air masking. [default=4], # NOTE: Capped at 5 internally
- "Sinus_detection_method": Sinus and airway detection method to employ (leave empty for none). [default=""],

## Status

Production, first release. 

## Roadmap: 

- [x] Bias correction
- [x] Advanced masking
    - [x] Denoising - e.g. non-local means (in sci-kit image)
- [x] Inversion
- [x] Boundary smoothing
- [x] Write out DICOMS



