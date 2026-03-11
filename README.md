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

- Median_filter_size: size of the median filter to apply to the volume data.
- Output_format: format of the output files. Currently supported formats are: "mha", "vti", "nii", "nii.gz".

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



