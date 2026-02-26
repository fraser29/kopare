# kopare
Mask and inversion of head MRI

Kōpare is Māori for "facial mask". 

## Main role

Mask external and internal air and invert ZTE MRI to generate pseudoCT images. 

## Installation

Clone this repository: 
```bash
git clone https://
```

Activate your virtual envirmonment and then install the project and requirements
```bash
cd kopare
pip install -e .
```

## Run 

```bash
kopare -h
```


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



