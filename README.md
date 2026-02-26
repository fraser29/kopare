# kopare
Mask and inversion of head MRI

Kōpare is Māori for "facial mask". 

## Main role

Mask external and internal air and invert ZTE MRI to generate pseudoCT images. 

## Installation

Clone this repository: 
```bash
git clone git@github.com:fraser29/kopare.git
```

Activate your virtual envirmonment and then install the project and requirements
```bash
cd kopare
pip install -e .
```



## Status

Beta. 

Current status is a basic implementation with simple image filtering and thresholding. 

## Roadmap: 

- [ ] Bias correction
- [ ] Advanced masking
    - [ ] Per slice external air masking
    - [ ] 
- [ ] Inversion
- [ ] Boundary smoothing



