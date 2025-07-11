# KADANet

## Dependencies

- **OS**: Ubuntu 22.04  
- **NVIDIA**:
  - CUDA: 12.2  
  - cuDNN: 9.1.0  
- **Python**: 3.8  
- **PyTorch**: >= 2.4.1  
- **Python packages**:
  - `numpy`
  - `opencv-python`
  - `lmdb`
  - `pyyaml`

## Dataset Preparation

We use DIV2K and Flickr2K as our training datasets (totally 3450 images).

To transform datasets to binary files for efficient IO, run:

```bash
python3 codes/scripts/create_lmdb.py



bash
python3 codes/scripts/generate_mod_blur_LR_bic.py
For evaluation of Anisotropic Gaussian kernels, we use DIV2KRK.
(Note: You need to modify the file paths manually.)
