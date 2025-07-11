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

For evaluation of Isotropic Gaussian kernels (Gaussian8), we use five datasets, i.e., Set5, Set14, Urban100, BSD100 and Manga109.
