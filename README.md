# Kernel-Aware Dual-Domain Adaptive Network: Enhancing Blind Super-Resolution Performance
The above content contains all the code details of the "Kernel-Aware Dual-Domain Adaptive Network: Enhancing Blind Super-Resolution Performance". This paper is currently submitted to The Vision Computer.

## Network Architecture

## Dependencies

- **OS**: Ubuntu 22.04  
- **NVIDIA**:
  - CUDA: 12.2  
  - cuDNN: 9.1.0  
- **Python**: 3.8  
- **PyTorch**: >= 2.4.1  

## Environmental installation
All experimental environment dependency packages in this paper are located in the 'requirements.txt' file, and the installation command is:
```
pip install -r requirements.txt
``` 


## Dataset Preparation

We use DIV2K and Flickr2K as our training datasets (totally 3450 images).  
``` 
python3 codes/scripts/create_lmdb.py
```
For evaluation of Isotropic Gaussian kernels (Gaussian8), we use four datasets, Set5, Set14, Urban100 and BSD100 .

To generate LRblur/LR/HR/Bicubic datasets paths, run:
``` 
python3 codes/scripts/generate_mod_blur_LR_bic.py
```
For evaluation of Anisotropic Gaussian kernels, we use DIV2KRK.

(You need to modify the file paths by yourself.)
## Train

1. The core algorithm is in `codes/config/KADANet`.

2. Please modify `codes/config/KADANet/options` to set path, iterations, and other parameters.

3. To train the model(s) in the paper, run below commands.

**For single GPU:**
``` 
cd codes/config/KADANet
python3 train.py -opt options/setting1/train_setting1_x4.yml
```
**For distributed training:**
```
cd codes/config/KADANet
python3 -m torch.distributed.launch --nproc_per_node=4 --master_poer=4321 train.py -opt=options/setting1/train_setting1_x4.yml --launcher pytorch
```
## Evaluation
To evalute our method, please modify the benchmark path and model path and run
```
cd codes/config/KADANet
python3 test.py -opt=options/setting1/test_setting1_x4.yml
```
