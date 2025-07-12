import os
import torch
import torchvision.utils as tvutils


# ------------------------------------------------------
# -----------Reformulate degradation kernel-------------
#重新构造退化内核
def normkernel_to_downkernel(rescaled_blur_hr, rescaled_hr, ksize, eps=1e-10):
    blur_img = torch.rfft(rescaled_blur_hr, 3, onesided=False)
    img = torch.rfft(rescaled_hr, 3, onesided=False)

    denominator = img[:, :, :, :, 0] * img[:, :, :, :, 0] \
                  + img[:, :, :, :, 1] * img[:, :, :, :, 1] + eps

    # denominator[denominator==0] = eps

    inv_denominator = torch.zeros_like(img)
    inv_denominator[:, :, :, :, 0] = img[:, :, :, :, 0] / denominator
    inv_denominator[:, :, :, :, 1] = -img[:, :, :, :, 1] / denominator

    kernel = torch.zeros_like(blur_img).cuda()
    kernel[:, :, :, :, 0] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 0] \
                            - inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 1]
    kernel[:, :, :, :, 1] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 1] \
                            + inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 0]

    ker = convert_otf2psf(kernel, ksize)

    return ker


# ------------------------------------------------------
# -----------Constraint Least Square Filter-------------
#约束最小二乘滤波器
#使用两个核（kernel 和 grad_kernel）来对图像 img 进行去模糊处理
def get_uperleft_denominator(img, kernel, grad_kernel):
    ker_f = convert_psf2otf(kernel, img.size()) # discrete fourier transform of kernel
    # kernel的离散傅里叶变换

    ker_p = convert_psf2otf(grad_kernel, img.size()) # discrete fourier transform of kernel
    # grad_kernel的离散傅里叶变换

    denominator = inv_fft_kernel_est(ker_f, ker_p)


    numerator = torch.rfft(img, 3, onesided=False)
    # 将图像从空间域转换到频率域

    deblur = deconv(denominator, numerator)
    return deblur


# --------------------------------
# --------------------------------
#计算卷积核的伪逆，主要在频域中进行操作
def inv_fft_kernel_est(ker_f, ker_p):
    inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] \
                      + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] \
                      + ker_p[:, :, :, :, 0] * ker_p[:, :, :, :, 0] \
                      + ker_p[:, :, :, :, 1] * ker_p[:, :, :, :, 1]
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
    inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
    return inv_ker_f

# --------------------------------
# --------------------------------？？？？？？？？？？
#使用伪逆卷积核对输入图像进行去模糊。
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
                            - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
                            + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur = torch.irfft(deblur_f, 3, onesided=False)
    return deblur

# --------------------------------
# --------------------------------
# 旨在将一个点扩散函数（PSF）转换为其对应的光学传递函数（OTF），去模糊
#这是进行卷积计算的准备工作
def  convert_psf2otf(ker, size):
    # 创建一个与给定大小相同的零张量，并将其移动到CUDA设备（如果可用）
    psf = torch.zeros(size).cuda()

    # circularly shift
    centre = ker.shape[2]//2 + 1
    ## 首先，我们假设ker是一个四维张量（例如，在批量处理时），但实际上可能只需要处理最后一个两个维度（H, W）
    #计算中心索引

    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    #将ker的不同部分复制到psf中，但这并不是循环移位
    # 它实际上是在重新排列ker的元素，，但这种方式不会将PSF的中心移动到张量的中心

    # compute the otf
    otf = torch.rfft(psf, 3, onesided=False)
    #将这个重新排列的psf视为PSF，并计算其傅里叶变换
    return otf

# --------------------------------
# --------------------------------
#将光学传递函数 (OTF) 转换回点扩散函数 (PSF)。
def convert_otf2psf(otf, size):
    #循环位移
    ker = torch.zeros(size).cuda()
    psf = torch.irfft(otf, 3, onesided=False)

    # circularly shift
    ksize = size[-1]
    centre = ksize//2 + 1

    ker[:, :, (centre-1):, (centre-1):] = psf[:, :, :centre, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, (centre-1):, :(centre-1)] = psf[:, :, :centre, -(centre-1):]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), (centre-1):] = psf[:, :, -(centre-1):, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), :(centre-1)] = psf[:, :, -(centre-1):, -(centre-1):]#.mean(dim=1, keepdim=True)

    return ker

#将卷积核中可以忽略的值归零，以确保卷积核的稀疏性。
def zeroize_negligible_val(k, n=40):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    ## Sort K's values in order to find the n-th largest
    #“”“将相对于k中的值可忽略不计的值归零”“”
    #对K的值进行排序，以找到第n个最大的值
    pc = k.shape[-1]//2 + 1
    k_sorted, indices = torch.sort(k.flatten(start_dim=1))
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[:, -n - 1]
    # Clip values lower than the minimum value
    filtered_k = torch.clamp(k - k_n_min.view(-1, 1, 1, 1), min=0, max=1.0)
    filtered_k[:, :, pc, pc] += 1e-20
    # Normalize to sum to 1
    norm_k = filtered_k / torch.sum(filtered_k, dim=(2, 3), keepdim=True)
    return norm_k

def postprocess(*images, rgb_range):
    #对图像进行后处理，包括调整像素范围。
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]