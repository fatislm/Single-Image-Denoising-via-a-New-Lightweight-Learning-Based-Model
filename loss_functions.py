import torch
import torch.nn as nn
from gaussian_smooth_layer import GaussianSmoothLayer

def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

def loss_func(noisy_img, model):
    # Gaussian Smoothing Layers
    BGBlur_kernel = [3, 9, 15]
    BlurWeight = [0.01, 0.1, 1.0]
    BlurNet = [GaussianSmoothLayer(3, k_size, 25).cuda() for k_size in BGBlur_kernel]

    # 1
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    # 2
    noisy11, noisy21 = pair_downsampler1(noisy_img)
    pred11 = noisy11 - model(noisy11)
    pred21 = noisy21 - model(noisy21)

    # 3
    noisy12, noisy22 = pair_downsampler2(noisy_img)
    pred12 = noisy12 - model(noisy12)
    pred22 = noisy22 - model(noisy22)

    loss_res = (1/6) * sum([
        (mse(noisy1, pred2) + mse(noisy2, pred1)) +
        (mse(noisy11, pred21) + mse(noisy21, pred11)) +
        (mse(noisy12, pred22) + mse(noisy22, pred12))
        for mse in [torch.nn.MSELoss()] * 6
    ])

    # 1
    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    denoised11, denoised21 = pair_downsampler1(noisy_denoised)
    denoised12, denoised22 = pair_downsampler2(noisy_denoised)

    loss_cons = (1/6) * (
        (mse(pred1, denoised1) + mse(pred2, denoised2)) +
        (mse(pred11, denoised11) + mse(pred21, denoised21)) +
        (mse(pred12, denoised12) + mse(pred22, denoised22))
    )

    loss = loss_res + loss_cons

    # Adding Gaussian Smoothing loss
    bgm_loss = 0
    for index, weight in enumerate(BlurWeight):
        out_b1 = BlurNet[index](noisy_img)
        out_real_b1 = BlurNet[index](noisy_img)
        grad_loss_b1 = mse(out_b1, out_real_b1)
        bgm_loss += weight * grad_loss_b1

    loss += bgm_loss

    return loss
