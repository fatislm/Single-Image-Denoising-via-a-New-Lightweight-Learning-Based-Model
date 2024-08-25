
![image](https://github.com/user-attachments/assets/bb61fe79-089c-4e7a-a601-57fd7b7a172d)



**ABSTRACT** Restoring a high-quality image from a noisy version poses a significant challenge in computer
vision, particularly in today’s context where high-resolution and large-sized images are prevalent. As such,
fast and efficient techniques are required to effectively address noise reduction in such images. Deep
CNN-based image-denoising algorithms have gained popularity due to the rapid growth of deep learning
and convolutional neural networks (CNNs). However, many existing deep learning models require paired
clean/noisy images for training, limiting their utility in real-world denoising scenarios. To address this
limitation, we propose a fast residual denoising framework (FRDF) designed based on zero-shot learning.
The FRDF first employs a novel downsampling technique to generate six different images from the noisy
input, which are then fed into a lightweight residual network with 23K parameters. The network effectively
utilizes a hybrid loss function, including residual, regularization, and guidance losses, to produce high-
quality denoised images. Our innovative downsampling approach leverages zero-shot learning principles,
enabling our framework to generalize to unseen noise types and adapt to diverse noise conditions without the
need for labelled data. Extensive experiments conducted on synthetic and real image confirm the superiority
of our proposed approach over existing dataset-free methods. Extensive experiments conducted on synthetic
and real images show that our method achieves up to 2 dB improvements in PSNR on the McMaster and
Kodak24 datasets. This renders our approach applicable in scenarios with limited data availability and
computational resources

# Image Denoising with Custom Neural Network

This project uses a custom neural network for image denoising, supporting both Gaussian and Poisson noise.

## Installation

1. Clone the repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
