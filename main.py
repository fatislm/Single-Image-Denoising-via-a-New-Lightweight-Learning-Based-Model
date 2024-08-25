import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import transform_image, add_noise, train, test, denoise
from models import Network

device = 'cuda' if torch.cuda.is_available() else 'cpu'
folder_path = "./data/"
n_chan = 3  # Assuming RGB images
noise_type = 'gauss'
noise_level = 25

# Initialize the model
model = Network(n_chan).to(device)
print("The number of parameters of the network is: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

PSNRs = []
losses = []

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)

        clean_img = transform_image(image_path)
        plt.imshow(clean_img.squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.show()

        noisy_img = add_noise(clean_img, noise_level, noise_type)

        clean_img = clean_img.to(device)
        noisy_img = noisy_img.to(device)

        max_epoch = 2000
        lr = 0.001
        step_size = 1500
        gamma = 0.5

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in tqdm(range(max_epoch)):
            loss = train(model, optimizer, noisy_img)
            scheduler.step()
            losses.append(loss)

        denoised_img = denoise(model, noisy_img)
        plt.imshow(denoised_img.cpu().squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.show()

        PSNR = test(model, noisy_img, clean_img)
        print(f"PSNR: {PSNR}")
        PSNRs.append(PSNR)

avg_psnr = sum(PSNRs) / len(PSNRs)
print(f"Average PSNR: {avg_psnr}")
