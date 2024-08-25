import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import transform_image, add_noise, train, test, denoise
from models import Network

def main():
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define parameters
    input_image_path = "./data/example.png"  # Replace with the actual path to the input image
    n_chan = 3  # Number of channels, assuming RGB images
    noise_type = 'gauss'
    noise_level = 25

    # Initialize the model
    model = Network(n_chan).to(device)
    print("The number of parameters of the network is: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Transform and display the clean image
    clean_img = transform_image(input_image_path)
    plt.imshow(clean_img.squeeze().permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    # Add noise to the clean image
    noisy_img = add_noise(clean_img, noise_level, noise_type)

    # Move images to device (GPU or CPU)
    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)

    # Training parameters
    max_epoch = 2000
    lr = 0.001
    step_size = 1500
    gamma = 0.5

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train the model
    losses = []
    for epoch in tqdm(range(max_epoch)):
        loss = train(model, optimizer, noisy_img)
        scheduler.step()
        losses.append(loss)

    # Denoise the image using the trained model
    denoised_img = denoise(model, noisy_img)

    # Display the denoised image
    plt.imshow(denoised_img.cpu().squeeze().permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    # Calculate and print PSNR
    PSNR = test(model, noisy_img, clean_img)
    print(f"PSNR: {PSNR}")

if __name__ == "__main__":
    main()
