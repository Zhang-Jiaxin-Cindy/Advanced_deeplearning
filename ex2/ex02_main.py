import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, plot_beta_schedulers
from torchvision.utils import save_image, make_grid
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path, class_labels=None, guidance_weight=None):
    # TODO: Implement - adapt code and method signature as needed
    os.makedirs(store_path, exist_ok=True)
    model.eval()
    model.to(device)

    with torch.inference_mode():
        generated_images = diffusor.sample(
            model=model,
            image_size=diffusor.img_size,
            batch_size=n_images,
            channels=3,
            class_labels=class_labels,
            guidance_weight=guidance_weight
        ).to(device)
    if generated_images.ndim == 5:
        generated_images = generated_images[:, -1]
 
    generated_images = (generated_images.clamp(-1, 1) + 1) / 2

  
    grid = make_grid(generated_images, nrow=int(n_images**0.5), padding=2)
    save_image(grid, os.path.join(store_path, "generated_grid.png"))

    for i in range(n_images):
        save_image(
            generated_images[i], 
            os.path.join(store_path, f"sample_{i:03d}.png")
        )

    print(f"Images saved to {store_path}")


def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            t = torch.randint(0, args.timesteps, (len(images),), device=device).long()

            loss = diffusor.p_losses(model, images, t, class_labels=labels)
     
            total_loss += loss.item()
    avg_loss = total_loss / len(testloader)
    print(f"Test Loss: {avg_loss:.5f}")
    return avg_loss


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break





def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = "./img"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path)
    save_dir = os.path.join(os.path.expanduser("~"), "models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(save_dir, "ckpt.pt")
    )



if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)

    print("--- Visualizing Results ---")
    images_path = Path(__file__).parent / "images"
    files = list(images_path.glob("*.png"))
    if len(files) == 0:
        print(f"Warning: No images found in {images_path}. Skipping visualization.")
    else:
        n_images = 3
        sample_count = min(n_images, len(files))
        selected_files = np.random.choice(files, sample_count, replace=False) # type: ignore
        fig, axs = plt.subplots(1, sample_count, figsize=(15, 5))
        if sample_count == 1:
            axs = [axs]

        for i, file in enumerate(selected_files):
            img = Image.open(file)
            axs[i].imshow(img)
            axs[i].set_title(file.name) 
            axs[i].axis('off')         
        
        plt.tight_layout()
        plt.show() 

    try:
        plot_beta_schedulers(
            timesteps=args.timesteps,
            beta_start=0.0001,
            beta_end=0.02
        )
    except NameError:
        print("Warning: plot_beta_schedulers function is not defined.")
