## a very long test on image reconstruction (like the original paper)
import torch
import torchvision
import ssl
import os
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from itertools import chain
from tqdm import tqdm
# need this to download the dataset
ssl._create_default_https_context = ssl._create_unverified_context

from energy_transformer import EnergyTransformer

im2tensor = torchvision.transforms.ToTensor()

def collate(x, im2tensor = im2tensor):
    x = [im2tensor(x_i[0]) for x_i in x]
    return torch.stack(x, dim = 0)

def tensor2im(x):
    return torchvision.transforms.ToPILImage()(x)

def save_im(x, path):
    tensor2im(x).save(path)

def save_features(feature_list, patch_deembedder, depatcher, epoch = 0):
    """Function to save each step in the recurrent iteration of the energy transformer as an image, gives
    a sense of what is actually happening as the energy gradient is surfed."""
    for i, x in enumerate(feature_list):
        x = patch_deembedder(x)
        x = depatcher(x)
        save_im(x[0], f"tmp/epoch_{epoch}_feature_{i}.png")

if __name__ == "__main__":
    # making a tmp folder to store the images
    os.makedirs("tmp/", exist_ok = True)

    # data root
    data_root = "C:/Projects/" # you should only need to change this

    # data and patcher params
    h, w = 32, 32
    patch_size = 4
    patch_dim = patch_size**2 * 3
    n_patches = 64
    batch_size = 32

    # masking task params
    mask_fraction = 0.5
    mask_dropout = 0.1

    # energy transformer params
    token_dim = 256
    hopfield_hidden_dim = 1024
    n_heads = 8
    n_iters = 12 # number of  reccurent iterations of the energy transformer
    alpha = 0.1 # step size for energy gradient descent

    # training params
    n_epochs = 10
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cifar = torchvision.datasets.CIFAR100(root = data_root, train = True, download = True)
    dataloader = torch.utils.data.DataLoader(cifar, 
                                             batch_size = batch_size, 
                                             shuffle = True,
                                             collate_fn = collate)

    # these break a batch of images into patches and the converse, respectively
    patcher = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size).to(device)
    depatcher = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_size, p2 = patch_size, h = h // patch_size, w = w // patch_size).to(device)

    # feedforward layers to convert the last dim of the patched images to an embedding dimension
    patch_embedder = torch.nn.Sequential(
        torch.nn.Linear(patch_dim, token_dim),
        torch.nn.LayerNorm(token_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(token_dim, token_dim)).to(device)
    patch_deembedder = torch.nn.Sequential(
        torch.nn.Linear(token_dim, token_dim),
        torch.nn.LayerNorm(token_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(token_dim, patch_dim)).to(device)

    # token for masking
    mask_token = torch.randn(token_dim).to(device).requires_grad_(True)

    model = EnergyTransformer(token_dim, 
                              hopfield_hidden_dim, 
                              n_heads,
                              hopfield_type = "softmax",
                              context_length = n_patches,
                              n_iters_default = n_iters, 
                              alpha = alpha,
                              return_energy = True).to(device)

    # including the embedders, disable them to see how they affect learning (hint: the ET still learns, but slower)
    optimizer = torch.optim.Adam(chain(
                                       model.parameters(),
                                       patch_embedder.parameters(),
                                       patch_deembedder.parameters()
                                       ), 
                                    lr = lr)
    criterion = torch.nn.MSELoss()


    losses = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for i, x in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            x = x.to(device)
            x_orig = x.clone()

            x = patcher(x)
            x = patch_embedder(x)

            # only masked input is in the loss, but a small fraction are left unmasked
            # dropping a few masks is extremely helpful for training (see end of first paragraph in paper appendix A)
            mask_sample = torch.rand(x.shape[0], n_patches)
            mask = mask_sample < mask_fraction
            mask_dropped_out = mask_sample < mask_fraction * (1 - mask_dropout)

            x[mask_dropped_out] = mask_token

            x, features, energies = model(x)
            x = patch_deembedder(x)

            x = depatcher(x)
            # coerce mask shape for deprojection
            mask_vector = mask.unsqueeze(-1).repeat(1, 1, patch_dim)
            img_mask = depatcher(mask_vector)

            # only compute loss on the masked sections
            loss = criterion(x[img_mask], x_orig[img_mask])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if i % 100 == 0:
                save_im(x_orig[0], f"tmp/epoch_{epoch}_{i}_orig.png")
                save_im(x[0], f"tmp/epoch_{epoch}_{i}_recon.png")
                save_features(features, patch_deembedder, depatcher, epoch = epoch)