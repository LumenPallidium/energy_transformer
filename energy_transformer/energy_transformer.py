import torch

# See https://ml-jku.github.io/hopfield-layers/, https://openreview.net/pdf?id=4nrZXPFN1c4 for mathematical reference
# Implementation based on jax here: https://github.com/bhoov/energy-transformer-jax

def value_and_grad(f, x):
    """Compute value and gradient of f at x, typically for energy functions. It uses jacobian
    because we are trying to get the gradient of the energy with respect to the
    input, and autograd.grad doesn't work as well with batched inputs."""
    y = x.detach().clone().requires_grad_(False)
    y = torch.func.vmap(f)(y)

    grads = torch.func.vmap(torch.func.jacrev(f))(x)
    return y, grads

class EnergyMHA(torch.nn.Module):
    """Multi-headed attention, but energetic."""
    def __init__(self,
                 embed_dim,
                 n_heads,
                 beta = None,
                 scale = 0.002):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if beta is None:
            # default to the standard scaling factor for attention
            scale_beta = 1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
            self.beta = torch.nn.Parameter(torch.ones(self.n_heads) * scale_beta) 
        else:
            self.beta = torch.nn.Parameter(beta)

        self.Wq = torch.nn.Parameter(torch.randn((self.n_heads, self.head_dim, self.embed_dim)) * scale)
        self.Wk = torch.nn.Parameter(torch.randn((self.n_heads, self.head_dim, self.embed_dim)) * scale)

    def energy(self, x):
        """Input is length, embed_dim"""
        k = torch.einsum("ld,hzd->lhz", x, self.Wk) # length, n_heads, head_dim
        q = torch.einsum("ld,hzd->lhz", x, self.Wq)

        # attention, where each head has its own scaling factor
        attention = torch.einsum("h,qhz,khz->hqk", self.beta, q, k) # n_heads, length, length

        attention = torch.logsumexp(attention, dim = -1) # n_heads, length
        attention = attention.sum(dim = -1) # n_heads

        return ((-1 / self.beta) * attention).sum(dim = -1) # scalar
    
    def manual_grad(self, x):
        """For testing (matches the jax implementation)"""
        k = torch.einsum("ld,hzd->lhz", x, self.Wk) # (batch, length, n_heads, head_dim)
        q = torch.einsum("ld,hzd->lhz", x, self.Wq)

        F1 = torch.einsum("hzd,lhz->lhd", self.Wq, k)
        F2 = torch.einsum("hzd,lhz->lhd", self.Wk, q)

        # attention, where each head has its own scaling factor
        attention = torch.einsum("h,qhz,khz->hqk", self.beta, q, k) # (batch, n_heads, length, length)
        attention = torch.nn.functional.softmax(attention, dim = -1) # (batch, n_heads, length)
        
        t1 = -torch.einsum("lhd,hqk->ld", F1, attention)
        t2 = -torch.einsum("lhd,hqk->ld", F2, attention)

        return t1 + t2
 
    def forward(self, x):
        return value_and_grad(self.energy, x)
    
class BaseModernHopfield(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 use_bias = False,
                 beta = None,
                 scale = 0.002):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        self._init_activation(beta = beta)
               
        # in order to match the jax implementation, not doing a Linear layer here
        self.W = torch.nn.Parameter(torch.randn(self.in_dim, self.hidden_dim) * scale)

    def _init_activation(self, beta):
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.activation = torch.nn.ReLU()

    def activation_energy(self, x):
        energy = self.activation(x)
        return -0.5*(energy**2).sum()
    
    def energy(self, x):
        h = self.beta * torch.einsum("ld,dh->lh", x, self.W)
        return self.activation_energy(h)
    
    def forward(self, x):
        return value_and_grad(self.energy, x)

class SoftmaxModernHopfield(BaseModernHopfield):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 use_bias = False,
                 beta = None):
        super().__init__(in_dim,
                         hidden_dim,
                         use_bias = use_bias,
                         beta = beta)
        
    def _init_activation(self, beta):
        beta = torch.tensor(0.01) if beta is None else torch.tensor(beta)
        self.beta = torch.nn.Parameter(beta)
        self.activation = torch.nn.Identity()

    def activation_energy(self, x):
        energy = self.activation(x)
        # note the derivative of logsumexp is softmax (hence the name)
        energy = torch.logsumexp(energy, dim = -1)
        return -1 / self.beta * energy.sum()
        
class EnergyTransformer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 n_heads,
                 context_length = 0,
                 n_iters_default = 12,
                 alpha = 0.1,
                 beta = None,
                 hopfield_type = "relu",
                 use_positional_embedding = True,
                 norm = torch.nn.LayerNorm):
        """Implements the energy-based transformer from https://openreview.net/pdf?id=4nrZXPFN1c4. Unlike a conventional transformer,
        this model does not have a feedforward output layer, instead it has a parallel modern Hopfield network that works with the
        transformer. This network is recurrent: it repeatedly acts on the input, which serves to descend the gradient of the 
        energy function.

        Parameters
        ----------
        in_dim : int
            Dimension of the input and output
        hidden_dim : int
            Dimension of the hidden layer in the modern Hopfield network. Note that the number of memories stored in the Hopfield
            network is bounded by this number.
        n_heads : int
            Number of heads in the multi-head attention layer
        context_length : int
            Integer that determines the size of the positional embedding. If 0, then no positional embedding is used.
        n_iters_default : int
            Default number of times the model will be recurrently applied to the input.
        alpha : float
            A step-size multiplier for the recurrent (energy descent) step.
        beta : float
            Inverse temperature for the energy function. It is learnable.
        hopfield_type : str
            Either "relu" or "softmax". Determines the type of modern Hopfield network used.
        use_positional_embedding : bool
            Whether to use a positional embedding.
        norm : torch.nn.Module
            A normalization layer applied to the input (at each recurrent step)
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.context_length = context_length
        self.n_iters = n_iters_default
        self.num_quantizers = self.n_iters # this is for convenience when using with vq-vae

        self.alpha = alpha
        if use_positional_embedding and context_length:
            self.pos_embedding = torch.nn.Parameter(torch.randn(context_length, in_dim))
        else:
            self.pos_embedding = 0

        self.mha = EnergyMHA(self.in_dim, self.n_heads, beta = beta)
        self.hopfield = SoftmaxModernHopfield(self.in_dim, self.hidden_dim, beta = beta) if hopfield_type == "softmax" else BaseModernHopfield(self.in_dim, self.hidden_dim, beta = beta)
        self.norm = norm(self.in_dim)

    def energy(self, x):
        mha_energy = self.mha.energy(x)
        hopfield_energy = self.hopfield.energy(x)
        return mha_energy + hopfield_energy
    
    def forward_step(self, x):
        return value_and_grad(self.energy, x)
    
    def forward(self, x, n_iters = None, 
                **kwargs # for compatibility with other models
                ):
        if n_iters is None:
            n_iters = self.n_iters

        x = x + self.pos_embedding

        energies = []
        features = []
        for i in range(n_iters):
            g = self.norm(x)
            energy, step, = self.forward_step(g)
            x = x - self.alpha * step

            energies.append(energy)
            features.append(x.detach().clone())
        return x, features, energies
    

if __name__ == "__main__":
    ## a very long test on image reconstruction (like the original paper)
    import torchvision
    import ssl
    import os
    import matplotlib.pyplot as plt
    from einops.layers.torch import Rearrange
    from itertools import chain
    from tqdm import tqdm
    # need this to download the dataset
    ssl._create_default_https_context = ssl._create_unverified_context

    im2tensor = torchvision.transforms.ToTensor()

    def collate(x, im2tensor = im2tensor):
        x = [im2tensor(x_i[0]) for x_i in x]
        return torch.stack(x, dim = 0)
    def tensor2im(x):
        return torchvision.transforms.ToPILImage()(x)
    def save_im(x, path):
        tensor2im(x).save(path)

    def save_features(feature_list, patch_deembedder, depatcher, epoch = 0):
        for i, x in enumerate(feature_list):
            x = patch_deembedder(x)
            x = depatcher(x)
            save_im(x[0], f"tmp/epoch_{epoch}_feature_{i}.png")
    
    # making a tmp folder to store the images
    os.makedirs("tmp/", exist_ok = True)

    cifar = torchvision.datasets.CIFAR100(root = "C:/Projects/", train = True, download = True)
    dataloader = torch.utils.data.DataLoader(cifar, 
                                             batch_size = 32, 
                                             shuffle = True,
                                             collate_fn = collate)

    patch_size = 4
    patch_dim = patch_size**2 * 3
    n_patches = 64 # adding 1 for CLS token
    mask_fraction = 0.5
    mask_dropout = 0.1

    token_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patcher = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_size, p2 = patch_size).to(device)
    depatcher = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_size, p2 = patch_size, h = 32 // patch_size, w = 32 // patch_size).to(device)

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
    
    mask_token = torch.randn(token_dim).to(device).requires_grad_(True)

    model = EnergyTransformer(token_dim, 1024, 8,
                              hopfield_type = "softmax",
                              context_length = n_patches, 
                              n_iters_default = 12, alpha = 0.1).to(device)
    
    optimizer = torch.optim.Adam(chain(
                                       model.parameters(),
                                       patch_embedder.parameters(),
                                       patch_deembedder.parameters()
                                       ), 
                                 lr = 1e-3)
    criterion = torch.nn.MSELoss()

    tmp = model.mha.Wq.clone()

    losses = []
    for epoch in range(10):
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

    


    

