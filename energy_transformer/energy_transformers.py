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
    
    def filtered_forward(self, x, indices, n_iters = None):
        """Given a tensor of the form [context, masked target] and indices describing where on the image
        they come from, add positional embedding and pass through the transformer."""
        if n_iters is None:
            n_iters = self.n_iters

        pos_embedding = self.pos_embedding[:, indices, :]

        x = x + pos_embedding

        energies = []
        features = []
        for i in range(n_iters):
            g = self.norm(x)
            energy, step, = self.forward_step(g)
            x = x - self.alpha * step

            energies.append(energy)
            features.append(x.detach().clone())
        return x, features, energies
    
    def ema_update(self, new_model):
        for ema_param, new_param in zip(self.parameters(), new_model.parameters()):
            ema_param.data.copy_(ema_param.data * self.ema_decay + (1 - self.ema_decay) * new_param.data)
    