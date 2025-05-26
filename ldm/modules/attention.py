from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import random
import time
import os
from ldm.modules.diffusionmodules.util import checkpoint
from typing import Optional, Any
from ldm.modules.diffusionmodules.util import normalization
from pytorch_lightning.utilities.distributed import rank_zero_only
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def l2norm(t):
    return F.normalize(t, dim = -1)

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def otsu2(mask_in):
    # normalize
    mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
       (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])
    
    N = 10
    bs = mask_in.shape[0]
    h = mask_in.shape[1]
    mask = []
    for i in range(bs):
        mask_i = mask_norm[i]
        low = [mask_i[mask_i < t/N] for t in range(1, N)]
        high = [mask_i[mask_i >= t/N] for t in range(1, N)]
        low_num = torch.tensor([i.shape[0]/h for i in low], device=mask_in.device)
        high_num = torch.tensor([i.shape[0]/h for i in high], device=mask_in.device)
        low_mean = torch.stack([i.mean() for i in low])
        high_mean = torch.stack([i.mean() for i in high])
        g = low_num*high_num*((low_mean-high_mean)**2)
        index = torch.argmax(g)
        t = index+1
        threshold_t = t/N
            
        mask_i[mask_i < threshold_t] = 0
        mask_i[mask_i > threshold_t] = 1
        mask.append(mask_i)
    mask_out = torch.stack(mask, dim=0)
            
    return mask_out

def otsu(mask_in):
    # normalize
    mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
       (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])
    
    N = 10
    bs = mask_in.shape[0]
    h = mask_in.shape[1]
    mask = []
    for i in range(bs):
        threshold_t = 0.
        max_g = 0.
        for t in range(N):
            mask_i = mask_norm[i]
            low = mask_i[mask_i < t/N]
            high = mask_i[mask_i >= t/N]
            low_num = low.shape[0]/h
            high_num = high.shape[0]/h
            low_mean = low.mean()
            high_mean = high.mean()
        
            g = low_num*high_num*((low_mean-high_mean)**2)
            if g > max_g:
                max_g = g
                threshold_t = t/N
            
        mask_i[mask_i < threshold_t] = 0
        mask_i[mask_i > threshold_t] = 1
        mask.append(mask_i)
    mask_out = torch.stack(mask, dim=0)
            
    return mask_out
# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        print(f"CONTEXT DIM {context_dim}")

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    @rank_zero_only
    def print_kv(self, k):
        print(f"shape {k.shape}")
        
    def forward(self, x, context=None, knn_memory=None, mask=None, return_attn=False):
       
        h = self.heads

        q = self.to_q(x)
       
        #print(k.shape)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        #print(f"shape {.shape}")
        #self.print_kv(k)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)



class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, return_attn=False, mask=None):
        return checkpoint(self._forward, (x, context, return_attn, mask), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, return_attn=False, mask=None):
        x = self.attn1(self.norm1(x)) + x
        if not return_attn:
            x = self.attn2(self.norm2(x), context=context, mask=mask) + x
            x = self.ff(self.norm3(x)) + x
            return x
        if return_attn:
            x_, attn = self.attn2(self.norm2(x), context=context, mask=mask, return_attn=True)
            x = x_ + x
            x = self.ff(self.norm3(x)) + x
            return x, attn

class BasicTransformerBlock_Memory(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, memory_context_dim=None, gated_ff=True, checkpoint=True, disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)    # Cross-attention with text
        self.attn_mem = attn_cls(query_dim=dim, context_dim=memory_context_dim,  # Cross-attention with memory
                                       heads=n_heads, dim_head=d_head, dropout=dropout)
        self.linear_cat = nn.Linear(2*dim, dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)      # Before self-attention (attn1)
        self.norm2 = nn.LayerNorm(dim)      # Before cross-attention (attn2)
        self.norm2_mem = nn.LayerNorm(dim)  # Before memory attention (attn_mem)
        self.norm3 = nn.LayerNorm(dim)      # Before feed-forward network
        self.checkpoint = checkpoint

    def forward(self, x, context=None, context_mem=None, return_attn=False, mask=None, knn_memory=None, save_memory=False, timesteps=None):
        if self.checkpoint:
            # Adjust the checkpoint function to handle memories
            def run_function(x, context, context_mem, return_attn, mask, knn_memory, save_memory, timesteps):
                return self._forward(x, context, context_mem, return_attn, mask, knn_memory, save_memory, timesteps)
            x = checkpoint(run_function, (x, context, context_mem, return_attn, mask, knn_memory, save_memory, timesteps), self.parameters(), self.checkpoint)
        else:
            x = self._forward(x, context, context_mem, return_attn, mask, knn_memory, save_memory, timesteps)
        return x

    def _forward(self, x, context=None, context_mem=None, return_attn=False, mask=None, knn_memory=None, save_memory=False, timesteps=None):
        # Apply memory attention first
        x_mem = self.attn_mem(self.norm2_mem(x), context=context_mem, mask=mask) if context_mem is not None else None

        # Apply self-attention
        x_attn1 = self.attn1(self.norm1(x))

        # Concatenate x_attn1 and x_mem if x_mem is not None
        if x_mem is not None:
            x_cat = torch.cat([x_attn1, x_mem], dim=-1)
            x_proj = self.linear_cat(x_cat)
            x = x + x_proj
        else:
            x = x + x_attn1

        # Apply cross-attention with context
        if context is not None:
            x_context = self.attn2(self.norm2(x), context=context, mask=mask)
            x = x + x_context

        # Apply feed-forward network
        x = x + self.ff(self.norm3(x))

        return x



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, image_cross=False,  disable_self_attn=False, use_linear=True,
                 use_checkpoint=True, return_attn=False):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.image_cross = image_cross
        
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d])
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, xr, context=None, cc_init=None, ph_pos=None, use_img_cond=True, return_attn=False, label = None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        
      
        return x + x_in, None, None, None #xr + xr_in, loss_reg, attn_save
     

class SpatialTransformer_Memory(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, image_cross=False,  disable_self_attn=False, use_linear=True,
                 use_checkpoint=True,  memory_context_dim = 1280):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.image_cross = image_cross
        
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock_Memory(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d], memory_context_dim = inner_dim)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear


    def forward(self, x, xr=None, context=None, knn_memory=None, save_memory = False, timesteps = None, label = None):
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if xr is not None:
            # Assuming xr is of shape [b, c_mem, h_mem, w_mem]
            xr = rearrange(xr, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], context_mem = xr)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
       
        return x + x_in

