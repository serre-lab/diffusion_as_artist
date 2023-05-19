import torch.nn as nn
import torch
from typing import Dict, Tuple
import numpy as np
from functools import partial
from .block import ConvNextBlock, SinusoidalPositionEmbeddings, PreNorm, Residual, LinearAttention
from .block import Attention, Upsample, Downsample
from utils.helper import default, exists


class Unet_V2_Omniglot(nn.Module):
    def __init__(self,
                 in_channels,
                 n_feat=256,
                 embedding_model=None,
                 dim_mults=(1, 2, 4)):
        super(Unet_V2_Omniglot, self).__init__()

        self.in_channels = in_channels

        init_dim = (n_feat//3) * 2

        if embedding_model == 'stack':
            self.init_conv = nn.Conv2d(2 * self.in_channels, init_dim, 7, padding=3)
        else :
            self.init_conv = nn.Conv2d(self.in_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: n_feat * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ConvNextBlock, mult=2)

        time_dim = n_feat * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(n_feat),
            nn.Linear(n_feat, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.embedding_model = embedding_model

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim, cond_emb_dim=None),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, cond_emb_dim=None)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, cond_emb_dim=None)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim, cond_emb_dim=None),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, cond_emb_dim=None),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(None, in_channels)

        self.final_conv = nn.Sequential(
            block_klass(n_feat, n_feat), nn.Conv2d(n_feat, out_dim, 1)
        )

    def forward(self, x, t, c=None, context_mask=None):
        if (c is not None) and (context_mask is not None):
            # mask out context if context_mask == 1
            context_mask = context_mask[:, None, None, None]
            context_mask = context_mask.repeat(1, x.size(1), x.size(2), x.size(3))
            context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
            c = c * context_mask

        if self.embedding_model == "stack":
            x = torch.cat([x, c], dim=1)
        x = self.init_conv(x)
        t = self.time_mlp(t)

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            x = block2(x, t, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)
            x = block2(x, t, c)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)