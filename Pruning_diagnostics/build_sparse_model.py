import torch 
import torch.nn as nn
from typing import Dict, Iterable, List, Optional, Tuple
import math
from ViT_Model import *

class SparseBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SparseVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SparseBlock(embed_dim, num_heads, hidden_dims[i], dropout) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build_sparse_model_from_masks(
    src_model: nn.Module,
    masks: Dict[str, torch.Tensor],
    cfg,
    device: torch.device,
) -> SparseVisionTransformer:
    # Active embedding channels are shared across the whole ViT.
    active_embed_idx = masks["cls_token"][0, 0].detach().bool().nonzero(as_tuple=False).flatten().cpu()
    if active_embed_idx.numel() == 0:
        active_embed_idx = torch.tensor([0], dtype=torch.long)

    old_embed_dim = int(src_model.cls_token.shape[-1])
    new_embed_dim = int(active_embed_idx.numel())

    # Keep attention valid: embed_dim must be divisible by num_heads.
    if new_embed_dim % max(1, cfg.num_heads) == 0:
        new_num_heads = cfg.num_heads
    else:
        new_num_heads = max(1, math.gcd(new_embed_dim, cfg.num_heads))

    hidden_dims: List[int] = []
    for b in range(cfg.depth):
        key = f"blocks.{b}.mlp.fc1.bias"
        if key in masks:
            hidden_dims.append(int(masks[key].detach().bool().sum().item()))
        else:
            hidden_dims.append(int(src_model.blocks[b].mlp.fc1.weight.shape[0]))

    dst_model = SparseVisionTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_classes=cfg.num_classes,
        embed_dim=new_embed_dim,
        depth=cfg.depth,
        num_heads=new_num_heads,
        hidden_dims=hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    ae = active_embed_idx.to(device)
    qkv_idx = torch.cat([ae, ae + old_embed_dim, ae + 2 * old_embed_dim], dim=0)

    with torch.no_grad():
        # Patch embedding and tokens.
        dst_model.patch_embed.proj.weight.copy_(src_model.patch_embed.proj.weight[ae, :, :, :])
        if src_model.patch_embed.proj.bias is not None:
            dst_model.patch_embed.proj.bias.copy_(src_model.patch_embed.proj.bias[ae])
        dst_model.cls_token.copy_(src_model.cls_token[:, :, ae])
        dst_model.pos_embed.copy_(src_model.pos_embed[:, :, ae])

        # Per-block transfer.
        for bi, (dst_block, src_block) in enumerate(zip(dst_model.blocks, src_model.blocks)):
            hid_key = f"blocks.{bi}.mlp.fc1.bias"
            hid_mask = masks.get(hid_key, None)
            if hid_mask is None:
                hid_idx = torch.arange(src_block.mlp.fc1.weight.shape[0], device=device)
            else:
                hid_idx = hid_mask.detach().bool().nonzero(as_tuple=False).flatten().to(device)
                if hid_idx.numel() == 0:
                    hid_idx = torch.tensor([0], dtype=torch.long, device=device)

            # LayerNorms
            dst_block.norm1.weight.copy_(src_block.norm1.weight[ae])
            dst_block.norm1.bias.copy_(src_block.norm1.bias[ae])
            dst_block.norm2.weight.copy_(src_block.norm2.weight[ae])
            dst_block.norm2.bias.copy_(src_block.norm2.bias[ae])

            # Attention
            dst_block.attn.in_proj_weight.copy_(src_block.attn.in_proj_weight[qkv_idx][:, ae])
            dst_block.attn.in_proj_bias.copy_(src_block.attn.in_proj_bias[qkv_idx])
            dst_block.attn.out_proj.weight.copy_(src_block.attn.out_proj.weight[ae][:, ae])
            dst_block.attn.out_proj.bias.copy_(src_block.attn.out_proj.bias[ae])

            # MLP
            dst_block.mlp.fc1.weight.copy_(src_block.mlp.fc1.weight[hid_idx][:, ae])
            dst_block.mlp.fc1.bias.copy_(src_block.mlp.fc1.bias[hid_idx])
            dst_block.mlp.fc2.weight.copy_(src_block.mlp.fc2.weight[ae][:, hid_idx])
            dst_block.mlp.fc2.bias.copy_(src_block.mlp.fc2.bias[ae])

        # Final norm and head.
        dst_model.norm.weight.copy_(src_model.norm.weight[ae])
        dst_model.norm.bias.copy_(src_model.norm.bias[ae])
        dst_model.head.weight.copy_(src_model.head.weight[:, ae])
        dst_model.head.bias.copy_(src_model.head.bias)

    return dst_model