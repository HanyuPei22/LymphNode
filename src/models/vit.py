# src/models/vit.py
"""
Vision Transformer (ViT) for CIFAR-10/100
Adapted for control plugin injection
"""
from turtle import forward
from uuid import RESERVED_FUTURE
from requests import patch
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0,"embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Batch size, Number of token(64 patches and 1 class label), embed_dim(#channel)
        B, N, C = x.shape 
        # qkv := (batchsize, #Tokens, 3, num of heads, dim for each head)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # qkv := (3, batchsize, num of heads, #Tokens, dim for each head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #(B, 8, 65, 64) @ (B, 8, 64, 65) --> (B, 8, 65, 65), Corr Matrix of token i and j (65,65)
        # **(-0.5) to normalize q@(k.T) to N(0,1), otherwise softmax's gradient may explode
        attn = (q @ k.transpose(-2,-1)) * (self.head_dim ** (-0.5))
        # Trans attn score to possibility in [0,1]
        attn = attn.softmax(dim = -1)
        attn = self.dropout(attn)
        # output = sum(attn[i,j] * value[j]) = weighted combination of value of all the tokens
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # (B, Numheads, NumTokens, embed_dim) --> (B, NumTokens, Numheads, embed_dim)
        x = self.proj(x)
        x = self.dropout(x)

        return x
    
class FeedForward(nn.Module):
    """
    Feed-forward network
    """

    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer encoder block
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class ViT(nn.Module):
    """
    Vision transformer for CIFAR10
    
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_class=10, embed_dim=512,
                 depth=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.conv1 = self.patch_embed.conv1
        self.bn1 = nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        # why there is no self before these variables?
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_class)

        self._init_weight()

    def _init_weight(self):

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim = 1)

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        output = self.linear(cls_output)

        return output

def ViT_Small(num_class=10):
    """ViT-Small for CIFAR"""
    return ViT(
        img_size=32,
        patch_size=4,
        num_class=num_class,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        dropout=0.1
    )


def ViT_Base(num_class=10):
    """ViT-Base for CIFAR"""
    return ViT(
        img_size=32,
        patch_size=4,
        num_class=num_class,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1
    )


def ViT_Tiny(num_class=10):
    """ViT-Tiny for CIFAR (lightweight)"""
    return ViT(
        img_size=32,
        patch_size=4,
        num_class=num_class,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1
    )