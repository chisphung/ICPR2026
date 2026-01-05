"""
Video Swin Transformer for Spatio-Temporal Fusion.

Implements 3D shifted window attention for processing aligned frame features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-Head Self-Attention.
    
    Computes attention within local 3D windows for efficiency.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (T, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_t = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (num_windows*B, N, C) where N = T*H*W within window
            mask: Attention mask
            
        Returns:
            Output features (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer Block with shifted window attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 4, 4),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (B, T, H, W, C)
            
        Returns:
            Output (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # Apply shift if needed
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x, 
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x
        
        # Partition into windows
        x_windows = self.window_partition(shifted_x)  # (num_windows*B, window_T*window_H*window_W, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        shifted_x = self.window_reverse(attn_windows, T, H, W)
        
        # Reverse shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x
        
        # Residual + FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition into non-overlapping windows."""
        B, T, H, W, C = x.shape
        wT, wH, wW = self.window_size
        
        # Pad if necessary
        pad_t = (wT - T % wT) % wT
        pad_h = (wH - H % wH) % wH
        pad_w = (wW - W % wW) % wW
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        
        _, Tp, Hp, Wp, _ = x.shape
        
        x = x.view(B, Tp // wT, wT, Hp // wH, wH, Wp // wW, wW, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, wT * wH * wW, C)
        
        return windows
    
    def window_reverse(self, windows: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """Reverse window partition."""
        wT, wH, wW = self.window_size
        
        # Account for padding
        Tp = math.ceil(T / wT) * wT
        Hp = math.ceil(H / wH) * wH
        Wp = math.ceil(W / wW) * wW
        
        B = windows.shape[0] // (Tp // wT * Hp // wH * Wp // wW)
        
        x = windows.view(B, Tp // wT, Hp // wH, Wp // wW, wT, wH, wW, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, Tp, Hp, Wp, -1)
        
        # Remove padding
        x = x[:, :T, :H, :W, :]
        
        return x


class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer for spatio-temporal feature fusion.
    
    Lightweight version for license plate restoration.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2),
        num_heads: Tuple[int, ...] = (3, 6),
        window_size: Tuple[int, int, int] = (2, 4, 4),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Input projection
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )
        
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = embed_dim * (2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else (
                        window_size[0] // 2,
                        window_size[1] // 2,
                        window_size[2] // 2,
                    ),
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            # Downsample between stages (except last)
            if i_layer < self.num_layers - 1:
                downsample = nn.Conv3d(dim, dim * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.layers.append(downsample)
        
        # Output projection
        final_dim = embed_dim * (2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.out_proj = nn.Linear(final_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, T, H, W)
            
        Returns:
            Fused features (B, C, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T, H, W)
        x = self.pos_drop(x)
        
        # Reshape to (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        # Process through layers
        for i, module in enumerate(self.layers):
            if isinstance(module, nn.ModuleList):
                # Transformer blocks
                for block in module:
                    x = block(x)
            else:
                # Downsample
                x = x.permute(0, 4, 1, 2, 3).contiguous()
                x = module(x)
                x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        # Output
        x = self.norm(x)
        x = self.out_proj(x)
        
        # Average over temporal dimension
        x = x.mean(dim=1)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        return x
