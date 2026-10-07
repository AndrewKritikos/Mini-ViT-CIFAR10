import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    def __init__(self, in_channels=3, embedded_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, embedded_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedded_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    
class HybridViTEmbeddings(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, num_patches=64):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        return x
    
class TransformBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        identity = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = identity + attn_out
        identity = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = identity + mlp_out

        return x
    
class HybridVisionTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, num_patches=64,
                 embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.embeddings = HybridViTEmbeddings(in_channels, embed_dim, num_patches)

        self.encoder = nn.Sequential(*[
            TransformBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.norm(x)

        cls_token_final = x[:,0]
        logits = self.head(cls_token_final)
        return logits
    
