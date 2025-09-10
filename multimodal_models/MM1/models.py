import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ----------------- Tabular Component ----------------- #
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForwardGEGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.act = GEGLU()
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        x = self.out(x)
        return self.dropout(x)

class FTTransformerBlock(nn.Module):
    def __init__(self, dim, heads, attn_dropout, ffn_hidden, ffn_dropout, residual_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.residual_dropout1 = nn.Dropout(residual_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardGEGLU(dim, ffn_hidden, ffn_dropout)
        self.residual_dropout2 = nn.Dropout(residual_dropout)
    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.residual_dropout1(attn_out)
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.residual_dropout2(ffn_out)
        return x

class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, num_categories, token_dim, bias=True):
        super().__init__()
        self.embeddings = nn.Embedding(sum(num_categories), token_dim)
        self.bias = nn.Parameter(torch.zeros(len(num_categories), token_dim)) if bias else None
        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)

    def forward(self, x):
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, in_features, token_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, token_dim))
        self.bias = nn.Parameter(torch.zeros(in_features, token_dim)) if bias else None

    def forward(self, x):
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class FTTransformer(nn.Module):
    def __init__(
        self,
        num_numerical,
        num_categories,
        token_dim=192,
        hidden_size=192,
        num_blocks=3,
        attention_n_heads=8,
        attention_dropout=0.2,
        residual_dropout=0.0,
        ffn_dropout=0.1,
        ffn_hidden_size=192,
        pooling_mode="cls",
        num_classes=2, 
    ):
        super().__init__()
        # Tokenizers
        self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(num_categories, token_dim) if num_categories else None
        self.numerical_feature_tokenizer = NumericalFeatureTokenizer(num_numerical, token_dim) if num_numerical else None

        # Adapters
        self.categorical_adapter = nn.Linear(token_dim, hidden_size) if num_categories else None
        self.numerical_adapter = nn.Linear(token_dim, hidden_size) if num_numerical else None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Transformer blocks
        self.transformer = nn.ModuleList([
            FTTransformerBlock(
                hidden_size,
                attention_n_heads,
                attention_dropout,
                ffn_hidden_size,
                ffn_dropout,
                residual_dropout
            )
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.pooling_mode = pooling_mode

    def forward(self, batch, return_feature=False):
        # batch: expects dict with keys "categorical" and/or "numerical"
        B = batch['numerical'].shape[0] if 'numerical' in batch else batch['categorical'].shape[0]
        multimodal_tokens = []

        if self.categorical_feature_tokenizer:
            x_cat = self.categorical_feature_tokenizer(batch['categorical'])  # (B, num_cat, token_dim)
            x_cat = self.categorical_adapter(x_cat)
            multimodal_tokens.append(x_cat)

        if self.numerical_feature_tokenizer:
            x_num = self.numerical_feature_tokenizer(batch['numerical'])  # (B, num_num, token_dim)
            x_num = self.numerical_adapter(x_num)
            multimodal_tokens.append(x_num)

        tokens = torch.cat(multimodal_tokens, dim=1)  # (B, total_num_tokens, hidden_size)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        tokens = torch.cat([tokens, cls_token], dim=1)  # (B, total_num_tokens+1, hidden_size)

        for block in self.transformer:
            tokens = block(tokens)

        features = tokens[:, -1, :]
        logits = self.head(features)

        if return_feature:
            return features, logits
        return logits
    
# ----------------- Image Component ----------------- #
class TimmModel(nn.Module):
    def __init__(self, num_classes=2, max_img_num_per_col=3, use_learnable_image=True):
        super().__init__()
        self.max_img_num_per_col = max_img_num_per_col
        self.use_learnable_image = use_learnable_image

        self.base_model = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=True,
            num_classes=0
        )
        self.feature_dim = self.base_model.num_features  # 1024 for swin_base_patch4_window7_224
        self.head = nn.Linear(self.feature_dim, num_classes)

        if use_learnable_image:
            self.learnable_image = nn.Parameter(torch.zeros(3, 224, 224))

    def forward(self, x, image_valid_num=None, return_feature=False):
        B, N, C, H, W = x.shape  # N should be 3
        device = x.device

        if self.use_learnable_image and image_valid_num is not None:
            for b in range(B):
                n_valid = image_valid_num[b].item()
                if n_valid < N:
                    x[b, n_valid:] = self.learnable_image

        x_flat = x.reshape(B * N, C, H, W)
        features = self.base_model(x_flat)
        features = features.view(B, N, -1)

        if image_valid_num is None:
            image_valid_num = torch.full((B,), N, dtype=torch.long, device=device)
        mask = torch.arange(N, device=device)[None, :] < image_valid_num[:, None]
        features = features * mask.unsqueeze(-1)
        avg_features = features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.head(avg_features)

        if return_feature:
            return avg_features, logits
        return logits

# ----------------- Fusion Component ----------------- #
class FusionMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='leaky_relu', dropout=0.1, normalization='layer_norm'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features) if normalization == 'layer_norm' else nn.Identity()
        self.act = nn.LeakyReLU() if activation == 'leaky_relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class FusionMLP(nn.Module):
    def __init__(
        self,
        tabular_dim,      # output dim of FTTransformer before its head (not logits!)
        image_dim,        # output dim of TimmModel before its head (not logits!)
        hidden_sizes=[128],
        num_classes=2,
        adapt_in_features='max',
        activation='leaky_relu',
        dropout=0.1,
        normalization='layer_norm',
    ):
        super().__init__()
        # Feature adaptation
        if adapt_in_features == 'max':
            base_dim = max(tabular_dim, image_dim)
        elif adapt_in_features == 'min':
            base_dim = min(tabular_dim, image_dim)
        else:
            raise ValueError(f"Unknown adapt_in_features: {adapt_in_features}")
        self.tabular_adapter = nn.Linear(tabular_dim, base_dim)
        self.image_adapter = nn.Linear(image_dim, base_dim)
        fusion_in_dim = 2 * base_dim

        # MLP layers
        layers = []
        for h in hidden_sizes:
            layers.append(FusionMLPBlock(
                in_features=fusion_in_dim,
                out_features=h,
                activation=activation,
                dropout=dropout,
                normalization=normalization
            ))
            fusion_in_dim = h
        self.fusion_mlp = nn.Sequential(*layers)
        self.head = nn.Linear(fusion_in_dim, num_classes)

    def forward(self, tabular_features, image_features):
        # tabular_features: [B, tabular_dim], image_features: [B, image_dim]
        tabular_proj = self.tabular_adapter(tabular_features)
        image_proj = self.image_adapter(image_features)
        fused = torch.cat([tabular_proj, image_proj], dim=1)
        fused = self.fusion_mlp(fused)
        logits = self.head(fused)
        return logits

# -------------- Final Model with all Components -------------- #
class MultimodalModel(nn.Module):
    def __init__(self, fttransformer, timm_model, fusion_mlp):
        super().__init__()
        self.fttransformer = fttransformer
        self.timm_model = timm_model
        self.fusion_mlp = fusion_mlp

    def forward(self, batch):
        # batch: expects dict with keys "numerical", "categorical", "images", and optionally "image_valid_num"
        tabular_feat, _ = self.fttransformer(batch, return_feature=True)
        image_feat, _ = self.timm_model(
            batch["images"],
            image_valid_num=batch.get("image_valid_num"),
            return_feature=True
        )
        logits = self.fusion_mlp(tabular_feat, image_feat)
        return logits
    
def build_multimodal_model(num_numerical, num_categories, tabular_token_dim, tabular_hidden_dim):
    fttransformer = FTTransformer(
        num_numerical=num_numerical,
        num_categories=num_categories,
        token_dim=tabular_token_dim,
        hidden_size=tabular_hidden_dim,
        num_blocks=3,
        attention_n_heads=8,
        num_classes=2 
    )
    timm_model = TimmModel(num_classes=2)
    fusion_mlp = FusionMLP(
        tabular_dim=tabular_hidden_dim,
        image_dim=timm_model.feature_dim,
        hidden_sizes=[128],
        num_classes=2
    )
    multimodal_model = MultimodalModel(fttransformer, timm_model, fusion_mlp)
    return multimodal_model