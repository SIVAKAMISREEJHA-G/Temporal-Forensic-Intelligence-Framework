import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """Self-attention over time steps."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        scores = self.attn(x).squeeze(-1)          # (B, T)
        weights = torch.softmax(scores, dim=-1)    # (B, T)
        out = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return out, weights


class TFIFClassifier(nn.Module):
    """
    Input : (B, T, 960) — pre-extracted MobileNetV3 frame features
    Output: (B, num_classes) — logits
    """
    def __init__(self, input_dim=960, hidden_dim=64, num_classes=7, num_layers=1, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 960)
        out, _ = self.gru(x)        # (B, T, 128)
        ctx, attn_w = self.attention(out)  # (B, 128), (B, T)
        logits = self.classifier(ctx)      # (B, 7)
        return logits, attn_w


if __name__ == "__main__":
    m = TFIFClassifier()
    dummy = torch.randn(4, 16, 960)
    logits, attn = m(dummy)
    print("Logits shape:", logits.shape)
    print("Attn weights shape:", attn.shape)
