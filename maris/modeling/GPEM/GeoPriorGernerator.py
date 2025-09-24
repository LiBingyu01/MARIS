"""
GPEM with DepthAnything structure features.

This module extracts structure cues using a DepthAnything backbone (if available),
projects them to match the visual feature channels, and fuses with the visual feature
map via a lightweight residual/MLP fusion.
"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from depth_anything_v2.dpt import DepthAnythingV2


def _build_depthanything(model_type: str = "small") -> Optional[nn.Module]:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[model_type])
    model.load_state_dict(torch.load(f'pretrained/depth_anything_v2_{model_type}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class GeoPriorGernerator(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        struct_channels: list[int] = [512, 1024, 1024],
        fusion: str = "add",  # add | mlp | dformer_like
        depthanything_model_type: str = "small",
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.struct_channels = struct_channels
        self.fusion = fusion

        # DepthAnything as primary structure extractor when available
        self.depth_model = _build_depthanything(depthanything_model_type)

        if self.fusion == "mlp":
            self.mlp = nn.Sequential(
                nn.Conv2d(2 * feat_channels, feat_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True),
            )
            self.proj = nn.ModuleList([
            nn.Conv2d(in_ch, feat_channels, kernel_size=1)
            for in_ch in self.struct_channels
            ])
        elif self.fusion == "alpha_fusion":
            self.proj_v = nn.ModuleList([
                nn.Conv2d(feat_channels, feat_channels, kernel_size=1) for in_ch in self.struct_channels
            ])
            self.proj_g = nn.ModuleList([
                nn.Conv2d(in_ch, feat_channels, kernel_size=1) for in_ch in self.struct_channels
            ])
            self.alpha_proj = nn.ModuleList([
                nn.Conv2d(2 * feat_channels, feat_channels, kernel_size=1) for in_ch in self.struct_channels
            ])
            self.mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
                ) for in_ch in self.struct_channels
            ])
        elif self.fusion == "add":
            self.alpha = 0.5
            self.proj = nn.ModuleList([
                nn.Conv2d(in_ch, feat_channels, kernel_size=1)
                for in_ch in self.struct_channels
            ])

    def forward_features_extra(self, x, masks=None):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        out_features = self.depth_model.pretrained.get_intermediate_layers(x, 
                                                        self.depth_model.intermediate_layer_idx[self.depth_model.encoder], 
                                                        return_class_token=True)
        
        out_feats = []
        for i, x in enumerate(out_features):
            x, cls_token = x[0], x[1]
            cls_token = cls_token.unsqueeze(1).expand_as(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            cls_token = cls_token.permute(0, 2, 1).reshape((cls_token.shape[0], cls_token.shape[-1], patch_h, patch_w))
            out_feats.append(x)

        return out_feats, cls_token

    @torch.no_grad()
    def compute_depth(self, x_rgb: torch.Tensor, input_hw: int = None) -> Optional[torch.Tensor]:
        """
        Run DepthAnything ONCE to produce a base depth map.
        Returns B×C×h×w depth list and cls_token, or None if DA is unavailable.
        input_hw: optional resize before DA to control compute (e.g., (384,384)).
        """
        if self.depth_model is None:
            return None
        if input_hw is not None:
            x_rgb = F.interpolate(x_rgb, size=input_hw, mode="bilinear", align_corners=False)
        depth_list, cls_token = self.forward_features_extra(x_rgb) # depth_list
        return depth_list[1:], cls_token

    def forward(self) -> torch.Tensor:
        """
        """

    def enhance_with_depth_list(self, depth_list, feats_list):
        enhance_feats_list = []
        for i in range(len(depth_list)):
            enhance_feats = self.enhance_with_depth(feats_list[i], 
                                                    depth_list[i], 
                                                    i)
            enhance_feats_list.append(enhance_feats)
        return enhance_feats_list

    def enhance_with_depth(self, 
                            v: torch.Tensor, 
                            g: torch.Tensor, 
                            i) -> torch.Tensor:
        """
            v: (B, C1, Hv, Wv) rgb_feats;
            g: (B, C2, H, W) dep_feats;
        """

        if self.fusion == "alpha_fusion":
            B, _, Hv, Wv = v.shape
            Fv = self.proj_v[i](v)  # [B, C1, Hv, Wv]
            Fg = self.proj_g[i](g)  # [B, C1, Hg, Wg]
            if Fg.shape[2:] != (Hv, Wv):
                Fg = F.interpolate(Fg, size=(Hv, Wv), mode="bilinear", align_corners=False)
            concat = torch.cat([Fv, Fg], dim=1)  # [B, 2*C1, H, W]
            alpha = torch.sigmoid(self.alpha_proj[i](concat))  # [B, C1, H, W]
            Fvg = Fv + alpha * Fg
            out = self.mlp[i](Fvg)  # [B, d, H, W]
            return out
        
        elif self.fusion == "mlp":
            B, _, Hv, Wv = v.shape
            Fg = self.proj[i](g)
            if Fg.shape[2:] != (Hv, Wv):
                Fg = F.interpolate(Fg, size=(Hv, Wv), mode="bilinear", align_corners=False)
            return self.mlp(torch.cat([v, Fg], dim=1))
        
        elif self.fusion == "add":
            B, _, Hv, Wv = v.shape
            Fg = self.proj[i](g)
            if Fg.shape[2:] != (Hv, Wv):
                Fg = F.interpolate(Fg, size=(Hv, Wv), mode="bilinear", align_corners=False)
            return self.alpha * v + (1.0 - self.alpha) * Fg
        

            