import torch
import torch.nn as nn


class SwapCT(nn.Module):

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)
