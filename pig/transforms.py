import torch
import torch.nn as nn


class SwapCT(nn.Module):

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 2, 1, 3, 4)
