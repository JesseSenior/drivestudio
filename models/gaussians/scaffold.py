import logging

import torch
from omegaconf import OmegaConf
from torch import nn

from models.gaussians.basics import *

logger = logging.getLogger()


class ScaffoldGaussians(nn.Module):
    def __init__(
        self,
        ctrl_cfg: OmegaConf,
        reg_cfg: OmegaConf,
        scene_scale: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.ctrl_cfg = ctrl_cfg
        self.reg_cfg = reg_cfg
        self.scene_scale = scene_scale
        self.device = device
