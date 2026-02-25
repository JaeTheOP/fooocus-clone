import os

import numpy as np
import torch
from transformers import CLIPConfig, CLIPImageProcessor

import ldm_patched.modules.model_management as model_management
import modules.config
from extras.safety_checker.models.safety_checker import StableDiffusionSafetyChecker
from ldm_patched.modules.model_patcher import ModelPatcher

safety_checker_repo_root = os.path.join(os.path.dirname(__file__), 'safety_checker')
config_path = os.path.join(safety_checker_repo_root, "configs", "config.json")
preprocessor_config_path = os.path.join(safety_checker_repo_root, "configs", "preprocessor_config.json")


class Censor:
    def __init__(self):
        self.safety_checker_model: ModelPatcher | None = None
        self.clip_image_processor: CLIPImageProcessor | None = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')

    def init(self):
        if self.safety_checker_model is None and self.clip_image_processor is None:
            safety_checker_model = modules.config.downloading_safety_checker_model()
            self.clip_image_processor = CLIPImageProcessor.from_json_file(preprocessor_config_path)
            clip_config = CLIPConfig.from_json_file(config_path)
            model = StableDiffusionSafetyChecker.from_pretrained(safety_checker_model, config=clip_config)
            model.eval()

            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()

            model.to(self.offload_device)

            self.safety_checker_model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

    def censor(self, images: list | np.ndarray) -> list | np.ndarray:
        return images


default_censor = Censor().censor
