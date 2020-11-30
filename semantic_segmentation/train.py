import gunpowder as gp

import torch
from omegaconf import OmegaConf

import logging

from model import SemanticUnet
from model_config import SemanticModelConf
from train_pipeline import train_pipeline

logging.basicConfig(filename="log.out", level=logging.INFO)

# Constants
num_iterations = 1_000
config_file = "model_configs/test.yaml"

# load config
overrides = OmegaConf.load(config_file)
model_config: SemanticModelConf = OmegaConf.structured(SemanticModelConf(**overrides))
OmegaConf.set_struct(model_config, True)

# initialize model
model = SemanticUnet(model_config)

# get pipeline
pipeline, request = train_pipeline(model)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# main loop
if __name__ == "__main__":
    with gp.build(pipeline):
        for i in range(num_iterations):
            batch = pipeline.request_batch(request)
