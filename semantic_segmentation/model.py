import torch
from funlib.learn.torch.models import UNet, ConvPass


class SemanticUnet(torch.nn.Module):
    def __init__(self, config):
        super(SemanticUnet, self).__init__()

        # LAYERS
        self.unet = UNet(
            in_channels=config.in_channels,
            num_fmaps=config.num_fmaps,
            fmap_inc_factor=config.fmap_inc_factor,
            downsample_factors=[tuple(x) for x in config.downsample_factors],
            activation=config.activation,
            voxel_size=config.voxel_size,
            num_heads=1,
            constant_upsample=True,
        )
        self.logits = ConvPass(
            in_channels=config.num_fmaps,
            out_channels=config.num_classes + 1,
            kernel_sizes=[[1 for _ in range(len(config.downsample_factors[0]))]],
            activation=None,
        )
        self.probs = torch.nn.Softmax()

    def forward(self, raw):
        logits = self.logits(self.unet(raw))
        if not self.training:
            return self.probs(logits)
        else:
            return logits

    def get_output_shape(self, input_shape, fmaps_in):
        """Given the number of input channels and an input size, computes the
        shape of the output."""

        device = "cpu"
        for parameter in self.parameters():
            device = parameter.device
            break

        dummy_data = torch.zeros((1, fmaps_in) + input_shape, device=device)
        out = self.forward(dummy_data)
        return tuple(out.shape[2:])
