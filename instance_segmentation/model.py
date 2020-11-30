import torch

from funlib.learn.torch.models import UNet

class AffinitiesUnet(torch.nn.Module):
    def __init__(self, config):
        super(AffinitiesUnet, self).__init__()

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
        
        self.dims = 3

        self.neighborhood = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        conv = torch.nn.Conv3d
        affs = [
            self.unet,
            conv(config.num_fmaps, self.dims, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.affs = torch.nn.Sequential(*affs)
        self.prediction_channels = self.dims
        self.target_channels = self.dims


    def forward(self, raw):
        affs = self.affs(raw)
        return affs

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