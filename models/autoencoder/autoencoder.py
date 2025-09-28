from omegaconf import OmegaConf

import torch
from models.autoencoder.quantization import VectorQuantizer2 as VectorQuantizer
from models.autoencoder.modules import Encoder, Decoder, DiagonalGaussianDistribution


class VQEncoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, encoder_state_dict_path: str):
        super().__init__()

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        embed_dim = config.params.embed_dim

        self.encoder = Encoder(**dd_config)
        self.quant_conv = torch.nn.Conv2d(dd_config["z_channels"], embed_dim, 1)

        state_dict = torch.load(encoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h


class VQDecoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, decoder_state_dict_path: str):
        super().__init__()

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        embed_dim = config.params.embed_dim
        n_embed = config.params.n_embed

        self.decoder = Decoder(**dd_config)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, dd_config["z_channels"], 1)

        state_dict = torch.load(decoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class KLEncoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, encoder_state_dict_path: str):
        super().__init__()

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        embed_dim = config.params.embed_dim

        self.encoder = Encoder(**dd_config)
        self.quant_conv = torch.nn.Conv2d(2*dd_config["z_channels"], 2*embed_dim, 1)

        state_dict = torch.load(encoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        return z


class KLDecoderInterface(torch.nn.Module):

    def __init__(self, first_stage_config_path: str, decoder_state_dict_path: str):
        super().__init__()

        config = OmegaConf.load(first_stage_config_path)
        dd_config = config.params.ddconfig

        embed_dim = config.params.embed_dim

        self.decoder = Decoder(**dd_config)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, dd_config["z_channels"], 1)

        state_dict = torch.load(decoder_state_dict_path)
        self.load_state_dict(state_dict)

    def forward(self, h):
        quant = self.post_quant_conv(h)
        dec = self.decoder(quant)
        return dec


if __name__ == "__main__":
    latent_encoder = VQEncoderInterface(
        first_stage_config_path="vq_f8_first_stage_config.yaml",
        encoder_state_dict_path="vq_f8_encoder.pt"
    )
    latent_decoder = VQDecoderInterface(
        first_stage_config_path="vq_f8_first_stage_config.yaml",
        decoder_state_dict_path="vq_f8_decoder.pt"
    )
    x = torch.ones(2, 3, 512, 512)
    y = latent_encoder(x)
    print(y.shape)

    kl_encoder = KLEncoderInterface(
        first_stage_config_path="kl_f8_first_stage_config.yaml",
        encoder_state_dict_path="kl_f8_encoder.pt"
    )
    kl_decoder = KLDecoderInterface(
        first_stage_config_path="kl_f8_first_stage_config.yaml",
        decoder_state_dict_path="kl_f8_decoder.pt"
    )
    x_1 = torch.ones(2, 3, 512, 512)
    y_1 = kl_encoder(x_1)
    print(y_1.shape)
