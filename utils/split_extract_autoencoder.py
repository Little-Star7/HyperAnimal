import torch
from omegaconf import OmegaConf
from ldm.models.autoencoder import VQModelInterface, AutoencoderKL


config_path = './vq_first_stage_config.yaml'
cfg = OmegaConf.load(config_path)
print(cfg)

model = VQModelInterface(embed_dim=cfg.params.embed_dim, n_embed=cfg.params.n_embed, ddconfig=cfg.params.ddconfig, lossconfig=cfg.params.lossconfig)
model.init_from_ckpt('d:/repository/temp/vq-f4.ckpt')


model_state_dict = model.state_dict()
encoder_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder.") or k.startswith("quant_conv.")}
decoder_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith("decoder.") or k.startswith("quantize.") or k.startswith("post_quant_conv.")}

torch.save(encoder_state_dict, 'd:/repository/temp/vq_encoder_model.pt')
torch.save(decoder_state_dict, 'd:/repository/temp/vq_decoder_model.pt')
#
# config_path = './kl_first_stage_config.yaml'
# cfg = OmegaConf.load(config_path)
# print(cfg)
#
# model = AutoencoderKL(embed_dim=cfg.params.embed_dim, ddconfig=cfg.params.ddconfig, lossconfig=cfg.params.lossconfig)
# model.init_from_ckpt('d:/repository/temp/kl-f4.ckpt')
#
#
# model_state_dict = model.state_dict()
# encoder_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith("encoder.") or k.startswith("quant_conv.")}
# decoder_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith("decoder.") or k.startswith("post_quant_conv.")}
#
# torch.save(encoder_state_dict, 'd:/repository/temp/kl_encoder_model.pt')
# torch.save(decoder_state_dict, 'd:/repository/temp/kl_decoder_model.pt')


import os
from omegaconf import OmegaConf, DictConfig
from models.autoencoder.autoencoder import VQEncoderInterface, VQDecoderInterface, KLEncoderInterface, KLDecoderInterface
from utils.helpers import print_status, count_model_parameters, ensure_path_join, denormalize_to_zero_to_one, \
    normalize_to_neg_one_to_one


# latent_encoder = VQEncoderInterface(
#                 first_stage_config_path=os.path.join(".", "models", "autoencoder", "first_stage_config.yaml"),
#                 encoder_state_dict_path=os.path.join(".", "models", "autoencoder", "first_stage_encoder_state_dict.pt")
#             )
# latent_encoder_1 = VQEncoderInterface(
#                 first_stage_config_path=os.path.join(".", "models", "autoencoder", "first_stage_config.yaml"),
#                 encoder_state_dict_path=os.path.join(".", "models", "autoencoder", "encoder_model.pt")
#             )
# trainable_params, _, total_params = count_model_parameters(latent_encoder)
# print(f"#Params Latent Encoder Model 0: {trainable_params} (Total: {total_params})")
# trainable_params, _, total_params = count_model_parameters(latent_encoder_1)
# print(f"#Params Latent Encoder Model 2: {trainable_params} (Total: {total_params})")

# latent_encoder = KLEncoderInterface(
#                 first_stage_config_path=os.path.join("../", "models", "autoencoder", "kl_f8_first_stage_config.yaml"),
#                 encoder_state_dict_path=os.path.join("../", "models", "autoencoder", "kl_f8_encoder.pt")
#             )
# trainable_params, _, total_params = count_model_parameters(latent_encoder)
# print(f"#Params Latent Encoder Model 0: {trainable_params} (Total: {total_params})")
#
#
# latent_decoder = KLDecoderInterface(
#                 first_stage_config_path=os.path.join("../", "models", "autoencoder", "kl_f8_first_stage_config.yaml"),
#                 decoder_state_dict_path=os.path.join("../", "models", "autoencoder", "kl_f8_decoder.pt")
#             )
# trainable_params, _, total_params = count_model_parameters(latent_decoder)
# print(f"#Params Latent Decoder Model 0: {trainable_params} (Total: {total_params})")
