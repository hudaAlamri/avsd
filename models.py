import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import Encoder, LateFusionEncoder
from decoders import Decoder


class AVSD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args, self.encoder)

    def _load_state_dict_(self, components):
        self.encoder.load_state_dict(components['encoder'])
        self.decoder.load_state_dict(components['decoder'])

    def forward(self, batch):
        enc_out = self.encoder(batch)
        dec_out = self.decoder(enc_out, batch)
        return dec_out
