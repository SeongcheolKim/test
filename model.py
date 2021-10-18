import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .embed import DataEmbedding
from ..masknn import activations
from ..utils import has_arg

class Informer(nn.Module):
    def __init__(self, enc_in, n_feats, n_src, c_out=64, seq_len=96, factor=5, d_model=256, n_heads=8, e_layers=3, d_layers=4, d_ff=512, 
                dropout=0.0, attn='prob', embed='timeF', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True, mask_act="softmax",
                device=torch.device('cuda')):
        super(Informer, self).__init__()
        self.enc_in = enc_in
        self.n_feats = n_feats
        self.n_src = n_src
        self.c_out = c_out
        self.seq_len = seq_len
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.mask_act = mask_act
        #self.pred_len = out_len
        self.attn = attn
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.distil = distil
        self.mix = mix
        self.device = device
        self.output_attention = output_attention
        # Encoding

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        # self.decoder = Decoder(
        #    [
        #        DecoderLayer(
        #            AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
        #                        d_model, n_heads, mix=mix),
        #            AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
        #                        d_model, n_heads, mix=False),
        #            d_model,
        #            d_ff,
        #            dropout=dropout,
        #            activation=activation,
        #        )
        #        for l in range(d_layers)
        #    ],
        #    norm_layer=torch.nn.LayerNorm(d_model)
        # )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(self.d_model, self.n_src*self.c_out, bias=True)
        
        mask_n1_class = activations.get(mask_act)
        if has_arg(mask_n1_class, "dim"):
            self.output_act = mask_n1_class(dim=1)
        else:
            self.output_act = mask_n1_class()
    def forward(self, x_enc, x_mark_enc=128, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        batch, _, n_frames = x_enc.size()
        enc_out = enc_out.permute(0,2,1)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #enc_out = enc_out.reshape(batch, n_frames, self.c_out*self.n_src)
        enc_out = self.projection(enc_out)
        output = self.output_act(enc_out)
        output = output.view(batch, n_frames, self.c_out, self.n_src)
        output = output.permute(0,3,2,1)
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return output, attns
        else:
            return output # [B, L, D]

    def get_config(self):
        config = {
                "enc_in": self.enc_in,
                "c_out": self.c_out,
                "seq_len": self.seq_len,
                "factor": self.factor,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "e_layers": self.e_layers,
                "d_layers": self.d_layers,
                "d_ff": self.d_ff,
                "dropout": self.dropout,
                "attn": self.attn,
                "embed": self.embed,
                "freq": self.freq,
                "activation": self.activation,
                "output_attention": self.output_attention,
                "distil": self.distil,
                "mix": self.mix,
                "mask_act": self.mask_act,
                "device": self.device,
                }
        return config


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
import torch
import torch.nn as nn
