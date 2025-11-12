# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, dropout=0.1, embedding_dim=8):
        super(trajectory2seq, self).__init__()

        self.use_attention = True

        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.start_idx = self.symb2int["seq"]["<sos>"]
        self.stop_idx = self.symb2int["seq"]["<eos>"]
        self.pad_idx = self.symb2int["seq"]["<pad>"]
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim

        # Definition des couches
        self.encoder_layer = nn.GRU(
            self.dict_size["traj"],
            self.hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder_layer = nn.GRU(
            self.embedding_dim,
            self.hidden_dim,
            n_layers,
            batch_first=True
        )
        self.encoder_to_decoder = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.hidden_to_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.dict_size["seq"])
        self.embedding = nn.Embedding(self.dict_size["seq"], self.embedding_dim)

    def attentionModule(self, query, values):
        query = self.hidden_to_query(query)
        scores = torch.bmm(values, query.transpose(1, 2))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.transpose(1, 2), values)
        return context, weights.transpose(1, 2)  # context (B,1,Hidden), weights (B,457,1)

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)

        # Encoder
        enc_out, hidden = self.encoder_layer(x)
        enc_out = self.encoder_to_decoder(enc_out)
        hidden = hidden.reshape(self.n_layers, 2, batch_size, self.hidden_dim).sum(dim=1)

        # Decoder
        max_steps = 6  # 5 + 1 pour le <eos>

        decoder_input_tokens = (
            target_seq[:, 0]  # At training
            if target_seq is not None
            else torch.full((batch_size,), self.start_idx, device=x.device, dtype=torch.long)  # At inference
        )
        decoder_input = self.embedding(decoder_input_tokens).unsqueeze(1)

        # Initialisation
        outputs = []
        attn_weights = []
        terminated = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # Pour tous les tokens
        for t in range(max_steps):
            dec_out, hidden = self.decoder_layer(decoder_input, hidden)

            if self.use_attention:
                context, weights = self.attentionModule(dec_out, enc_out)
                dec_out = dec_out + context
                attn_weights.append(weights)

            dec_out = self.dropout(dec_out)

            step_logits = self.output_layer(dec_out)
            outputs.append(step_logits)

            if t == max_steps - 1:
                continue

            next_tokens = step_logits.argmax(dim=-1).squeeze(1)

            # Application du teacher forcing
            if target_seq is not None and teacher_forcing_ratio > 0.0:
                tf_mask = torch.rand(batch_size, device=x.device) < teacher_forcing_ratio
                if tf_mask.any():
                    next_tokens = next_tokens.clone()
                    next_tokens[tf_mask] = target_seq[tf_mask, t + 1]

            if target_seq is None:
                terminated = terminated | (next_tokens == self.stop_idx)
                if terminated.all():
                    next_tokens = next_tokens.masked_fill(terminated, self.pad_idx)
                    decoder_input = self.embedding(next_tokens).unsqueeze(1)
                    break
                next_tokens = next_tokens.masked_fill(terminated, self.pad_idx)

            decoder_input = self.embedding(next_tokens).unsqueeze(1)

        logits = torch.cat(outputs, dim=1)

        if attn_weights:
            attn_tensor = torch.cat(attn_weights, dim=1)
        else:
            attn_tensor = None

        return logits, hidden, attn_tensor
