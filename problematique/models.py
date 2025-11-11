# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, dropout=0.1, embedding_dim=8):
        super(trajectory2seq, self).__init__()

        self.use_attention = True

        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.start_idx = self.symb2int["seq"]["<sos>"]
        self.stop_idx = self.symb2int["seq"]["<eos>"]
        self.pad_idx = self.symb2int["seq"]["<pad>"]
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim

        # Definition des couches
        # Couches pour rnn
        self.encoder_layer = nn.GRU(
            self.dict_size["traj"],
            self.hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
        )  # traj to latent space
        # Projection to merge both encoder directions before decoding
        self.encoder_to_decoder = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.decoder_layer = nn.GRU(self.embedding_dim, self.hidden_dim, n_layers, batch_first=True)
        # the first hidden dim for input comes from decoder and the second comes from the hidden output of the encoder

        # Couches pour attention
        self.hidden_to_query = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Couche dense pour la sortie
        self.output_layer = nn.Linear(self.hidden_dim, self.dict_size["seq"])
        self.embedding = nn.Embedding(self.dict_size["seq"], self.embedding_dim)

    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden_to_query(query)  # (B, T, H)

        # Attention

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
        # values: (B, S, H), query: (B, 1, H)
        # Scores par produit scalaire: (B, S, 1)
        scores = torch.bmm(values, query.transpose(1, 2))  # (B, S, T)
        weights = torch.softmax(scores, dim=1)  # (B, S, T)
        # Contexte: somme pondérée des valeurs -> (B, T, H)
        context = torch.bmm(weights.transpose(1, 2), values)  # (B, T, H)

        # ---------------------- Laboratoire 2 - Question 4 - Fin de la section à compléter -----------------

        return context, weights.transpose(1, 2)  # context (B,T,H), weights (B,T,S)

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)

        # encoder
        enc_out, hidden = self.encoder_layer(x)

        # Merge the bidirectional outputs to match decoder dimensionality
        enc_out = self.encoder_to_decoder(enc_out)
        hidden = hidden.reshape(self.n_layers, 2, batch_size, self.hidden_dim).sum(dim=1)

        # decoder
        if target_seq is not None:
            max_steps = max(0, target_seq.size(1) - 1)
        else:
            max_steps = self.maxlen

        if max_steps == 0:
            self.last_attention_weights = None
            return torch.empty(batch_size, 0, self.dict_size["seq"], device=x.device), hidden

        decoder_input_tokens = (
            target_seq[:, 0]
            if target_seq is not None
            else torch.full((batch_size,), self.start_idx, device=x.device, dtype=torch.long)
        )
        decoder_input = self.embedding(decoder_input_tokens).unsqueeze(1)

        outputs = []
        attn_weights = []
        terminated = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for t in range(max_steps):
            dec_out, hidden = self.decoder_layer(decoder_input, hidden)

            if self.use_attention:
                context, weights = self.attentionModule(dec_out, enc_out)
                dec_out = dec_out + context
                attn_weights.append(weights)

            dec_out = self.dropout(dec_out)  # regularize decoder hidden state

            step_logits = self.output_layer(dec_out)
            outputs.append(step_logits)

            if t == max_steps - 1:
                continue

            next_tokens = step_logits.argmax(dim=-1).squeeze(1)

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

        if self.use_attention and attn_weights:
            self.last_attention_weights = torch.cat(attn_weights, dim=1)
        else:
            self.last_attention_weights = None

        return logits, hidden
