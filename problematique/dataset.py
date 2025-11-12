import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle


class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol = '<pad>'
        self.start_symbol = '<sos>'
        self.stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)
        self.data = list(self.data)

        self.symb2int = {}
        # Vocab: symbols → ints
        self.symb2int = {
            'seq': {
                self.start_symbol: 0,
                self.stop_symbol:  1,
                self.pad_symbol:   2,
                'a': 3,
                'b': 4,
                'c': 5,
                'd': 6,
                'e': 7,
                'f': 8,
                'g': 9,
                'h': 10,
                'i': 11,
                'j': 12,
                'k': 13,
                'l': 14,
                'm': 15,
                'n': 16,
                'o': 17,
                'p': 18,
                'q': 19,
                'r': 20,
                's': 21,
                't': 22,
                'u': 23,
                'v': 24,
                'w': 25,
                'x': 26,
                'y': 27,
                'z': 28
            }
        }

        # Dictionnaires d'entiers vers symboles
        self.int2symb = dict()
        self.int2symb['seq'] = {v: k for k, v in self.symb2int['seq'].items()}

        # Extraction des symboles
        maxWordLength = 0
        maxDataLength = 0

        for word, data in self.data:
            maxWordLength = max(maxWordLength, len(word) + 2)  # +2 (<sos>, <eos>)
            maxDataLength = max(maxDataLength, len(data[0]))

        # Ajout du padding et conversion en tenseurs utilisables par PyTorch
        for i, (word, (x_seq, y_seq)) in enumerate(self.data):
            word_ids = [self.symb2int['seq'][self.start_symbol]]

            # Conversion du mot en liste d'entiers avec stop
            for char in word:
                word_ids.append(self.symb2int['seq'][char])
            word_ids.append(self.symb2int['seq'][self.stop_symbol])

            # Ajout du padding au mot
            wordPadLength = maxWordLength - len(word_ids)
            word_ids.extend([self.symb2int['seq'][self.pad_symbol]] * wordPadLength)

            # Conversion des sequences x et v en tableau numpy
            x_seq = np.asarray(x_seq, dtype=np.float32)
            y_seq = np.asarray(y_seq, dtype=np.float32)
            traj = np.stack([x_seq, y_seq], axis=-1)

            # Ajout du padding à la trajectoire (dernière position répétée)
            pad_len = maxDataLength - traj.shape[0]
            if pad_len > 0:
                last_point = traj[-1, :]
                pad_values = np.repeat(last_point[None, :], pad_len, axis=0)
                traj = np.concatenate([traj, pad_values], axis=0)

            # Sauvegarde des données transformées
            self.data[i] = (traj.astype(np.float32), np.asarray(word_ids, dtype=np.int64))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj, word_ids = self.data[idx]
        traj_tensor = torch.from_numpy(traj)
        word_tensor = torch.from_numpy(word_ids)
        return traj_tensor, word_tensor

    def visualisation(self, idx):
        traj, word_ids = self.data[idx]

        valid_mask = ~np.all(np.isclose(traj, 0.0), axis=1)
        traj = traj[valid_mask]
        x = traj[:, 0]
        y = traj[:, 1]

        chars = []
        for token in word_ids:
            symbol = self.int2symb['seq'].get(int(token), self.pad_symbol)
            if symbol == self.stop_symbol:
                break
            if symbol in {self.start_symbol, self.pad_symbol}:
                continue
            chars.append(symbol)
        word = ''.join(chars)

        plt.figure(figsize=(6, 1.6))
        plt.plot(x, y, linewidth=0.6)
        plt.scatter(x, y, s=8)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.figtext(0.5, 0, f"cible: {word}", ha='center', fontsize=12)
        plt.show()


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(2):
        a.visualisation(np.random.randint(0, len(a)))
