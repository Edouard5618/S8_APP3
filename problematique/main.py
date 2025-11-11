# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *


if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                    # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    hidden_dim = 16
    embedding_dim = 8
    n_layers = 2
    max_len = 512
    lr = 0.001
    batch_size = 32
    teacher_forcing_ratio = 0.5
    dropout = 0.1

    # À compléter
    n_epochs = 50

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    torch.manual_seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')

    # Séparation train / validation
    train_percentage = 0.8
    n_train = int(train_percentage * len(dataset))
    n_val = len(dataset) - n_train

    train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])

    # Instanciation des dataloaders
    dataload_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Instanciation du model
    vocab_sizes = {
        'seq': len(dataset.symb2int['seq']),
        'traj': dataset.symb2int['traj']
    }
    model = trajectory2seq(hidden_dim, n_layers, dataset.int2symb,
                           dataset.symb2int, vocab_sizes, device, max_len, dropout=dropout, embedding_dim=embedding_dim)
    model = model.to(device)

    # print model summary (nb of parameters, layers, ...)
    print(model)
    print('Nombre de paramètres du modèle :', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Initialisation des variables
    # À compléter

    if trainning:

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if learning_curves:
            train_loss_hist = []
            val_loss_hist = []
            train_dist_hist = []
            val_dist_hist = []
            fig_loss, ax_loss = plt.subplots()
            fig_dist, ax_dist = plt.subplots()

        for epoch in range(1, n_epochs + 1):
            running_loss_train = 0
            dist = 0
            current_tf = teacher_forcing_ratio * max(0.0, 1 - (epoch - 1) / n_epochs)

            model.train()

            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                traj_seq, word_seq = data
                # print('traj_seq: ', traj_seq[1])
                traj_seq = traj_seq.to(device).float()
                word_seq = word_seq.to(device).long()

                optimizer.zero_grad()
                output, hidden = model(traj_seq, word_seq, current_tf)

                target_tokens = word_seq[:, 1:]
                logits = output.reshape(-1, model.dict_size['seq'])
                targets = target_tokens.reshape(-1)
                loss = criterion(logits, targets)

                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                word_seq_list = word_seq.cpu().tolist()
                batch_len = len(word_seq_list)
                for i in range(batch_len):
                    a = word_seq_list[i]
                    b = [dataset.symb2int['seq'][dataset.start_symbol]] + output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    dist += edit_distance(a[:Ma], b[:Mb]) / batch_len

                    word_a = []
                    for idx in a[:Ma]:
                        sym = dataset.int2symb['seq'][idx]
                        if sym in {dataset.start_symbol, dataset.pad_symbol}:
                            continue
                        word_a.append(sym)

                    word_b = []
                    for idx in b[:Mb]:
                        sym = dataset.int2symb['seq'].get(idx, dataset.pad_symbol)
                        if sym in {dataset.start_symbol, dataset.pad_symbol}:
                            continue
                        word_b.append(sym)
                    if batch_idx == 1:
                        print('Word :', ''.join(word_a), ' - Predicted :',
                              ''.join(word_b), 'distance:', edit_distance(a[:Ma], b[:Mb]))

                # Affichage pendant l'entraînement
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx * batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, (batch_idx+1) * batch_size, len(dataload_train.dataset),
                100. * (batch_idx+1) * batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                dist/len(dataload_train)), end='\r')
            print('\n')

            train_loss_epoch = running_loss_train/len(dataload_train)
            train_dist_epoch = dist/len(dataload_train)

            model.eval()
            val_running_loss = 0
            val_dist = 0
            with torch.no_grad():
                for val_traj, val_word in val_loader:
                    val_traj = val_traj.to(device).float()
                    val_word = val_word.to(device).long()

                    val_output, _ = model(val_traj, val_word, teacher_forcing_ratio=0.0)
                    val_targets = val_word[:, 1:]
                    val_logits = val_output.reshape(-1, model.dict_size['seq'])
                    val_targets_flat = val_targets.reshape(-1)
                    val_loss = criterion(val_logits, val_targets_flat)
                    val_running_loss += val_loss.item()

                    val_output_list = torch.argmax(val_output, dim=-1).detach().cpu().tolist()
                    val_word_list = val_word.cpu().tolist()
                    batch_len = len(val_word_list)
                    for i in range(batch_len):
                        a = val_word_list[i]
                        b = [dataset.symb2int['seq'][dataset.start_symbol]] + val_output_list[i]
                        Ma = a.index(1)
                        Mb = b.index(1) if 1 in b else len(b)
                        val_dist += edit_distance(a[:Ma], b[:Mb]) / batch_len

            if len(val_loader) > 0:
                val_loss_epoch = val_running_loss / len(val_loader)
                val_dist_epoch = val_dist / len(val_loader)
            else:
                val_loss_epoch = float('nan')
                val_dist_epoch = float('nan')

            # Affichage graphique
            if learning_curves:
                train_loss_hist.append(train_loss_epoch)
                val_loss_hist.append(val_loss_epoch)
                train_dist_hist.append(train_dist_epoch)
                val_dist_hist.append(val_dist_epoch)

                ax_loss.cla()
                ax_loss.plot(train_loss_hist, label='train loss')
                ax_loss.plot(val_loss_hist, label='val loss')
                ax_loss.set_title('Loss per epoch')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.legend()

                ax_dist.cla()
                ax_dist.plot(train_dist_hist, label='train distance')
                ax_dist.plot(val_dist_hist, label='val distance')
                ax_dist.set_title('Edit distance per epoch')
                ax_dist.set_xlabel('Epoch')
                ax_dist.set_ylabel('Distance')
                ax_dist.legend()

                plt.draw()
                plt.pause(0.01)

            # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close('all')

        torch.save(model, 'model.pt')

if test:
    model.eval()

    # Load a trained model only if we are not training now
    if not trainning:
        model_path = 'model.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Impossible de charger le modèle entraîné: {model_path}")

        loaded = torch.load(model_path, map_location=device, weights_only=False)
        model = loaded.to(device)
        model.eval()

    test_dataset = HandwrittenWords('data_test.p')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    total_edit_distance = 0.0
    total_samples = 0
    true_labels = []
    pred_labels = []
    missing_characters = 0
    extra_characters = 0

    start_idx = test_dataset.symb2int['seq'][test_dataset.start_symbol]
    stop_idx = test_dataset.symb2int['seq'][test_dataset.stop_symbol]
    pad_idx = test_dataset.symb2int['seq'][test_dataset.pad_symbol]

    # Helper: strip special tokens from an integer sequence
    def strip_tokens(seq):
        out = []
        for t in seq:
            if t == stop_idx:
                break
            out.append(t)
        return out

    with torch.no_grad():
        for traj_seq, word_seq in test_loader:
            traj_seq = traj_seq.to(device).float()
            word_seq = word_seq.to(device).long()

            # Inference: no teacher forcing
            output_logits, _ = model(traj_seq, teacher_forcing_ratio=0.0)  # [B, T, V]
            pred_sequences = output_logits.argmax(dim=-1).cpu().tolist()
            true_sequences = word_seq.cpu().tolist()

            print('Pred_sequences:', pred_sequences)
            print('True_sequences:', true_sequences)

            for true_seq, pred_seq in zip(true_sequences, pred_sequences):
                true_tokens = strip_tokens(true_seq)
                pred_tokens = strip_tokens(pred_seq)

                total_edit_distance += edit_distance(true_tokens, pred_tokens)
                total_samples += 1

                # Length diagnostics
                missing_characters += max(0, len(true_tokens) - len(pred_tokens))
                extra_characters += max(0, len(pred_tokens) - len(true_tokens))

                # Collect per-symbol labels for confusion matrix on overlapping part only
                paired_len = min(len(true_tokens), len(pred_tokens))
                if paired_len > 0:
                    true_labels.extend(true_tokens[:paired_len])
                    pred_labels.extend(pred_tokens[:paired_len])

    if total_samples == 0:
        print("Test - Aucun échantillon disponible pour l'évaluation.")
    else:
        avg_edit_distance = total_edit_distance / total_samples
        print(f"\nTest - Average Edit Distance: {avg_edit_distance:.4f}")

        if missing_characters or extra_characters:
            print(f"Test - Caractères manquants (séquence prédite trop courte): {missing_characters}")
            print(f"Test - Caractères supplémentaires (séquence prédite trop longue): {extra_characters}")

        if true_labels and pred_labels:
            conf_mat, class_indices = confusion_matrix(true_labels, pred_labels, ignore=[start_idx, stop_idx, pad_idx])
            class_labels = [test_dataset.int2symb['seq'].get(idx, str(idx)) for idx in class_indices]
            print('Test - Confusion Matrix (indexes):')
            print(conf_mat)
            plot_confusion_matrix(conf_mat, class_labels)
        else:
            print("Test - Impossible de calculer la matrice de confusion (aucune paire symbole-vérité/prédiction).")
