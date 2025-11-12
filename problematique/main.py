# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import *
from dataset import *
from metrics import *
from visualization import *
# from visualization import strip_special_tokens, safe_matplotlib_call


if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    training = False             # Entraînement?
    test = True                 # Test?
    learning_curves = True      # Affichage des courbes d'entraînement?
    gen_test_images = True      # Génération images test?
    use_attention = True        # Utiliser le module d'attention?
    seed = 1                    # Pour répétabilité
    n_workers = 0               # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    hidden_dim = 15
    embedding_dim = 9
    n_layers = 2
    lr = 0.0015
    batch_size = 32
    teacher_forcing_ratio = 0.5
    dropout = 0.12
    n_epochs = 50
    grad_clip = 1.0
    best_model_path = 'model.pt'
    torch.manual_seed(seed)

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Chargement des données
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    dataset = HandwrittenWords('data_trainval.p')

    train_percentage = 0.8
    n_train = int(train_percentage * len(dataset))
    n_val = len(dataset) - n_train

    train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])

    dataload_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Instanciation du modèle
    vocab_sizes = {
        'seq': len(dataset.symb2int['seq']),
        'traj': 2,  # x et y
    }
    model = trajectory2seq(hidden_dim, n_layers, dataset.int2symb,
                           dataset.symb2int, vocab_sizes, device, dropout=dropout, embedding_dim=embedding_dim, use_attention=use_attention)
    model = model.to(device)

    print(model)
    print('Nombre de paramètres du modèle :', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if training:
        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Initialisation des historiques pour affichage
        train_loss_hist = []
        val_loss_hist = []
        train_dist_hist = []
        val_dist_hist = []
        best_val_loss = float('inf')
        fig_loss, ax_loss = plt.subplots()
        fig_dist, ax_dist = plt.subplots()

        ### Training ###
        for epoch in range(1, n_epochs + 1):
            running_loss_train = 0
            dist = 0
            current_tf = teacher_forcing_ratio * max(0.0, 1 - (epoch - 1) / n_epochs)

            model.train()

            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                traj_seq, word_seq = data
                traj_seq = traj_seq.to(device).float()
                word_seq = word_seq.to(device).long()

                # Passage avant
                optimizer.zero_grad()
                output, hidden, _ = model(traj_seq, word_seq, current_tf)
                target_tokens = word_seq[:, 1:]
                logits = output.reshape(-1, model.dict_size['seq'])
                targets = target_tokens.reshape(-1)

                # Calcul de la perte
                loss = criterion(logits, targets)
                running_loss_train += loss.item()

                # Rétropropagation et optimisation
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                # Distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                word_seq_list = word_seq.cpu().tolist()
                batch_len = len(word_seq_list)
                for i in range(batch_len):
                    a = word_seq_list[i]
                    b = [dataset.symb2int['seq'][dataset.start_symbol]] + output_list[i]
                    length_a = a.index(1)
                    length_b = b.index(1) if 1 in b else len(b)
                    dist += edit_distance(a[:length_a], b[:length_b]) / batch_len

                    # Affichage d'un exemple de prédiction (première batch)
                    if batch_idx == 1:
                        word_a = []
                        for idx in a[:length_a]:
                            sym = dataset.int2symb['seq'][idx]
                            if sym in {dataset.start_symbol, dataset.pad_symbol}:
                                continue
                            word_a.append(sym)

                        word_b = []
                        for idx in b[:length_b]:
                            sym = dataset.int2symb['seq'].get(idx, dataset.pad_symbol)
                            if sym in {dataset.start_symbol, dataset.pad_symbol}:
                                continue
                            word_b.append(sym)

                        print('Word :', ''.join(word_a), ' - Predicted :',
                              ''.join(word_b), 'distance:', edit_distance(a[:length_a], b[:length_b]))

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

            #### Validation ###
            model.eval()

            val_running_loss = 0
            val_dist = 0
            with torch.no_grad():
                for val_traj, val_word in val_loader:
                    # Formatage des données
                    val_traj = val_traj.to(device).float()
                    val_word = val_word.to(device).long()

                    # Passage avant
                    val_output, _, _ = model(val_traj, val_word, teacher_forcing_ratio=0.0)
                    val_targets = val_word[:, 1:]
                    val_logits = val_output.reshape(-1, model.dict_size['seq'])
                    val_targets_flat = val_targets.reshape(-1)

                    # Calcul de la perte
                    val_loss = criterion(val_logits, val_targets_flat)
                    val_running_loss += val_loss.item()
                    val_output_list = torch.argmax(val_output, dim=-1).detach().cpu().tolist()
                    val_word_list = val_word.cpu().tolist()
                    batch_len = len(val_word_list)

                    # Distance d'édition
                    for i in range(batch_len):
                        a = val_word_list[i]
                        b = [dataset.symb2int['seq'][dataset.start_symbol]] + val_output_list[i]
                        length_a = a.index(1)
                        length_b = b.index(1) if 1 in b else len(b)
                        val_dist += edit_distance(a[:length_a], b[:length_b]) / batch_len

            val_loss_epoch = val_running_loss / len(val_loader)
            val_dist_epoch = val_dist / len(val_loader)

            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                torch.save(model, best_model_path)
                print(f"Nouveau meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")

            # Affichage graphique
            if learning_curves:
                train_loss_hist.append(train_loss_epoch)
                val_loss_hist.append(val_loss_epoch)
                train_dist_hist.append(train_dist_epoch)
                val_dist_hist.append(val_dist_epoch)

                ax_loss.cla()
                ax_loss.plot(train_loss_hist, label='train loss')
                ax_loss.plot(val_loss_hist, label='val loss')
                ax_loss.set_title('Loss pour chaque epoch')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.legend()

                ax_dist.cla()
                ax_dist.plot(train_dist_hist, label='train distance')
                ax_dist.plot(val_dist_hist, label='val distance')
                ax_dist.set_title('Distance d\'édition pour chaque epoch')
                ax_dist.set_xlabel('Epoch')
                ax_dist.set_ylabel('Distance')
                ax_dist.legend()

                plt.draw()
                safe_matplotlib_call(plt.pause, 0.1)

            # Terminer l'affichage d'entraînement
            if learning_curves:
                safe_matplotlib_call(plt.show, block=False)


### Testing ###
if test:
    model.eval()

    # Load a trained model only if we are not training now
    if not training:
        model_path = 'model.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Impossible de charger le modèle entraîné: {model_path}")

        loaded = torch.load(model_path, map_location=device, weights_only=False)
        model = loaded.to(device)
        model.eval()

    # Chargement des données de test
    test_dataset = HandwrittenWords('data_test.p')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Initialisation
    total_edit_distance = 0.0
    total_samples = 0
    true_labels = []
    pred_labels = []
    missing_characters = 0
    extra_characters = 0
    start_idx = test_dataset.symb2int['seq'][test_dataset.start_symbol]
    stop_idx = test_dataset.symb2int['seq'][test_dataset.stop_symbol]
    pad_idx = test_dataset.symb2int['seq'][test_dataset.pad_symbol]

    with torch.no_grad():
        for traj_seq, word_seq in test_loader:
            traj_seq = traj_seq.to(device).float()
            word_seq = word_seq.to(device).long()

            # Passage avant
            output_logits, _, _ = model(traj_seq, teacher_forcing_ratio=0.0)
            pred_sequences = output_logits.argmax(dim=-1).cpu().tolist()
            true_sequences = word_seq.cpu().tolist()

            # Calcul de la distance d'édition
            for idx, (true_seq, pred_seq) in enumerate(zip(true_sequences, pred_sequences)):
                true_tokens = strip_special_tokens(true_seq, start_idx, stop_idx, pad_idx)
                pred_tokens = strip_special_tokens(pred_seq, start_idx, stop_idx, pad_idx)

                total_edit_distance += edit_distance(true_tokens, pred_tokens)
                total_samples += 1

                # Données pour matrice de confusion
                missing_characters += max(0, len(true_tokens) - len(pred_tokens))
                extra_characters += max(0, len(pred_tokens) - len(true_tokens))
                paired_len = min(len(true_tokens), len(pred_tokens))
                if paired_len > 0:
                    true_labels.extend(true_tokens[:paired_len])
                    pred_labels.extend(pred_tokens[:paired_len])

    avg_edit_distance = total_edit_distance / total_samples
    print(f"\nTest - Average Edit Distance: {avg_edit_distance:.4f}")

    if missing_characters or extra_characters:
        print(f"Test - Caractères manquants (séquence prédite trop courte): {missing_characters}")
        print(f"Test - Caractères supplémentaires (séquence prédite trop longue): {extra_characters}")

    conf_mat, class_indices = confusion_matrix(true_labels, pred_labels, ignore=[start_idx, stop_idx, pad_idx])
    class_labels = [test_dataset.int2symb['seq'].get(idx, str(idx)) for idx in class_indices]
    plot_confusion_matrix(conf_mat, class_labels)
