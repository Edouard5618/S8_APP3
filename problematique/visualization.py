import numpy as np
import matplotlib.pyplot as plt


def visualize_attention_batch(traj_batch, attn_batch, pred_batch, true_batch, dataset, rendered_count, max_examples):
    if attn_batch is None or rendered_count >= max_examples:
        return rendered_count

    stop_symbol = dataset.stop_symbol
    pad_symbol = dataset.pad_symbol

    for idx, (pred_seq, true_seq) in enumerate(zip(pred_batch, true_batch)):
        if rendered_count >= max_examples:
            break

        pred_symbols = _tokens_to_symbols(pred_seq, dataset)
        if not pred_symbols:
            continue

        target_symbols = _tokens_to_symbols(true_seq, dataset)
        attention_slice = attn_batch[idx]
        step_capacity = min(len(pred_symbols), attention_slice.shape[0])
        if step_capacity == 0:
            continue

        if pred_symbols[-1] in {stop_symbol, pad_symbol} and step_capacity > 1:
            step_capacity -= 1

        tokens_for_plot = pred_symbols[:step_capacity]
        trajectory_slice = traj_batch[idx]

        predicted_caption = _build_caption(tokens_for_plot, stop_symbol, pad_symbol)
        target_caption = _build_caption(target_symbols, stop_symbol, pad_symbol)

        plot_attention_over_trajectory(trajectory_slice, attention_slice[:step_capacity],
                                       tokens_for_plot, predicted_caption, target_caption)
        rendered_count += 1

    return rendered_count


def trim_trailing_duplicates(traj):
    if len(traj) < 2:
        return traj
    end = len(traj) - 1
    while end > 0 and np.allclose(traj[end], traj[end - 1]):
        end -= 1
    return traj[:end + 1]


def strip_special_tokens(seq, start_idx, stop_idx, pad_idx):
    cleaned = []
    for token in seq:
        if token == stop_idx:
            break
        if token == start_idx or token == pad_idx:
            continue
        cleaned.append(token)
    return cleaned


def _tokens_to_symbols(sequence, dataset):
    symbols = []
    int2symb = dataset.int2symb['seq']
    stop_symbol = dataset.stop_symbol
    pad_symbol = dataset.pad_symbol
    for token_idx in sequence:
        symbol = int2symb.get(int(token_idx), pad_symbol)
        if symbol == dataset.start_symbol:
            continue
        symbols.append(symbol)
        if symbol in {stop_symbol, pad_symbol}:
            break
    return symbols


def _build_caption(symbols, stop_symbol, pad_symbol):
    caption = ''.join(ch for ch in symbols if ch not in {stop_symbol, pad_symbol})
    if stop_symbol in symbols:
        caption += stop_symbol
    return caption


def plot_attention_over_trajectory(traj, attn_matrix, token_labels, predicted_word, target_word):
    traj = trim_trailing_duplicates(traj)
    if traj.size == 0:
        return

    seq_len = traj.shape[0]
    attn_matrix = attn_matrix[:, :seq_len]

    x = traj[:, 0]
    y = traj[:, 1]
    n_tokens = len(token_labels)

    fig, axes = plt.subplots(n_tokens, 1, figsize=(6, 1.4 * n_tokens), sharex=True, sharey=True)
    if n_tokens == 1:
        axes = [axes]

    colorbar_ref = None
    for ax, token, weights in zip(axes, token_labels, attn_matrix):
        vmax = max(float(np.max(weights)), 1e-6)
        ax.plot(x, y, color='lightgray', linewidth=0.8)
        colorbar_ref = ax.scatter(x, y, c=weights, cmap='magma', s=18, vmin=0.0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        ax.text(0.01, 0.7, token, transform=ax.transAxes, fontsize=10)

    cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(colorbar_ref, cax=cbar_ax)
    cbar.set_label('Attention weight', rotation=90)
    fig.text(0.5, 0.04, f"pred: {predicted_word}", ha='center', fontsize=11)
    fig.text(0.5, 0.02, f"target: {target_word}", ha='center', fontsize=11)
    safe_matplotlib_call(fig.tight_layout, rect=[0.03, 0.08, 0.87, 0.98])
    safe_matplotlib_call(plt.show)
    plt.close(fig)


def safe_matplotlib_call(func, *args, **kwargs):
    """Attempt a Matplotlib call and swallow the Tk null pointer error once."""
    try:
        return func(*args, **kwargs)
    except ValueError as exc:
        if "PyCapsule_New called with null pointer" not in str(exc):
            raise
        if not getattr(safe_matplotlib_call, "_warned", False):
            print("Attention: affichage Matplotlib désactivé (backend Tk indisponible).")
            safe_matplotlib_call._warned = True
