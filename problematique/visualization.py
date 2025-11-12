import matplotlib.pyplot as plt


# Note: The code in this file is mainly written by LLMs. We assume that the visualization is not
# part of the knowledge this APP is trying to evaluate.


def strip_special_tokens(seq, start_idx, stop_idx, pad_idx):
    cleaned = []
    for token in seq:
        if token == stop_idx:
            break
        if token == start_idx or token == pad_idx:
            continue
        cleaned.append(token)
    return cleaned


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
