"""Etapa 5: evaluación Top-K (Precision, Recall, NDCG y HR @K)."""
import numpy as np
import torch


def _csr(df, n_users):
    """Representación por filas (puntero + índices) de las interacciones de cada lector."""
    u, i = np.array(df["user_idx"]), np.array(df["item_idx"])
    order = np.argsort(u, kind="stable")
    u, i = u[order], i[order]
    return np.searchsorted(u, np.arange(n_users + 1)), i


@torch.no_grad()
def evaluate(score_fn, target, exclude, n_users, n_items, k=20, batch=1024, device="cpu"):
    """Evalúa un recomendador.

    score_fn(users) -> matriz (len(users), n_items) con los puntajes.
    target:  interacciones a predecir (val o test).
    exclude: lista de DataFrames cuyos libros no se pueden recomendar (ya vistos).
    Las métricas se promedian sobre los lectores que tienen libros en `target`.
    """
    t_ptr, t_idx = _csr(target, n_users)
    x_csrs = [_csr(df, n_users) for df in exclude]
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=device).float())
    ideal = torch.cumsum(discounts, 0)           # DCG ideal con 1..k aciertos
    sums = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "hr": 0.0}
    n_valid = 0

    def fill(mat, ptr, idx, b0, b1, value):
        rows = np.repeat(np.arange(b1 - b0), np.diff(ptr[b0:b1 + 1]))
        cols = idx[ptr[b0]:ptr[b1]]
        mat[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = value

    for b0 in range(0, n_users, batch):
        b1 = min(b0 + batch, n_users)
        scores = score_fn(torch.arange(b0, b1, device=device)).clone()
        for ptr, idx in x_csrs:
            fill(scores, ptr, idx, b0, b1, float("-inf"))
        tgt = torch.zeros((b1 - b0, n_items), dtype=torch.bool, device=device)
        fill(tgt, t_ptr, t_idx, b0, b1, True)

        top = scores.topk(k, dim=1).indices
        hits = tgt.gather(1, top).float()
        n_rel = tgt.sum(1).float()
        ok = n_rel > 0
        n_hits = hits.sum(1)
        idcg = ideal[(n_rel.clamp(min=1, max=k) - 1).long()]
        sums["precision"] += (n_hits / k)[ok].sum().item()
        sums["recall"] += (n_hits / n_rel.clamp(min=1))[ok].sum().item()
        sums["ndcg"] += ((hits * discounts).sum(1) / idcg)[ok].sum().item()
        sums["hr"] += (n_hits > 0).float()[ok].sum().item()
        n_valid += int(ok.sum().item())

    return {f"{m}@{k}": v / max(n_valid, 1) for m, v in sums.items()}
