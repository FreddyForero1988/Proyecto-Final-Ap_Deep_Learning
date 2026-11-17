"""Etapa 4 (entrenamiento) y ejecución completa del pipeline.

Uso:
    python -m src.train --model all
    python -m src.train --model ihdt --epochs 30
    python -m src.train --model all --user_frac 0.05 --epochs 2    # prueba rápida
"""
import argparse
import copy
import json
import time
from dataclasses import asdict

import pandas as pd
import torch
import torch.nn.functional as F

from .config import MODELS_DIR, RESULTS_DIR, Config
from .data import build_graph, prepare_data
from .evaluate import evaluate
from .models import IHDT, LightGCN, popularity_scores

MODEL_NAMES = ["popularity", "mf", "lightgcn", "ihdt"]


def build_model(name, data, cfg):
    if name == "mf":                       # MF-BPR = LightGCN sin capas de propagación
        return LightGCN(data.n_users, data.n_items, cfg.dim, n_layers=0)
    if name == "lightgcn":
        return LightGCN(data.n_users, data.n_items, cfg.dim, cfg.n_layers)
    if name == "ihdt":
        return IHDT(data.n_users, data.n_items, data.genre_ids, data.author_ids,
                    len(data.genre_vocab), len(data.author_vocab),
                    cfg.dim, cfg.n_layers, cfg.dropout)
    raise ValueError(name)


def sample_negatives(users, sorted_keys, n_items, tries=3):
    """Un libro negativo por interacción: uno que el lector no tiene en train."""
    neg = torch.randint(0, n_items, users.shape, device=users.device)
    for _ in range(tries):
        key = users * n_items + neg
        pos = torch.searchsorted(sorted_keys, key).clamp(max=len(sorted_keys) - 1)
        clash = sorted_keys[pos] == key
        if not clash.any():
            break
        neg = torch.where(clash, torch.randint_like(neg, n_items), neg)
    return neg


def score_fn_from(model, graph):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.embeddings(graph)
    return lambda users: user_emb[users] @ item_emb.T


def train_model(name, data, cfg, graph, device):
    """Entrena con pérdida BPR y early stopping sobre Recall@K de validación."""
    model = build_model(name, data, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    u_all = torch.as_tensor(data.train["user_idx"].to_numpy().copy(), device=device)
    i_all = torch.as_tensor(data.train["item_idx"].to_numpy().copy(), device=device)
    keys = (u_all * data.n_items + i_all).sort().values
    key_metric = f"recall@{cfg.topk}"
    best, best_epoch, best_state, history = -1.0, 0, None, []
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(u_all), device=device)
        total, steps = 0.0, 0
        for b in range(0, len(perm), cfg.batch_size):
            idx = perm[b:b + cfg.batch_size]
            u, p = u_all[idx], i_all[idx]
            n = sample_negatives(u, keys, data.n_items)
            user_emb, item_emb = model.embeddings(graph)
            eu = user_emb[u]
            loss = -F.logsigmoid((eu * item_emb[p]).sum(1) - (eu * item_emb[n]).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1

        val = evaluate(score_fn_from(model, graph), data.val, [data.train],
                       data.n_users, data.n_items, cfg.topk, device=device)
        history.append({"epoch": epoch, "loss": total / steps, **val})
        print(f"[{name}] época {epoch:3d} | pérdida {total / steps:.4f} | "
              f"val {key_metric} {val[key_metric]:.4f} | {time.time() - start:.0f}s", flush=True)
        if val[key_metric] > best:
            best, best_epoch = val[key_metric], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        elif epoch - best_epoch >= cfg.patience:
            print(f"[{name}] early stopping (mejor época: {best_epoch})")
            break

    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS_DIR / f"{name}.pt")
    return model, history, best_epoch, time.time() - start


def run(models, cfg, rebuild=False):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    data = prepare_data(cfg, rebuild)
    print(f"Lectores: {data.n_users} | libros: {data.n_items} | "
          f"train/val/test: {len(data.train)}/{len(data.val)}/{len(data.test)}")
    graph = build_graph(data.train, data.n_users, data.n_items, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"
    results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    results["_config"] = asdict(cfg)

    for name in models:
        if name == "popularity":
            pop = popularity_scores(data.train, data.n_items).to(device)
            score_fn = lambda users: pop.expand(len(users), -1)
            extra = {}
        else:
            model, history, best_epoch, secs = train_model(name, data, cfg, graph, device)
            pd.DataFrame(history).to_csv(RESULTS_DIR / f"history_{name}.csv", index=False)
            score_fn = score_fn_from(model, graph)
            extra = {"best_epoch": best_epoch, "train_seconds": round(secs)}
        val = evaluate(score_fn, data.val, [data.train], data.n_users, data.n_items,
                       cfg.topk, device=device)
        test = evaluate(score_fn, data.test, [data.train, data.val], data.n_users,
                        data.n_items, cfg.topk, device=device)
        results[name] = {"val": val, "test": test, **extra}
        print(f"[{name}] TEST", {k: round(v, 4) for k, v in test.items()}, flush=True)
        metrics_path.write_text(json.dumps(results, indent=2))

    table = pd.DataFrame({m: results[m]["test"] for m in models if m in results}).T
    table.to_csv(RESULTS_DIR / "comparison_test.csv")
    return results, table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--min_pos", type=int, help="mínimo de interacciones positivas por lector")
    ap.add_argument("--keep_fiction", action="store_true", help="no excluir el género 'fiction'")
    ap.add_argument("--user_frac", type=float, help="fracción de lectores (pruebas rápidas)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--rebuild", action="store_true", help="recalcular datos procesados")
    a = ap.parse_args()

    cfg = Config()
    for arg, field in [("epochs", "epochs"), ("patience", "patience"), ("lr", "lr"),
                       ("dim", "dim"), ("layers", "n_layers"), ("batch_size", "batch_size"),
                       ("min_pos", "min_pos_per_user"), ("user_frac", "user_frac"), ("seed", "seed")]:
        if getattr(a, arg) is not None:
            setattr(cfg, field, getattr(a, arg))
    if a.keep_fiction:
        cfg.drop_genres = ()
    models = MODEL_NAMES if a.model == "all" else [a.model]
    run(models, cfg, a.rebuild)


if __name__ == "__main__":
    main()
