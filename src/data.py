"""Etapas 1 a 3 del pipeline: adquisición, preprocesamiento, división y grafo."""
import json
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import (BASE_URL, GENRE_MAP, PROCESSED_DIR, RAW_DIR, RAW_FILES,
                     SPLITS_DIR, Config)


# --------------------------------------------------------------------------
# Etapa 1. Adquisición
# --------------------------------------------------------------------------
def download_raw(force: bool = False):
    """Descarga los 4 CSV de goodbooks-10k a data/raw (si no existen)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dst = RAW_DIR / name
        if dst.exists() and not force:
            continue
        print(f"Descargando {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", dst)
    return RAW_DIR


# --------------------------------------------------------------------------
# Etapa 2. Preprocesamiento
# --------------------------------------------------------------------------
def _item_genres(books: pd.DataFrame, book_tags: pd.DataFrame, tags: pd.DataFrame,
                 cfg: Config) -> pd.Series:
    """Top-N géneros por libro, a partir de las etiquetas (estantes) de los usuarios."""
    tag2genre = {t: g for g, ts in GENRE_MAP.items() if g not in cfg.drop_genres for t in ts}
    bt = book_tags.merge(tags, on="tag_id")
    bt["genre"] = bt["tag_name"].map(tag2genre)
    bt = bt.dropna(subset=["genre"])
    bg = bt.groupby(["goodreads_book_id", "genre"], as_index=False)["count"].sum()
    bg = (bg.sort_values(["goodreads_book_id", "count", "genre"], ascending=[True, False, True])
            .groupby("goodreads_book_id").head(cfg.genres_per_book))
    genres = bg.groupby("goodreads_book_id")["genre"].apply(list)
    return books["goodreads_book_id"].map(genres)


def preprocess(cfg: Config):
    """Devuelve (interacciones, ítems) ya reindexados y limpios."""
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    books = pd.read_csv(RAW_DIR / "books.csv")
    book_tags = pd.read_csv(RAW_DIR / "book_tags.csv")
    tags = pd.read_csv(RAW_DIR / "tags.csv")

    # 1. Verificaciones mínimas
    if not ratings["book_id"].isin(books["book_id"]).all():
        raise ValueError("Hay calificaciones de libros que no existen en books.csv")
    n_dup = int(ratings.duplicated(["user_id", "book_id"]).sum())
    if n_dup:
        print(f"Aviso: {n_dup} pares usuario-libro duplicados eliminados")
        ratings = ratings.drop_duplicates(["user_id", "book_id"])

    # 2. Interacción positiva: rating >= umbral (los demás no se usan)
    pos = ratings.loc[ratings["rating"] >= cfg.pos_threshold, ["user_id", "book_id"]]

    # 3. Filtro por lector
    counts = pos.groupby("user_id").size()
    keep = counts[counts >= cfg.min_pos_per_user].index.to_numpy()
    if cfg.user_frac < 1.0:
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(keep, size=int(len(keep) * cfg.user_frac), replace=False)
    pos = pos[pos["user_id"].isin(keep)]

    # 5. Reindexado contiguo (se conservan los 10 000 libros)
    user_ids = np.sort(pos["user_id"].unique())
    book_ids = np.sort(books["book_id"].unique())
    user2idx = pd.Series(np.arange(len(user_ids)), index=user_ids)
    book2idx = pd.Series(np.arange(len(book_ids)), index=book_ids)
    inter = pos.assign(user_idx=pos["user_id"].map(user2idx).to_numpy(),
                       item_idx=pos["book_id"].map(book2idx).to_numpy())
    inter = inter.sort_values(["user_idx", "item_idx"]).reset_index(drop=True)

    # 4. Atributos del libro: géneros y autores
    items = books.sort_values("book_id").reset_index(drop=True)
    items["item_idx"] = book2idx.loc[items["book_id"]].to_numpy()
    genres = _item_genres(items, book_tags, tags, cfg)
    items["genres"] = genres.apply(lambda g: "|".join(g) if isinstance(g, list) else "")
    items["authors"] = items["authors"].fillna("").apply(
        lambda s: "|".join([a.strip() for a in s.split(",") if a.strip()][:cfg.max_authors_per_book]))
    items = items[["item_idx", "book_id", "title", "genres", "authors"]]
    return inter, items


# --------------------------------------------------------------------------
# Etapa 3. División y grafo
# --------------------------------------------------------------------------
def split_interactions(inter: pd.DataFrame, cfg: Config):
    """División por lector: 10 % test; del resto, 10 % validación (semilla fija)."""
    df = inter.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    rank = df.groupby("user_idx").cumcount().to_numpy()
    n = df.groupby("user_idx")["item_idx"].transform("size").to_numpy()
    n_test = np.maximum(1, np.round(cfg.test_frac * n)).astype(int)
    n_val = np.maximum(1, np.round(cfg.val_frac * (n - n_test))).astype(int)
    part = np.where(rank < n_test, "test", np.where(rank < n_test + n_val, "val", "train"))
    cols = ["user_idx", "item_idx"]
    return tuple(df.loc[part == p, cols].reset_index(drop=True) for p in ("train", "val", "test"))


def _encode_items(items: pd.DataFrame):
    """Convierte géneros y autores de cada libro en matrices de IDs (0 = relleno)."""
    def encode(col, width):
        lists = items[col].apply(lambda s: s.split("|") if s else [])
        vocab = sorted({x for l in lists for x in l})
        v2i = {x: i + 1 for i, x in enumerate(vocab)}
        out = np.zeros((len(items), width), dtype=np.int64)
        for r, l in enumerate(lists):
            for c, x in enumerate(l[:width]):
                out[r, c] = v2i[x]
        return out, vocab

    width_g = max(1, int(items["genres"].str.count(r"\|").add(1).where(items["genres"] != "", 0).max()))
    width_a = max(1, int(items["authors"].str.count(r"\|").add(1).where(items["authors"] != "", 0).max()))
    genre_ids, genre_vocab = encode("genres", width_g)
    author_ids, author_vocab = encode("authors", width_a)
    return genre_ids, author_ids, genre_vocab, author_vocab


@dataclass
class Data:
    n_users: int
    n_items: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame
    genre_ids: np.ndarray
    author_ids: np.ndarray
    genre_vocab: list
    author_vocab: list


def prepare_data(cfg: Config, rebuild: bool = False) -> Data:
    """Ejecuta las etapas 1-3 (con caché en data/processed y data/splits)."""
    params_file = PROCESSED_DIR / "params.json"
    params = cfg.preprocessing_params()
    cached = (params_file.exists() and json.loads(params_file.read_text()) == params
              and (SPLITS_DIR / "test.csv").exists())
    if rebuild or not cached:
        download_raw()
        inter, items = preprocess(cfg)
        train, val, test = split_interactions(inter, cfg)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        inter.to_csv(PROCESSED_DIR / "interactions.csv", index=False)
        items.to_csv(PROCESSED_DIR / "item_features.csv", index=False)
        for name, df in (("train", train), ("val", val), ("test", test)):
            df.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        params_file.write_text(json.dumps(params))
    else:
        print("Usando datos procesados en caché (data/processed y data/splits)")
        items = pd.read_csv(PROCESSED_DIR / "item_features.csv").fillna("")
        train, val, test = (pd.read_csv(SPLITS_DIR / f"{n}.csv") for n in ("train", "val", "test"))
    items = items.fillna("")
    genre_ids, author_ids, gv, av = _encode_items(items)
    n_users = int(max(train.user_idx.max(), val.user_idx.max(), test.user_idx.max())) + 1
    return Data(n_users, len(items), train, val, test, items, genre_ids, author_ids, gv, av)


@dataclass
class Graph:
    adj: torch.Tensor        # (U+I, U+I), simétrica y normalizada D^-1/2 A D^-1/2
    ui_mean: torch.Tensor    # (U, I), promedio de los libros de cada lector
    iu_mean: torch.Tensor    # (I, U), promedio de los lectores de cada libro
    user_feat: torch.Tensor  # (U, 1), log(1 + nº de libros) estandarizado
    n_users: int
    n_items: int


def build_graph(train: pd.DataFrame, n_users: int, n_items: int, device="cpu") -> Graph:
    """Grafo usuario-libro construido SOLO con el conjunto de entrenamiento."""
    u = torch.as_tensor(np.array(train["user_idx"]), dtype=torch.long)
    i = torch.as_tensor(np.array(train["item_idx"]), dtype=torch.long)
    deg_u = torch.bincount(u, minlength=n_users).float()
    deg_i = torch.bincount(i, minlength=n_items).float()

    def sparse(rows, cols, vals, shape):
        return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, shape).coalesce().to(device)

    w = (deg_u[u] * deg_i[i]).rsqrt()
    adj = sparse(torch.cat([u, i + n_users]), torch.cat([i + n_users, u]),
                 torch.cat([w, w]), (n_users + n_items,) * 2)
    ui_mean = sparse(u, i, 1.0 / deg_u[u], (n_users, n_items))
    iu_mean = sparse(i, u, 1.0 / deg_i[i], (n_items, n_users))
    feat = torch.log1p(deg_u)
    feat = ((feat - feat.mean()) / (feat.std() + 1e-8)).unsqueeze(1).to(device)
    return Graph(adj, ui_mean, iu_mean, feat, n_users, n_items)
