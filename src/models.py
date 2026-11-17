"""Modelos: IHDT adaptado a libros y baselines (LightGCN, MF-BPR).

Todos los modelos exponen `embeddings(graph) -> (emb_usuarios, emb_libros)`;
el puntaje de un par (u, i) es el producto interno de sus embeddings.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Graph


class LightGCN(nn.Module):
    """LightGCN (He et al., 2020). Con n_layers=0 equivale a MF-BPR."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users, self.n_layers = n_users, n_layers
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def embeddings(self, g: Graph):
        e = torch.cat([self.user.weight, self.item.weight])
        layers = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(g.adj, e)
            layers.append(e)
        e = torch.stack(layers).mean(0)
        return e[:self.n_users], e[self.n_users:]


class IHDT(nn.Module):
    """Dos torres con aprendizaje interactivo y GCN de orden superior.

    Adaptación del modelo de He, Li y Cai (2024) a recomendación de libros:
      * Torre de usuario: embedding del ID + un rasgo del historial (nº de libros
        leídos) + a_u, el promedio de los libros que el lector calificó bien.
      * Torre de libro: embedding del ID + géneros + autores + a_v, el promedio de
        los lectores que lo calificaron bien. (a_u y a_v son la "interacción":
        cada torre recibe información de la otra.)
      * Cada torre: Linear -> ReLU -> normalización L2 (como en la Fig. 3 del paper).
      * GCN de orden superior sobre el grafo usuario-libro (capa estilo NGCF).
      * Embedding final = concatenación de las salidas de todas las capas.
    """

    def __init__(self, n_users, n_items, genre_ids, author_ids, n_genres, n_authors,
                 dim=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_users, self.n_layers = n_users, n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.genre_emb = nn.Embedding(n_genres + 1, dim, padding_idx=0)
        self.author_emb = nn.Embedding(n_authors + 1, dim, padding_idx=0)
        self.user_feat = nn.Linear(1, dim)
        self.user_tower = nn.Linear(3 * dim, dim)
        self.item_tower = nn.Linear(4 * dim, dim)
        self.w1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.w2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.register_buffer("genre_ids", torch.as_tensor(np.array(genre_ids), dtype=torch.long))
        self.register_buffer("author_ids", torch.as_tensor(np.array(author_ids), dtype=torch.long))
        for emb in (self.user_emb, self.item_emb, self.genre_emb, self.author_emb):
            nn.init.normal_(emb.weight, std=0.1)
        with torch.no_grad():
            self.genre_emb.weight[0].zero_()
            self.author_emb.weight[0].zero_()

    @staticmethod
    def _masked_mean(emb: nn.Embedding, ids: torch.Tensor):
        mask = (ids > 0).unsqueeze(-1).float()
        return (emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)

    def towers(self, g: Graph):
        genre = self._masked_mean(self.genre_emb, self.genre_ids)
        author = self._masked_mean(self.author_emb, self.author_ids)
        # Interacción: cada torre recibe un resumen del otro lado del grafo
        a_u = torch.sparse.mm(g.ui_mean, self.item_emb.weight + genre + author)
        a_v = torch.sparse.mm(g.iu_mean, self.user_emb.weight)
        u_in = torch.cat([self.user_emb.weight, self.user_feat(g.user_feat), a_u], dim=1)
        v_in = torch.cat([self.item_emb.weight, genre, author, a_v], dim=1)
        e_u = F.normalize(F.relu(self.user_tower(u_in)), dim=1)
        e_v = F.normalize(F.relu(self.item_tower(v_in)), dim=1)
        return e_u, e_v

    def embeddings(self, g: Graph):
        e_u, e_v = self.towers(g)
        e = torch.cat([e_u, e_v])
        layers = [e]
        for w1, w2 in zip(self.w1, self.w2):
            side = torch.sparse.mm(g.adj, e)                      # L·E
            e = F.leaky_relu(w1(side + e) + w2(side * e), 0.2)    # (L+I)·E·W1 + (L·E ⊙ E)·W2
            e = F.normalize(self.drop(e), dim=1)
            layers.append(e)
        e = torch.cat(layers, dim=1)
        return e[:self.n_users], e[self.n_users:]


def popularity_scores(train, n_items: int) -> torch.Tensor:
    """Baseline de popularidad: número de lectores que calificaron bien cada libro."""
    counts = np.bincount(np.array(train["item_idx"]), minlength=n_items)
    return torch.as_tensor(counts, dtype=torch.float32)
