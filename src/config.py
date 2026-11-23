"""Configuración central del proyecto: rutas, fuentes de datos y parámetros."""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Fuente de datos: repositorio público goodbooks-10k (Goodreads)
BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"
RAW_FILES = ["ratings.csv", "books.csv", "book_tags.csv", "tags.csv"]

# Las etiquetas originales son estantes de usuarios (ruidosas). Se agrupan en
# géneros; todo lo que no esté aquí (to-read, favorites, owned, kindle...) se descarta.
GENRE_MAP = {
    "fiction": ["fiction", "novel", "novels", "general-fiction"],
    "non-fiction": ["non-fiction", "nonfiction"],
    "fantasy": ["fantasy"],
    "urban-fantasy": ["urban-fantasy"],
    "science-fiction": ["science-fiction", "sci-fi", "scifi"],
    "sci-fi-fantasy": ["sci-fi-fantasy", "fantasy-sci-fi"],
    "young-adult": ["young-adult", "ya", "ya-fiction", "teen"],
    "new-adult": ["new-adult"],
    "middle-grade": ["middle-grade"],
    "children": ["children", "childrens", "children-s", "children-s-books",
                 "childrens-books", "kids", "picture-books"],
    "classics": ["classics", "classic"],
    "romance": ["romance", "contemporary-romance", "historical-romance"],
    "paranormal-romance": ["paranormal-romance"],
    "paranormal": ["paranormal", "supernatural", "vampires", "vampire"],
    "mystery": ["mystery", "mystery-thriller"],
    "thriller": ["thriller", "suspense"],
    "crime": ["crime"],
    "historical-fiction": ["historical-fiction", "historical"],
    "horror": ["horror"],
    "dystopian": ["dystopian", "dystopia"],
    "contemporary": ["contemporary", "contemporary-fiction", "realistic-fiction"],
    "literary-fiction": ["literary-fiction", "literature"],
    "adventure": ["adventure"],
    "humor": ["humor"],
    "chick-lit": ["chick-lit"],
    "graphic-novels": ["graphic-novels", "graphic-novel", "comics"],
    "manga": ["manga"],
    "memoir": ["memoir"],
    "biography": ["biography"],
    "history": ["history"],
    "science": ["science"],
    "philosophy": ["philosophy"],
    "psychology": ["psychology"],
    "poetry": ["poetry"],
    "drama": ["drama", "plays"],
    "short-stories": ["short-stories"],
    "mythology": ["mythology"],
}


@dataclass
class Config:
    # --- Preprocesamiento (etapa 2) ---
    pos_threshold: int = 4          # rating >= 4 => interacción positiva
    min_pos_per_user: int = 5       # mínimo de interacciones positivas por lector (>= 3)
    drop_genres: tuple = ("fiction",)  # géneros a excluir (fiction está en ~77 % de los libros)
    genres_per_book: int = 5        # géneros más frecuentes que se conservan por libro
    max_authors_per_book: int = 3
    user_frac: float = 1.0          # < 1.0 usa una muestra de lectores (pruebas rápidas)
    # --- División (etapa 3) ---
    test_frac: float = 0.1          # 10 % de cada lector a test
    val_frac: float = 0.1           # 10 % del resto a validación
    seed: int = 42
    # --- Modelo y entrenamiento (etapa 4) ---
    dim: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    lr: float = 1e-3                # el paper usa 1e-4; ver README
    weight_decay: float = 1e-5      # regularización L2 (valor del paper)
    batch_size: int = 4096
    epochs: int = 50
    patience: int = 10              # el paper usa 50; se acorta por el tiempo de Colab
    topk: int = 20
    # --- Ablation study ---
    use_genres: bool = True
    use_authors: bool = True
    use_interaction: bool = True

    def __post_init__(self):
        if self.min_pos_per_user < 3:
            raise ValueError("min_pos_per_user debe ser >= 3 (train, val y test no pueden quedar vacíos)")

    def preprocessing_params(self) -> dict:
        keys = ["pos_threshold", "min_pos_per_user", "drop_genres", "genres_per_book",
                "max_authors_per_book", "user_frac", "test_frac", "val_frac", "seed"]
        d = asdict(self)
        return {k: (list(d[k]) if isinstance(d[k], tuple) else d[k]) for k in keys}
