"""
embeddings/embedding_generator.py
----------------------------------
Wraps SentenceTransformer to produce dense vector embeddings
from text. Supports single encoding and batch encoding.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates vector embeddings from text using a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        Name of the pre-trained SentenceTransformer model.
        Default: 'all-MiniLM-L6-v2' — good speed/accuracy trade-off (dim=384).
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): 
        # all-MiniLM-L6-v2 : it allows you to transform text into embeddings 
        print(f"[EmbeddingGenerator] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingGenerator] Embedding dimension: {self.dim}")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a vector embedding.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        np.ndarray of shape (dim,)
        """
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into a matrix of embeddings.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        show_progress : bool
            Show tqdm progress bar during encoding.

        Returns
        -------
        np.ndarray of shape (N, dim)
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def encode_with_metadata(
        self, text: str, genre: str, genres_list: list[str]
    ) -> np.ndarray:
        """
        Augmented embedding: concatenate text embedding with one-hot genre vector.
        Used by the Hybrid approach.

        Parameters
        ----------
        text : str
            Movie description.
        genre : str
            Movie genre label.
        genres_list : list[str]
            Full ordered list of possible genres (defines one-hot length).

        Returns
        -------
        np.ndarray of shape (dim + len(genres_list),)
        """
        text_emb = self.encode(text)
        one_hot = np.zeros(len(genres_list), dtype=np.float32)
        if genre in genres_list:
            one_hot[genres_list.index(genre)] = 1.0
        return np.concatenate([text_emb, one_hot])
