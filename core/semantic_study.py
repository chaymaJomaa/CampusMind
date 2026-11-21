
"""
CampusMind - Assistant de révision sémantique.

Fonctionnalités :
- Extraction PDF et TXT
- Découpage des documents
- Embeddings multilingues
- Recherche sémantique
- Références aux sources
"""

from io import BytesIO
from functools import lru_cache

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


@lru_cache(maxsize=1)
def load_semantic_model():
    """
    Charge le modèle une seule fois
    dans le processus Python.
    """
    return SentenceTransformer(MODEL_NAME)


def extract_document(filename, content):
    """
    Extrait le texte d'un PDF ou TXT.

    Retourne une liste de pages :
    {
        "source": nom du fichier,
        "page": numéro,
        "text": contenu
    }
    """

    pages = []

    if filename.lower().endswith(".pdf"):

        reader = PdfReader(BytesIO(content))

        for page_number, page in enumerate(
            reader.pages, start=1
        ):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "source": filename,
                    "page": page_number,
                    "text": text
                })

    elif filename.lower().endswith(".txt"):

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("cp1252")

        pages.append({
            "source": filename,
            "page": 1,
            "text": text
        })

    else:
        raise ValueError("Format non pris en charge.")

    return pages


def build_chunks(pages, chunk_size=850):
    """
    Découpe les documents en passages
    avec un chevauchement de contexte.
    """

    chunks = []
    overlap = 150

    for page in pages:

        text = " ".join(page["text"].split())

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            passage = text[start:end]

            if passage.strip():

                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "text": passage
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks


def build_index(chunks):
    """
    Calcule les embeddings des passages.
    """

    if not chunks:
        return None

    model = load_semantic_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


def semantic_search(
    question,
    chunks,
    embeddings,
    top_k=3
):
    """
    Recherche les passages les plus proches
    sémantiquement de la question.
    """

    if (
        not question.strip()
        or not chunks
        or embeddings is None
    ):
        return []

    model = load_semantic_model()

    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    scores = embeddings @ question_vector

    best_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in best_indices:

        chunk = chunks[int(index)]

        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": round(
                float(scores[index]),
                4
            )
        })

    return results
