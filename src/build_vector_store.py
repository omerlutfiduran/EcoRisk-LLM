import json
import os
import numpy as np

def build_index(kb_path="../data/knowledge_base.json",
                index_path="../data/faiss_index.bin",
                meta_path="../data/kb_metadata.json",
                model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Encodes knowledge base entries into vectors and saves them in a FAISS index.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"[*] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    texts = [entry["icerik"] for entry in kb]
    
    print(f"[*] Encoding {len(texts)} entries...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Initialize FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, index_path)
    
    # Save metadata for retrieval reference
    metadata = []
    for i, entry in enumerate(kb):
        metadata.append({
            "idx": i,
            "id": entry["id"],
            "kategori": entry["kategori"],
            "icerik": entry["icerik"]
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[*] Vector store created successfully at {index_path}")
    return index, metadata

if __name__ == "__main__":
    build_index()
