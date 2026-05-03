import json
import os
import random
import time
import numpy as np
from groq import Groq
from prompt_builder import build_llama_prompt

def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

_load_env()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in environment.")

_faiss_index = None
_kb_metadata = None
_embed_model = None

def _load_rag_components():
    global _faiss_index, _kb_metadata, _embed_model

    if _faiss_index is not None:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    index_path = os.path.join(base_dir, "data", "faiss_index.bin")
    meta_path  = os.path.join(base_dir, "data", "kb_metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("FAISS index files not found.")

    _faiss_index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        _kb_metadata = json.load(f)

    _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def retrieve_knowledge(hedef_veri, top_k=3):
    _load_rag_components()

    sorgu = (
        f"Yukseklik {hedef_veri['Cografya']['Yukseklik']} metre, "
        f"egim yuzde {hedef_veri['Cografya']['Egim']}, "
        f"sicaklik {hedef_veri['Meteoroloji']['Sicaklik']} derece, "
        f"ruzgar hizi {hedef_veri['Meteoroloji']['Ruzgar_Hizi']} m/s, "
        f"NDVI {hedef_veri['Yakitlar']['NDVI']}, "
        f"NDWI {hedef_veri['Yakitlar']['NDWI']}"
    )

    sorgu_vektoru = _embed_model.encode([sorgu], normalize_embeddings=True)
    sorgu_vektoru = np.array(sorgu_vektoru, dtype="float32")

    skorlar, indeksler = _faiss_index.search(sorgu_vektoru, top_k)

    bulunan_parcalar = []
    for skor, idx in zip(skorlar[0], indeksler[0]):
        if idx < 0: continue
        entry = _kb_metadata[idx]
        bulunan_parcalar.append({
            "id": entry["id"],
            "kategori": entry["kategori"],
            "icerik": entry["icerik"],
            "benzerlik": float(skor)
        })

    metin_parcalari = [f"- [{p['kategori']}] {p['icerik']}" for p in bulunan_parcalar]
    return "\n".join(metin_parcalari), bulunan_parcalar

def run_inference_on_sample(json_file_path, output_file_path, num_samples=20, random_sample=False, seed=42):
    client = Groq(api_key=GROQ_API_KEY)

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if random_sample:
        random.seed(seed)
        yanginli   = [r for r in data if r.get("Gercek_Yangin_Durumu", -1) == 1]
        yanginssiz = [r for r in data if r.get("Gercek_Yangin_Durumu", -1) == 0]
        yari = num_samples // 2
        secilen = random.sample(yanginli, min(yari, len(yanginli))) + \
                  random.sample(yanginssiz, min(num_samples - yari, len(yanginssiz)))
        random.shuffle(secilen)
        test_data = secilen[:num_samples]
    else:
        test_data = data[:num_samples]

    results = []
    print(f"[*] Starting inference for {len(test_data)} records...")

    for i, record in enumerate(test_data):
        rec_id = record.get("id", i)
        data_source = record.get("Girdi_Verisi", record)

        hedef_veri = {
            "Cografya":    data_source["Cografya"],
            "Meteoroloji": data_source["Meteoroloji"],
            "Yakitlar":    data_source["Yakitlar"],
        }

        retrieved_text, rag_detay = retrieve_knowledge(hedef_veri, top_k=3)
        prompt = build_llama_prompt(hedef_veri, retrieved_text)

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=300,
            )

            model_cevabi = chat_completion.choices[0].message.content.strip()
            results.append({
                "id":                     rec_id,
                "Gercek_Yangin_Durumu":   record.get("Gercek_Yangin_Durumu", -1),
                "Girdi_Verisi":           hedef_veri,
                "RAG_Parcalari":          [p["id"] for p in rag_detay],
                "RAG_Benzerlik_Skorlari": [round(p["benzerlik"], 3) for p in rag_detay],
                "Kullanilan_RAG_Metni":   retrieved_text,
                "Model_Ciktisi":          model_cevabi,
            })
            print(f"[{i+1}/{len(test_data)}] ID {rec_id}: Success")
            time.sleep(3.0) 

        except Exception as e:
            print(f"[{i+1}/{len(test_data)}] ID {rec_id}: Error -> {e}")
            results.append({
                "id": rec_id,
                "Gercek_Yangin_Durumu": record.get("Gercek_Yangin_Durumu", -1),
                "Girdi_Verisi": hedef_veri,
                "Error": str(e)
            })

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Main dataset evaluation
    run_inference_on_sample(
        json_file_path   = "../data/formatted_data.json",
        output_file_path = "../outputs/llm_results.json",
        num_samples      = 20,
        random_sample    = True
    )

    # Edge-case scenarios evaluation
    run_inference_on_sample(
        json_file_path   = "../data/edge_cases.json",
        output_file_path = "../outputs/edge_cases_results.json",
        num_samples      = 3,
        random_sample    = False
    )