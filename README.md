# EcoRisk LLM — Ekolojik Yangın Riski Değerlendirme Sistemi

LLM tabanlı (Llama-3.1-8B/70B) gerçek bir **Retrieval-Augmented Generation (RAG)** pipeline kullanarak, Antalya bölgesindeki coğrafi, meteorolojik ve vejetasyon verilerinden otomatik orman yangını risk değerlendirme raporu üreten bir sistemdir.

Sistem, kural tabanlı (hardcoded) yaklaşımların ötesine geçerek, **FAISS** tabanlı bir vektör veritabanı ve **Sentence Transformers** (Semantic Search) kullanarak en ilgili ekolojik bilgileri dinamik olarak enjekte eder.

## Proje Yapısı

```
EcoRisk_LLM/
├── main.py                    # CSV → JSON dönüştürücü (veri hazırlık)
├── requirements.txt           # Gerekli Python kütüphaneleri
├── .env                       # API anahtarı (Git'e yüklenmez)
├── data/
│   ├── dataset_spatial_filter.csv   # Ham veri seti (5000+ kayıt)
│   ├── formatted_data.json          # JSON formatına dönüştürülmüş veri
│   ├── knowledge_base.json          # 26 parçalık bilimsel bilgi bankası
│   ├── faiss_index.bin              # FAISS vektör indeksi (otomatik oluşur)
│   ├── kb_metadata.json             # Vektör metadata (otomatik oluşur)
│   └── edge_cases.json              # Edge-case test senaryoları
├── src/
│   ├── build_vector_store.py  # Vektör veritabanını oluşturan script
│   ├── llm_inference.py       # FAISS RAG + LLM inference pipeline
│   ├── prompt_builder.py      # 4 seviyeli risk prompt oluşturucu
│   └── evaluate_metrics.py    # Değerlendirme metrikleri (Accuracy, F1, Exact-Match)
└── outputs/
    ├── llm_results.json       # Ana değerlendirme sonuçları (20 örnek)
    └── edge_cases_results.json # Edge-case test sonuçları
```

## Kurulum

### 1. Gereksinimler

- Python 3.10+
- Groq API anahtarı ([console.groq.com](https://console.groq.com) adresinden ücretsiz alınabilir)

### 2. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 3. API Anahtarını Ayarla

Proje kök dizininde `.env` dosyası oluşturun:
```
GROQ_API_KEY=gsk_XXXXXXXXXXXX
```

## Çalıştırma

### Adım 1: Vektör Veritabanını Oluşturma

Knowledge base'i (bilgi bankası) semantik arama için vektörlere dönüştürmek gerekir:

```bash
cd src
python build_vector_store.py
```
Bu işlem `data/` klasörü altında `faiss_index.bin` dosyasını oluşturacaktır.

### Adım 2: LLM Inference (RAG Pipeline)

FAISS tabanlı gerçek RAG inference çalıştırmak için:

```bash
python llm_inference.py
```

Bu komut:
1. Girdi verisini doğal dil sorgusuna çevirir.
2. FAISS ile bilgi bankasından en ilgili 3 ekolojik bilgiyi çeker.
3. Bu bilgileri prompta enjekte eder ve LLM'den rapor üretir.
4. Sonuçları `outputs/` klasörüne kaydeder.

### Adım 3: Değerlendirme Metrikleri

```bash
python evaluate_metrics.py
```

Üretilen metrikler:
- **Accuracy/F1**: Risk seviyesinin doğruluğu (Düşük/Orta/Yüksek).
- **Exact-Match Tutarlılık**: LLM'in sayısal verileri %100 doğrulukla koruma oranı.
- **ROUGE-L**: Dilsel akıcılık ve zenginlik göstergesi.

## Metodoloji: Real RAG

Sistem şu 4 adımlı pipeline'ı kullanır:

1. **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` modeli ile 26 ekolojik bilgi parçası 384 boyutlu vektörlere dönüştürülür.
2. **Retrieval (FAISS)**: Girdi verisi (Sıcaklık, Rüzgar, NDVI vb.) bir sorgu cümlesine çevrilir ve vektör uzayında en yakın (Cosine Similarity) 3 bilgi parçası bulunur.
3. **Prompt Augmentation**: Çekilen bilgiler ve 3-shot örnekler sisteme enjekte edilir.
4. **Generation**: Llama-3.1-8B (veya 3.3-70B) modeli bilimsel temelli bir rapor üretir.

## Gerekli Kütüphaneler

| Kütüphane | Amaç |
|---|---|
| `faiss-cpu` | Vektör veritabanı ve hızlı semantik arama |
| `sentence-transformers`| Metin embedding (vektörizasyon) |
| `groq` | Llama API erişimi |
| `rouge_score` | Dilsel başarı analizi |
| `pandas` | Veri işleme |
