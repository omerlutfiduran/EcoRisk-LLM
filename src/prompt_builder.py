import json

def build_llama_prompt(target_json_data, retrieved_rules_text=""):
    """
    Constructs the final prompt for Llama-3 including system instructions,
    few-shot examples, and RAG context.
    """

    system_prompt = """Sen, Akdeniz ekosistemi (ozellikle Antalya bolgesi) orman yangini dinamikleri uzerine uzmanlasmis kidemli bir ekolojik risk analistisin. Gorevin, sana JSON formatinda verilen cografi, meteorolojik ve yakit (NDVI, NDWI) verilerini analiz ederek kisa, net ve akademik bir risk degerlendirme metni olusturmaktir.

KESIN KURALLAR:
1. Sadece ve sadece sana verilen JSON verisindeki sayisal degerleri kullan. Asla disaridan bir istatistik uydurma.
2. Eger sana "EKOLOJIK BILGI BANKASI (RAG)" bilgileri verilmisse, raporunu yazarken mutlaka bu bilimsel gercekleri referans al ve metne yedir.
3. Analizini 3-5 cumleyi gecmeyecek sekilde, nesnel ve akademik bir dille raporla.
4. Raporunun SON CUMLESINDE asagidaki 4 seviyeden birini MUTLAKA acikca belirt:
   - "DUSUK RISK"
   - "ORTA RISK"
   - "YUKSEK RISK"
   - "COK YUKSEK RISK"

RISK BELIRLEMEDE REHBER ESIKLER:
- Sicaklik > 30 VE Ruzgar > 2.5 => en az YUKSEK RISK
- NDVI > 0.3 VE NDWI < 0 (kuru yakit var) => riski bir kademe artir
- Egim > 15 VE Ruzgar > 2 (baca etkisi) => riski bir kademe artir
- NDVI < 0.15 (yanacak bitki yok) => DUSUK RISK
- NDWI > 0.1 VE Sicaklik < 22 => DUSUK RISK"""

    few_shot_examples = """
--- ORNEK 1 (Low Risk) ---
GIRDI JSON:
{
 "Cografya": {"Yukseklik": 150.0, "Egim": 2.5},
 "Meteoroloji": {"Sicaklik": 18.5, "Ruzgar_Hizi": 0.5},
 "Yakitlar": {"NDVI": 0.65, "NDWI": 0.15}
}
CIKTI RAPOR:
Bolge, 150.0 metre rakimda ve %2.5 gibi dusuk bir egimde yer almaktadir. 18.5 derece sicaklik ve 0.5 m/s ruzgar hizi meteorolojik olarak stabil bir ortam sunarken, 0.65 NDVI ve 0.15 NDWI degerleri bitki ortusunun nemli ve canli oldugunu gostermektedir. Bu sayisal verilere dayanarak, alanin mevcut orman yangini riski DUSUK RISK seviyesindedir.

--- ORNEK 2 (Very High Risk) ---
GIRDI JSON:
{
 "Cografya": {"Yukseklik": 850.0, "Egim": 25.4},
 "Meteoroloji": {"Sicaklik": 32.8, "Ruzgar_Hizi": 5.2},
 "Yakitlar": {"NDVI": 0.22, "NDWI": -0.18}
}
CIKTI RAPOR:
Bolge, 850.0 metre rakimda ve %25.4 gibi oldukca dik bir egime sahip topografyada bulunmaktadir. 32.8 derece yuksek sicaklik ve 5.2 m/s siddetli ruzgar hizi, baca etkisi yaratarak yanginin yayilma hizini dramatik sekilde artiracaktir. 0.22 NDVI ve -0.18 NDWI degerleri bitki ortusundeki ciddi kurakligi ve su stresini kanitlamaktadir. Tum bu parametreler bir arada degerlendirildiginde, bolgede COK YUKSEK RISK seviyesinde orman yangini tehlikesi bulunmaktadir.
"""

    rag_section = ""
    if retrieved_rules_text:
        rag_section = f"\n--- EKOLOJIK BILGI BANKASI (RAG) ---\nReferans alacagin ekolojik bilgiler:\n{retrieved_rules_text}\n"

    target_prompt = f"""
--- ASIL ANALIZ ---
Asagidaki verileri RAG bilgileriyle sentezle. Son cumlede risk seviyesini belirt.
{rag_section}
GIRDI JSON:
{json.dumps(target_json_data, indent=1)}
CIKTI RAPOR:
"""

    return f"{system_prompt}\n{few_shot_examples}\n{target_prompt}"