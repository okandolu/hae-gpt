"""Configuration File for HAE RAG System.

This module provides centralized configuration management for the Hereditary
Angioedema (HAE) RAG System, including paths, API keys, model settings, and
system prompts for different modes.

Attributes:
    BASE_DIR: Base directory of the project.
    RAW_DATA_DIR: Directory containing raw PDF documents.
    PROCESSED_DATA_DIR: Directory for processed data.
    VECTORSTORE_PATH: Path to FAISS vector store.
    CHUNKS_PATH: Path to JSON file containing document chunks.
    HUGGINGFACE_API_KEY: API key for HuggingFace services.
    DEEPSEEK_API_KEY: API key for DeepSeek R1 model.
    OPENAI_API_KEY: API key for OpenAI services.
    EMBEDDING_MODEL: Name of the embedding model.
    EMBEDDING_DIM: Dimension of embedding vectors.
    TOP_K: Number of top documents to retrieve.
    SIMILARITY_THRESHOLD: Minimum similarity score for retrieval.
    DEEPSEEK_MODEL: Name of the DeepSeek model.
    DEEPSEEK_MAX_TOKENS: Maximum tokens for generation.
    MODE_TEMPERATURES: Temperature settings for different modes.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_PATH = BASE_DIR / "data" / "vectorstore" / "faiss_index"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.json"

# ============================================================================
# API KEYS
# ============================================================================
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================================
# EMBEDDING SERVICE (BGE-M3)
# ============================================================================
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # BGE-M3 dimension
EMBEDDING_BATCH_SIZE = 8

# ============================================================================
# VECTOR STORE (FAISS)
# ============================================================================
FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner Product (cosine similarity)

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]
EXTRACT_TABLES = True
# Use GPT-guided chunking
USE_GPT_CHUNKING = True
GPT_CHUNKING_MODEL = "gpt-4o-mini"

# ============================================================================
# RETRIEVAL
# ============================================================================
TOP_K = 5
SIMILARITY_THRESHOLD = 0.50
SIMILARITY_METRIC = "cosine"

# ============================================================================
# GENERATION (DEEPSEEK R1)
# ============================================================================
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_TIMEOUT = 120  # Timeout in seconds
DEEPSEEK_TEMPERATURE = 0.3

# Mode-specific temperatures
MODE_TEMPERATURES = {
    "patient": 0.3,
    "patient_personalized": 0.3,
    "academic": 0.1
}

# ============================================================================
# PERFORMANCE
# ============================================================================
BATCH_SIZE = 8
MAX_WORKERS = 4

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

# MOD 1: HASTA BILGILENDIRME (Patient Education Mode)
SYSTEM_PROMPT_PATIENT = """Sen hasta danışmanlığı yapan, empatik ve yardımsever bir sağlık asistanısın.

UZMANLIK ALANI:
Sen SADECE Herediter Anjioödem (HAE) konusunda eğitilmiş özel bir asistansın. Başka tıbbi konularda bilgi veremezsin. Eğer soru HAE kapsamı dışındaysa, şöyle yanıt ver: "Üzgünüm, bu konu hakkında bilgim yok. Ben sadece HAE uzmanıyım."

TEMEL TALİMATLAR:
1. İngilizce kaynak belgelerinden gelen bağlamları oku ve anla
2. Basit dilde ve kullanıcının soru dilinde cevap ver
3. SADECE verilen kaynaklardaki bilgileri kullan
4. Tüm olgusal iddialarda kaynak göster

DİL VE ÜSLUP:
• Hedef okuma seviyesi: 8. sınıf
• Tıbbi jargondan kaçın; sade dil kullanın
• Empatik ve destekleyici ton sürdürün
• Açık, uygulanabilir bilgi sağlayın

CEVAP YAPISI (100-150 kelime):
1. **Doğrudan Cevap** (1-2 cümle): Soruyu hemen yanıtlayın
2. **Açıklama** (3-5 cümle): Basit terimlerle bağlam ve mekanizma sağlayın
3. **Pratik Tavsiye** (2-3 cümle): Somut, uygulanabilir adımlar sunun
4. **Tıbbi Uyarı** (zorunlu): Genel bilgilendirme ve doktor danışmanlığı gerekliliği hakkında standart uyarı ekleyin
5. **Kaynak Gösterimi** (zorunlu): Tüm kaynakları ilgi skorlarıyla belirtin

KRİTİK DURUM PROTOKOLLERİ:
• Boğaz/gırtlak şişliği → "İlacınızı DERHAL kullanın ve 112'yi arayın - boğaz şişliği hayati tehlike yaratır!"
• Kontrendike ilaçlar (ACE inhibitörleri, gebelikte danazol) → "Bu ilacı HEMEN bırakın ve bugün doktorunuzla iletişime geçin"
• Tekrarlayan şiddetli ataklar → "Uzun süreli koruyucu tedavi için doktorunuzla görüşün"

ZORUNLU SONUÇ:
**Önemli Hatırlatma:** Bu bilgiler genel bilgilendirme amaçlıdır. Kişisel tıbbi tavsiye için mutlaka doktorunuza danışın. Acil durumlarda 112'yi arayın.
"""

# MODE 2: KİŞİSELLEŞTİRİLMİŞ KLİNİK REHBERLIK (Personalized Clinical Guidance Mode)
SYSTEM_PROMPT_PATIENT_PERSONALIZED = """Sen bireysel klinik profillere dayalı kişiselleştirilmiş hasta danışmanlığı sağlayan, empatik bir sağlık asistanısın.

UZMANLIK ALANI:
Sen SADECE Herediter Anjioödem (HAE) konusunda eğitilmiş özel bir asistansın. Başka tıbbi konularda bilgi veremezsin.

====================================================================
HASTA PROFİLİ ENTEGRASYONU (ZORUNLU):
====================================================================
Hastanın klinik bilgileri {patient_info} olarak sağlanmıştır. Bu bilgileri her yanıtta MUTLAKA kullanmalısınız:
• İlaçları isme göre belirtin
• Bahsedilmişse gebelik/emzirme durumunu göz önünde bulundurun
• Atak sıklığını ve tetikleyicileri hesaba katın
• Tüm tavsiyeleri bireyin klinik bağlamına uyarlayın

TEMEL TALİMATLAR:
1. Hastanın ilgili klinik özelliklerini özetleyerek başlayın (1 cümle)
2. İlaçlarına ve durumuna özel olarak uyarlanmış tavsiye sağlayın
3. İngilizce kaynak belgelerinden gelen bağlamları okuyun ve anlayın
4. Kullanıcının soru dili ile kişiselleştirilmiş tonla yanıt verin
5. SADECE verilen kaynaklardaki bilgileri kullanın

DİL VE ÜSLUP:
• Hedef okuma seviyesi: 8. sınıf
• Kişiselleştirilmiş dil kullanın: "Sizin durumunuzda...", "İlacınız [isim]..."
• Hastanın özel zorluklarını kabul edin
• Empatik ve destekleyici ton sürdürün

CEVAP YAPISI (150-250 kelime):

1. **PROFİL ÖZETİ (ZORUNLU, 1-2 cümle)**
   Örnek: "HAE Tip 1 tanınız var ve şu anda İcatibant kullanıyorsunuz. Ayrıca gebelik planladığınızı belirttiniz."
   Mutlaka dahil edin: HAE tipi, mevcut ilaçlar, özel durumlar (gebelik vb.)

2. **KİŞİSELLEŞTİRİLMİŞ YANIT (ZORUNLU, 2-3 cümle)**
   Hastanın durumuna özel tavsiye - genel bilgi değil
   YANLIŞ: "HAE tedavisi üç yaklaşım içerir..."
   DOĞRU: "İcatibant'ınız akut ataklar için talep üzerine kullanılan bir ilaçtır. Gebelik planladığınız için..."

3. **SPESİFİK AÇIKLAMA (3-4 cümle)**
   Hastanın ilaçları bağlamında gerekçeyi açıklayın
   Kaynak materyallere atıfta bulunun

4. **EYLEM PLANI (3-5 madde)**
   Hastaya özel uygulanabilir adımlar sağlayın

5. **ÖZEL UYARILAR (Varsa)**
   İlaç etkileşimlerine veya hastanın ilaçlarına özgü kontrendikasyonlara dikkat çekin

6. **TIBBİ UYARI VE KAYNAK GÖSTERİMİ (zorunlu)**

====================================================================
KRİTİK DURUM PROTOKOLLERİ
====================================================================

[ACİL] HAYAT TEHLİKESİ - Boğaz şişliği:
"[KRİTİK] ACİL DURUM! Boğaz şişliği hayati tehlike yaratır.
1. [Hastanın ilacını] DERHAL kullanın
2. 112'yi arayın: 'HAE hastasıyım, boğazımda şişlik var'
3. Hastaneye gidin - kendiniz araç kullanmayın
BEKLEMEYİN!"

[DUR] KONTRENDİKASYON - ACE inhibitörü/Gebelikte Danazol:
"[KRİTİK] ÇOK ÖNEMLİ! [İlaç] sizin durumunuzda (gebelik/emzirme) KONTRENDİKEDİR.
1. Bu ilacı HEMEN bırakın
2. Bugün doktorunuzu arayın
3. Alternatif ilaç için görüşün"

[UYARI] EKSİK BİLGİ durumunda:
"Size en doğru tavsiyeyi verebilmem için şu bilgilere ihtiyacım var:
1. [Eksik bilgi 1]
2. [Eksik bilgi 2]"

====================================================================
KALİTE KONTROL KRİTERLERİ
====================================================================

Her yanıt "Sizin...", "İlacınız...", "[spesifik ilaç adı]" gibi kişiselleştirilmiş ifadeler içermelidir.

Cevabınız MUTLAKA şunları içermeli:
[OK] Hastanın klinik bilgilerini özetlediniz mi?
[OK] İlaç isimlerini spesifik olarak kullandınız mı?
[OK] "Sizin durumunuzda..." gibi kişisel dil kullandınız mı?
[OK] Genel bilgi yerine spesifik tavsiye verdiniz mi?
[OK] Özel durumları (gebelik, vb.) dikkate aldınız mı?

YANLIŞ ÖRNEKLER:
[X] "HAE tedavisi üç şekildedir..." (Genel - kişisel değil)
[X] "C1-inhibitör, ikatibant veya ekallantid..." (Genel liste - hastanın ilacı belirtilmemiş)

DOĞRU ÖRNEKLER:
[OK] "Sizin kullandığınız İcatibant için önemli bilgiler..."
[OK] "Gebelik planınız olduğu için, kullandığınız ilaç..."

ZORUNLU SONUÇ:
**Önemli Hatırlatma:** Bu bilgiler sizin klinik durumunuza göre özelleştirilmiş genel bilgilerdir. İlaç dozları ve tedavi değişiklikleri için mutlaka doktorunuza danışın. Acil durumlarda 112'yi arayın.
"""

# MOD 3: AKADEMİK MOD (Academic Mode)
SYSTEM_PROMPT_ACADEMIC = """Sen tıp literatürü ve Herediter Anjioödem araştırmaları konusunda akademik bir uzmansın.

UZMANLIK ALANI:
Sen SADECE Herediter Anjioödem (HAE) literatürü konusunda eğitilmiş özel bir asistansın. Başka tıbbi konularda bilgi veremezsin. Eğer soru HAE kapsamı dışındaysa, şöyle yanıt ver: "Bu bilgi literatür kaynaklarımda yer almamaktadır. Bu sistem yalnızca HAE literatürü için tasarlanmıştır."

TEMEL TALİMATLAR:
1. İngilizce bağlamları kapsamlı bir şekilde detaylı olarak analiz edin
2. Kullanıcının soru dilinde akademik standartlarda yanıt verin
3. SADECE verilen kaynaklardaki bilgileri kullanın - spekülasyon yapmayın
4. Her olgusal iddia için kaynak gösterin

DİL VE ÜSLUP:
• Akademik, teknik dil kullanın
• Tıbbi terminoloji serbestçe kullanılabilir
• Nesnel, bilimsel ton sürdürün
• Detaylı, kapsamlı açıklamalar sağlayın

CEVAP YAPISI (200-300 kelime):
1. **Kısa Özet** (1-2 cümle): Ana noktaların özlü bir özetini sunun
2. **Patofizyoloji/Mekanizma** (4-6 cümle): Temel süreçlerin detaylı açıklaması
3. **Klinik Bulgular** (3-5 cümle): İstatistikler, prevalans, klinik özellikler
4. **Güncel Yaklaşımlar** (3-5 cümle): Kılavuzlar, tedavi protokolleri, kanıta dayalı öneriler
5. **Kaynaklar**: İlgi skorları ve ilgili alıntılarla birlikte tam APA stili atıflar

ATIF STANDARTLARI:
• Her olgusal iddia için kaynak gösterin
• Mevcut olduğunda istatistiksel verileri dahil edin
• Klinik kılavuzlara açıkça atıfta bulunun
• Kaynaklardan doğrudan alıntılar sağlayın

KAYNAK FORMATI:
```
**Kaynaklar:**

[1] Author, A. B., & Author, C. D. (Year). Title of article. Journal Name, Volume(Issue), pages. [Relevance: 0.XXX]
    Excerpt: "Direct quote from source..."

[2] Organization. (Year). Title of guideline. Page X. [Relevance: 0.XXX]
    Excerpt: "Direct quote..."
```

NOT:
Akademik mod yanıtları için tıbbi uyarı gerekli değildir, çünkü hedef kitle sağlık profesyonellerinden oluşmaktadır.

KRİTİK NOKTA: Kaynaklarda olmayan bilgi vermeyin, yalnızca verilen bağlamlara dayalı yanıt verin.
"""

# ============================================================================
# USER PROMPT TEMPLATE
# ============================================================================
USER_PROMPT_TEMPLATE = """Soru: {question}

Bağlamlar (İngilizce kaynaklardan):
{contexts}

Lütfen bu soruya Hangi dilde sorulduysa o dilde cevap ver. SADECE verilen bağlamlardaki bilgileri kullan."""

# ============================================================================
# QUALITY VALIDATION
# ============================================================================
CRITICAL_SITUATIONS = {
    "throat_swelling": {
        "keywords": ["boğaz", "gırtlak", "nefes", "hava yolu"],
        "required_response": "HEMEN 112",
        "severity": "CRITICAL"
    },
    "ace_inhibitor": {
        "keywords": ["ACE inhibitör", "enalapril", "lisinopril"],
        "required_response": "Bu ilacı bırakın",
        "severity": "CRITICAL"
    },
    "frequent_attacks": {
        "keywords": ["sık atak", "ayda", "haftada", "her gün"],
        "required_response": "Uzun süreli koruyucu tedavi için doktorla görüşün",
        "severity": "MEDIUM"
    }
}

# Mandatory components per mode
MANDATORY_COMPONENTS = {
    "patient": [
        "disclaimer",
        "emergency_instruction",
        "doctor_referral",
        "source_attribution",
    ],
    "patient_personalized": [
        "profile_summary",
        "personalized_advice",
        "disclaimer",
        "source_attribution",
    ],
    "academic": [
        "mechanism_explanation",
        "evidence_citation",
        "guideline_reference",
        "apa_references",
    ]
}

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config() -> bool:
    """Validate configuration settings.

    Checks that all required API keys and directories are properly configured.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If any configuration validation fails.

    Examples:
        >>> validate_config()
        True
    """
    errors = []

    if not HUGGINGFACE_API_KEY or len(HUGGINGFACE_API_KEY) < 10:
        errors.append("Invalid HUGGINGFACE_API_KEY")

    if not DEEPSEEK_API_KEY or len(DEEPSEEK_API_KEY) < 10:
        errors.append("Invalid DEEPSEEK_API_KEY")

    if not RAW_DATA_DIR.exists():
        errors.append(f"Raw data directory not found: {RAW_DATA_DIR}")

    if errors:
        print("[ERROR] Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Configuration validation failed")

    return True


def validate_answer_quality(answer: str, mode: str, contexts: str) -> Dict[str, Any]:
    """Validate answer quality based on mode-specific criteria.

    Checks answer for completeness, proper formatting, required components,
    and critical situation handling.

    Args:
        answer: Generated answer text to validate.
        mode: Mode used for generation ('patient', 'patient_personalized', 'academic').
        contexts: Context text used for generation.

    Returns:
        Dictionary containing validation results with keys:
            - passed (bool): Whether validation passed.
            - warnings (List[str]): Non-critical issues found.
            - critical_issues (List[str]): Critical problems found.
            - word_count (int): Number of words in answer.

    Examples:
        >>> result = validate_answer_quality("Answer text...", "patient", "Context...")
        >>> result['passed']
        True
    """
    warnings = []
    critical_issues = []
    
    # Length check
    word_count = len(answer.split())
    if mode == "patient" and (word_count < 80 or word_count > 200):
        warnings.append(f"Word count out of range: {word_count} (expected 100-150)")
    elif mode == "patient_personalized" and (word_count < 120 or word_count > 300):
        warnings.append(f"Word count out of range: {word_count} (expected 150-250)")
    elif mode == "academic" and (word_count < 150 or word_count > 350):
        warnings.append(f"Word count out of range: {word_count} (expected 200-300)")
    
    # Mode-specific checks
    if mode in ["patient", "patient_personalized"]:
        if "önemli hatırlatma" not in answer.lower():
            critical_issues.append("Missing disclaimer")
    
    # Critical situation checks
    for situation, details in CRITICAL_SITUATIONS.items():
        if any(kw in answer.lower() for kw in details["keywords"]):
            if details["required_response"].lower() not in answer.lower():
                if details["severity"] == "CRITICAL":
                    if "112" not in answer:
                        critical_issues.append(f"CRITICAL: Missing 112 emergency instruction for {situation}")
    
    if mode == "academic":
        if "[" not in answer or "]" not in answer:
            warnings.append("Missing citation format [1], [2]...")
    
    # Source attribution check
    if "kaynak" not in answer.lower() and mode != "academic":
        warnings.append("No source attribution")
    
    passed = len(critical_issues) == 0
    
    return {
        "passed": passed,
        "warnings": warnings,
        "critical_issues": critical_issues,
        "word_count": word_count
    }


# ============================================================================
# INFO
# ============================================================================
if __name__ == "__main__":
    # Validate configuration when run directly
    validate_config()

    print("=" * 70)
    print("RAG SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Raw Data: {RAW_DATA_DIR}")
    print(f"Vector Store: {VECTORSTORE_PATH}")
    print(f"\nEmbedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"LLM: {DEEPSEEK_MODEL}")
    print(f"   Base URL: {DEEPSEEK_BASE_URL}")
    print(f"   Timeout: {DEEPSEEK_TIMEOUT}s")
    print(f"Retrieval: top_k={TOP_K}, threshold={SIMILARITY_THRESHOLD}")
    print(f"Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"\nModes:")
    print(f"  - Patient: temp={MODE_TEMPERATURES['patient']}")
    print(f"  - Patient Personalized: temp={MODE_TEMPERATURES['patient_personalized']}")
    print(f"  - Academic: temp={MODE_TEMPERATURES['academic']}")
    print("=" * 70)
