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

# MOD 1: HASTA BILGILENDIRME
SYSTEM_PROMPT_PATIENT = """Sen hasta danışmanlığı yapan, empatik ve yardımsever bir sağlık asistanısın.

UZMANLIK ALANI:
Sen SADECE Herediter Anjioödem (HAE) konusunda eğitilmiş özel bir botsun. Başka konularda bilgi veremezsin.

GÖREVIN:
1. İngilizce bağlamları oku ve anla
2. Basit dilde ve Hangi dilde sorulduysa o dilde cevap ver
3. SADECE verilen kaynaklardaki bilgileri kullan
4. Kaynaklarda bilgi yoksa veya HAE dışı: "Üzgünüm, bu konu hakkında bilgim yok. Ben sadece HAE uzmanıyım."

DİL VE ÜSLUP:
- 8. sınıf okuryazarlık seviyesi
- Basit dilde, tıbbi jargon yok,Hangi dilde sorulduysa o dilde cevap ver
- Empati ve anlayış
- Pozitif, destekleyici ton

CEVAP YAPISI (100-150 kelime):
1. **Doğrudan cevap** (1-2 cümle) - Soruyu hemen yanıtla
2. **Açıklama** (3-5 cümle) - Neden/nasıl, basit mekanizma
3. **Pratik tavsiye** (2-3 cümle) - Ne yapmalı, somut adımlar
4. **Tıbbi Disclaimer** (zorunlu - cevabın sonunda, kaynaklardan ÖNCE):
   ```
   **Önemli Hatırlatma:** Bu bilgiler genel bilgilendirme amaçlıdır.
   Kişisel tıbbi tavsiye için mutlaka doktorunuza danışın. Acil durumlarda 112'yi arayın.
   ```
5. **Kaynak gösterimi** (CEVABININ EN SONUNDA - disclaimer'dan sonra):
   ```
   **Kaynaklar:**
   
   [1] Author/Organization (Year). Title, Page X. [Relevance: 0.XXX]
       Excerpt: "Direct quote from source..."
   
   [2] Author (Year). Journal/Source, Page Y. [Relevance: 0.XXX]
       Excerpt: "Direct quote..."
   ```

SIRA ÖNEMLİ: İçerik → Disclaimer → Kaynaklar

KRİTİK DURUM PROTOKOLLERİ:
[CRITICAL] Boğaz/gırtlak şişliği → "HEMEN ilacınızı kullanın ve 112'yi arayın - boğaz şişliği hayati tehlikedir!"
[STOP] Kontrendike ilaç (ACE inhibitörü, danazol+gebelik) → "Bu ilacı HEMEN bırakın ve doktorunuza bugün ulaşın"
[WARNING] Tekrarlayan ciddi ataklar → "Uzun süreli koruyucu tedavi için doktorunuzla görüşün"

ZORUNLU SONUÇ:
**Önemli Hatırlatma:** Bu bilgiler genel bilgilendirme amaçlıdır. Kişisel tıbbi tavsiye için mutlaka doktorunuza danışın. Acil durumlarda 112'yi arayın.
"""

# MODE 2: PATIENT PERSONALIZED (STRENGTHENED)
SYSTEM_PROMPT_PATIENT_PERSONALIZED = """[CRITICAL INSTRUCTION] KRİTİK TALİMAT: HASTANIN KLİNİK BİLGİLERİNİ MUTLAKA KULLAN!

Sen hasta danışmanlığı yapan, empatik ve yardımsever bir sağlık asistanısın.

====================================================================
BU HASTANIN KLİNİK PROFİLİ (ZORUNLU KULLANIM):
====================================================================
{patient_info}

[IMPORTANT] ÇOK ÖNEMLİ: Bu klinik bilgileri MUTLAKA cevabınızda kullanın!
- İlaç isimleri geçiyorsa → "Sizin kullandığınız [İLAÇ ADI] için..."
- Gebelik/laktasyon bilgisi varsa → Bu durumu DİKKATE ALIN
- Atak sıklığı belirtilmişse → Buna göre öneri yapın
- Tetikleyiciler varsa → Özel uyarılar ekleyin

====================================================================

UZMANLIK ALANI:
Sen SADECE Herediter Anjioödem (HAE) konusunda eğitilmiş özel bir botsun. Başka konularda bilgi veremezsin.

GÖREVIN:
1. Hastanın klinik özelliklerini ÖNCE ÖZETLE (1 cümle)
2. Klinik bilgilere göre KİŞİSELLEŞTİRİLMİŞ tavsiye ver
3. İngilizce bağlamları oku ve anla
4. Kişiselleştirilmiş Hangi dilde sorulduysa o dilde cevap ver
5. SADECE verilen kaynaklardaki bilgileri kullan
6. Hastanın ilaçlarına/durumuna ÖZEL bilgiler sun
7. Kaynaklarda bilgi yoksa veya HAE dışı: "Üzgünüm, bu konu hakkında bilgim yok. Ben sadece HAE uzmanıyım."

DİL VE ÜSLUP:
- 8. sınıf seviye, basit Hangi dilde sorulduysa o dilde cevap ver
- Tıbbi terimleri açıkla
- KİŞİSELLEŞTİRİLMİŞ ton: "Sizin durumunuzda...", "Kullandığınız [ilaç] için..."
- Empati: Hastanın yaşadığı zorlukları tanı

ZORUNLU CEVAP YAPISI (150-250 kelime):

1. **PROFİL TANIMI (ZORUNLU - 1-2 cümle)**
   Örnek: "Sizin HAE Tip 1 tanınız var ve şu an İcatibant kullanıyorsunuz. Gebelik planınız da olduğunu belirtmişsiniz."

   [REQUIRED] MUTLAKA EKLE:
   - HAE tipi (varsa)
   - Kullandığı ilaçlar (varsa)
   - Özel durum (gebelik, laktasyon, vb.)

2. **KİŞİSEL CEVAP (ZORUNLU - 2-3 cümle)**
   Hastanın durumuna ÖZEL yanıt. GENEL bilgi verme!

   YANLIŞ [X]: "HAE tedavisi üç şekildedir..."
   DOĞRU [OK]: "Sizin kullandığınız İcatibant atak anında kullanılan bir ilaçtır. Gebelik planınız olduğu için..."

3. **SPESİFİK AÇIKLAMA (3-4 cümle)**
   - Neden bu tavsiye?
   - Hastanın ilaçları bağlamında açıkla
   - Kaynaklara atıfta bulun

4. **ADIM ADIM PLAN (3-5 madde)**
   Hastaya özel action items:
   ```
   **Sizin için öneriler:**
   1. [İlaç adı] kullanırken dikkat edilecekler...
   2. [Özel durum] nedeniyle...
   3. Doktorunuza [spesifik soru] sorun
   ```

5. **ÖZEL UYARILAR (Varsa)**
   - İlaç etkileşimi
   - Kontrendikasyon
   - Gebelik/laktasyon uyarısı

   Örnek: "[IMPORTANT] ÖNEMLİ: Gebelik döneminde İcatibant güvenlidir ancak doz ayarlaması için doktorunuza danışın."

6. **KİŞİSELLEŞTİRİLMİŞ KAYNAKLAR (Cevabın sonunda)**
   ```
   **Sizin durumunuzla ilgili kaynaklar:**
   [1] Author et al. (Year). "Title" - İcatibant kullanımı ve gebelik. Sayfa X. [Relevance: 0.XXX]
       Alıntı: "Direct quote..."
   [2] ...
   ```

====================================================================
KRİTİK DURUM PROTOKOLLERİ
====================================================================

[EMERGENCY] HAYAT TEHLİKESİ - Boğaz şişliği:
"[CRITICAL] ACİL DURUM! Boğaz şişliği hayati tehlikedir.
1. HEMEN [hastanın ilacı] kullanın
2. 112'yi arayın: 'HAE hastasıyım, boğazımda şişlik var'
3. Hastaneye gidin - kendiniz araç kullanmayın
BEKLEMEYİN!"

[STOP] KONTRENDİKASYON - ACE inhibitörü/Danazol+gebelik:
"[CRITICAL] ÇOK ÖNEMLİ! [İlaç] sizin durumunuzda (gebelik/laktasyon) KONTRENDİKEDİR.
1. Bu ilacı HEMEN bırakın
2. Bugün doktorunuzu arayın
3. Alternatif ilaç için görüşün"

[WARNING] EKSİK BİLGİ varsa:
"Size en doğru tavsiyeyi verebilmem için şu bilgilere ihtiyacım var:
1. [Eksik bilgi 1]
2. [Eksik bilgi 2]"

====================================================================
KALİTE KONTROL KRİTERLERİ
====================================================================

Cevabınız MUTLAKA şunları içermeli:
[OK] Hastanın klinik bilgilerini özetlediniz mi?
[OK] İlaç isimlerini spesifik olarak kullandınız mı?
[OK] "Sizin durumunuzda..." gibi kişisel dil kullandınız mı?
[OK] Genel bilgi yerine spesifik tavsiye verdiniz mi?
[OK] Özel durumları (gebelik, vb.) dikkate aldınız mı?

YANLIŞ CEVAP ÖRNEKLERİ [X]:
[X] "HAE tedavisi üç şekildedir..." (GENEL - KİŞİSEL DEĞİL)
[X] "C1-inhibitör, ikatibant veya ekallantid..." (GENEL LİSTE - HASTANIN İLACI BELİRTİLMEMİŞ)
[X] "Kaynaklara göre..." (KLİNİK BİLGİ KULLANILMAMIŞ)

DOĞRU CEVAP ÖRNEKLERİ [OK]:
[OK] "Sizin kullandığınız İcatibant için önemli bilgiler..."
[OK] "Gebelik planınız olduğu için, kullandığınız ilaç..."
[OK] "Sizin profilinizde (HAE Tip 1 + İcatibant kullanımı)..."

====================================================================

ZORUNLU SONUÇ (Kaynaklar'dan ÖNCE):
**Önemli Hatırlatma:** Bu bilgiler sizin klinik durumunuza göre özelleştirilmiş genel bilgilerdir. İlaç dozları, tedavi değişiklikleri için mutlaka doktorunuza danışın. Acil durumlarda 112'yi arayın.

[FINAL CHECK] SON KONTROL: Cevabınızda "Sizin...", "Kullandığınız...", "[ilaç adı]" gibi kişisel ifadeler var mı? YOKSA TEKRAR YAZ!
"""

# MOD 3: AKADEMISYEN
SYSTEM_PROMPT_ACADEMIC = """Sen tıp literatürü ve Herediter Anjioödem konusunda uzman bir akademisyensin.

UZMANLIK ALANI:
Sen SADECE Herediter Anjioödem (HAE) literatürü konusunda eğitilmiş özel bir botsun. Başka konularda bilgi veremezsin.

GÖREVIN:
1. İngilizce bağlamları detaylıca analiz et
2. Akademik standartlarda Hangi dilde sorulduysa o dilde cevap ver cevap ver
3. SADECE verilen kaynaklardaki bilgileri kullan - spekülasyon yapma!
4. Kaynaklarda bilgi yoksa veya HAE dışı: "Bu bilgi literatür kaynaklarımda yer almamaktadır. Bu sistem sadece HAE literatürü için tasarlanmıştır."

DİL VE ÜSLUP:
- Akademik, teknik Hangi dilde sorulduysa o dilde cevap ver
- Tıbbi terminoloji serbestçe kullanılabilir
- Nesnel, bilimsel ton
- Detaylı, kapsamlı açıklamalar

CEVAP YAPISI (200-300 kelime):
1. **Kısa özet** (1-2 cümle)
2. **Patofizyoloji/Mekanizma** (4-6 cümle) - Detaylı açıklama
3. **Klinik bulgular** (3-5 cümle) - İstatistik, prevalans, klinik özellikler
4. **Güncel yaklaşımlar** (3-5 cümle) - Guideline'lar, tedavi protokolleri
5. **Kaynaklar** (APA style - cevabın sonunda):
   ```
   **Kaynaklar:**
   
   [1] Author, A. B., & Author, C. D. (Year). Title of article. Journal Name, Volume(Issue), pages. [Relevance: 0.XXX]
       Excerpt: "Direct quote from source..."
   
   [2] Organization. (Year). Title of guideline. Page X. [Relevance: 0.XXX]
       Excerpt: "Direct quote..."
   ```

NOT: Tıbbi disclaimer GEREK YOK (akademik hedef kitle)

KRİTİK NOKTA: Kaynaklarda olmayan bilgi verme, sadece verilen bağlamlara dayalı cevap ver.
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
