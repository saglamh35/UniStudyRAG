"""
UniStudyRAG - Configuration Management
Tüm sabitler ve environment variable yönetimi
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


class Config:
    """
    Uygulama yapılandırma sınıfı.
    Tüm sabitler ve environment variable'lar burada yönetilir.
    """
    
    # Model Ayarları
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemma3:4b")
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "llama3.2-vision")
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # RAG Ayarları
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "8"))
    
    # Vision Ayarı
    ENABLE_VISION: bool = os.getenv("ENABLE_VISION", "true").lower() == "true"
    
    # Path Ayarları
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    CHROMA_DB_DIR: Path = Path(os.getenv("CHROMA_DB_DIR", "chroma_db"))
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "cache_data"))
    
    # System Prompt (Elite Version - Gizli Düşünce Zinciri)
    SYSTEM_PROMPT: str = """SEN ELİT BİR AKADEMİK ASİSTANSIN. Görevin, verilen bağlamı (Context) analiz ederek kullanıcıya NET, DOĞRU ve AKICI bir cevap vermektir.

### ÇALIŞMA PRENSİBİ (İÇSEL SÜREÇ - BUNU ÇIKTIYA YAZMA!):

1. Önce sorunun dilini ve niyetini anla.

2. Bağlamdaki metinleri ve [GÖRSEL ANALİZİ] etiketli kısımları tara. Başlıkları, üniversite isimlerini ve şekil altı yazılarını kaçırma.

3. Bilgileri sentezle. Eğer bağlamda bilgi yoksa dürüstçe "Bilmiyorum" de.

### GÖRSEL VERİ KULLANIMI:

Bağlam içinde `--- OCR START ---` ve `--- VISUAL DESC START ---` bloklarını görürsen, bunları o sayfanın **kesin ve doğru** içeriği olarak kabul et. Özellikle 'Bu belge ne hakkında?' gibi sorularda OCR kısmındaki başlıklara ve kurum isimlerine öncelik ver.

### ÇIKTI KURALLARI (KULLANICIYA GÖRÜNEN):

* **DOĞRUDAN CEVAP VER:** "Analiz ediyorum...", "Sentez yapıyorum..." gibi ara adımları ASLA yazma. Direkt konuya gir.

* **DİL KİLİDİ:** Kullanıcı Türkçe sorduysa %100 Türkçe, İngilizce sorduysa İngilizce cevap ver.

* **TERİM KORUMA:** Teknik terimleri (Örn: 'Flip-Flop', 'Gleitkommazahlen') orijinal haliyle parantez içinde koru.

* **YAPISAL:** Cevabı maddeler halinde veya kısa paragraflarla sun. Önemli yerleri **kalın** yaz."""
    
    # LLM Ayarları
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "4096"))  # Context window
    
    # Retriever Ayarları (MMR)
    RETRIEVER_SEARCH_TYPE: str = "mmr"
    RETRIEVER_FETCH_K: int = int(os.getenv("RETRIEVER_FETCH_K", "20"))
    RETRIEVER_LAMBDA_MULT: float = float(os.getenv("RETRIEVER_LAMBDA_MULT", "0.6"))
    
    @classmethod
    def get_vision_prompt(cls) -> str:
        """
        Vision modeli için kullanılan prompt (Çok Katmanlı Analiz - Aggressive OCR & Detail).
        
        Returns:
            str: Vision analiz prompt'u
        """
        return """Analyze this document image with EXTREME DETAIL.

PHASE 1: OCR (TEXT EXTRACTION)
- Transcribe ALL visible text exactly as it appears.
- Pay special attention to: HEADERS, TITLES, UNIVERSITY NAMES, COURSE CODES, and FOOTERS.
- Do not summarize the text, just extract it.

PHASE 2: VISUAL ANALYSIS
- If this is a diagram/chart: Describe every node, arrow, label, and connection. Explain the flow.
- If this is a slide: Describe the layout and any visual hierarchy.

OUTPUT FORMAT:

--- OCR START ---
[Insert Extracted Text Here]
--- OCR END ---

--- VISUAL DESC START ---
[Insert Detailed Visual Description Here]
--- VISUAL DESC END ---
"""

