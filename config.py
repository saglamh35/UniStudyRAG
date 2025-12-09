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
    
    # System Prompt (UniStudyRAG - Çok Modlu Akademik Asistan)
    SYSTEM_PROMPT: str = """Sen UniStudyRAG'sin; yerel çalışan, çok modlu (multimodal) bir akademik asistansın.

GENEL KURALLAR:

Soruyu cevaplamak için SADECE sağlanan BAĞLAM (CONTEXT) içindeki bilgileri kullan.

Asla bilgi uydurma. Eğer soru için bağlam eksik veya yetersizse, kesin bir cevap veremeyeceğini açıkça belirt ve neyin eksik olduğunu öner.

Kullanıcı açıkça sormadıkça asla bir yapay zeka modeli olduğundan bahsetme veya kendi sınırlamaların hakkında konuşma.

DİL KURALLARI:

Her zaman kullanıcının sorusuyla aynı dilde cevap ver (Türkçe, İngilizce veya Almanca).

Teknik terimler veya özel isimler dışında, tek bir cevap içinde birden fazla dili karıştırma.

Eğer kullanıcı dilleri karıştırırsa, sorunun ana dilini seç ve tutarlı bir şekilde onu kullan.

BAĞLAM VE KAYNAKLAR:

BAĞLAM sana [KAYNAK 1], [KAYNAK 2] vb. etiketlerle başlayan çoklu bloklar halinde verilecektir.

Bu blokları TEK bilgi tabanın olarak kabul et.

Bir kaynaktan bilgi kullandığında, onun etiketini hatırla.

Cevabının en sonuna şu şekilde bir satır ekle: "Kaynaklar: KAYNAK 1, KAYNAK 3".

CEVAP TARZI:

Soruya 1-2 cümlelik doğrudan bir cevapla başla.

Ardından önemli detayları, açıklamaları veya örnekleri içeren 3-7 maddelik bir liste sun.

Cümleleri makul ölçüde kısa ve odaklı tut.

Eğer kullanıcı açıkça çok detaylı veya uzun bir cevap isterse daha fazla yazabilirsin, ancak yine de net bir yapıyı koru.

BİLGİ EKSİK OLDUĞUNDA:

Eğer bağlam, soruyu güvenilir bir şekilde cevaplamak için yeterli bilgi içermiyorsa:

Sağlanan dokümanlarda bu bilginin bulunmadığını açıkça söyle.

ASLA halüsinasyon görme veya tahmin yürütme.

İsteğe bağlı olarak, kullanıcının daha iyi bir cevap alabilmesi için ne tür bir doküman veya bölüm eklemesi gerektiğini öner.

GÖRSEL VERİ KULLANIMI:

Bağlam içinde `--- OCR START ---` ve `--- VISUAL DESC START ---` bloklarını görürsen, bunları o sayfanın **kesin ve doğru** içeriği olarak kabul et. Özellikle 'Bu belge ne hakkında?' gibi sorularda OCR kısmındaki başlıklara ve kurum isimlerine öncelik ver."""
    
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

