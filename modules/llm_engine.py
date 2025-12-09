"""
UniStudyRAG - LLM Engine Module
LLM başlatma, System Prompt ve Query (cevap üretme) mantığı
"""

from typing import Iterator, List, Optional, Tuple, Union
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from config import Config
from utils import logger


class LLMEngine:
    """
    LLM işlemlerini yöneten servis.
    Streaming desteği ile cevap üretme.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        LLMEngine'i başlatır.
        
        Args:
            model_name: LLM model adı (None ise Config'den alınır)
            ollama_base_url: Ollama sunucu URL'i (None ise Config'den alınır)
            temperature: LLM temperature (None ise Config'den alınır)
            num_ctx: Context window boyutu (None ise Config'den alınır)
            system_prompt: System prompt (None ise Config'den alınır)
        """
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self.ollama_base_url = ollama_base_url or Config.OLLAMA_BASE_URL
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.num_ctx = num_ctx or Config.LLM_NUM_CTX
        self.system_prompt = system_prompt or Config.SYSTEM_PROMPT
        
        # Model caching - Tekrar tekrar yüklenmesin
        self._llm: Optional[ChatOllama] = None
        
        logger.info(f"LLMEngine başlatıldı - Model: {self.model_name}, Context Window: {self.num_ctx}")
    
    def _get_llm(self) -> ChatOllama:
        """
        LLM modelini döndürür (caching ile).
        Context window 4096'ya genişletilmiş (uzun belgeler için).
        
        Returns:
            ChatOllama: LLM modeli nesnesi
        """
        if self._llm is None:
            logger.info(f"LLM modeli yükleniyor: {self.model_name}")
            self._llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_base_url,
                temperature=self.temperature,
                num_ctx=self.num_ctx  # Context window'u 4096'ya çıkar
            )
        return self._llm
    
    def query(
        self,
        question: str,
        retriever=None
    ) -> Tuple[Union[str, Iterator[str]], List[Document]]:
        """
        Kullanıcı sorusunu işler ve streaming cevap döndürür.
        
        Args:
            question: Kullanıcının sorusu
            retriever: Retriever nesnesi (None ise hata döner)
            
        Returns:
            Tuple[Union[str, Iterator[str]], List[Document]]: 
                - Eğer doküman yoksa: (Hata mesajı string, boş liste)
                - Normal durumda: (Streaming response iterator, Kaynak dokümanlar)
        """
        if retriever is None:
            return "⚠️ Retriever bulunamadı. Lütfen önce vectorstore oluşturun.", []
        
        # İlgili dokümanları al
        logger.info(f"Soru işleniyor: {question[:50]}...")
        relevant_docs = retriever.invoke(question)
        
        # Eğer doküman yoksa
        if len(relevant_docs) == 0:
            logger.warning("Kaynak bulunamadı.")
            return "⚠️ Kaynak bulunamadı. Lütfen farklı bir soru deneyin.", []
        
        logger.info(f"✅ {len(relevant_docs)} adet ilgili doküman bulundu")
        
        # Dokümanları kaynak etiketleriyle birleştir (context)
        context_parts = []
        for idx, doc in enumerate(relevant_docs, 1):
            source_label = f"[KAYNAK {idx}]"
            context_parts.append(f"{source_label}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Promptu hazırla
        prompt = f"""{self.system_prompt}

BAĞLAM: {context}

SORU: {question}"""
        
        # LLM'den streaming cevap al
        llm = self._get_llm()
        response_stream = llm.stream(prompt)
        
        # Generator fonksiyonu: Her chunk'tan sadece içeriği çıkar
        def content_generator():
            """Stream'den sadece content'i çıkarır."""
            for chunk in response_stream:
                if hasattr(chunk, 'content'):
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
                elif isinstance(chunk, dict) and 'content' in chunk:
                    yield chunk['content']
        
        logger.info("Streaming cevap üretiliyor...")
        return content_generator(), relevant_docs

