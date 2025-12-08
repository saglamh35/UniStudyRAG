"""
UniStudyRAG - Vector Store Module
ChromaDB başlatma, Embedding modeli ve Retriever oluşturma mantığı
"""

import gc
import time
import shutil
from pathlib import Path
from typing import Optional
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import Config
from utils import logger


class VectorStoreService:
    """
    Vektör veritabanı, embedding ve retriever işlemlerini yöneten servis.
    """
    
    def __init__(
        self,
        embed_model_name: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        retrieval_k: Optional[int] = None
    ):
        """
        VectorStoreService'i başlatır.
        
        Args:
            embed_model_name: Embedding model adı (None ise Config'den alınır)
            ollama_base_url: Ollama sunucu URL'i (None ise Config'den alınır)
            retrieval_k: Retriever'dan döndürülecek doküman sayısı (None ise Config'den alınır)
        """
        self.embed_model_name = embed_model_name or Config.EMBED_MODEL_NAME
        self.ollama_base_url = ollama_base_url or Config.OLLAMA_BASE_URL
        self.retrieval_k = retrieval_k or Config.RETRIEVAL_K
        
        # Model caching - Tekrar tekrar yüklenmesin
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._vectorstore: Optional[Chroma] = None
        self._retriever = None
        
        logger.info(f"VectorStoreService başlatıldı - Embed Model: {self.embed_model_name}")
    
    def _get_embeddings(self) -> OllamaEmbeddings:
        """
        Embedding modelini döndürür (caching ile).
        
        Returns:
            OllamaEmbeddings: Embedding modeli nesnesi
        """
        if self._embeddings is None:
            logger.info(f"Embedding modeli yükleniyor: {self.embed_model_name}")
            self._embeddings = OllamaEmbeddings(
                model=self.embed_model_name,
                base_url=self.ollama_base_url
            )
        return self._embeddings
    
    def build_vectorstore(
        self, 
        chunks: list[Document], 
        persist_directory: Optional[Path] = None
    ) -> Chroma:
        """
        Vektör veritabanını oluşturur.
        
        Args:
            chunks: Chunk'lara bölünmüş dokümanlar
            persist_directory: Veritabanının kaydedileceği klasör (None ise geçici)
            
        Returns:
            Chroma: Vektör veritabanı nesnesi
        """
        embeddings = self._get_embeddings()
        
        # Eğer persist_directory belirtilmişse, mevcut veritabanını sil
        if persist_directory and persist_directory.exists():
            logger.info(f"Mevcut veritabanı siliniyor: {persist_directory}")
            
            # Önce mevcut vectorstore nesnesini temizle (dosya kilidini serbest bırakmak için)
            if self._vectorstore is not None:
                self._vectorstore = None
                self._retriever = None
            
            # Belleği temizle ve dosya kilidini serbest bırakmayı dene
            gc.collect()
            
            # Windows'ta dosya kilidinin serbest bırakılması için kısa bir bekleme
            time.sleep(0.5)
            
            # Klasörü silmeyi dene (hata durumunda devam et)
            try:
                shutil.rmtree(persist_directory)
                persist_directory.mkdir(exist_ok=True)
            except (PermissionError, OSError) as e:
                logger.warning(f"Dosya silinemedi, üzerine yazılacak: {e}")
                # ChromaDB zaten üzerine yazabilir, devam et
        
        # Vektör veritabanını oluştur
        logger.info(f"Vektör veritabanı oluşturuluyor ({len(chunks)} chunk)...")
        if persist_directory:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(persist_directory)
            )
            logger.info(f"✅ Vektör veritabanı oluşturuldu ve kaydedildi: {persist_directory}")
        else:
            # Geçici veritabanı (memory-only)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            logger.info("✅ Geçici vektör veritabanı oluşturuldu")
        
        self._vectorstore = vectorstore
        return vectorstore
    
    def get_retriever(self):
        """
        Retriever nesnesini döndürür.
        MMR (Maximal Marginal Relevance) kullanarak retrieval bias'ı azaltır.
        Belgenin farklı yerlerinden (başından ve sonundan) çeşitli metinleri getirir.
        
        Returns:
            Retriever nesnesi
        """
        if self._vectorstore is None:
            raise ValueError("Vectorstore henüz oluşturulmamış. Önce build_vectorstore() çağrılmalı.")
        
        if self._retriever is None:
            logger.info("Retriever oluşturuluyor (MMR)...")
            self._retriever = self._vectorstore.as_retriever(
                search_type=Config.RETRIEVER_SEARCH_TYPE,  # MMR kullan
                search_kwargs={
                    "k": self.retrieval_k,  # Döndürülecek doküman sayısı
                    "fetch_k": Config.RETRIEVER_FETCH_K,  # İlk 20 benzer dokümanı getir
                    "lambda_mult": Config.RETRIEVER_LAMBDA_MULT  # Çeşitlilik/benzerlik dengesi
                }
            )
            logger.info(f"✅ Retriever hazır (k={self.retrieval_k}, fetch_k={Config.RETRIEVER_FETCH_K})")
        
        return self._retriever
    
    def reset_vectorstore(self):
        """
        Vectorstore'u sıfırlar (yeni PDF yüklendiğinde kullanılır).
        """
        logger.info("Vectorstore sıfırlanıyor...")
        self._vectorstore = None
        self._retriever = None

