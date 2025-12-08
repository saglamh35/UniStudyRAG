"""
UniStudyRAG - RAG Engine (Backend Mantığı)
Katmanlı mimari ile refactor edilmiş RAG yöneticisi
Multimodal (Görsel + Metin) PDF dokümanlarından bilgi çıkarma ve soru-cevap sistemi
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union, Iterator
from langchain_core.documents import Document

from config import Config
from utils import logger
from modules.ingestion import PDFIngestionService
from modules.vectorstore import VectorStoreService
from modules.llm_engine import LLMEngine


class RAGManager:
    """
    RAG (Retrieval-Augmented Generation) işlemlerini yöneten ana sınıf.
    Katmanlı mimari ile modüler yapıda çalışır.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        vision_model_name: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        k_retrieval: Optional[int] = None,
        enable_vision: Optional[bool] = None
    ):
        """
        RAGManager'ı başlatır.
        
        Args:
            model_name: LLM model adı (None ise Config'den alınır)
            embed_model_name: Embedding model adı (None ise Config'den alınır)
            vision_model_name: Vision model adı (None ise Config'den alınır, kullanılmıyor ama uyumluluk için)
            ollama_base_url: Ollama sunucu URL'i (None ise Config'den alınır)
            chunk_size: Chunk boyutu (None ise Config'den alınır)
            chunk_overlap: Chunk overlap miktarı (None ise Config'den alınır)
            k_retrieval: Retriever'dan döndürülecek doküman sayısı (None ise Config'den alınır)
            enable_vision: Görsel analiz özelliğini etkinleştir/devre dışı bırak (None ise Config'den alınır)
        """
        # Modül servislerini başlat
        self.ingestion_service = PDFIngestionService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_vision=enable_vision
        )
        
        self.vectorstore_service = VectorStoreService(
            embed_model_name=embed_model_name,
            ollama_base_url=ollama_base_url,
            retrieval_k=k_retrieval
        )
        
        self.llm_engine = LLMEngine(
            model_name=model_name,
            ollama_base_url=ollama_base_url
        )
        
        logger.info("RAGManager başlatıldı (Katmanlı Mimari)")
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        """
        Byte formatındaki PDF dosyasını yükler (Streamlit için).
        Multimodal: Hem metin hem görsel analiz yapar.
        
        Args:
            pdf_bytes: PDF dosyasının byte içeriği
            filename: Dosya adı
            
        Returns:
            List[Document]: Yüklenen dokümanların listesi
        """
        return self.ingestion_service.load_pdf_from_bytes(pdf_bytes, filename)
    
    def load_pdfs_from_directory(self, data_dir: Path) -> List[Document]:
        """
        Belirtilen klasördeki tüm PDF dosyalarını yükler.
        Multimodal: Hem metin hem görsel analiz yapar.
        
        Args:
            data_dir: PDF dosyalarının bulunduğu klasör
            
        Returns:
            List[Document]: Yüklenen dokümanların listesi
        """
        return self.ingestion_service.load_pdfs_from_directory(data_dir)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Dokümanları chunk'lara böler.
        
        Args:
            documents: Bölünecek dokümanların listesi
            
        Returns:
            List[Document]: Chunk'lara bölünmüş dokümanların listesi
        """
        return self.ingestion_service.split_documents(documents)
    
    def build_vectorstore(
        self, 
        chunks: List[Document], 
        persist_directory: Optional[Path] = None
    ):
        """
        Vektör veritabanını oluşturur.
        
        Args:
            chunks: Chunk'lara bölünmüş dokümanlar
            persist_directory: Veritabanının kaydedileceği klasör (None ise geçici)
            
        Returns:
            Chroma: Vektör veritabanı nesnesi
        """
        return self.vectorstore_service.build_vectorstore(chunks, persist_directory)
    
    def get_retriever(self):
        """
        Retriever nesnesini döndürür.
        MMR (Maximal Marginal Relevance) kullanarak retrieval bias'ı azaltır.
        
        Returns:
            Retriever nesnesi
        """
        return self.vectorstore_service.get_retriever()
    
    def query(
        self,
        question: str,
        retriever=None
    ) -> Tuple[Union[str, Iterator[str]], List[Document]]:
        """
        Kullanıcı sorusunu işler ve streaming cevap döndürür.
        
        Args:
            question: Kullanıcının sorusu
            retriever: Retriever nesnesi (None ise otomatik alınır)
            
        Returns:
            Tuple[Union[str, Iterator[str]], List[Document]]: 
                - Eğer doküman yoksa: (Hata mesajı string, boş liste)
                - Normal durumda: (Streaming response iterator, Kaynak dokümanlar)
        """
        # Retriever'ı al
        if retriever is None:
            retriever = self.get_retriever()
        
        return self.llm_engine.query(question, retriever)
    
    def reset_vectorstore(self):
        """
        Vectorstore'u sıfırlar (yeni PDF yüklendiğinde kullanılır).
        """
        self.vectorstore_service.reset_vectorstore()