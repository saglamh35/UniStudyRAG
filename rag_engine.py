"""
UniStudyRAG - RAG Engine (Backend Mantığı)
Multimodal (Görsel + Metin) PDF dokümanlarından bilgi çıkarma ve soru-cevap sistemi
"""

import os
import shutil
import tempfile
import base64
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Union, Iterator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Vision kütüphaneleri (opsiyonel - graceful degradation)
try:
    from pdf2image import convert_from_path, convert_from_bytes
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️  pdf2image veya PIL yüklü değil. Görsel analiz devre dışı.")


class RAGManager:
    """
    RAG (Retrieval-Augmented Generation) işlemlerini yöneten ana sınıf.
    Multimodal PDF yükleme (metin + görsel), chunking, embedding ve retrieval işlemlerini içerir.
    """
    
    def __init__(
        self,
        model_name: str = "gemma3:4b",
        embed_model_name: str = "nomic-embed-text",
        vision_model_name: str = "llama3.2-vision",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_retrieval: int = 5,
        enable_vision: bool = True
    ):
        """
        RAGManager'ı başlatır.
        
        Args:
            model_name: LLM model adı
            embed_model_name: Embedding model adı
            vision_model_name: Vision model adı (görsel analiz için)
            ollama_base_url: Ollama sunucu URL'i
            chunk_size: Chunk boyutu
            chunk_overlap: Chunk overlap miktarı
            k_retrieval: Retriever'dan döndürülecek doküman sayısı
            enable_vision: Görsel analiz özelliğini etkinleştir/devre dışı bırak
        """
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.vision_model_name = vision_model_name
        self.ollama_base_url = ollama_base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.enable_vision = enable_vision and PDF2IMAGE_AVAILABLE
        
        # Model caching - Tekrar tekrar yüklenmesin
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._llm: Optional[ChatOllama] = None
        self._vectorstore: Optional[Chroma] = None
        self._retriever = None
        
        # Güncellenmiş system prompt (Chain of Thought - CoT yaklaşımı ile)
        self.system_prompt = """Sen UniStudyRAG, üniversite seviyesinde akademik bir asistansın.

GÖREVİN: Verilen BAĞLAM (Context) ve Görsel Analizlerini kullanarak soruyu yanıtlamak.

YANITLAMA STRATEJİSİ (ZİNCİRLEME DÜŞÜNCE):
Cevabı vermeden önce şu adımları takip et:
1. **Analiz:** Sorunun ne istediğini ve hangi dilde olduğunu anla.
2. **Yapısal Kontrol:** Eğer belge bir CV veya Makale ise; Başlıktaki Kişi (Sahibi) ile Referanslar/Yazarlar kısmındaki kişileri ayırt et. İsimleri karıştırma.
3. **Sentez:** Metin ve görsel verileri birleştir.
4. **Yanıt:** Cevabı kullanıcının dilinde oluştur.

KURALLAR:
- Kullanıcı Türkçe sorduysa Türkçe cevapla.
- Teknik terimlerin orijinalini parantez içinde koru.
- Asla uydurma."""
    
    def _get_embeddings(self) -> OllamaEmbeddings:
        """
        Embedding modelini döndürür (caching ile).
        
        Returns:
            OllamaEmbeddings: Embedding modeli nesnesi
        """
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.embed_model_name,
                base_url=self.ollama_base_url
            )
        return self._embeddings
    
    def _get_llm(self) -> ChatOllama:
        """
        LLM modelini döndürür (caching ile).
        Context window 4096'ya genişletilmiş (uzun belgeler için).
        
        Returns:
            ChatOllama: LLM modeli nesnesi
        """
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_base_url,
                temperature=0,
                num_ctx=4096  # Context window'u 4096'ya çıkar (CV ve uzun belgeler için)
            )
        return self._llm
    
    def _analyze_image(self, image) -> Optional[str]:
        """
        Görseli vision modeli ile analiz eder.
        
        Args:
            image: PIL Image nesnesi
            
        Returns:
            Optional[str]: Görsel analiz sonucu (hata durumunda None veya boş string)
        """
        if not self.enable_vision:
            return None
        
        try:
            # Resmi base64'e çevir
            buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image.save(buffered, format="PNG")
            buffered.seek(0)
            
            # Base64 encoding
            with open(buffered.name, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Geçici dosyayı sil
            os.unlink(buffered.name)
            
            # Ollama API'ye istek gönder
            prompt = "Describe this technical image, diagram, or chart in detail. Extract all numbers and text visible."
            
            payload = {
                "model": self.vision_model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"⚠️  Vision model hatası: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Vision model bağlantı hatası: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Görsel analiz hatası: {e}")
            return None
    
    def _process_pdf_with_vision(
        self,
        pdf_path: str,
        filename: str,
        is_bytes: bool = False
    ) -> List[Document]:
        """
        PDF'i hem metin hem görsel olarak işler (multimodal).
        
        Args:
            pdf_path: PDF dosya yolu veya byte içeriği
            filename: Dosya adı
            is_bytes: pdf_path'in byte içeriği olup olmadığı
            
        Returns:
            List[Document]: İşlenmiş dokümanların listesi
        """
        documents = []
        
        try:
            # Önce metin olarak yükle
            if is_bytes:
                # Byte içeriği için geçici dosya oluştur
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_path)
                    tmp_path = tmp_file.name
            else:
                tmp_path = pdf_path
            
            loader = PyPDFLoader(tmp_path)
            text_docs = loader.load()
            
            # Eğer vision aktif değilse, sadece metni döndür
            if not self.enable_vision:
                for doc in text_docs:
                    doc.metadata["source"] = filename
                return text_docs
            
            # PDF'i resimlere çevir
            try:
                if is_bytes:
                    images = convert_from_bytes(pdf_path)
                else:
                    images = convert_from_path(tmp_path)
            except Exception as e:
                print(f"⚠️  PDF'den resim çıkarılamadı: {e}. Sadece metin kullanılıyor.")
                for doc in text_docs:
                    doc.metadata["source"] = filename
                return text_docs
            
            # Her sayfa için metin + görsel analizi birleştir
            for page_num, (text_doc, image) in enumerate(zip(text_docs, images), start=1):
                print(f"Görseller analiz ediliyor (Sayfa {page_num})...")
                
                # Orijinal metin
                original_text = text_doc.page_content
                
                # Görsel analizi
                vision_analysis = self._analyze_image(image)
                
                # Birleştirilmiş içerik
                if vision_analysis:
                    combined_content = f"""{original_text}

--- GÖRSEL ANALİZİ ---
Modelin Gördüğü: {vision_analysis}"""
                else:
                    combined_content = original_text
                
                # Yeni doküman oluştur
                new_doc = Document(
                    page_content=combined_content,
                    metadata={
                        **text_doc.metadata,
                        "source": filename,
                        "page": page_num - 1  # 0-indexed
                    }
                )
                documents.append(new_doc)
            
            # Geçici dosyayı sil
            if is_bytes and os.path.exists(tmp_path):
                os.remove(tmp_path)
            
            return documents
            
        except Exception as e:
            print(f"⚠️  PDF işleme hatası: {e}")
            # Hata durumunda sadece metin döndür
            try:
                loader = PyPDFLoader(tmp_path if is_bytes else pdf_path)
                text_docs = loader.load()
                for doc in text_docs:
                    doc.metadata["source"] = filename
                return text_docs
            except:
                return []
    
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
        return self._process_pdf_with_vision(pdf_bytes, filename, is_bytes=True)
    
    def load_pdfs_from_directory(self, data_dir: Path) -> List[Document]:
        """
        Belirtilen klasördeki tüm PDF dosyalarını yükler.
        Multimodal: Hem metin hem görsel analiz yapar.
        
        Args:
            data_dir: PDF dosyalarının bulunduğu klasör
            
        Returns:
            List[Document]: Yüklenen dokümanların listesi
        """
        data_dir.mkdir(exist_ok=True)
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            return []
        
        all_documents = []
        for pdf_file in pdf_files:
            try:
                docs = self._process_pdf_with_vision(str(pdf_file), pdf_file.name, is_bytes=False)
                all_documents.extend(docs)
            except Exception as e:
                print(f"⚠️  HATA: {pdf_file.name} yüklenemedi ({e}) - Atlanıyor...")
                continue
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Dokümanları chunk'lara böler.
        
        Args:
            documents: Bölünecek dokümanların listesi
            
        Returns:
            List[Document]: Chunk'lara bölünmüş dokümanların listesi
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def build_vectorstore(self, chunks: List[Document], persist_directory: Optional[Path] = None) -> Chroma:
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
            shutil.rmtree(persist_directory)
            persist_directory.mkdir(exist_ok=True)
        
        # Vektör veritabanını oluştur
        if persist_directory:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(persist_directory)
            )
        else:
            # Geçici veritabanı (memory-only)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
        
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
            self._retriever = self._vectorstore.as_retriever(
                search_type="mmr",  # MMR kullan (Retrieval Bias'ı azaltır)
                search_kwargs={
                    "k": 5,  # Döndürülecek doküman sayısı
                    "fetch_k": 20,  # İlk 20 benzer dokümanı getir
                    "lambda_mult": 0.6  # Çeşitlilik/benzerlik dengesi (0.6 = daha çeşitli)
                }
            )
        
        return self._retriever
    
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
        
        # İlgili dokümanları al
        relevant_docs = retriever.invoke(question)
        
        # Eğer doküman yoksa
        if len(relevant_docs) == 0:
            return "⚠️ Kaynak bulunamadı. Lütfen farklı bir soru deneyin.", []
        
        # Dokümanları birleştir (context)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
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
        
        return content_generator(), relevant_docs
    
    def reset_vectorstore(self):
        """
        Vectorstore'u sıfırlar (yeni PDF yüklendiğinde kullanılır).
        """
        self._vectorstore = None
        self._retriever = None
