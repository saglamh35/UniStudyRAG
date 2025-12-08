"""
UniStudyRAG - PDF Ingestion Module
PDF yükleme, Vision entegrasyonu ve Chunking mantığı
Smart Caching sistemi ile optimize edilmiş
"""

import os
import json
import hashlib
import tempfile
import base64
import requests
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config
from utils import logger

# Vision kütüphaneleri (opsiyonel - graceful degradation)
try:
    from pdf2image import convert_from_path, convert_from_bytes
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image veya PIL yüklü değil. Görsel analiz devre dışı.")


class PDFIngestionService:
    """
    PDF yükleme, görsel analiz ve chunking işlemlerini yöneten servis.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_vision: Optional[bool] = None
    ):
        """
        PDFIngestionService'i başlatır.
        
        Args:
            chunk_size: Chunk boyutu (None ise Config'den alınır)
            chunk_overlap: Chunk overlap miktarı (None ise Config'den alınır)
            enable_vision: Görsel analiz özelliği (None ise Config'den alınır)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.enable_vision = (enable_vision if enable_vision is not None 
                             else Config.ENABLE_VISION) and PDF2IMAGE_AVAILABLE
        
        # Cache dizinini oluştur
        Config.CACHE_DIR.mkdir(exist_ok=True)
        
        logger.info(f"PDFIngestionService başlatıldı - Vision: {self.enable_vision}, Cache: {Config.CACHE_DIR}")
    
    def _compute_file_hash(self, pdf_bytes: bytes) -> str:
        """
        PDF dosyasının byte içeriğinden MD5 hash üretir.
        
        Args:
            pdf_bytes: PDF dosyasının byte içeriği
            
        Returns:
            str: MD5 hash değeri (hex formatında)
        """
        return hashlib.md5(pdf_bytes).hexdigest()
    
    def _get_cache_path(self, file_hash: str) -> Path:
        """
        Cache dosyasının tam yolunu döndürür.
        
        Args:
            file_hash: Dosya hash değeri
            
        Returns:
            Path: Cache dosyasının yolu
        """
        return Config.CACHE_DIR / f"{file_hash}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[List[Document]]:
        """
        Cache'den dokümanları yükler.
        
        Args:
            cache_path: Cache dosyasının yolu
            
        Returns:
            Optional[List[Document]]: Yüklenen dokümanlar (hata durumunda None)
        """
        try:
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # JSON'dan Document nesnelerini yeniden oluştur
            documents = []
            for item in cached_data:
                doc = Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {})
                )
                documents.append(doc)
            
            return documents
            
        except json.JSONDecodeError as e:
            logger.warning(f"Cache dosyası bozuk (JSON hatası): {cache_path} - {e}")
            return None
        except Exception as e:
            logger.warning(f"Cache okuma hatası: {cache_path} - {e}")
            return None
    
    def _save_to_cache(self, documents: List[Document], cache_path: Path) -> bool:
        """
        Dokümanları cache'e kaydeder.
        
        Args:
            documents: Kaydedilecek dokümanlar
            cache_path: Cache dosyasının yolu
            
        Returns:
            bool: Başarılı ise True, hata durumunda False
        """
        try:
            # Document nesnelerini JSON'a çevir
            cached_data = []
            for doc in documents:
                cached_data.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # JSON dosyasına kaydet
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache yazma hatası: {cache_path} - {e}")
            return False
    
    def _analyze_image(self, image: Image.Image) -> Optional[str]:
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
            try:
                image.save(buffered, format="PNG")
                buffered.flush()  # Ensure all data is written
                buffered.close()  # Close file handle before unlinking (Windows compatibility)
                
                # Base64 encoding
                with open(buffered.name, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            finally:
                # Geçici dosyayı sil (Windows'ta dosya kapatıldıktan sonra)
                if os.path.exists(buffered.name):
                    try:
                        os.unlink(buffered.name)
                    except OSError:
                        pass  # Ignore if file is already deleted
            
            # Ollama API'ye istek gönder (Güçlendirilmiş prompt)
            prompt = Config.get_vision_prompt()
            
            payload = {
                "model": Config.VISION_MODEL_NAME,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            response = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.warning(f"Vision model hatası: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Vision model bağlantı hatası: {e}")
            return None
        except Exception as e:
            logger.warning(f"Görsel analiz hatası: {e}")
            return None
    
    def _process_pdf_with_vision(
        self,
        pdf_path: str,
        filename: str,
        is_bytes: bool = False,
        pdf_bytes: Optional[bytes] = None
    ) -> List[Document]:
        """
        PDF'i hem metin hem görsel olarak işler (multimodal).
        Smart Caching ile optimize edilmiş.
        
        Args:
            pdf_path: PDF dosya yolu veya byte içeriği
            filename: Dosya adı
            is_bytes: pdf_path'in byte içeriği olup olmadığı
            pdf_bytes: PDF byte içeriği (cache kontrolü için)
            
        Returns:
            List[Document]: İşlenmiş dokümanların listesi
        """
        # Cache kontrolü (sadece byte içeriği varsa)
        if is_bytes and pdf_bytes is not None:
            file_hash = self._compute_file_hash(pdf_bytes)
            cache_path = self._get_cache_path(file_hash)
            
            # Cache'den yükle
            cached_docs = self._load_from_cache(cache_path)
            if cached_docs is not None:
                logger.info(f"Cache HIT: {filename} diskten yükleniyor...")
                return cached_docs
            
            logger.info(f"Cache MISS: {filename} işleniyor ve kaydediliyor...")
        
        documents = []
        tmp_path = None
        
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
                
                # Cache'e kaydet (vision olmasa bile)
                if is_bytes and pdf_bytes is not None:
                    self._save_to_cache(text_docs, cache_path)
                
                return text_docs
            
            # PDF'i resimlere çevir
            try:
                if is_bytes:
                    images = convert_from_bytes(pdf_path)
                else:
                    images = convert_from_path(tmp_path)
            except Exception as e:
                logger.warning(f"PDF'den resim çıkarılamadı: {e}. Sadece metin kullanılıyor.")
                for doc in text_docs:
                    doc.metadata["source"] = filename
                
                # Cache'e kaydet
                if is_bytes and pdf_bytes is not None:
                    self._save_to_cache(text_docs, cache_path)
                
                return text_docs
            
            # Her sayfa için metin + görsel analizi birleştir
            for page_num, (text_doc, image) in enumerate(zip(text_docs, images), start=1):
                logger.info(f"Görseller analiz ediliyor (Sayfa {page_num})...")
                
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
            
            # Cache'e kaydet
            if is_bytes and pdf_bytes is not None:
                self._save_to_cache(documents, cache_path)
            
            # Geçici dosyayı sil
            if is_bytes and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            
            return documents
            
        except Exception as e:
            logger.error(f"PDF işleme hatası: {e}")
            # Hata durumunda sadece metin döndür
            try:
                loader = PyPDFLoader(tmp_path if is_bytes else pdf_path)
                text_docs = loader.load()
                for doc in text_docs:
                    doc.metadata["source"] = filename
                
                # Cache'e kaydet (hata durumunda bile)
                if is_bytes and pdf_bytes is not None:
                    self._save_to_cache(text_docs, cache_path)
                
                return text_docs
            except:
                return []
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        """
        Byte formatındaki PDF dosyasını yükler (Streamlit için).
        Multimodal: Hem metin hem görsel analiz yapar.
        Smart Caching ile optimize edilmiş.
        
        Args:
            pdf_bytes: PDF dosyasının byte içeriği
            filename: Dosya adı
            
        Returns:
            List[Document]: Yüklenen dokümanların listesi
        """
        return self._process_pdf_with_vision(
            pdf_bytes, 
            filename, 
            is_bytes=True, 
            pdf_bytes=pdf_bytes
        )
    
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
            logger.warning(f"Klasörde PDF dosyası bulunamadı: {data_dir}")
            return []
        
        all_documents = []
        for pdf_file in pdf_files:
            try:
                docs = self._process_pdf_with_vision(str(pdf_file), pdf_file.name, is_bytes=False)
                all_documents.extend(docs)
                logger.info(f"✅ {pdf_file.name} yüklendi ({len(docs)} sayfa)")
            except Exception as e:
                logger.error(f"⚠️  HATA: {pdf_file.name} yüklenemedi ({e}) - Atlanıyor...")
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
        logger.info(f"✅ {len(chunks)} adet chunk oluşturuldu")
        return chunks

