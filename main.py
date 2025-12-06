"""
UniStudyRAG - Yerel RAG Uygulaması Backend
PDF dokümanlarından bilgi çıkarma ve soru-cevap sistemi
"""

import os
import sys
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# Sabitler
DATA_DIR = Path("data")
CHROMA_DB_DIR = Path("chroma_db")
MODEL_NAME = "gemma3:4b"
EMBED_MODEL_NAME = "nomic-embed-text"


def load_pdfs() -> list:
    """
    PDF dosyalarını yükler.
    
    Returns:
        list: Yüklenen dokümanların listesi
        
    Exits:
        Eğer klasör boşsa veya PDF yoksa programı sonlandırır.
    """
    # Data klasörünü oluştur
    DATA_DIR.mkdir(exist_ok=True)
    
    # PDF dosyalarını bul
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ HATA: data klasöründe PDF dosyası bulunamadı!")
        print(f"   Lütfen PDF dosyalarını {DATA_DIR} klasörüne ekleyin.")
        sys.exit(1)
    
    print(f"✅ {len(pdf_files)} adet PDF dosyası bulundu.")
    
    # PDF'leri yükle
    print("\n📄 PDF dosyaları yükleniyor...")
    documents = []
    total_pages = 0
    
    for pdf_file in pdf_files:
        print(f"   Yükleniyor: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            total_pages += len(docs)
            print(f"   ✓ {len(docs)} sayfa yüklendi")
        except Exception as e:
            print(f"   ⚠️  HATA: {pdf_file.name} yüklenemedi ({e}) - Atlanıyor...")
            continue
    
    if not documents:
        print("\n❌ HATA: Hiçbir PDF dosyası yüklenemedi!")
        sys.exit(1)
    
    print(f"\n✅ Toplam {total_pages} sayfa yüklendi ({len(documents)} doküman).")
    return documents


def split_documents(documents: list) -> list:
    """
    Dokümanları chunk'lara böler.
    
    Args:
        documents: Bölünecek dokümanların listesi
        
    Returns:
        list: Chunk'lara bölünmüş dokümanların listesi
    """
    print("\n✂️  Metinler bölünüyor (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)} adet chunk oluşturuldu.")
    return chunks


def build_embeddings() -> OllamaEmbeddings:
    """
    Embedding modelini oluşturur.
    
    Returns:
        OllamaEmbeddings: Embedding modeli nesnesi
    """
    print("\n🔤 Embedding modeli hazırlanıyor...")
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL_NAME,
        base_url="http://localhost:11434"
    )
    print(f"✅ Embedding modeli hazır ({EMBED_MODEL_NAME}).")
    return embeddings


def build_vectorstore(chunks: list, embeddings: OllamaEmbeddings) -> Chroma:
    """
    Vektör veritabanını oluşturur (her seferinde sıfırdan).
    
    Args:
        chunks: Chunk'lara bölünmüş dokümanlar
        embeddings: Embedding modeli
        
    Returns:
        Chroma: Vektör veritabanı nesnesi
    """
    print("\n💾 Vektör veritabanı hazırlanıyor...")
    
    # Mevcut veritabanını sil (clean start)
    if CHROMA_DB_DIR.exists():
        print("   Mevcut veritabanı siliniyor...")
        shutil.rmtree(CHROMA_DB_DIR)
    
    # Klasörü yeniden oluştur
    CHROMA_DB_DIR.mkdir(exist_ok=True)
    
    print("   Veritabanı sıfırdan oluşturuluyor (bu biraz zaman alabilir)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR)
    )
    print("✅ Vektör veritabanı oluşturuldu ve kaydedildi.")
    return vectorstore


def build_retriever(vectorstore: Chroma):
    """
    Retriever oluşturur.
    
    Args:
        vectorstore: Vektör veritabanı nesnesi
        
    Returns:
        Retriever nesnesi
    """
    print("\n🔗 Retriever hazırlanıyor...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("✅ Retriever hazır (k=5).")
    return retriever


def build_llm() -> ChatOllama:
    """
    LLM modelini oluşturur.
    
    Returns:
        ChatOllama: LLM modeli nesnesi
    """
    print("\n🤖 LLM modeli hazırlanıyor...")
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url="http://localhost:11434",
        temperature=0
    )
    print(f"✅ LLM modeli hazır ({MODEL_NAME}).")
    return llm


def chat_loop(retriever, llm: ChatOllama):
    """
    Soru-cevap döngüsünü yönetir.
    
    Args:
        retriever: Retriever nesnesi
        llm: LLM modeli nesnesi
    """
    print("\n" + "="*60)
    print("🎓 UniStudyRAG Hazır!")
    print("="*60)
    print("Soru sormak için yazın. Çıkmak için 'q' veya 'quit' yazın.\n")
    
    # Sistem promptu
    system_prompt = """Sen üniversite öğrencilerine yardımcı olan bir asistansın. Aşağıdaki BAĞLAM (Context) bilgisini kullanarak soruyu cevapla. Bağlamda bilgi yoksa 'Bilgim yok' de, uydurma."""
    
    while True:
        # Kullanıcıdan soru al
        question = input("❓ Sorunuz: ").strip()
        
        # Çıkış kontrolü
        if question.lower() in ['q', 'quit', 'exit', 'çıkış']:
            print("\n👋 Görüşmek üzere!")
            break
        
        if not question:
            print("⚠️  Lütfen bir soru girin.\n")
            continue
        
        # Soruyu işle
        print("\n🔍 Cevap aranıyor...\n")
        try:
            # İlgili dokümanları al
            relevant_docs = retriever.invoke(question)
            
            # Debug: Bulunan doküman sayısını göster
            print(f"📊 Bulunan doküman sayısı: {len(relevant_docs)}")
            
            # Eğer doküman yoksa uyarı ver ve devam et
            if len(relevant_docs) == 0:
                print("⚠️  Kaynak bulunamadı.\n")
                continue
            
            # Debug: İlk dokümanın bilgilerini göster
            if len(relevant_docs) > 0:
                first_doc = relevant_docs[0]
                print("=" * 60)
                print("🔍 DEBUG - İlk Doküman Bilgileri:")
                print("-" * 60)
                source = first_doc.metadata.get("source", "Bilinmeyen")
                page = first_doc.metadata.get("page", "Bilinmeyen")
                file_name = Path(source).name if source != "Bilinmeyen" else "Bilinmeyen"
                print(f"Kaynak dosya: {file_name}")
                print(f"Sayfa: {page}")
                print(f"İlk 200 karakter: {first_doc.page_content[:200]}...")
                print("=" * 60)
                print()
            
            # Dokümanları birleştir (context)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Promptu hazırla
            prompt = f"""{system_prompt}

BAĞLAM: {context}

SORU: {question}"""
            
            # LLM'den cevap al
            response = llm.invoke(prompt)
            
            # Cevabı göster
            print("💬 Cevap:")
            print("-" * 60)
            print(response.content)
            print("-" * 60)
            
            # Kaynak dokümanları göster
            if relevant_docs:
                print("\n📚 Kaynak Dokümanlar:")
                seen_sources = set()
                for i, doc in enumerate(relevant_docs, 1):
                    source = doc.metadata.get("source", "Bilinmeyen")
                    page = doc.metadata.get("page", "Bilinmeyen")
                    
                    # Dosya adını al (tam yol yerine)
                    file_name = Path(source).name if source != "Bilinmeyen" else "Bilinmeyen"
                    
                    # Aynı kaynağı tekrar gösterme
                    source_key = f"{file_name}|{page}"
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        print(f"   {i}. Kaynak: {file_name} | Sayfa: {page}")
            
            print("\n")
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}\n")


def main():
    """Ana fonksiyon - Tüm işlemleri sırasıyla yürütür."""
    # 1. PDF'leri yükle
    documents = load_pdfs()
    
    # 2. Dokümanları chunk'lara böl
    chunks = split_documents(documents)
    
    # 3. Embedding modelini oluştur
    embeddings = build_embeddings()
    
    # 4. Vektör veritabanını oluştur (sıfırdan)
    vectorstore = build_vectorstore(chunks, embeddings)
    
    # 5. Retriever oluştur
    retriever = build_retriever(vectorstore)
    
    # 6. LLM modelini oluştur
    llm = build_llm()
    
    # 7. Soru-cevap döngüsünü başlat
    chat_loop(retriever, llm)


if __name__ == "__main__":
    main()
