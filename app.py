"""
UniStudyRAG - Streamlit Web Arayüzü
Modern ve kullanıcı dostu chat arayüzü
"""

import streamlit as st
from pathlib import Path
from rag_engine import RAGManager
from config import Config

# Sayfa yapılandırması
st.set_page_config(
    page_title="UniStudyRAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile stil iyileştirmeleri
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_rag_manager():
    """
    RAGManager'ı cache'ler (her tıklamada yeniden yüklenmesin).
    Config'den değerleri otomatik alır.
    
    Returns:
        RAGManager: RAG yöneticisi nesnesi
    """
    return RAGManager(
        # Tüm parametreler None olduğu için Config'den otomatik alınacak
        # İsterseniz burada override edebilirsiniz
    )


def initialize_session_state():
    """
    Session state'i başlatır.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []


def process_uploaded_files(uploaded_files, rag_manager: RAGManager):
    """
    Yüklenen PDF dosyalarını işler ve vectorstore oluşturur.
    
    Args:
        uploaded_files: Streamlit file_uploader'dan gelen dosyalar
        rag_manager: RAGManager nesnesi
    """
    if not uploaded_files:
        return
    
    # Yeni dosyalar yüklendi mi kontrol et
    current_file_names = {f.name for f in uploaded_files}
    previous_file_names = {f for f in st.session_state.uploaded_files}
    
    if current_file_names != previous_file_names:
        # Yeni dosyalar var, işle
        with st.spinner("📄 PDF dosyaları yükleniyor..."):
            all_documents = []
            
            for uploaded_file in uploaded_files:
                # PDF'i byte olarak oku
                pdf_bytes = uploaded_file.read()
                
                # RAGManager ile yükle
                try:
                    docs = rag_manager.load_pdf_from_bytes(pdf_bytes, uploaded_file.name)
                    all_documents.extend(docs)
                    st.success(f"✅ {uploaded_file.name} yüklendi ({len(docs)} sayfa)")
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name} yüklenemedi: {e}")
            
            if all_documents:
                # Dokümanları chunk'lara böl
                with st.spinner("✂️ Metinler bölünüyor (chunking)..."):
                    chunks = rag_manager.split_documents(all_documents)
                    st.info(f"📊 {len(chunks)} adet chunk oluşturuldu")
                
                # Vectorstore oluştur
                with st.spinner("💾 Vektör veritabanı oluşturuluyor (bu biraz zaman alabilir)..."):
                    # Kalıcı klasör kullan (Config'den al)
                    chroma_db_dir = Config.CHROMA_DB_DIR
                    rag_manager.build_vectorstore(chunks, persist_directory=chroma_db_dir)
                    st.session_state.vectorstore_ready = True
                    st.session_state.uploaded_files = list(current_file_names)
                    st.success("✅ Vektör veritabanı hazır! Soru sorabilirsiniz.")
            else:
                st.warning("⚠️ Hiçbir PDF dosyası yüklenemedi.")


def display_sources(relevant_docs):
    """
    Kaynak dokümanları gösterir (Expander içinde).
    
    Args:
        relevant_docs: Kaynak dokümanların listesi
    """
    if not relevant_docs:
        return
    
    # Tekrar eden kaynakları ele
    seen_sources = set()
    unique_sources = []
    
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Bilinmeyen")
        page = doc.metadata.get("page", "Bilinmeyen")
        file_name = Path(source).name if source != "Bilinmeyen" else "Bilinmeyen"
        source_key = f"{file_name}|{page}"
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_sources.append({
                "file": file_name,
                "page": page,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
    
    # Expander ile göster
    with st.expander(f"📚 Kaynak Dokümanlar ({len(unique_sources)} adet)", expanded=False):
        for i, source in enumerate(unique_sources, 1):
            st.markdown(f"**{i}. {source['file']}** (Sayfa: {source['page']})")
            st.caption(f"İçerik önizleme: {source['content']}")
            st.divider()


def main():
    """
    Ana Streamlit uygulaması.
    """
    # Session state'i başlat
    initialize_session_state()
    
    # RAGManager'ı al (cache'lenmiş)
    rag_manager = get_rag_manager()
    
    # Başlık
    st.markdown('<p class="main-header">🎓 UniStudyRAG</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Dosya yükleme
    with st.sidebar:
        st.header("📁 Dosya Yükleme")
        
        uploaded_files = st.file_uploader(
            "PDF dosyalarını seçin",
            type=["pdf"],
            accept_multiple_files=True,
            help="Birden fazla PDF dosyası seçebilirsiniz"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files, rag_manager)
        
        st.markdown("---")
        st.info("💡 **Kullanım:**\n1. PDF dosyalarını yükleyin\n2. İşleme tamamlanınca soru sorun\n3. Cevap ve kaynaklar otomatik gösterilir")
        
        # Vectorstore durumu
        if st.session_state.vectorstore_ready:
            st.success("✅ Sistem hazır")
        else:
            st.warning("⚠️ PDF yükleyin")
    
    # Ana alan - Chat arayüzü
    if not st.session_state.vectorstore_ready:
        st.info("👈 Lütfen sol taraftan PDF dosyalarını yükleyin.")
        return
    
    # Chat mesajlarını göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Eğer assistant mesajıysa ve kaynaklar varsa göster
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])
    
    # Kullanıcıdan soru al
    if prompt := st.chat_input("Sorunuzu yazın..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Kullanıcı mesajını göster
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant cevabını oluştur
        with st.chat_message("assistant"):
            try:
                # RAG ile sorgu yap
                response_stream_or_error, relevant_docs = rag_manager.query(prompt)
                
                # Eğer hata mesajı (string) döndüyse
                if isinstance(response_stream_or_error, str):
                    st.error(response_stream_or_error)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_stream_or_error
                    })
                else:
                    # Streaming response (generator)
                    # st.write_stream akış bittiğinde tam metni döndürür
                    full_response = st.write_stream(response_stream_or_error)
                    
                    # Kaynakları göster
                    display_sources(relevant_docs)
                    
                    # Mesajı session state'e ekle (akış tamamlandıktan sonra)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": relevant_docs
                    })
                    
            except Exception as e:
                error_msg = f"❌ Hata oluştu: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


if __name__ == "__main__":
    main()

