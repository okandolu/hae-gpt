# -*- coding: utf-8 -*-
"""Streamlit UI for RAG System - Multi-Mode HAE Q&A.

Web interface for Hereditary Angioedema Q&A system with multiple operational modes.

Modes:
    1. Hasta Bilgilendirme: 8. sınıf okuryazarlık, empatik
    2. Hasta Özel: Klinik bilgilerle kişiselleştirilmiş (HER SORU İÇİN AYRI KLİNİK)
    3. Akademisyen: Teknik, detaylı, APA citations

Features:
    - Single question interface with real-time results
    - Batch query processing with database persistence
    - Multiple citation formats (Markdown, JSON, Table)
    - Question and patient info history tracking
"""

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from batch_processor import BatchQueryProcessor
from batch_query_db import BatchQueryDB
from src import CitationFormatter, Generator, Retriever

# Page config
st.set_page_config(
    page_title="HAE Q&A System - RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.citation-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #4CAF50;
    margin: 10px 0;
}
.context-preview {
    background-color: #e8f4f8;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    font-size: 0.9em;
    max-height: 200px;
    overflow-y: auto;
}
.full-context {
    background-color: #fff;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 5px;
    font-family: monospace;
    font-size: 0.85em;
    line-height: 1.6;
    max-height: 500px;
    overflow-y: auto;
}
.mode-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9em;
}
.mode-patient {
    background-color: #e3f2fd;
    color: #1565c0;
}
.mode-academic {
    background-color: #f3e5f5;
    color: #6a1b9a;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """Load RAG system components with caching.

    Returns:
        Tuple containing (retriever, generator, formatter, error_message).
        If successful, error_message is None. If failed, components are None.
    """
    try:
        retriever = Retriever()
        generator = Generator()
        formatter = CitationFormatter()
        return retriever, generator, formatter, None
    except Exception as e:
        return None, None, None, str(e)


def main():
    """Main application entry point.

    Initializes the Streamlit UI with session state, loads the RAG system,
    and renders the multi-tab interface for single and batch queries.
    """
    # Initialize session state
    if 'query_results' not in st.session_state:
        st.session_state.query_results = None
    if 'last_question' not in st.session_state:
        st.session_state.last_question = None
    if 'mode' not in st.session_state:
        st.session_state.mode = "patient"
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = ""

    # Check vectorstore and load system
    if not config.VECTORSTORE_PATH.with_suffix(".index").exists():
        st.error("❌ Vector store bulunamadı!")
        st.info("**Kurulum gerekli:**")
        st.code("python setup_pipeline.py", language="bash")
        st.stop()

    with st.spinner("Sistem yükleniyor..."):
        retriever, generator, formatter, error = load_rag_system()
    if error:
        st.error(f"❌ Sistem yüklenemedi: {error}")
        st.stop()
    st.success("✓ Sistem hazır!")

    # -----------------------
    # Batch Query Tab (INNER FUNCTION)
    # -----------------------
    def render_batch_query_tab():
        """Render the batch query processing interface.

        Provides UI for processing multiple questions in batch mode with
        database persistence and export capabilities.
        """
        st.title("📊 Toplu Soru-Cevap Sistemi")
        st.markdown("**Birden fazla soruyu toplu olarak işleyin ve sonuçları database'e kaydedin**")

        # Initialize
        if 'batch_processor' not in st.session_state:
            st.session_state.batch_processor = None
        if 'batch_db' not in st.session_state:
            st.session_state.batch_db = BatchQueryDB()

        # Sidebar - Batch Settings
        with st.sidebar:
            st.header("⚙️ Batch Ayarları")

            # Mode selection
            batch_mode = st.selectbox(
                "Mod",
                options=[
                    ("patient", "👤 Hasta Bilgilendirme"),
                    ("patient_personalized", "👥 Hasta Özel"),
                    ("academic", "🎓 Akademisyen")
                ],
                format_func=lambda x: x[1],
                key="batch_mode_select"
            )[0]

            # Patient info (if personalized)
            batch_patient_info = None
            if batch_mode == "patient_personalized":
                st.info("💡 Hasta Özel modda her soru için ayrı klinik bilgi girebilirsiniz")
                batch_patient_info = st.text_area(
                    "Genel Klinik Bilgiler (Opsiyonel - Fallback)",
                    height=80,
                    placeholder="Tüm sorular için kullanılacak varsayılan klinik bilgi...",
                    help="Her soru için ayrı klinik girilmezse bu bilgiler kullanılır",
                    key="batch_patient_info"
                )

            # Parameters
            st.subheader("🔍 Parametreler")
            batch_top_k = st.slider("Context Sayısı (k)", 1, 10, 5, key="batch_top_k_slider")
            batch_threshold = st.slider("Benzerlik Eşiği", 0.0, 1.0, 0.5, 0.05, key="batch_threshold_slider")

            st.markdown("---")

            # Database stats
            st.subheader("📊 Database İstatistikleri")
            db_count = st.session_state.batch_db.get_query_count()
            st.metric("Kayıtlı Sorular", db_count)

            # Clear database
            if st.button("🗑️ Database'i Temizle", type="secondary", key="batch_db_clear_btn"):
                if st.session_state.batch_db.get_query_count() > 0:
                    if st.checkbox("Eminim, tüm kayıtları sil", key="batch_db_clear_confirm"):
                        st.session_state.batch_db.clear_all()
                        st.success("✓ Database temizlendi!")
                        st.rerun()

        # Main content - Tabs
        tab1, tab2, tab3 = st.tabs(["➕ Yeni Batch", "📋 Kayıtlar", "📥 Export"])

        # TAB 1: New Batch
        with tab1:
            st.subheader("Yeni Batch İşleme")

            # SPECIAL CASE: Different input for Patient Personalized mode
            if batch_mode == "patient_personalized":
                st.info("👥 **Hasta Özel Mod:** Her soru için ayrı klinik bilgi girebilirsiniz")
                
                # Input methods
                input_method = st.radio(
                    "Giriş yöntemi:",
                    ["Manuel Giriş (Soru | Klinik)", "Dosyadan Yükle"],
                    horizontal=True,
                    key="batch_input_method_personalized"
                )
                
                questions = []
                patient_infos = []
                
                if input_method == "Manuel Giriş (Soru | Klinik)":
                    st.markdown("""
                    **Format:** Her satırda bir soru-klinik çifti, pipe `|` ile ayrılmış
                    - Format: `Soru metni | Klinik bilgiler`
                    - Klinik boş bırakılabilir: `Soru metni |`
                    """)
                    
                    questions_text = st.text_area(
                        "Soru ve klinik bilgilerini girin:",
                        height=250,
                        placeholder="HAE nedir? | HAE Tip 1, 35 yaş, son atak 2 hafta önce\nTedavi seçenekleri nelerdir? | İcatibant kullanıyor, gebelik planı var\nProfilaksi gerekli mi? |",
                        key="batch_questions_personalized_text"
                    )
                    
                    if questions_text:
                        for line in questions_text.split('\n'):
                            line = line.strip()
                            if not line or '|' not in line:
                                continue
                            
                            parts = line.split('|', 1)
                            question = parts[0].strip()
                            patient_info = parts[1].strip() if len(parts) > 1 else ""
                            
                            if question:
                                questions.append(question)
                                patient_infos.append(patient_info)
                
                else:  # Dosyadan Yükle
                    st.markdown("**Excel Şablon Formatı:**")
                    st.markdown("- **Sütun 1:** Soru metni")
                    st.markdown("- **Sütun 2:** Klinik bilgiler (opsiyonel)")
                    
                    # Template download button
                    if st.button("📋 Örnek Excel Şablonu İndir", key="download_template_btn"):
                        template_df = pd.DataFrame({
                            'Soru': [
                                'HAE nedir?',
                                'Tedavi seçenekleri nelerdir?',
                                'Profilaksi gerekli mi?'
                            ],
                            'Klinik Bilgiler': [
                                'HAE Tip 1, 35 yaş, son atak 2 hafta önce',
                                'İcatibant kullanıyor, gebelik planı var',
                                'Yılda 3-4 atak görülüyor'
                            ]
                        })
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            template_df.to_excel(writer, index=False, sheet_name='Sorular')
                        
                        st.download_button(
                            label="⬇️ Şablonu İndir",
                            data=output.getvalue(),
                            file_name="batch_sorular_sablon.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="template_download_btn"
                        )
                    
                    uploaded_file = st.file_uploader(
                        "Excel dosyası yükle (.xlsx) - 2 sütun: Soru | Klinik",
                        type=['xlsx'],
                        key="batch_file_uploader_personalized"
                    )
                    
                    if uploaded_file:
                        try:
                            df = pd.read_excel(uploaded_file)
                            
                            if len(df.columns) < 1:
                                st.error("❌ Dosyada en az 1 sütun olmalı (Soru)")
                            else:
                                questions = df.iloc[:, 0].dropna().astype(str).tolist()
                                
                                if len(df.columns) >= 2:
                                    patient_infos = df.iloc[:, 1].fillna("").astype(str).tolist()
                                else:
                                    patient_infos = [""] * len(questions)
                                
                                st.success(f"✓ {len(questions)} soru-klinik çifti yüklendi")
                        except Exception as e:
                            st.error(f"❌ Dosya okuma hatası: {e}")
                
                # Preview
                if questions:
                    st.info(f"📝 **{len(questions)} soru-klinik çifti** işlenmeye hazır")
                    with st.expander("👁️ Soru-Klinik çiftlerini önizle"):
                        for i, (q, p) in enumerate(zip(questions, patient_infos), 1):
                            st.markdown(f"**{i}. Soru:** {q}")
                            if p:
                                st.markdown(f"   *Klinik:* {p}")
                            else:
                                st.markdown(f"   *Klinik:* (boş)")
                            if i < len(questions):
                                st.markdown("---")
            
            else:
                # NORMAL MODES: Standard input
                input_method = st.radio(
                    "Soru girişi:",
                    ["Manuel Giriş", "Dosyadan Yükle"],
                    horizontal=True,
                    key="batch_input_method"
                )

                questions = []
                patient_infos = []

                if input_method == "Manuel Giriş":
                    questions_text = st.text_area(
                        "Soruları girin (her satırda bir soru):",
                        height=200,
                        placeholder="HAE nedir?\nHAE tedavi yöntemleri nelerdir?\nC1 inhibitör eksikliği nedir?",
                        key="batch_questions_text"
                    )
                    if questions_text:
                        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                        patient_infos = [""] * len(questions)

                else:  # Dosyadan Yükle
                    uploaded_file = st.file_uploader(
                        "Soru dosyası yükle (.txt veya .xlsx)",
                        type=['txt', 'xlsx'],
                        key="batch_file_uploader"
                    )

                    if uploaded_file:
                        try:
                            if uploaded_file.name.endswith('.txt'):
                                questions_text = uploaded_file.read().decode('utf-8')
                                questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                                patient_infos = [""] * len(questions)
                            elif uploaded_file.name.endswith('.xlsx'):
                                df = pd.read_excel(uploaded_file)
                                questions = df.iloc[:, 0].dropna().astype(str).tolist()
                                patient_infos = [""] * len(questions)

                            st.success(f"✓ {len(questions)} soru yüklendi")
                        except Exception as e:
                            st.error(f"❌ Dosya okuma hatası: {e}")

            # Preview questions
            if questions:
                st.info(f"📝 **{len(questions)} soru** işlenmeye hazır")
                with st.expander("👁️ Soruları önizle"):
                    for i, q in enumerate(questions[:10], 1):
                        st.caption(f"{i}. {q}")
                    if len(questions) > 10:
                        st.caption(f"... ve {len(questions) - 10} soru daha")

                # Process button
                if st.button("🚀 Batch İşleme Başlat", type="primary", use_container_width=True, key="batch_process_btn"):
                    # Initialize processor
                    if st.session_state.batch_processor is None:
                        st.session_state.batch_processor = BatchQueryProcessor()

                    # Process
                    with st.spinner(f"⏳ {len(questions)} soru işleniyor..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        results = []
                        for i, question in enumerate(questions, 1):
                            status_text.text(f"İşleniyor: {i}/{len(questions)} - {question[:50]}...")
                            progress_bar.progress(i / len(questions))

                            try:
                                current_patient_info = patient_infos[i-1] if patient_infos and i-1 < len(patient_infos) else batch_patient_info
                                
                                result = st.session_state.batch_processor.process_single_question(
                                    question=question,
                                    mode=batch_mode,
                                    patient_info=current_patient_info,
                                    threshold=batch_threshold,
                                    top_k=batch_top_k
                                )
                                results.append(result)

                                # Insert to database
                                st.session_state.batch_db.insert_query(result)

                            except Exception as e:
                                st.warning(f"⚠️ Hata (soru {i}): {str(e)}")

                        progress_bar.empty()
                        status_text.empty()

                    st.success(f"✓ {len(results)} soru başarıyla işlendi ve database'e kaydedildi!")

                    # Show summary
                    if results:
                        st.subheader("📊 Özet İstatistikler")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            results_with_citations = [r for r in results if r.get('citations')]
                            if results_with_citations:
                                avg_sim = sum(r['citations'][0]['similarity'] for r in results_with_citations) / len(results_with_citations)
                                st.metric("Ort. Benzerlik", f"{avg_sim:.3f}")
                            else:
                                st.metric("Ort. Benzerlik", "N/A")
                        with col2:
                            avg_tokens = sum(r['total_tokens'] for r in results) / len(results)
                            st.metric("Ort. Token", f"{avg_tokens:.0f}")
                        with col3:
                            total_time = sum(r.get('query_time', 0) for r in results)
                            st.metric("Toplam Süre", f"{total_time:.1f}s")

        # TAB 2: View Records
        with tab2:
            st.subheader("Kayıtlı Sorular")

            if st.session_state.batch_db.get_query_count() == 0:
                st.info("ℹ️ Henüz kayıtlı soru yok. 'Yeni Batch' sekmesinden soru ekleyin.")
            else:
                # Load data
                df = st.session_state.batch_db.get_all_queries()

                # Display options
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"Toplam {len(df)} kayıt")
                with col2:
                    show_full = st.checkbox("Tüm sütunları göster", value=False, key="batch_view_show_full")

                # Column selection
                if show_full:
                    display_cols = df.columns.tolist()
                else:
                    display_cols = [
                        'id', 'timestamp', 'question', 'answer', 'mode',
                        'context1_reference', 'context1_similarity',
                        'total_tokens'
                    ]
                    display_cols = [c for c in display_cols if c in df.columns]

                # Display
                st.dataframe(
                    df[display_cols],
                    use_container_width=True,
                    height=400
                )

                # Details expander
                with st.expander("👁️ Detaylı Görünüm (İlk 5 kayıt)"):
                    for idx, row in df.head(5).iterrows():
                        st.markdown(f"### Kayıt #{row['id']}")
                        st.markdown(f"**Soru:** {row['question']}")
                        st.markdown(f"**Cevap:** {row['answer'][:200]}...")
                        st.markdown(f"**Mod:** {row['mode']} | **Tokens:** {row['total_tokens']}")

                        # Contexts
                        st.markdown("**Kaynaklar:**")
                        for i in range(1, 6):
                            colname_ref = f'context{i}_reference'
                            if colname_ref in row and pd.notna(row[colname_ref]):
                                sim = row.get(f'context{i}_similarity', 0)
                                st.markdown(f"- [{i}] {row[colname_ref]} (Benzerlik: {sim:.3f})")

                        st.markdown("---")

        # TAB 3: Export
        with tab3:
            st.subheader("📥 Export")

            db_count_export = st.session_state.batch_db.get_query_count()
            if db_count_export == 0:
                st.info("ℹ️ Database boş. Export edilecek veri yok.")
            else:
                st.write(f"**{db_count_export} kayıt** export edilecek.")

                # Export options
                export_filename_base = f"batch_queries_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

                col1, col2 = st.columns(2)

                # CSV Export
                with col1:
                    st.markdown("### 📊 CSV Export")
                    
                    csv_filename = st.text_input(
                        "CSV Dosya adı:",
                        value=f"{export_filename_base}.csv",
                        key="csv_export_filename"
                    )

                    if st.button("📥 CSV İndir", type="primary", use_container_width=True, key="csv_export_btn"):
                        try:
                            output_path = f"data/exports/{csv_filename}"
                            Path("data/exports").mkdir(parents=True, exist_ok=True)

                            st.session_state.batch_db.export_to_csv(output_path)

                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ CSV Dosyasını İndir",
                                    data=f,
                                    file_name=csv_filename,
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="csv_export_download_btn"
                                )

                            st.success(f"✓ CSV Export tamamlandı: {csv_filename}")

                        except Exception as e:
                            st.error(f"❌ CSV export hatası: {str(e)}")

                # Excel Export
                with col2:
                    st.markdown("### 📗 Excel Export")
                    
                    excel_filename = st.text_input(
                        "Excel Dosya adı:",
                        value=f"{export_filename_base}.xlsx",
                        key="excel_export_filename"
                    )

                    if st.button("📥 Excel İndir", type="primary", use_container_width=True, key="excel_export_btn"):
                        try:
                            output_path = f"data/exports/{excel_filename}"
                            Path("data/exports").mkdir(parents=True, exist_ok=True)

                            st.session_state.batch_db.export_to_excel_smart(output_path)

                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Excel Dosyasını İndir",
                                    data=f,
                                    file_name=excel_filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                    key="excel_export_download_btn"
                                )

                            st.success(f"✓ Excel Export tamamlandı: {excel_filename}")

                        except Exception as e:
                            st.error(f"❌ Excel export hatası: {str(e)}")


    # -----------------------
    # Header
    # -----------------------
    st.title("🏥 Hereditary Angioedema Q&A System")
    st.markdown("**Sorular sorun, İngilizce tıbbi literatürden cevap alın**")

    # Sekmeler
    main_tab, batch_tab = st.tabs(["💬 Tekli Soru", "📊 Toplu Soru-Cevap"])
    
    with batch_tab:
        render_batch_query_tab()
    
    with main_tab:
        # -----------------------
        # Sidebar - Settings
        # -----------------------
        with st.sidebar:
            st.header("⚙️ Ayarlar")

            st.markdown("---")
            st.subheader("🎭 Kullanım Modu")
            mode = st.radio(
                "Mod seçin:",
                options=[
                    ("patient", "👤 Hasta Bilgilendirme", "8. sınıf seviye, empatik, anlaşılır"),
                    ("patient_personalized", "👥 Hasta Özel", "Klinik bilgilerinize özel, kişiselleştirilmiş"),
                    ("academic", "🎓 Akademisyen", "Teknik, detaylı, akademik üslup")
                ],
                format_func=lambda x: f"{x[1]}: {x[2]}",
                key="mode_select"
            )[0]

            # Save to session state
            st.session_state.mode = mode
            selected_mode = mode

            # Show mode description
            mode_descriptions = {
                "patient": "🎯 **Basit ve empatik dil**, 8. sınıf okuryazarlık seviyesi, jargonsuz açıklamalar",
                "patient_personalized": "🎯 **Kişiselleştirilmiş tavsiyeler**, klinik verilerinize göre özel bilgiler",
                "academic": "🎯 **Teknik ve bilimsel dil**, detaylı açıklamalar, akademik kaynak formatı"
            }
            st.info(mode_descriptions[selected_mode])

            # Patient info (if personalized mode)
            if selected_mode == "patient_personalized":
                st.markdown("---")
                st.subheader("👤 Klinik Bilgileriniz")
                patient_info = st.text_area(
                    "Klinik bilgiler (opsiyonel):",
                    value=st.session_state.patient_info,
                    height=120,
                    placeholder="Örn: HAE Tip 1, 35 yaş, son atak 2 hafta önce, Icatibant kullanıyor...",
                    help="Bu bilgiler LLM'e gönderilir ve cevabı kişiselleştirir. Boş bırakabilirsiniz.",
                    key="patient_info_input"
                )
                st.session_state.patient_info = patient_info

            st.markdown("---")
            st.subheader("🔍 Retrieval Parametreleri")
            top_k = st.slider("Context sayısı (k)", 1, 10, 5, key="top_k_slider")
            threshold = st.slider("Benzerlik eşiği", 0.0, 1.0, 0.50, 0.05, key="threshold_slider")

            st.markdown("---")
            st.subheader("🎨 Display Options")
            show_citations = st.checkbox("Citations göster", value=True, key="show_citations")
            show_reasoning = st.checkbox("Reasoning göster", value=False, key="show_reasoning")
            show_metadata = st.checkbox("Metadata göster", value=False, key="show_metadata")
            show_full_context = st.checkbox("Tüm context'leri göster", value=False, key="show_full_context")
            show_quality_check = st.checkbox("Kalite kontrolü yap", value=False, key="show_quality_check")

        # -----------------------
        # Query Interface
        # -----------------------
        st.subheader("💬 Soru Sorun")

        question = st.text_input(
            "Sorunuz:",
            value="",
            placeholder="Örn: HAE nedir? Nasıl tedavi edilir?",
            key="question_input"
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            query_button = st.button("🔍 Sorgula", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Temizle", use_container_width=True)
        with col3:
            if st.button("🔄 Yenile", use_container_width=True):
                st.rerun()

        if clear_button:
            st.session_state.query_results = None
            st.session_state.last_question = None
            st.rerun()

        # Process query
        if query_button and question:
            st.session_state.last_question = question

            try:
                with st.spinner("🔍 Benzer bağlamlar aranıyor..."):
                    # 1. Retrieve
                    results = retriever.retrieve(
                        query=question,
                        k=top_k,
                        threshold=threshold
                    )

                    formatted_contexts = ""
                    if not results:
                        st.warning(f"⚠️ Eşik ({threshold:.2f}) üzerinde benzerlik bulunamadı.")
                        st.info("""Bu soru muhtemelen Herediter Anjioödem (HAE) ile ilgili değil veya kaynaklarda bu spesifik bilgi yok.""")
                    else:
                        st.success(f"✓ {len(results)} ilgili bağlam bulundu")
                        formatted_contexts = retriever.format_contexts_for_llm(results)

                with st.spinner("🤖 DeepSeek R1 cevap üretiyor..."):
                    # 2. Generate with mode
                    patient_info_to_use = st.session_state.patient_info if selected_mode == "patient_personalized" else None

                    result = generator.generate_with_citations(
                        question=question,
                        retrieval_results=results,
                        formatted_contexts=formatted_contexts,
                        mode=selected_mode,
                        patient_info=patient_info_to_use,
                        validate_quality=show_quality_check
                    )

                    # Save to session state
                    st.session_state.query_results = {
                        'question': question,
                        'results': results,
                        'result': result,
                        'mode': selected_mode,
                        'patient_info': patient_info_to_use
                    }

            except Exception as e:
                st.error(f"❌ Hata: {e}")
                with st.expander("🐛 Debug Info"):
                    import traceback
                    st.code(traceback.format_exc())
                st.stop()

        # -----------------------
        # Display Results
        # -----------------------
        if st.session_state.query_results:
            results = st.session_state.query_results['results']
            result = st.session_state.query_results['result']
            question = st.session_state.query_results['question']
            display_mode = st.session_state.query_results['mode']

            # Show contexts BEFORE answer (if enabled AND results exist)
            if show_full_context and results:
                with st.expander("📚 Retrieved Contexts (Tam Bağlamlar)", expanded=False):
                    for i, (doc, score) in enumerate(results, 1):
                        st.markdown(f"### Context {i}")

                        # Metadata
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.caption(f"📄 **{doc.metadata['filename']}**")
                        with c2:
                            st.caption(f"📖 Sayfa {doc.metadata['page']}")
                        with c3:
                            st.caption(f"⭐ Skor: {score:.3f}")

                        if show_metadata:
                            st.caption(f"🗂️ Bölüm: {doc.metadata.get('section', 'Unknown')}")
                            st.caption(f"🔢 Chunk: {doc.metadata.get('chunk_id', '?')}/{doc.metadata.get('total_chunks', '?')}")
                            if doc.metadata.get('has_table'):
                                st.caption("📊 İçerik: Tablo içerir")

                        # Full text
                        st.markdown('<div class="full-context">', unsafe_allow_html=True)
                        st.markdown(doc.page_content)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown("---")

            # Main answer
            st.markdown("---")
            st.subheader("💡 Cevap")

            # Mode badge
            mode_names = {
                "patient": "👤 Hasta Bilgilendirme",
                "patient_personalized": "👥 Hasta Özel",
                "academic": "🎓 Akademisyen"
            }
            st.markdown(f'<span class="mode-badge mode-{display_mode.split("_")[0]}">{mode_names[display_mode]}</span>', unsafe_allow_html=True)

            # Answer text
            st.markdown(result["answer"])

            # Quality check (if enabled)
            if show_quality_check and "quality_check" in result:
                qc = result["quality_check"]
                with st.expander("✅ Kalite Kontrolü", expanded=False):
                    if "score" in qc:
                        st.markdown(f"**Skor:** {qc['score']}/10")
                    
                    if "reasoning" in qc:
                        st.markdown(f"**Değerlendirme:** {qc['reasoning']}")
                    
                    if "passed" in qc:
                        status = "✅ Geçti" if qc["passed"] else "❌ Sorunlar var"
                        st.markdown(f"**Durum:** {status}")
                    
                    if "warnings" in qc and qc["warnings"]:
                        st.markdown("**⚠️ Uyarılar:**")
                        for warning in qc["warnings"]:
                            st.markdown(f"- {warning}")
                    
                    if "critical_issues" in qc and qc["critical_issues"]:
                        st.markdown("**🚨 Kritik Sorunlar:**")
                        for issue in qc["critical_issues"]:
                            st.markdown(f"- {issue}")
                    
                    if "word_count" in qc:
                        st.caption(f"📝 Kelime sayısı: {qc['word_count']}")

            # Citations (if enabled and exist)
            if show_citations and result.get("citations"):
                st.markdown("---")
                st.subheader("📚 Kaynaklar")

                for i, citation in enumerate(result["citations"], 1):
                    # Use reference if available, otherwise use filename
                    ref_text = citation.get('reference', citation.get('filename', f'Kaynak {i}'))
                    with st.expander(f"[{i}] {ref_text}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"📄 **{citation['filename']}**")
                        with col2:
                            st.caption(f"📖 Sayfa {citation['page']}")
                        with col3:
                            st.caption(f"⭐ Benzerlik: {citation['similarity']:.3f}")

                        st.caption(f"🗂️ Bölüm: {citation['section']}")
                        if citation.get('has_table'):
                            st.caption("📊 Bu bölüm tablo içeriyor")

            # Reasoning (if enabled)
            if show_reasoning and result.get("reasoning"):
                st.markdown("---")
                with st.expander("🧠 Reasoning (DeepSeek R1)", expanded=False):
                    st.markdown(result["reasoning"])

            # Save to database and Export
            st.markdown("---")
            st.subheader("💾 Kaydetme ve Export")

            # First row: Save to database
            col_db1, col_db2 = st.columns([2, 1])
            
            with col_db1:
                st.markdown("**📊 Database'e Kaydet**")
                st.caption("Bu soruyu ve cevabını database'e kaydedin. Daha sonra 'Toplu Soru-Cevap > Kayıtlar' sekmesinden görebilirsiniz.")
            
            with col_db2:
                if st.button("💾 Database'e Kaydet", type="primary", use_container_width=True, key="save_to_db_btn"):
                    try:
                        # BatchQueryDB'yi initialize et
                        if 'single_query_db' not in st.session_state:
                            st.session_state.single_query_db = BatchQueryDB()
                        
                        # Database'e kaydet
                        db = st.session_state.single_query_db
                        
                        # Prepare data for database
                        db_data = {
                            'question': question,
                            'answer': result['answer'],
                            'mode': display_mode,
                            'threshold': threshold,
                            'top_k': top_k,
                            'patient_info': st.session_state.patient_info if display_mode == "patient_personalized" else "",
                            'citations': result.get('citations', []),
                            'retrieval_results': results,
                            'reasoning': result.get('reasoning', ''),
                            'prompt_tokens': result.get('prompt_tokens', 0),
                            'completion_tokens': result.get('completion_tokens', 0),
                            'total_tokens': result.get('total_tokens', 0),
                            'query_time': 0.0
                        }
                        
                        row_id = db.insert_query(db_data)
                        st.success(f"✅ Database'e kaydedildi! (ID: {row_id})")
                        st.info("💡 'Toplu Soru-Cevap > Kayıtlar' sekmesinden tüm kayıtları görebilirsiniz.")
                    
                    except Exception as e:
                        st.error(f"❌ Kaydetme hatası: {e}")

            st.markdown("---")

            # Second row: Export options
            st.markdown("**📥 Dosya Olarak İndir**")
            
            c1, c2, c3 = st.columns(3)

            with c1:
                # JSON export
                export_data = {
                    "question": question,
                    "answer": result["answer"],
                    "mode": display_mode,
                    "citations": result.get("citations", []),
                    "reasoning": result.get("reasoning", ""),
                    "metadata": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["total_tokens"],
                        "temperature": result.get("temperature", 0.1)
                    }
                }

                st.download_button(
                    label="📄 JSON İndir",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"answer_{question[:20]}.json",
                    mime="application/json",
                    key="export_json_download_btn"
                )

            with c2:
                # Markdown export
                md_content = f"# {question}\n\n"
                md_content += f"**Mod:** {mode_names[display_mode]}\n\n"
                md_content += f"## Cevap\n\n{result['answer']}\n\n"

                if result.get("citations"):
                    md_content += "## Kaynaklar\n\n"
                    for i, cit in enumerate(result["citations"], 1):
                        ref_text = cit.get('reference', cit.get('filename', f'Kaynak {i}'))
                        md_content += f"{i}. {ref_text} (Sayfa {cit['page']})\n"

                st.download_button(
                    label="📝 Markdown İndir",
                    data=md_content,
                    file_name=f"answer_{question[:20]}.md",
                    mime="text/markdown",
                    key="export_md_download_btn"
                )

            with c3:
                # Copy to clipboard
                if st.button("📋 Metni Kopyala", key="copy_answer_btn"):
                    st.text_area(
                        "Kopyalanacak Metin",
                        result["answer"],
                        height=200,
                        key="copy_text_area"
                    )

            # Token usage
            st.caption(
                f"🪙 Token kullanımı: {result['total_tokens']} total "
                f"({result['prompt_tokens']} prompt + {result['completion_tokens']} completion) | "
                f"Temperature: {result.get('temperature', 0.1)}"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>
        🌐 Cross-lingual RAG | 🧬 BGE-M3 (1024-dim) | 🤖 DeepSeek R1 (Reasoning) | 💾 Faiss Vector Store<br>
        📚 146 PDF Documents | 7190 Semantic Chunks | GPT-4o-mini Guided Chunking<br>
        🎭 3 Modes: Hasta Bilgilendirme | Hasta Özel | Akademisyen
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
