import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.embeddings.clip_encoder import CLIPEmbeddingPipeline
from src.ingestion.video_processor import KeyframeExtractor
from src.reasoning.vlm_auditor import VLMAuditor
from src.retrieval.vector_search import VectorDatabaseManager

# Page Configuration
st.set_page_config(
    page_title="OmniVision-RAG | Multimodal Video Search & Audit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Enterprise Tech Look
st.markdown("""
<style>
    .metric-box {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .badge-risk-alto {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-risk-baixo {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipelines():
    """Initializes and caches model singletons to avoid re-loading on each UI interaction."""
    db_manager = VectorDatabaseManager()
    clip_encoder = CLIPEmbeddingPipeline()
    vlm_auditor = VLMAuditor()
    extractor = KeyframeExtractor(output_dir="storage/keyframes")
    return db_manager, clip_encoder, vlm_auditor, extractor


db_manager, clip_encoder, vlm_auditor, extractor = get_pipelines()

# Sidebar: System Diagnostics
with st.sidebar:
    st.title("⚡ OmniVision-RAG")
    st.caption("Multimodal Video Retrieval & Compliance Engine")
    st.divider()

    st.subheader("System Architecture")
    backend_type = "In-Memory Store" if db_manager.is_memory_fallback else "PostgreSQL + pgvector"
    st.markdown(f"**Database:** `{backend_type}`")
    st.markdown(f"**Index Type:** `HNSW (Cosine Similarity)`")
    st.markdown(f"**Encoder:** `{clip_encoder.model_name}` (512d)")
    
    total_events = db_manager.count_events()
    st.metric("Total Indexed Keyframes", total_events)
    st.divider()

    st.subheader("Synthetic Test Data")
    if st.button("Generate Sample Demo Video"):
        os.makedirs("storage/demo", exist_ok=True)
        demo_path = "storage/demo/sample_security_feed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(demo_path, fourcc, 10.0, (640, 480))

        labels = [
            ("Operador com Capacete", (0, 255, 0)),
            ("Area Segura Sem Risco", (255, 255, 0)),
            ("Operador Sem Capacete em Area Industrial", (0, 0, 255)),
            ("Carga Suspensa em Movimento", (255, 0, 255)),
        ]

        for sec, (text, color) in enumerate(labels):
            for f in range(10):  # 1 second each
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(frame, (80, 80), (560, 400), color, 3)
                cv2.putText(frame, f"CAM-01 [T={sec}s]", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, text, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                out.write(frame)

        out.release()
        st.success(f"Sample video generated: {demo_path}")
        st.session_state["sample_video_path"] = demo_path

# Main Dashboard View
st.title("🔍 OmniVision-RAG: Visual Search & Compliance")
st.markdown(
    "Pesquisa semântica em vídeo via **espaço latente multimodal (OpenCLIP)** e auditoria técnica automatizada com **Vision-Language Models (VLMs)**."
)

tab_search, tab_ingest, tab_audit = st.tabs([
    "🔎 1. Busca Semântica em Vídeo",
    "📥 2. Ingestão & Indexação Vetorial",
    "🤖 3. Parecer Multimodal (VLM)"
])

# Tab 1: Semantic Natural Language Search
with tab_search:
    st.subheader("Consulta em Linguagem Natural")
    col_query, col_topk = st.columns([4, 1])

    with col_query:
        query_text = st.text_input(
            "Digite o evento ou objeto visual desejado:",
            value="operador sem capacete",
            placeholder="Ex: pessoa correndo, empilhadeira, operador sem EPI...",
        )
    with col_topk:
        top_k = st.slider("Top K Resultados", min_value=1, max_value=8, value=3)

    if st.button("Executar Busca Vetorial", type="primary"):
        if not query_text.strip():
            st.warning("Por favor, digite uma consulta válida.")
        else:
            with st.spinner("Codificando texto e executando busca por similaridade de cosseno..."):
                t_start = time.perf_counter()
                query_vec = clip_encoder.encode_text(query_text)
                results = db_manager.search_similar_frames(query_vec, top_k=top_k)
                t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            st.success(f"Busca vetorial concluída em **{t_elapsed_ms} ms**! Encontrados {len(results)} frames relevantes.")
            st.session_state["last_search_results"] = results
            st.session_state["last_query"] = query_text

    # Display Results Grid
    results = st.session_state.get("last_search_results", [])
    if results:
        cols = st.columns(len(results))
        for idx, item in enumerate(results):
            with cols[idx]:
                st.markdown(f"**Similaridade: {item['similarity'] * 100:.1f}%**")
                st.caption(f"⏱️ Timestamp: `{item['timestamp_sec']}s`")
                if os.path.exists(item["storage_path"]):
                    img = Image.open(item["storage_path"])
                    st.image(img, use_container_width=True)
                else:
                    st.warning("Frame físico não encontrado.")

                if st.button(f"Auditar Frame #{item['id']}", key=f"btn_audit_{item['id']}"):
                    st.session_state["selected_audit_frame"] = item
                    st.info(f"Frame #{item['id']} selecionado. Acesse a aba 'Parecer Multimodal'!")

# Tab 2: Video Ingestion & Processing
with tab_ingest:
    st.subheader("Ingestão de Mídia & Vetorização HNSW")
    uploaded_file = st.file_uploader("Selecione um arquivo de vídeo (MP4 / AVI):", type=["mp4", "avi"])
    
    use_sample = st.checkbox("Usar vídeo de demonstração gerado pela barra lateral", value=True)
    video_source_path = None

    if uploaded_file is not None:
        os.makedirs("storage/uploads", exist_ok=True)
        video_source_path = os.path.join("storage/uploads", uploaded_file.name)
        with open(video_source_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"Arquivo recebido: `{uploaded_file.name}`")
    elif use_sample and st.session_state.get("sample_video_path"):
        video_source_path = st.session_state["sample_video_path"]
        st.info(f"Usando vídeo de teste sintético: `{video_source_path}`")

    sample_rate = st.slider("Taxa de amostragem (frames por segundo de vídeo):", 0.5, 5.0, 1.0, 0.5)

    if st.button("Iniciar Indexação Vetorial"):
        if not video_source_path or not os.path.exists(video_source_path):
            st.error("Nenhum vídeo disponível para processamento. Gere o vídeo de teste na barra lateral ou faça upload.")
        else:
            with st.spinner("Extraindo keyframes e gerando vetores de embedding..."):
                frames = extractor.extract_keyframes(video_source_path, fps_sample=sample_rate)
                progress_bar = st.progress(0.0)

                for i, f_info in enumerate(frames):
                    # Encode frame with OpenCLIP
                    vec = clip_encoder.encode_image(f_info["storage_path"])
                    # Save to PostgreSQL
                    db_manager.insert_event(
                        video_name=os.path.basename(video_source_path),
                        storage_path=f_info["storage_path"],
                        timestamp_sec=f_info["timestamp_sec"],
                        scene_embedding=vec,
                        metadata={"frame_index": f_info["frame_index"]},
                    )
                    progress_bar.progress((i + 1) / len(frames))

            st.success(f"Indexação concluída! {len(frames)} novos keyframes armazenados no índice vetorial.")
            st.rerun()

# Tab 3: Multimodal Compliance Auditor
with tab_audit:
    st.subheader("Auditoria de Conformidade e Raciocínio Visual (VLM)")
    selected_frame = st.session_state.get("selected_audit_frame")

    if not selected_frame:
        st.info("Nenhum frame selecionado. Execute uma pesquisa na Aba 1 e clique em 'Auditar Frame', ou selecione um frame indexado.")
    else:
        col_img, col_report = st.columns([1, 1])

        with col_img:
            st.markdown(f"#### Frame em Análise (Evento #{selected_frame['id']})")
            st.caption(f"Vídeo: `{selected_frame['video_name']}` | Timestamp: `{selected_frame['timestamp_sec']}s`")
            if os.path.exists(selected_frame["storage_path"]):
                st.image(Image.open(selected_frame["storage_path"]), use_container_width=True)

        with col_report:
            st.markdown("#### Parecer Estruturado Gerado por IA")
            query_ctx = st.session_state.get("last_query", "")

            with st.spinner("Executando raciocínio com Vision-Language Model..."):
                report = vlm_auditor.audit_frame(selected_frame["storage_path"], context_query=query_ctx)

            if report["irregularidade_detectada"]:
                st.markdown(f"<span class='badge-risk-alto'>IRREGULARIDADE DETECTADA — RISCO {report['nivel_risco']}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='badge-risk-baixo'>CONFORME — RISCO {report['nivel_risco']}</span>", unsafe_allow_html=True)

            st.markdown("##### Descrição Visual Objetiva")
            st.write(report["descricao_visual"])

            st.markdown("##### Ação Corretiva Sugerida")
            st.info(report["acao_corretiva_sugerida"])

            st.markdown("##### JSON Estruturado de Auditoria (Schema Pydantic)")
            st.json(report)
