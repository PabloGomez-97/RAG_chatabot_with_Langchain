import streamlit as st
import traceback
from pathlib import Path

# Import backend services
from backend.aereo.config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, 
    LOCAL_VECTOR_STORE_DIR, TMP_DIR,
    extract_airports_from_query, detect_air_freight_query_type,
    get_airport_region, analyze_route_direction
)
from backend.aereo.core import (
    load_air_freight_documents,
    create_air_freight_chain,
    create_air_freight_retriever,
    validate_air_freight_route,
    analyze_air_freight_sources
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class AereoPageService:
    """Servicio para la página de carga aérea - maneja la lógica de UI"""
    
    def __init__(self):
        pass
    
    def get_air_freight_response(self, prompt):
        """Función principal para respuestas de carga aérea"""
        try:
            with st.spinner("🔍 Buscando tarifas de carga aérea..."):
                
                # Verificar que existe base de datos
                if not hasattr(st.session_state, 'vector_store'):
                    st.error("❌ No hay base de datos cargada")
                    return
                
                # Buscar documentos relevantes
                retriever = st.session_state.retriever
                docs = retriever.get_relevant_documents(prompt)
                
                # Analizar la consulta para detectar aeropuertos y rutas
                airport_info = extract_airports_from_query(prompt)
                query_type = detect_air_freight_query_type(prompt)
                
                # Validar ruta y mostrar análisis
                validation = validate_air_freight_route(prompt, docs)
                
                with st.expander("🔍 **Análisis de Consulta**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Tipo de consulta:** {query_type}")
                        st.write(f"**Estado:** {validation['verification_status']}")
                        if airport_info['aol_detected'] and airport_info['aod_detected']:
                            st.write(f"**Ruta detectada:** {airport_info['aol_detected']} → {airport_info['aod_detected']}")
                        elif airport_info['airports_found']:
                            st.write(f"**Aeropuertos detectados:** {', '.join(airport_info['airports_found'])}")
                    
                    with col2:
                        st.write(f"**Documentos encontrados:** {len(docs)}")
                        if validation['route_exists']:
                            st.success("✅ Ruta encontrada en tarifario")
                        elif validation['suggestions']:
                            st.info(f"💡 {len(validation['suggestions'])} rutas similares")
                        else:
                            st.warning("⚠️ Ruta no encontrada")
                
                # Ejecutar chain
                response = st.session_state.chain.invoke({"question": prompt})
                answer = response["answer"]
                
                # Agregar mensajes al historial
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    st.markdown(answer)

        except Exception as e:
            st.error(f"Error en consulta de carga aérea: {str(e)}")
            with st.expander("🔧 **Detalles técnicos del error**"):
                st.code(traceback.format_exc())

    def display_air_freight_metrics(self, validation: dict, airport_info: dict, source_docs: list):
        """Muestra métricas específicas de carga aérea"""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if validation['route_exists']:
                st.success("✅ **Ruta Disponible**")
            else:
                st.error("❌ **Ruta No Encontrada**")
        
        with col2:
            verified_docs = sum(1 for doc in source_docs if doc.metadata.get('verification_status') == 'VERIFIED')
            st.info(f"📄 **{verified_docs} Rutas Verificadas**")
        
        with col3:
            if airport_info['has_route_pattern']:
                operation = analyze_route_direction(airport_info['aol_detected'], airport_info['aod_detected'])
                st.info(f"🌍 **{operation}**")
            else:
                st.info("🔍 **Consulta General**")
        
        # Mostrar sugerencias si las hay
        if validation.get('suggestions'):
            with st.expander("💡 **Rutas Alternativas Disponibles**"):
                for suggestion in validation['suggestions'][:5]:
                    aol, aod = suggestion.split('_')
                    operation = analyze_route_direction(aol, aod)
                    st.write(f"✈️ **{aol} → {aod}** ({operation})")

    def enhanced_sidebar_air_freight(self):
        """Interfaz lateral para carga aérea"""
        with st.sidebar:
            st.markdown("### ✈️ **Sistema CRAFTTRANSWAY**")
            st.success("""
            ✅ Consulta de tarifas aéreas AOL → AOD
            ✅ Información de airlines y servicios
            ✅ Tarifas mínimas y por kilogramo
            ✅ Rutas de importación y exportación
            ✅ Verificación de disponibilidad
            ✅ Múltiples monedas (USD, EUR, GBP)
            """)
            
            st.markdown("---")
            
            if OPENAI_API_KEY:
                st.success("✅ OpenAI API conectada")
            else:
                st.error("❌ API Key no encontrada")
                return

        # Tabs para gestión
        tab2, tab1 = st.tabs(["📂 Cargar", "🔨 Crear Sistema"])

        with tab1:
            st.markdown("### 🔨 Crear Base de Datos de Carga Aérea")
            
            st.session_state.uploaded_file_list = st.file_uploader(
                "Selecciona archivo Excel CRAFTTRANSWAY:",
                accept_multiple_files=True,
                type=["xlsx", "xls"],
                help="Sube el archivo con las tarifas de carga aérea"
            )
            
            st.session_state.vector_store_name = st.text_input(
                "📊 Nombre Base de Datos:",
                placeholder="ej: crafttransway_2025",
                help="Nombre para la base de datos de tarifas aéreas"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔨 Crear Sistema Aéreo", type="primary"):
                    self.create_air_freight_system()
            with col2:
                if st.button("🗑️ Limpiar"):
                    self.delete_temp_files()

        with tab2:
            st.markdown("### 📂 Cargar Base de Datos Existente")
            
            available_stores = [
                f.name for f in LOCAL_VECTOR_STORE_DIR.iterdir() 
                if f.is_dir() and not f.name.startswith('.')
            ]
            
            if available_stores:
                st.session_state.selected_vectorstore_name = st.selectbox(
                    "🗂️ Bases de datos disponibles:",
                    options=[""] + available_stores
                )
            else:
                st.info("📁 No hay bases de datos disponibles")
            
            if st.button("📖 Cargar Base de Datos", type="primary"):
                self.load_existing_air_freight_vectorstore()

    def create_air_freight_system(self):
        """Pipeline de creación del sistema de carga aérea"""
        
        if not OPENAI_API_KEY:
            st.error("❌ Configura OpenAI API key")
            return

        if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
            st.error("❌ Selecciona archivo Excel y nombre de base de datos")
            return
        
        with st.spinner("🔨 Creando sistema de carga aérea..."):
            try:
                # Limpiar y guardar archivos
                self.delete_temp_files()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📤 Guardando archivos...")
                for i, uploaded_file in enumerate(st.session_state.uploaded_file_list):
                    temp_file_path = TMP_DIR / uploaded_file.name
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    progress_bar.progress((i + 1) / len(st.session_state.uploaded_file_list) * 0.1)
                
                # Procesar documentos de carga aérea
                status_text.text("✈️ Procesando tarifas de carga aérea...")
                documents = load_air_freight_documents()
                progress_bar.progress(0.5)
                
                if not documents:
                    st.error("❌ No se encontraron rutas válidas")
                    return
                
                # Mostrar estadísticas de procesamiento
                analysis = analyze_air_freight_sources(documents)
                
                st.success("✅ **Procesamiento Completado:**")
                cols = st.columns(4)
                with cols[0]:
                    st.metric("✈️ Rutas Procesadas", analysis['total_routes'])
                with cols[1]:
                    st.metric("🛫 Aeropuertos AOL", len(analysis['aol_airports']))
                with cols[2]:
                    st.metric("🛬 Aeropuertos AOD", len(analysis['aod_airports']))
                with cols[3]:
                    st.metric("🏢 Compañías", len(analysis['companies']))
                
                # Crear chunks
                status_text.text("📄 Creando chunks de documentos...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=100,
                    separators=[
                        "\nTARIFA CARGA AÉREA CRAFTTRANSWAY - RUTA VERIFICADA",
                        "\n=== INFORMACIÓN DE RUTA ===",
                        "\n=== TARIFAS VERIFICADAS ===",
                        "\n\n", "\n", " ", ""
                    ]
                )
                chunks = text_splitter.split_documents(documents)
                progress_bar.progress(0.7)
                
                st.info(f"📄 {len(chunks)} chunks creados")
                
                # Crear vectorstore
                status_text.text("🧠 Indexando rutas aéreas...")
                embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
                
                persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
                persist_path.mkdir(parents=True, exist_ok=True)
                
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_path.as_posix(),
                    collection_name="air_freight_routes"
                )
                progress_bar.progress(0.9)
                
                # Crear chain
                status_text.text("🔗 Configurando sistema de consultas...")
                st.session_state.retriever = create_air_freight_retriever(
                    vector_store=st.session_state.vector_store, k=15
                )
                
                st.session_state.chain, st.session_state.memory = create_air_freight_chain(
                    retriever=st.session_state.retriever
                )
                
                self.clear_chat_history()
                
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                
                st.success("✅ **Sistema de Carga Aérea creado exitosamente!**")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔧 Detalles del error"):
                    st.code(traceback.format_exc())

    def load_existing_air_freight_vectorstore(self):
        """Cargar vectorstore existente de carga aérea"""
        if not OPENAI_API_KEY or not st.session_state.selected_vectorstore_name:
            st.error("❌ Configura API key y selecciona base de datos")
            return

        vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
        
        if not vectorstore_path.exists():
            st.error("❌ Base de datos no existe")
            return

        with st.spinner("📖 Cargando sistema de carga aérea..."):
            try:
                embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
                
                st.session_state.vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=vectorstore_path.as_posix(),
                    collection_name="air_freight_routes"
                )
                
                collection_count = st.session_state.vector_store._collection.count()
                if collection_count == 0:
                    st.warning("⚠️ Base de datos vacía")
                    return
                
                st.session_state.retriever = create_air_freight_retriever(
                    vector_store=st.session_state.vector_store, k=15
                )
                
                st.session_state.chain, st.session_state.memory = create_air_freight_chain(
                    retriever=st.session_state.retriever
                )
                
                self.clear_chat_history()
                
                st.success("✅ **Sistema de carga aérea cargado exitosamente!**")
                st.info(f"📊 {collection_count} rutas indexadas")
                
            except Exception as e:
                st.error(f"❌ Error cargando: {str(e)}")
                with st.expander("🔧 Detalles del error"):
                    st.code(traceback.format_exc())

    def delete_temp_files(self):
        """Limpiar archivos temporales"""
        try:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            import glob
            import os
            files = glob.glob(TMP_DIR.as_posix() + "/*")
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass
        except:
            pass

    def clear_chat_history(self):
        """Limpiar historial de chat"""
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        if hasattr(st.session_state, 'memory') and st.session_state.memory:
            try:
                st.session_state.memory.clear()
            except:
                pass

    def render_chat_interface(self):
        """Renderiza la interfaz de chat principal"""
        
        # Header del sistema
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("💬 Consultor de Tarifas")

        # Manejo de ejemplos seleccionados
        if hasattr(st.session_state, 'ejemplo_selected'):
            prompt = st.session_state.ejemplo_selected
            del st.session_state.ejemplo_selected
            
            if hasattr(st.session_state, 'chain'):
                self.get_air_freight_response(prompt)
            else:
                st.warning("⚠️ Crea o carga sistema primero")

        # Mensajes del chat
        if "messages" not in st.session_state:
            self.clear_chat_history()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Input principal
        if prompt := st.chat_input("Consulta tarifas aéreas... (ej: ¿Cuál es la tarifa de MIA a SCL?)"):
            
            if not OPENAI_API_KEY:
                st.error("🔑 Configura OpenAI API key")
                st.stop()
            
            if not hasattr(st.session_state, 'chain'):
                st.warning("⚠️ Crea o carga sistema de carga aérea")
                st.stop()
            
            # Ejecutar consulta
            self.get_air_freight_response(prompt)


def render_aereo_page():
    """Función principal para renderizar la página de carga aérea"""
    service = AereoPageService()
    
    # Sidebar
    service.enhanced_sidebar_air_freight()
    
    st.markdown("---")
    
    # Chat interface
    service.render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    ✈️ Sistema de Consulta de Tarifas Aéreas | AOL → AOD | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)