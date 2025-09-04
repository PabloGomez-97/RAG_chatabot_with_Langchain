import streamlit as st
import os
import glob
from pathlib import Path
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Importar módulos locales
from config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, OPENAI_MODELS, 
    TMP_DIR, LOCAL_VECTOR_STORE_DIR
)
from core import (
    enhanced_seemann_document_loader_v2, 
    create_enhanced_conversational_chain_v2,
    create_enhanced_seemann_retriever,
    multi_query_retriever,
    validate_response_completeness,
    enhance_query_for_completeness,
    EnhancedFreightParser
)

####################################################################
#            CONFIGURACIÓN STREAMLIT
####################################################################

st.set_page_config(
    page_title="Seemann Group v2.0 - Consultor Avanzado",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 Seemann Group v2.0 - Consultor Avanzado de Tarifas")
st.markdown("*Sistema inteligente con capacidades exhaustivas de búsqueda y análisis*")

####################################################################
#            FUNCIONES DE INTERFAZ
####################################################################

def get_enhanced_seemann_response_v2(prompt):
    """Función principal de respuesta v2.0 con búsqueda exhaustiva"""
    try:
        with st.spinner("🔍 Ejecutando búsqueda exhaustiva v2.0..."):
            
            # Búsqueda multi-query
            if hasattr(st.session_state, 'vector_store'):
                all_relevant_docs = multi_query_retriever(st.session_state.vector_store, prompt)
                
                with st.expander("🔍 **Análisis de búsqueda exhaustiva**", expanded=False):
                    st.write(f"**Total documentos analizados:** {len(all_relevant_docs)}")
                    
                    sources_count = {}
                    processing_methods = {}
                    
                    for doc in all_relevant_docs:
                        source_name = Path(doc.metadata.get("source", "")).name
                        sources_count[source_name] = sources_count.get(source_name, 0) + 1
                        
                        method = doc.metadata.get("processing_method", "unknown")
                        processing_methods[method] = processing_methods.get(method, 0) + 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Por archivo:**")
                        for source, count in sources_count.items():
                            st.write(f"• {source}: {count} registros")
                    
                    with col2:
                        st.write("**Por método de procesamiento:**")
                        for method, count in processing_methods.items():
                            icon = "🔄" if "vertical" in method else "📊"
                            st.write(f"{icon} {method}: {count}")
            
            # Ejecutar chain principal
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            
            # Validar completitud
            validation = validate_response_completeness(answer, prompt, response.get("source_documents", []))
            
            # Segunda búsqueda si es necesario
            if validation['completeness'] < 0.7:
                st.warning("⚠️ Respuesta incompleta detectada. Ejecutando búsqueda adicional...")
                
                enhanced_query = enhance_query_for_completeness(prompt, answer)
                response2 = st.session_state.chain.invoke({"question": enhanced_query})
                
                if len(response2.get("source_documents", [])) > len(response.get("source_documents", [])):
                    answer = response2["answer"]
                    response["source_documents"].extend(response2.get("source_documents", []))
            
            # Agregar al historial
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Mostrar conversación
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Métricas de completitud
                completeness_score = validation.get('completeness', 0)
                if completeness_score >= 0.8:
                    st.success(f"✅ **Alta completitud** ({completeness_score:.0%}) - Análisis exhaustivo completado")
                elif completeness_score >= 0.5:
                    st.warning(f"⚠️ **Completitud media** ({completeness_score:.0%}) - Información parcial")
                else:
                    st.error(f"🔍 **Completitud baja** ({completeness_score:.0%}) - Información limitada")
                
                # Advertencias
                if validation.get('warnings'):
                    with st.expander("⚠️ **Advertencias de completitud**"):
                        for warning in validation['warnings']:
                            st.write(f"• {warning}")
                
                # Análisis de fuentes mejorado
                with st.expander("📋 **Análisis detallado de fuentes**"):
                    sources = response.get("source_documents", [])
                    if sources:
                        # Estadísticas
                        csv_vertical = sum(1 for doc in sources if "vertical" in doc.metadata.get("processing_method", ""))
                        csv_standard = sum(1 for doc in sources if "standard" in doc.metadata.get("processing_method", ""))
                        pdf_count = sum(1 for doc in sources if doc.metadata.get("document_type") == "pdf")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔄 CSV Verticales", csv_vertical)
                        with col2:
                            st.metric("📊 CSV Estándar", csv_standard)
                        with col3:
                            st.metric("📄 PDFs", pdf_count)
                        
                        # Detalle por archivo y carrier
                        st.write("**Fuentes por naviera:**")
                        carrier_sources = {}
                        
                        for doc in sources:
                            carrier = doc.metadata.get("carrier", "N/A")
                            source_name = Path(doc.metadata.get("source", "")).name
                            processing_method = doc.metadata.get("processing_method", "")
                            
                            if carrier not in carrier_sources:
                                carrier_sources[carrier] = []
                            
                            carrier_sources[carrier].append({
                                'source': source_name,
                                'method': processing_method
                            })
                        
                        for carrier, sources_list in carrier_sources.items():
                            if carrier and carrier != "N/A":
                                unique_sources = list(set([s['source'] for s in sources_list]))
                                methods = list(set([s['method'] for s in sources_list]))
                                method_icon = "🔄" if any("vertical" in m for m in methods) else "📊"
                                
                                st.write(f"{method_icon} **{carrier}:** {', '.join(unique_sources)}")
                    
                    else:
                        st.error("❌ No se encontraron fuentes relevantes")
                        st.info("💡 **Sugerencias para mejorar búsqueda:**")
                        st.write("• Especifica puertos exactos (Shanghai, San Antonio)")
                        st.write("• Incluye nombres de navieras (COSCO, MSK, CMA CGM)")
                        st.write("• Menciona tipo de contenedor (20', 40', FCL)")
                        st.write("• Usa términos como 'comparar', 'opciones', 'alternativas'")
                
    except Exception as e:
        st.error(f"❌ **Error en sistema v2.0:** {str(e)}")
        st.info("🔧 **Diagnóstico avanzado:**")
        st.write("• Verifica base de datos cargada con archivos v2.0")
        st.write("• Revisa conexión OpenAI API")
        st.write("• Confirma formato de archivos CSV")
        
        with st.expander("🐛 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

def enhanced_sidebar_seemann_v2():
    """Interfaz lateral v2.0"""
    with st.sidebar:
        st.markdown("### 🚀 **Sistema v2.0 Avanzado**")
        st.success("""
        ✅ Parser CSV vertical/horizontal
        ✅ Búsqueda multi-query exhaustiva
        ✅ Validación de completitud
        ✅ Extracción tarifas combinadas
        ✅ Soporte puertos múltiples
        ✅ Análisis de fuentes detallado
        """)
        
        st.markdown("---")
        
        # Estado del sistema
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API conectada")
        else:
            st.error("❌ API Key no encontrada")
            st.info("📝 Agrega `OPENAI_API_KEY` a tu archivo `.env`")
            return

    # Tabs mejoradas
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Crear v2.0", "📂 Cargar", "📊 Estadísticas", "🧪 Test"])

    with tab1:
        st.markdown("### 📤 Crear Base de Datos v2.0")
        
        st.session_state.uploaded_file_list = st.file_uploader(
            "Selecciona archivos para procesamiento avanzado:",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv", "xlsx"],
            help="CSVs serán procesados con parser v2.0 (vertical + horizontal)"
        )
        
        st.session_state.vector_store_name = st.text_input(
            "📊 Nombre Base de Datos v2.0:",
            placeholder="ej: seemann_v2_tarifas_2025",
            help="Incluye v2 para identificar versión avanzada"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Crear con Sistema v2.0", type="primary"):
                enhanced_chain_RAG_blocks_v2()
        with col2:
            if st.button("🗑️ Limpiar"):
                delete_temp_files()

    with tab2:
        st.markdown("### 📂 Cargar Base de Datos")
        
        available_stores = [
            f.name for f in LOCAL_VECTOR_STORE_DIR.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ]
        
        if available_stores:
            st.session_state.selected_vectorstore_name = st.selectbox(
                "🗂️ Bases disponibles:",
                options=[""] + available_stores
            )
        else:
            st.info("🔍 No hay bases de datos disponibles")
        
        if st.button("📖 Cargar Base de Datos", type="primary"):
            load_existing_vectorstore_v2()

    with tab3:
        st.markdown("### 📊 Estadísticas del Sistema v2.0")
        
        if hasattr(st.session_state, 'vector_store'):
            try:
                collection_count = st.session_state.vector_store._collection.count()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Total Docs", collection_count)
                with col2:
                    st.metric("🤖 Modelo", st.session_state.get('selected_model', 'gpt-4o'))
                with col3:
                    st.metric("🌡️ Temperatura", f"{st.session_state.get('temperature', 0.05)}")
            except:
                st.info("Carga una base de datos para ver estadísticas")
        else:
            st.info("No hay base de datos cargada")

    with tab4:
        st.markdown("### 🧪 Test Parser v2.0")
        if st.button("Ejecutar Test"):
            test_parser_improvements()

def enhanced_chain_RAG_blocks_v2():
    """Pipeline v2.0 con todas las mejoras"""
    
    if not OPENAI_API_KEY:
        st.error("❌ Configura OpenAI API key")
        return

    if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
        st.error("❌ Selecciona archivos y nombre de base de datos")
        return
    
    with st.spinner("🔄 Procesando con sistema v2.0..."):
        try:
            # Limpiar y guardar archivos
            delete_temp_files()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📤 Guardando archivos...")
            for i, uploaded_file in enumerate(st.session_state.uploaded_file_list):
                temp_file_path = TMP_DIR / uploaded_file.name
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_file_list) * 0.2)
            
            # Procesar con sistema v2.0
            status_text.text("🔍 Procesando con parser v2.0...")
            documents = enhanced_seemann_document_loader_v2()
            progress_bar.progress(0.4)
            
            if not documents:
                st.error("❌ No se procesaron documentos")
                return
            
            # Estadísticas detalladas
            csv_vertical = sum(1 for doc in documents if "vertical_parser_v2" in doc.metadata.get("processing_method", ""))
            csv_standard = sum(1 for doc in documents if "standard_parser_v2" in doc.metadata.get("processing_method", ""))
            pdf_docs = sum(1 for doc in documents if doc.metadata.get("document_type") == "pdf")
            
            st.success("📊 **Procesamiento v2.0 completado:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔄 CSV Verticales", csv_vertical)
            with col2:
                st.metric("📊 CSV Estándar", csv_standard)
            with col3:
                st.metric("📄 PDFs", pdf_docs)
            
            # Crear chunks optimizados
            status_text.text("✂️ Creando chunks optimizados...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=250,
                separators=[
                    "\nCOTIZACIÓN MARÍTIMA FCL",
                    "\n=== INFORMACIÓN DE RUTA ===",
                    "\n=== TARIFAS EN USD ===",
                    "\n\n", "\n", " ", ""
                ]
            )
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.6)
            
            st.info(f"🔍 {len(chunks)} chunks optimizados creados")
            
            # Crear vectorstore
            status_text.text("🧠 Generando embeddings...")
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            persist_path.mkdir(parents=True, exist_ok=True)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path.as_posix(),
                collection_name="seemann_v2_enhanced"
            )
            progress_bar.progress(0.8)
            
            # Crear chain v2.0
            status_text.text("🔗 Configurando sistema v2.0...")
            st.session_state.retriever = create_enhanced_seemann_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_enhanced_conversational_chain_v2(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ **Sistema v2.0 creado exitosamente!**")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def load_existing_vectorstore_v2():
    """Cargar vectorstore existente v2.0"""
    if not OPENAI_API_KEY or not st.session_state.selected_vectorstore_name:
        st.error("❌ Configura API key y selecciona base de datos")
        return

    vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
    
    if not vectorstore_path.exists():
        st.error("❌ Base de datos no existe")
        return

    with st.spinner("📖 Cargando sistema v2.0..."):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path.as_posix(),
                collection_name="seemann_v2_enhanced"
            )
            
            collection_count = st.session_state.vector_store._collection.count()
            if collection_count == 0:
                st.warning("⚠️ Base de datos vacía")
                return
            
            st.session_state.retriever = create_enhanced_seemann_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_enhanced_conversational_chain_v2(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            st.success("✅ **Sistema v2.0 cargado exitosamente!**")
            st.info(f"📊 {collection_count} documentos indexados")
            
        except Exception as e:
            st.error(f"❌ Error cargando: {str(e)}")

def delete_temp_files():
    """Limpiar archivos temporales"""
    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        files = glob.glob(TMP_DIR.as_posix() + "/*")
        for f in files:
            try:
                os.remove(f)
            except:
                pass
    except:
        pass

def clear_chat_history():
    """Limpiar historial"""
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    if hasattr(st.session_state, 'memory') and st.session_state.memory:
        try:
            st.session_state.memory.clear()
        except:
            pass

def test_parser_improvements():
    """Test funcionalidad parser v2.0"""
    st.markdown("### 🧪 Test Parser v2.0")
    
    parser = EnhancedFreightParser()
    
    # Test casos
    test_cases = {
        "Tarifa combinada": "USD2300/2800 per 20/40",
        "Puertos múltiples": "QINGDAO/SHANGHAI/NINGBO/SHENZHEN", 
        "Free time": "Free time:21days",
        "Normalización MSK": "msk",
        "Normalización puerto": "shanghai"
    }
    
    results = {
        "Tarifa combinada": parser.parse_combined_rates(test_cases["Tarifa combinada"]),
        "Puertos múltiples": parser.parse_multiple_ports(test_cases["Puertos múltiples"]),
        "Free time": parser.extract_free_time(test_cases["Free time"]),
        "Normalización MSK": parser.normalize_carrier_name(test_cases["Normalización MSK"]),
        "Normalización puerto": parser.normalize_port_name(test_cases["Normalización puerto"])
    }
    
    for test_name, result in results.items():
        st.write(f"**{test_name}:** {result}")
    
    st.success("✅ Parser v2.0 funcionando correctamente")

def seemann_chatbot_v2():
    """Chatbot principal v2.0"""
    enhanced_sidebar_seemann_v2()
    
    st.markdown("---")
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Consultor Avanzado v2.0 - Seemann Group")
        if hasattr(st.session_state, 'chain'):
            st.success("🟢 Sistema v2.0 Activo - Búsqueda Exhaustiva Habilitada")
        else:
            st.warning("🟡 Crear/Cargar Base de Datos v2.0")

    # Mensajes
    if "messages" not in st.session_state:
        clear_chat_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input principal
    if prompt := st.chat_input("Consulta avanzada v2.0... (ej: ¿Qué opciones completas tengo desde China a San Antonio?)"):
        
        if not OPENAI_API_KEY:
            st.error("🔑 Configura OpenAI API key")
            st.stop()
        
        if not hasattr(st.session_state, 'chain'):
            st.warning("⚠️ Crea o carga base de datos v2.0")
            st.stop()
        
        # Ejecutar respuesta v2.0
        get_enhanced_seemann_response_v2(prompt)

####################################################################
#            FUNCIÓN PRINCIPAL
####################################################################

def main():
    """Función principal de la aplicación"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.selected_model = "gpt-4o"
        st.session_state.temperature = 0.05
    
    # Ejecutar sistema v2.0
    seemann_chatbot_v2()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚢 Seemann Group v2.0 - Sistema Avanzado | Parser Inteligente | Búsqueda Exhaustiva | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()