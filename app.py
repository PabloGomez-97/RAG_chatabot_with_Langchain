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
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, detect_lcl_query_type,
    extract_ports_from_query, validate_lcl_document_relevance
)
from core import (
    load_lcl_excel_documents, 
    create_lcl_conversational_chain,
    create_lcl_retriever,
    multi_query_lcl_retriever,
    validate_lcl_response,
    analyze_lcl_sources
)

####################################################################
#            CONFIGURACIÓN STREAMLIT
####################################################################

st.set_page_config(
    page_title="Seemann Group - LCL Marítimo",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 Seemann Group - Sistema LCL Marítimo")  
st.markdown("*Consultor especializado en tarifas LCL (Less than Container Load) marítimas*")

####################################################################
#            FUNCIONES PRINCIPALES LCL
####################################################################

def get_lcl_response(prompt):
    """Función principal de respuesta LCL"""
    try:
        with st.spinner("🔍 Analizando consulta LCL..."):
            
            # Detectar tipo de consulta LCL
            query_type = detect_lcl_query_type(prompt)
            route_info = extract_ports_from_query(prompt)
            
            # Búsqueda especializada LCL
            if hasattr(st.session_state, 'vector_store'):
                all_relevant_docs = multi_query_lcl_retriever(
                    st.session_state.vector_store, prompt
                )
                
                # Filtrar documentos por relevancia LCL
                filtered_docs = validate_lcl_document_relevance(prompt, all_relevant_docs)
                
                with st.expander("🧠 **Análisis de Consulta LCL**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**Tipo de Consulta:** {query_type}")
                        
                        if route_info.get('has_route'):
                            st.success(f"**Ruta Detectada:** {route_info.get('origin_raw', '')} → {route_info.get('destination_raw', '')}")
                        else:
                            st.warning("**Consulta General:** Sin ruta específica")
                    
                    with col2:
                        st.write(f"**Documentos Encontrados:** {len(all_relevant_docs)}")
                        st.write(f"**Documentos Relevantes:** {len(filtered_docs)}")
                        
                        if len(filtered_docs) < len(all_relevant_docs):
                            st.warning(f"Se filtraron {len(all_relevant_docs) - len(filtered_docs)} documentos irrelevantes")
            
            # Ejecutar chain LCL
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            
            # Validar respuesta LCL
            validation = validate_lcl_response(
                answer, prompt, response.get("source_documents", [])
            )
            
            # Agregar al historial
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Mostrar conversación
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Métricas LCL
                display_lcl_metrics(validation, query_type, route_info)
                
                # Análisis de fuentes LCL
                with st.expander("📋 **Análisis de Fuentes LCL**"):
                    analyze_and_display_lcl_sources(response.get("source_documents", []))
                
    except Exception as e:
        st.error(f"Error en sistema LCL: {str(e)}")
        with st.expander("🔧 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

def display_lcl_metrics(validation: dict, query_type: str, route_info: dict):
    """Muestra métricas específicas para LCL"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        completeness = validation.get('completeness', 0)
        if completeness >= 0.8:
            st.success(f"✅ **Completitud** ({completeness:.0%})")
        elif completeness >= 0.5:
            st.warning(f"⚠️ **Completitud** ({completeness:.0%})")
        else:
            st.error(f"❌ **Completitud** ({completeness:.0%})")
    
    with col2:
        route_accuracy = validation.get('route_accuracy', 0)
        if route_accuracy >= 0.8:
            st.success(f"🎯 **Precisión Ruta** ({route_accuracy:.0%})")
        elif route_accuracy >= 0.5:
            st.warning(f"⚠️ **Precisión Ruta** ({route_accuracy:.0%})")
        else:
            st.error(f"❌ **Precisión Ruta** ({route_accuracy:.0%})")
    
    with col3:
        query_labels = {
            'route_specific': '🛣️ Ruta Específica',
            'region_query': '🌍 Por Región',
            'comparative': '📊 Comparativo',
            'country_specific': '🏳️ Por País',
            'general': '🔍 General'
        }
        st.info(f"**Tipo:** {query_labels.get(query_type, query_type)}")
    
    # Mostrar advertencias
    if validation.get('warnings'):
        with st.expander("⚠️ **Advertencias LCL**"):
            for warning in validation['warnings']:
                st.write(f"• {warning}")
    
    # Mostrar sugerencias
    if validation.get('suggestions'):
        with st.expander("💡 **Sugerencias**"):
            for suggestion in validation['suggestions']:
                st.write(f"• {suggestion}")

def analyze_and_display_lcl_sources(sources: list):
    """Analiza y muestra fuentes LCL"""
    
    if not sources:
        st.error("No se encontraron fuentes")
        return
    
    analysis = analyze_lcl_sources(sources)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Por Región:**")
        for region, count in analysis['regions'].items():
            st.write(f"🌍 {region}: {count}")
    
    with col2:
        st.write("**Puertos Principales:**")
        sorted_ports = sorted(analysis['ports'].items(), key=lambda x: x[1], reverse=True)[:5]
        for port, count in sorted_ports:
            st.write(f"⚓ {port}: {count}")
    
    with col3:
        st.write("**Por País:**")
        sorted_countries = sorted(analysis['countries'].items(), key=lambda x: x[1], reverse=True)[:5]
        for country, count in sorted_countries:
            st.write(f"🏳️ {country}: {count}")

def enhanced_sidebar_lcl():
    """Interfaz lateral especializada para LCL"""
    with st.sidebar:
        st.markdown("### 🚀 **Sistema LCL Marítimo**")
        st.success("""
        ✅ Procesamiento Excel especializado LCL
        ✅ Detección automática de rutas
        ✅ Información completa de costos
        ✅ Tiempos de tránsito y frecuencias
        ✅ Agentes locales por puerto
        ✅ Costos adicionales detallados
        ✅ Observaciones por ruta
        """)
        
        st.markdown("---")
        
        # Estado del sistema
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API conectada")
        else:
            st.error("❌ API Key no encontrada")
            return

    # Tabs para LCL
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Crear LCL", "📂 Cargar", "📊 Estadísticas", "🧪 Test"])

    with tab1:
        st.markdown("### 📤 Crear Base de Datos LCL")
        
        st.session_state.uploaded_file_list = st.file_uploader(
            "Selecciona archivo Excel de tarifas LCL:",
            accept_multiple_files=True,
            type=["xlsx", "xls"],
            help="El sistema procesará automáticamente las hojas regionales (AMERICA, EUROPA, NORTEAMERICA, ASIA)"
        )
        
        st.session_state.vector_store_name = st.text_input(
            "📊 Nombre Base de Datos LCL:",
            placeholder="ej: seemann_lcl_2025",
            help="Nombre identificativo para la base de datos LCL"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Crear Sistema LCL", type="primary"):
                create_lcl_rag_system()
        with col2:
            if st.button("🗑️ Limpiar"):
                delete_temp_files()

    with tab2:
        st.markdown("### 📂 Cargar Base de Datos LCL")
        
        available_stores = [
            f.name for f in LOCAL_VECTOR_STORE_DIR.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ]
        
        if available_stores:
            st.session_state.selected_vectorstore_name = st.selectbox(
                "🗂️ Bases LCL disponibles:",
                options=[""] + available_stores
            )
        else:
            st.info("📁 No hay bases de datos LCL disponibles")
        
        if st.button("📖 Cargar Base de Datos LCL", type="primary"):
            load_existing_lcl_vectorstore()

    with tab3:
        st.markdown("### 📊 Estadísticas del Sistema LCL")
        
        if hasattr(st.session_state, 'vector_store'):
            try:
                collection_count = st.session_state.vector_store._collection.count()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Total Registros LCL", collection_count)
                with col2:
                    st.metric("🤖 Modelo", st.session_state.get('selected_model', 'gpt-4o'))
                with col3:
                    st.metric("🌡️ Temperatura", f"{st.session_state.get('temperature', 0.05)}")
                    
                st.markdown("#### 📈 Estadísticas LCL")
                st.info("Sistema especializado en tarifas LCL marítimas activo")
            except:
                st.info("Carga una base de datos para ver estadísticas")
        else:
            st.info("No hay base de datos LCL cargada")

    with tab4:
        st.markdown("### 🧪 Test Sistema LCL")
        if st.button("Ejecutar Test LCL"):
            test_lcl_system()

def create_lcl_rag_system():
    """Pipeline de creación del sistema LCL"""
    
    if not OPENAI_API_KEY:
        st.error("❌ Configura OpenAI API key")
        return

    if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
        st.error("❌ Selecciona archivo Excel y nombre de base de datos")
        return
    
    with st.spinner("📄 Procesando Excel LCL..."):
        try:
            # Limpiar y guardar archivos
            delete_temp_files()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📤 Guardando archivos Excel...")
            for i, uploaded_file in enumerate(st.session_state.uploaded_file_list):
                temp_file_path = TMP_DIR / uploaded_file.name
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_file_list) * 0.2)
            
            # Procesar con sistema LCL
            status_text.text("🧠 Procesando tarifas LCL...")
            documents = load_lcl_excel_documents()
            progress_bar.progress(0.4)
            
            if not documents:
                st.error("❌ No se procesaron documentos LCL")
                return
            
            # Estadísticas de procesamiento
            analysis = analyze_lcl_sources(documents)
            
            st.success("📊 **Procesamiento LCL completado:**")
            cols = st.columns(4)
            with cols[0]:
                st.metric("📄 Total Registros", analysis['total'])
            with cols[1]:
                st.metric("🌍 Regiones", len(analysis['regions']))
            with cols[2]:
                st.metric("⚓ Puertos", len(analysis['ports']))
            with cols[3]:
                st.metric("🏳️ Países", len(analysis['countries']))
            
            # Crear chunks optimizados para LCL
            status_text.text("✂️ Creando chunks LCL...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                separators=[
                    "\nTARIFA LCL MARÍTIMA",
                    "\n=== INFORMACIÓN DE RUTA ===",
                    "\n=== TARIFAS Y COSTOS ===",
                    "\n\n", "\n", " ", ""
                ]
            )
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.6)
            
            st.info(f"📁 {len(chunks)} chunks LCL creados")
            
            # Crear vectorstore LCL
            status_text.text("🧠 Generando embeddings LCL...")
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            persist_path.mkdir(parents=True, exist_ok=True)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path.as_posix(),
                collection_name="seemann_lcl_maritime"
            )
            progress_bar.progress(0.8)
            
            # Crear chain LCL
            status_text.text("🔗 Configurando sistema conversacional LCL...")
            st.session_state.retriever = create_lcl_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_lcl_conversational_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ **Sistema LCL creado exitosamente!**")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def load_existing_lcl_vectorstore():
    """Cargar vectorstore LCL existente"""
    if not OPENAI_API_KEY or not st.session_state.selected_vectorstore_name:
        st.error("❌ Configura API key y selecciona base de datos")
        return

    vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
    
    if not vectorstore_path.exists():
        st.error("❌ Base de datos LCL no existe")
        return

    with st.spinner("📖 Cargando sistema LCL..."):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path.as_posix(),
                collection_name="seemann_lcl_maritime"
            )
            
            collection_count = st.session_state.vector_store._collection.count()
            if collection_count == 0:
                st.warning("⚠️ Base de datos LCL vacía")
                return
            
            st.session_state.retriever = create_lcl_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_lcl_conversational_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            st.success("✅ **Sistema LCL cargado exitosamente!**")
            st.info(f"📊 {collection_count} registros LCL indexados")
            
        except Exception as e:
            st.error(f"❌ Error cargando: {str(e)}")

def lcl_chatbot():
    """Chatbot principal LCL"""
    enhanced_sidebar_lcl()
    
    st.markdown("---")
    
    # Header LCL
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Consultor LCL Marítimo - Seemann Group")
        if hasattr(st.session_state, 'chain'):
            st.success("🟢 Sistema LCL Activo")
        else:
            st.warning("🟡 Crear/Cargar Base de Datos LCL")

    # Mensajes
    if "messages" not in st.session_state:
        clear_chat_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input principal
    if prompt := st.chat_input("Consulta tarifas LCL... (ej: ¿Cuánto cuesta desde Shanghai a Chile?)"):
        
        if not OPENAI_API_KEY:
            st.error("🔑 Configura OpenAI API key")
            st.stop()
        
        if not hasattr(st.session_state, 'chain'):
            st.warning("⚠️ Crea o carga base de datos LCL")
            st.stop()
        
        # Ejecutar respuesta LCL
        get_lcl_response(prompt)

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

def test_lcl_system():
    """Test funcionalidad LCL"""
    st.markdown("### 🧪 Test Sistema LCL")
    
    from config import extract_ports_from_query, normalize_port_name, detect_lcl_query_type
    
    # Test casos LCL
    test_cases = [
        "¿Cuánto cuesta desde Shanghai a San Antonio?",
        "Necesito tarifa desde Buenos Aires",
        "¿Qué opciones tengo desde Europa?",
        "Tiempo de tránsito desde Antwerp"
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        st.write(f"**Test {i}:** {test_query}")
        
        # Test detección
        query_type = detect_lcl_query_type(test_query)
        route_info = extract_ports_from_query(test_query)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"   **Tipo:** {query_type}")
        with col2:
            if route_info.get('has_route'):
                st.write(f"   **Ruta:** {route_info.get('origin_raw', '')} → {route_info.get('destination_raw', '')}")
            else:
                st.write("   **Ruta:** No específica")
    
    st.success("✅ Test LCL completado")

####################################################################
#            FUNCIÓN PRINCIPAL
####################################################################

def main():
    """Función principal de la aplicación LCL"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.selected_model = "gpt-4o"
        st.session_state.temperature = 0.05
    
    # Ejecutar sistema LCL
    lcl_chatbot()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚢 Seemann Group - Sistema LCL Marítimo | Especializado en Tarifas Less than Container Load | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()