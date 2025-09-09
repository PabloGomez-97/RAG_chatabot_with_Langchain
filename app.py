import streamlit as st
import os
import glob
from pathlib import Path
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Importar módulos locales MSL
from config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, OPENAI_MODELS, 
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, detect_msl_query_type,
    extract_ports_from_query, validate_msl_document_relevance
)
from core import (
    load_msl_excel_documents, 
    create_msl_conversational_chain,
    create_msl_retriever,
    multi_query_msl_retriever,
    validate_msl_response,
    analyze_msl_sources
)

####################################################################
#            CONFIGURACIÓN STREAMLIT
####################################################################

st.set_page_config(
    page_title="MSL - LCL Marítimo",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 MSL (Seemann Group) - Sistema LCL Marítimo")  
st.markdown("*Consultor especializado en tarifas LCL (Less than Container Load) marítimas MSL*")

####################################################################
#            FUNCIONES PRINCIPALES MSL
####################################################################

def get_msl_response(prompt):
    """Función principal de respuesta MSL"""
    try:
        with st.spinner("🔍 Analizando consulta LCL MSL..."):
            
            # PASO 1: Detectar tipo de consulta MSL
            query_type = detect_msl_query_type(prompt)
            route_info = extract_ports_from_query(prompt)
            
            # PASO 2: Buscar documentos MSL relevantes
            if hasattr(st.session_state, 'vector_store'):
                all_relevant_docs = multi_query_msl_retriever(st.session_state.vector_store, prompt)
                
                # PASO 3: Filtrar por relevancia MSL
                filtered_docs = validate_msl_document_relevance(prompt, all_relevant_docs)
                
                with st.expander("🧠 **Análisis MSL**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**Tipo Consulta:** {query_type}")
                        if route_info.get('has_route'):
                            origin = route_info.get('origin_raw', 'N/A')
                            st.write(f"**Ruta MSL:** {origin} → Chile")
                    
                    with col2:
                        st.write(f"**Documentos Totales:** {len(all_relevant_docs)}")
                        st.write(f"**Documentos Filtrados:** {len(filtered_docs)}")
            
            # PASO 4: Ejecutar chain MSL
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            
            # PASO 5: Validar respuesta MSL
            validation = validate_msl_response(
                answer, prompt, response.get("source_documents", [])
            )
            
            # PASO 6: Agregar al historial y mostrar
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Métricas MSL
                display_msl_metrics(validation, query_type, route_info)
                
                # Análisis de fuentes MSL
                with st.expander("📋 **Análisis de Fuentes MSL**"):
                    analyze_and_display_msl_sources(response.get("source_documents", []))
                
    except Exception as e:
        st.error(f"Error en sistema MSL: {str(e)}")
        with st.expander("🔧 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

def display_msl_metrics(validation: dict, query_type: str, route_info: dict):
    """Muestra métricas específicas para MSL"""
    
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
        with st.expander("⚠️ **Advertencias MSL**"):
            for warning in validation['warnings']:
                st.write(f"• {warning}")
    
    # Mostrar sugerencias
    if validation.get('suggestions'):
        with st.expander("💡 **Sugerencias**"):
            for suggestion in validation['suggestions']:
                st.write(f"• {suggestion}")

def analyze_and_display_msl_sources(sources: list):
    """Analiza y muestra fuentes MSL"""
    
    if not sources:
        st.error("No se encontraron fuentes MSL")
        return
    
    analysis = analyze_msl_sources(sources)
    
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

def enhanced_sidebar_msl():
    """Interfaz lateral especializada para MSL"""
    with st.sidebar:
        st.markdown("### 🚀 **Sistema MSL LCL Marítimo**")
        st.success("""
        ✅ Procesamiento especializado MSL
        ✅ Estructura específica del tarifario MSL
        ✅ Destino implícito: Chile (San Antonio/Valparaíso)
        ✅ Información completa de costos MSL
        ✅ Tiempos de tránsito y frecuencias
        ✅ Agentes locales por puerto
        ✅ Costos adicionales detallados (DDT, VGM)
        ✅ 4 regiones: AMERICA, EUROPA, NORTEAMERICA, ASIA
        """)
        
        st.markdown("---")
        
        # Estado del sistema
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API conectada")
        else:
            st.error("❌ API Key no encontrada")
            return

    # Tabs para MSL
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Crear MSL", "📂 Cargar", "📊 Estadísticas", "🧪 Test"])

    with tab1:
        st.markdown("### 📤 Crear Base de Datos MSL")
        
        st.session_state.uploaded_file_list = st.file_uploader(
            "Selecciona archivo Excel MSL:",
            accept_multiple_files=True,
            type=["xlsx", "xls"],
            help="El sistema procesará automáticamente las hojas MSL (AMERICA, EUROPA, NORTEAMERICA, ASIA)"
        )
        
        st.session_state.vector_store_name = st.text_input(
            "📊 Nombre Base de Datos MSL:",
            placeholder="ej: msl_lcl_2025",
            help="Nombre identificativo para la base de datos MSL"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Crear Sistema MSL", type="primary"):
                create_msl_rag_system()
        with col2:
            if st.button("🗑️ Limpiar"):
                delete_temp_files()

    with tab2:
        st.markdown("### 📂 Cargar Base de Datos MSL")
        
        available_stores = [
            f.name for f in LOCAL_VECTOR_STORE_DIR.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ]
        
        if available_stores:
            st.session_state.selected_vectorstore_name = st.selectbox(
                "🗂️ Bases MSL disponibles:",
                options=[""] + available_stores
            )
        else:
            st.info("📁 No hay bases de datos MSL disponibles")
        
        if st.button("📖 Cargar Base de Datos MSL", type="primary"):
            load_existing_msl_vectorstore()

    with tab3:
        st.markdown("### 📊 Estadísticas del Sistema MSL")
        
        if hasattr(st.session_state, 'vector_store'):
            try:
                collection_count = st.session_state.vector_store._collection.count()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Total Registros MSL", collection_count)
                with col2:
                    st.metric("🤖 Modelo", st.session_state.get('selected_model', 'gpt-4o'))
                with col3:
                    st.metric("🌡️ Temperatura", f"{st.session_state.get('temperature', 0.05)}")
                    
                st.markdown("#### 📈 Estadísticas MSL")
                st.info("Sistema especializado en tarifas LCL marítimas MSL activo")
            except:
                st.info("Carga una base de datos para ver estadísticas")
        else:
            st.info("No hay base de datos MSL cargada")

    with tab4:
        st.markdown("### 🧪 Test Sistema MSL")
        if st.button("Ejecutar Test MSL"):
            test_msl_system()

def create_msl_rag_system():
    """Pipeline de creación del sistema MSL"""
    
    if not OPENAI_API_KEY:
        st.error("❌ Configura OpenAI API key")
        return

    if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
        st.error("❌ Selecciona archivo Excel MSL y nombre de base de datos")
        return
    
    with st.spinner("📄 Procesando Excel MSL..."):
        try:
            # Limpiar y guardar archivos
            delete_temp_files()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📤 Guardando archivos Excel MSL...")
            for i, uploaded_file in enumerate(st.session_state.uploaded_file_list):
                temp_file_path = TMP_DIR / uploaded_file.name
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_file_list) * 0.2)
            
            # Procesar con sistema MSL
            status_text.text("🧠 Procesando tarifas MSL...")
            documents = load_msl_excel_documents()
            progress_bar.progress(0.4)
            
            if not documents:
                st.error("❌ No se procesaron documentos MSL")
                return
            
            # Estadísticas de procesamiento MSL
            analysis = analyze_msl_sources(documents)
            
            st.success("📊 **Procesamiento MSL completado:**")
            cols = st.columns(4)
            with cols[0]:
                st.metric("📄 Total Registros", analysis['total'])
            with cols[1]:
                st.metric("🌍 Regiones", len(analysis['regions']))
            with cols[2]:
                st.metric("⚓ Puertos", len(analysis['ports']))
            with cols[3]:
                st.metric("🏳️ Países", len(analysis['countries']))
            
            # Crear chunks optimizados para MSL
            status_text.text("✂️ Creando chunks MSL...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                separators=[
                    "\nTARIFA LCL MARÍTIMA - MSL",
                    "\n=== INFORMACIÓN DE RUTA MSL ===",
                    "\n=== TARIFAS MSL ===",
                    "\n\n", "\n", " ", ""
                ]
            )
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.6)
            
            st.info(f"📝 {len(chunks)} chunks MSL creados")
            
            # Crear vectorstore MSL
            status_text.text("🧠 Generando embeddings MSL...")
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            persist_path.mkdir(parents=True, exist_ok=True)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path.as_posix(),
                collection_name="msl_lcl_maritime"
            )
            progress_bar.progress(0.8)
            
            # Crear chain MSL
            status_text.text("🔗 Configurando sistema conversacional MSL...")
            st.session_state.retriever = create_msl_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_msl_conversational_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ **Sistema MSL creado exitosamente!**")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def load_existing_msl_vectorstore():
    """Cargar vectorstore MSL existente"""
    if not OPENAI_API_KEY or not st.session_state.selected_vectorstore_name:
        st.error("❌ Configura API key y selecciona base de datos")
        return

    vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
    
    if not vectorstore_path.exists():
        st.error("❌ Base de datos MSL no existe")
        return

    with st.spinner("📖 Cargando sistema MSL..."):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path.as_posix(),
                collection_name="msl_lcl_maritime"
            )
            
            collection_count = st.session_state.vector_store._collection.count()
            if collection_count == 0:
                st.warning("⚠️ Base de datos MSL vacía")
                return
            
            st.session_state.retriever = create_msl_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_msl_conversational_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            st.success("✅ **Sistema MSL cargado exitosamente!**")
            st.info(f"📊 {collection_count} registros MSL indexados")
            
        except Exception as e:
            st.error(f"❌ Error cargando: {str(e)}")

def msl_chatbot():
    """Chatbot principal MSL"""
    enhanced_sidebar_msl()
    
    st.markdown("---")
    
    # Header MSL
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Consultor LCL Marítimo - MSL (Seemann Group)")
        if hasattr(st.session_state, 'chain'):
            st.success("🟢 Sistema MSL Activo")
        else:
            st.warning("🟡 Crear/Cargar Base de Datos MSL")

    # Mensajes
    if "messages" not in st.session_state:
        clear_chat_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input principal
    if prompt := st.chat_input("Consulta tarifas MSL... (ej: ¿Cuánto cuesta desde Shanghai a Chile?)"):
        
        if not OPENAI_API_KEY:
            st.error("🔑 Configura OpenAI API key")
            st.stop()
        
        if not hasattr(st.session_state, 'chain'):
            st.warning("⚠️ Crea o carga base de datos MSL")
            st.stop()
        
        # Ejecutar respuesta MSL
        get_msl_response(prompt)

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

def test_msl_system():
    """Test funcionalidad MSL"""
    st.markdown("### 🧪 Test Sistema MSL")
    
    # Test casos MSL
    test_cases = [
        "¿Cuánto cuesta desde Shanghai a Chile?",
        "Necesito tarifa desde Buenos Aires",
        "¿Qué opciones MSL tengo desde Europa?",
        "Tiempo de tránsito desde Antwerp"
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        st.write(f"**Test {i}:** {test_query}")
        
        # Test detección MSL
        query_type = detect_msl_query_type(test_query)
        route_info = extract_ports_from_query(test_query)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"   **Tipo:** {query_type}")
        with col2:
            if route_info.get('has_route'):
                st.write(f"   **Ruta:** {route_info.get('origin_raw', '')} → Chile")
            else:
                st.write("   **Ruta:** No específica")
    
    st.success("✅ Test MSL completado")

####################################################################
#            FUNCIÓN PRINCIPAL
####################################################################

def main():
    """Función principal de la aplicación MSL"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.selected_model = "gpt-4o"
        st.session_state.temperature = 0.05
    
    # Ejecutar sistema MSL
    msl_chatbot()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚢 MSL (Seemann Group) - Sistema LCL Marítimo | Especializado en Tarifas Less than Container Load | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()