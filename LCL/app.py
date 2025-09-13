import streamlit as st
import os
import glob
from pathlib import Path
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Importar módulos de LCL marítimo
from .config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, OPENAI_MODELS, 
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, detect_lcl_maritime_query_type,
    extract_lcl_ports_from_query, get_lcl_port_region, analyze_lcl_route_direction,
    extract_region_from_query, normalize_port_name, matches_pod
)
from .core import (
    load_lcl_maritime_documents,
    create_lcl_maritime_chain,
    create_lcl_maritime_retriever,
    validate_lcl_maritime_route,
    analyze_lcl_maritime_sources
)

####################################################################
#            FUNCIONES PRINCIPALES
####################################################################

def get_lcl_maritime_response(prompt):
    """Función principal para respuestas de LCL marítimo"""
    try:
        with st.spinner("🔍 Buscando tarifas LCL marítimas..."):
            
            # Verificar que existe base de datos
            if not hasattr(st.session_state, 'vector_store'):
                st.error("❌ No hay base de datos cargada")
                return
            
            # Buscar documentos relevantes
            retriever = st.session_state.retriever
            docs = retriever.get_relevant_documents(prompt)
            
            # Analizar la consulta para detectar puertos y rutas
            port_info = extract_lcl_ports_from_query(prompt)
            query_type = detect_lcl_maritime_query_type(prompt)
            region_requested = extract_region_from_query(prompt)
            
            # Validar ruta y mostrar análisis
            validation = validate_lcl_maritime_route(prompt, docs)
            
            with st.expander("🔍 **Análisis de Consulta**", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Tipo de consulta:** {query_type}")
                    st.write(f"**Estado:** {validation['verification_status']}")
                    if port_info['pol_detected'] and port_info['pod_detected']:
                        st.write(f"**Ruta detectada:** {port_info['pol_detected']} → {port_info['pod_detected']}")
                    elif port_info['ports_found']:
                        st.write(f"**Puertos detectados:** {', '.join(port_info['ports_found'])}")
                    if region_requested != 'ALL':
                        st.write(f"**Región específica:** {region_requested}")
                
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
        st.error(f"Error en consulta de LCL marítimo: {str(e)}")
        with st.expander("🔧 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

def enhanced_sidebar_lcl_maritime():
    """Interfaz lateral para LCL marítimo"""
    with st.sidebar:
        st.markdown("### 🚢 **Sistema MSL LCL Marítimo**")
        st.success("""
        ✅ Consulta de tarifas LCL POL → POD
        ✅ Cobertura mundial (4 regiones)
        ✅ Tarifas TON/M3 y mínimos
        ✅ Agentes locales especializados
        ✅ Tiempos de tránsito
        ✅ Importaciones hacia Chile
        """)
        
        st.markdown("---")
        
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API conectada")
        else:
            st.error("❌ API Key no encontrada")
            return

    # Tabs para gestión
    tab2, tab1 = st.tabs(["📂 Cargar", "📁 Crear Sistema"])

    with tab1:
        st.markdown("### 📁 Crear Base de Datos LCL Marítimo")
        
        st.session_state.uploaded_file_list = st.file_uploader(
            "Selecciona archivo Excel MSL LCL:",
            accept_multiple_files=True,
            type=["xlsx", "xls"],
            help="Sube el archivo con las tarifas LCL marítimas multi-regionales"
        )
        
        st.session_state.vector_store_name = st.text_input(
            "📊 Nombre Base de Datos:",
            placeholder="ej: msl_lcl_2025",
            help="Nombre para la base de datos de tarifas LCL"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📁 Crear Sistema LCL", type="primary"):
                create_lcl_maritime_system()
        with col2:
            if st.button("🗑️ Limpiar"):
                delete_temp_files()

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
            load_existing_lcl_maritime_vectorstore()

def create_lcl_maritime_system():
    """Pipeline de creación del sistema LCL marítimo"""
    
    if not OPENAI_API_KEY:
        st.error("❌ Configura OpenAI API key")
        return

    if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
        st.error("❌ Selecciona archivo Excel y nombre de base de datos")
        return
    
    with st.spinner("📁 Creando sistema LCL marítimo..."):
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
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_file_list) * 0.1)
            
            # Procesar documentos LCL marítimo
            status_text.text("🚢 Procesando tarifas LCL marítimas...")
            documents = load_lcl_maritime_documents()
            progress_bar.progress(0.5)
            
            if not documents:
                st.error("❌ No se encontraron rutas válidas")
                return
            
            # Mostrar estadísticas de procesamiento
            analysis = analyze_lcl_maritime_sources(documents)
            
            st.success("✅ **Procesamiento Completado:**")
            cols = st.columns(4)
            with cols[0]:
                st.metric("🚢 Rutas Procesadas", analysis['total_routes'])
            with cols[1]:
                st.metric("🌍 Regiones", len(analysis['regional_distribution']))
            with cols[2]:
                st.metric("🚢 Puertos POL", len(analysis['pol_ports']))
            with cols[3]:
                st.metric("🏢 Compañías", len(analysis['companies']))
            
            # Mostrar distribución regional
            with st.expander("🌍 **Distribución Regional**"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Rutas por región:**")
                    for region, count in analysis['regional_distribution'].items():
                        st.write(f"📍 {region}: {count} rutas")
                
                with col2:
                    st.write("**Compañías:**")
                    for company in sorted(analysis['companies']):
                        st.write(f"🏢 {company}")
            
            # Mostrar resumen de puertos
            with st.expander("🚢 **Puertos Disponibles**"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**POL (Origen) - Ejemplos:**")
                    for pol in sorted(list(analysis['pol_ports']))[:10]:
                        region = get_lcl_port_region(pol)
                        st.write(f"🚢 {pol} ({region})")
                
                with col2:
                    st.write("**POD (Destino):**")
                    for pod in sorted(analysis['pod_ports']):
                        region = get_lcl_port_region(pod)
                        st.write(f"🏢 {pod} ({region})")
            
            # Crear chunks
            status_text.text("📄 Creando chunks de documentos...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100,
                separators=[
                    "\nTARIFA LCL MARÍTIMO MSL - RUTA VERIFICADA",
                    "\n=== INFORMACIÓN DE RUTA ===",
                    "\n=== TARIFAS VERIFICADAS ===",
                    "\n\n", "\n", " ", ""
                ]
            )
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.7)
            
            st.info(f"📄 {len(chunks)} chunks creados")
            
            # Crear vectorstore
            status_text.text("🧠 Indexando rutas LCL...")
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            persist_path.mkdir(parents=True, exist_ok=True)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path.as_posix(),
                collection_name="lcl_maritime_routes"
            )
            progress_bar.progress(0.9)
            
            # Crear chain
            status_text.text("🔗 Configurando sistema de consultas...")
            st.session_state.retriever = create_lcl_maritime_retriever(
                vector_store=st.session_state.vector_store, k=20
            )
            
            st.session_state.chain, st.session_state.memory = create_lcl_maritime_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ **Sistema LCL Marítimo creado exitosamente!**")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def load_existing_lcl_maritime_vectorstore():
    """Cargar vectorstore existente de LCL marítimo"""
    if not OPENAI_API_KEY or not st.session_state.selected_vectorstore_name:
        st.error("❌ Configura API key y selecciona base de datos")
        return

    vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
    
    if not vectorstore_path.exists():
        st.error("❌ Base de datos no existe")
        return

    with st.spinner("📖 Cargando sistema LCL marítimo..."):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path.as_posix(),
                collection_name="lcl_maritime_routes"
            )
            
            collection_count = st.session_state.vector_store._collection.count()
            if collection_count == 0:
                st.warning("⚠️ Base de datos vacía")
                return
            
            st.session_state.retriever = create_lcl_maritime_retriever(
                vector_store=st.session_state.vector_store, k=20
            )
            
            st.session_state.chain, st.session_state.memory = create_lcl_maritime_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            st.success("✅ **Sistema LCL marítimo cargado exitosamente!**")
            st.info(f"📊 {collection_count} rutas indexadas")
            
        except Exception as e:
            st.error(f"❌ Error cargando: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def lcl_maritime_chatbot():
    """Chatbot principal de LCL marítimo"""
    enhanced_sidebar_lcl_maritime()
    
    st.markdown("---")
    
    # Header del sistema
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Consultor de Tarifas")
        if hasattr(st.session_state, 'chain'):
            st.success("✅ Sistema Activo")

    # Manejo de ejemplos seleccionados
    if hasattr(st.session_state, 'ejemplo_selected'):
        prompt = st.session_state.ejemplo_selected
        del st.session_state.ejemplo_selected
        
        if hasattr(st.session_state, 'chain'):
            get_lcl_maritime_response(prompt)
        else:
            st.warning("⚠️ Crea o carga sistema primero")

    # Mensajes del chat
    if "messages" not in st.session_state:
        clear_chat_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input principal
    if prompt := st.chat_input("Consulta tarifas LCL... (ej: ¿Cuál es la tarifa desde SANTOS a Chile?)"):
        
        if not OPENAI_API_KEY:
            st.error("🔑 Configura OpenAI API key")
            st.stop()
        
        if not hasattr(st.session_state, 'chain'):
            st.warning("⚠️ Crea o carga sistema LCL marítimo")
            st.stop()
        
        # Ejecutar consulta
        get_lcl_maritime_response(prompt)

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
    """Limpiar historial de chat"""
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    if hasattr(st.session_state, 'memory') and st.session_state.memory:
        try:
            st.session_state.memory.clear()
        except:
            pass

####################################################################
#            FUNCIÓN PRINCIPAL
####################################################################

def main():
    """Función principal del sistema LCL marítimo"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Ejecutar chatbot LCL marítimo
    lcl_maritime_chatbot()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚢 MSL LCL Marítimo - Sistema de Consulta de Importaciones | POL → POD | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()