import streamlit as st
import os
import glob
from pathlib import Path
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Importar módulos de LCL marítimo
from config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, OPENAI_MODELS, 
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, detect_lcl_maritime_query_type,
    extract_lcl_ports_from_query, get_lcl_port_region, analyze_lcl_route_direction,
    extract_region_from_query, normalize_port_name, matches_pod
)
from core import (
    load_lcl_maritime_documents,
    create_lcl_maritime_chain,
    create_lcl_maritime_retriever,
    validate_lcl_maritime_route,
    analyze_lcl_maritime_sources
)

####################################################################
#            CONFIGURACIÓN STREAMLIT
####################################################################

st.set_page_config(
    page_title="MSL LCL Marítimo - Sistema de Consultas",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 MSL LCL MARÍTIMO - Sistema de Consulta de Tarifas de Importación")  
st.markdown("*Consulta tarifas LCL marítimas desde todo el mundo hacia Chile*")

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
                
                # Mostrar métricas y análisis
                display_lcl_maritime_metrics(validation, port_info, response.get("source_documents", []), region_requested)
                
                # Análisis detallado de fuentes
                with st.expander("📋 **Análisis Detallado de Rutas LCL**"):
                    display_lcl_route_analysis(response.get("source_documents", []), port_info)
                
    except Exception as e:
        st.error(f"Error en consulta de LCL marítimo: {str(e)}")
        with st.expander("🔧 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

def display_lcl_maritime_metrics(validation: dict, port_info: dict, source_docs: list, region_requested: str):
    """Muestra métricas específicas de LCL marítimo"""
    
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
        if port_info['has_route_pattern']:
            operation = analyze_lcl_route_direction(port_info['pol_detected'], port_info['pod_detected'])
            st.info(f"🌍 **{operation.split(' hacia ')[0] if ' hacia ' in operation else operation}**")
        else:
            st.info("🔍 **Consulta General**")
    
    # Mostrar información regional
    if region_requested != 'ALL':
        st.info(f"🌍 **Región específica: {region_requested}**")
    
    # Mostrar distribución regional de resultados
    if source_docs:
        regional_count = {}
        for doc in source_docs:
            region = doc.metadata.get('region', 'Unknown')
            regional_count[region] = regional_count.get(region, 0) + 1
        
        if len(regional_count) > 1:
            with st.expander("🌍 **Distribución Regional de Resultados**"):
                for region, count in regional_count.items():
                    st.write(f"📍 {region}: {count} rutas")
    
    # Mostrar sugerencias si las hay
    if validation.get('suggestions'):
        with st.expander("💡 **Rutas Alternativas Disponibles**"):
            for suggestion in validation['suggestions'][:5]:
                parts = suggestion.split('_')
                if len(parts) >= 3:
                    pol, pod, region = parts[0], parts[1], parts[2]
                    operation = analyze_lcl_route_direction(pol, pod)
                    st.write(f"🚢 **{pol} → {pod}** ({region}) - {operation}")

def display_lcl_route_analysis(sources: list, port_info: dict):
    """Muestra análisis detallado de rutas LCL encontradas"""
    
    if not sources:
        st.warning("⚠️ No se encontraron fuentes")
        return
    
    # Filtrar documentos relevantes
    relevant_docs = []
    if port_info.get('has_route_pattern'):
        # Buscar ruta específica
        pol_target = normalize_port_name(port_info['pol_detected'], "pol") if port_info['pol_detected'] else None
        pod_target = normalize_port_name(port_info['pod_detected'], "pod") if port_info['pod_detected'] else None

        for doc in sources:
            pol_doc = normalize_port_name(doc.metadata.get('pol', ''), "pol")
            pod_doc = normalize_port_name(doc.metadata.get('pod', ''), "pod")
            if pol_target and pod_target and pol_doc == pol_target and matches_pod(pod_target, pod_doc):
                relevant_docs.append(doc)

    
    elif port_info.get('ports_found'):
        ports_found_norm = [normalize_port_name(p, "any") for p in (port_info.get('ports_found') or [])]
        for doc in sources:
            pol_doc = normalize_port_name(doc.metadata.get('pol', ''), "pol")
            pod_doc = normalize_port_name(doc.metadata.get('pod', ''), "pod")
            if any( (p in {"SAN ANTONIO","VALPARAISO","SAI/VAP"} and matches_pod(p, pod_doc)) or     
                    (p == pol_doc) or (p == pod_doc)
                    for p in ports_found_norm ):
                relevant_docs.append(doc)

    else:
        relevant_docs = sources[:10]  # Mostrar primeros 10 si es consulta general
    
    if relevant_docs:
        st.write(f"**🔍 Análisis de {len(relevant_docs)} rutas LCL relevantes:**")
        st.markdown("---")
        
        for i, doc in enumerate(relevant_docs[:5], 1):  # Limitar a 5 para no saturar
            st.markdown(f"**Ruta {i}: {doc.metadata.get('pol', 'N/A')} → {doc.metadata.get('pod', 'N/A')} ({doc.metadata.get('region', 'N/A')})**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**POL:** {doc.metadata.get('pol', 'N/A')}")
                st.write(f"**País:** {doc.metadata.get('country', 'N/A')}")
                st.write(f"**POD:** {doc.metadata.get('pod', 'N/A')}")
                st.write(f"**Región:** {doc.metadata.get('region', 'N/A')}")
                st.write(f"**Compañía:** {doc.metadata.get('company', 'N/A')}")
            
            with col2:
                st.write(f"**Agente:** {doc.metadata.get('agent', 'No especificado')}")
                st.write(f"**Servicio:** {doc.metadata.get('service', 'No especificado')}")
                st.write(f"**Tiempo Tránsito:** {doc.metadata.get('transit_time', 'No especificado')}")
                st.write(f"**Frecuencia:** {doc.metadata.get('frequency', 'No especificado')}")
                st.write(f"**Fila Excel:** {doc.metadata.get('row_number', 'N/A')}")
            
            if i < len(relevant_docs[:5]):  # No agregar línea después del último
                st.markdown("---")
    else:
        # Mostrar análisis general de todas las fuentes
        analysis = analyze_lcl_maritime_sources(sources)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Puertos POL disponibles:**")
            for pol in sorted(list(analysis.get('pol_ports', set())))[:5]:
                region = get_lcl_port_region(pol)
                st.write(f"🚢 {pol} ({region})")
        
        with col2:
            st.write("**Puertos POD disponibles:**")
            for pod in sorted(list(analysis.get('pod_ports', set()))):
                region = get_lcl_port_region(pod)
                st.write(f"🏢 {pod} ({region})")
        
        with col3:
            st.write("**Distribución regional:**")
            for region, count in analysis.get('regional_distribution', {}).items():
                st.write(f"🌍 {region}: {count}")
            
            if analysis.get('companies'):
                st.write("**Compañías:**")
                for company in sorted(list(analysis.get('companies', set()))):
                    st.write(f"🏢 {company}")

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
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Crear Sistema", "📂 Cargar", "📊 Estadísticas", "🧪 Ejemplos"])

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

    with tab3:
        st.markdown("### 📊 Estadísticas del Sistema")
        
        if hasattr(st.session_state, 'vector_store'):
            try:
                collection_count = st.session_state.vector_store._collection.count()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🚢 Rutas LCL", collection_count)
                with col2:
                    st.metric("🤖 Modelo", "gpt-4o")
                with col3:
                    st.metric("🎯 Precisión", "Máxima")
                    
                st.markdown("#### 📈 Estado del Sistema")
                st.success("Sistema LCL marítimo activo")
            except:
                st.info("Carga una base de datos para ver estadísticas")
        else:
            st.info("No hay base de datos cargada")

    with tab4:
        st.markdown("### 🧪 Ejemplos de Consultas")
        
        ejemplos = [
            "¿Cuál es la tarifa desde SANTOS a Chile?",
            "¿Qué opciones hay desde Europa a San Antonio?",
            "¿Cuánto cuesta desde MIAMI por tonelada?",
            "¿Qué agentes tenemos en Asia?",
            "Tarifas desde Argentina a Chile",
            "¿Hay rutas directas desde HAMBURG?",
            "Comparar precios América vs Europa",
            "¿Cuál es el tiempo de tránsito desde SINGAPORE?"
        ]
        
        st.write("**💡 Prueba estas consultas:**")
        for ejemplo in ejemplos:
            if st.button(f"📝 {ejemplo}", key=f"ejemplo_{hash(ejemplo)}"):
                st.session_state.ejemplo_selected = ejemplo

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
        st.subheader("💬 Consultor de Tarifas LCL Marítimas MSL")
        if hasattr(st.session_state, 'chain'):
            st.success("✅ Sistema Activo")
        else:
            st.warning("📁 Crear/Cargar Sistema")

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