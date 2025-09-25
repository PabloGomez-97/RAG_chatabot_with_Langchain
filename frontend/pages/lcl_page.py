import streamlit as st
import traceback
from pathlib import Path

# Import backend services
from backend.lcl.config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, 
    LOCAL_VECTOR_STORE_DIR, TMP_DIR,
    extract_lcl_ports_from_query, detect_lcl_maritime_query_type,
    get_lcl_port_region, analyze_lcl_route_direction,
    extract_region_from_query, normalize_port_name, matches_pod
)
from backend.lcl.core import (
    load_lcl_maritime_documents,
    create_lcl_maritime_chain,
    create_lcl_maritime_retriever,
    validate_lcl_maritime_route,
    analyze_lcl_maritime_sources
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class LCLPageService:
    """Servicio para la página de LCL marítimo - maneja la lógica de UI"""
    
    def __init__(self):
        pass
    
    def get_lcl_maritime_response(self, prompt):
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
                    
                    # Mostrar métricas y análisis específicos de LCL
                    self.display_lcl_maritime_metrics(validation, port_info, response.get("source_documents", []), region_requested)
                    
                    # Análisis detallado de fuentes
                    with st.expander("📋 **Análisis Detallado de Rutas LCL**"):
                        self.display_lcl_route_analysis(response.get("source_documents", []), port_info)

        except Exception as e:
            st.error(f"Error en consulta de LCL marítimo: {str(e)}")
            with st.expander("🔧 **Detalles técnicos del error**"):
                st.code(traceback.format_exc())

    def display_lcl_maritime_metrics(self, validation: dict, port_info: dict, source_docs: list, region_requested: str):
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
        
        # Mostrar información de región
        if region_requested != 'ALL':
            st.info(f"🌎 **Región específica: {region_requested}**")
        
        # Mostrar distribución regional de resultados
        if source_docs:
            regions_found = {}
            for doc in source_docs:
                region = doc.metadata.get('region', 'Unknown')
                regions_found[region] = regions_found.get(region, 0) + 1
            
            if len(regions_found) > 1:
                with st.expander("🌎 **Distribución Regional de Resultados**"):
                    for region, count in sorted(regions_found.items()):
                        st.write(f"📍 **{region}:** {count} rutas")
        
        # Mostrar sugerencias si las hay
        if validation.get('suggestions'):
            with st.expander("💡 **Rutas Alternativas Disponibles**"):
                for suggestion in validation['suggestions'][:5]:
                    parts = suggestion.split('_')
                    if len(parts) >= 3:
                        pol, pod, region = parts[0], parts[1], parts[2]
                        operation = analyze_lcl_route_direction(pol, pod)
                        st.write(f"🚢 **{pol} → {pod}** ({region}) - {operation}")

    def display_lcl_route_analysis(self, sources: list, port_info: dict):
        """Muestra análisis detallado de rutas LCL encontradas"""
        
        if not sources:
            st.warning("⚠️ No se encontraron fuentes")
            return
        
        # Filtrar documentos relevantes
        relevant_docs = []
        if port_info.get('has_route_pattern'):
            # Buscar ruta específica
            pol_target = normalize_port_name(port_info['pol_detected'], "pol")
            pod_target = normalize_port_name(port_info['pod_detected'], "pod")
            
            for doc in sources:
                doc_pol = normalize_port_name(doc.metadata.get('pol', ''), "pol")
                doc_pod = normalize_port_name(doc.metadata.get('pod', ''), "pod")
                
                if doc_pol == pol_target and matches_pod(pod_target, doc_pod):
                    relevant_docs.append(doc)
        
        elif port_info.get('ports_found'):
            # Buscar documentos que contengan alguno de los puertos
            ports_found_norm = [normalize_port_name(p, "any") for p in port_info['ports_found']]
            
            for doc in sources:
                doc_pol = normalize_port_name(doc.metadata.get('pol', ''), "pol")
                doc_pod = normalize_port_name(doc.metadata.get('pod', ''), "pod")
                
                for p in ports_found_norm:
                    if p in {"SAN ANTONIO", "VALPARAISO", "SAI/VAP"}:
                        if matches_pod(p, doc_pod):
                            relevant_docs.append(doc)
                            break
                    else:
                        if p in (doc_pol, doc_pod):
                            relevant_docs.append(doc)
                            break
        else:
            relevant_docs = sources[:10]  # Mostrar primeros 10 si es consulta general
        
        if relevant_docs:
            st.write(f"**🔍 Análisis de {len(relevant_docs)} rutas LCL relevantes:**")
            st.markdown("---")
            
            for i, doc in enumerate(relevant_docs[:5], 1):  # Limitar a 5 para no saturar
                region = doc.metadata.get('region', 'N/A')
                st.markdown(f"**Ruta {i}: {doc.metadata.get('pol', 'N/A')} → {doc.metadata.get('pod', 'N/A')} ({region})**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**POL:** {doc.metadata.get('pol', 'N/A')}")
                    st.write(f"**País:** {doc.metadata.get('country', 'N/A')}")
                    st.write(f"**POD:** {doc.metadata.get('pod', 'N/A')}")
                    st.write(f"**Región:** {region}")
                    st.write(f"**Compañía:** {doc.metadata.get('company', 'N/A')}")
                    st.write(f"**Fila Excel:** {doc.metadata.get('row_number', 'N/A')}")
                
                with col2:
                    st.write(f"**Servicio:** {doc.metadata.get('service', 'No especificado')}")
                    st.write(f"**Agente Local:** {doc.metadata.get('agent', 'No especificado')}")
                    st.write(f"**Tiempo Tránsito:** {doc.metadata.get('transit_time', 'No especificado')}")
                    st.write(f"**Frecuencia:** {doc.metadata.get('frequency', 'No especificado')}")
                    st.write(f"**Moneda TON/M3:** {doc.metadata.get('ton_m3_currency', 'No especificado')}")
                    st.write(f"**Moneda Mínimo:** {doc.metadata.get('minimo_currency', 'No especificado')}")
                
                if i < len(relevant_docs[:5]):  # No agregar línea después del último
                    st.markdown("---")
        else:
            # Mostrar análisis general de todas las fuentes
            analysis = analyze_lcl_maritime_sources(sources)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Puertos POL disponibles (ejemplos):**")
                for pol in sorted(list(analysis.get('pol_ports', set())))[:8]:
                    region = get_lcl_port_region(pol)
                    st.write(f"🚢 {pol} ({region})")
            
            with col2:
                st.write("**Puertos POD disponibles:**")
                for pod in sorted(list(analysis.get('pod_ports', set()))):
                    region = get_lcl_port_region(pod)
                    st.write(f"🏢 {pod} ({region})")
            
            with col3:
                st.write("**Por región:**")
                for region, count in sorted(analysis.get('regional_distribution', {}).items()):
                    st.write(f"🌍 {region}: {count} rutas")
                
                if analysis.get('companies'):
                    st.write("**Compañías:**")
                    for company in sorted(list(analysis.get('companies', set())))[:3]:
                        st.write(f"🏢 {company}")

    def enhanced_sidebar_lcl_maritime(self):
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
        tab2, tab1 = st.tabs(["📂 Cargar", "🔨 Crear Sistema"])

        with tab1:
            st.markdown("### 🔨 Crear Base de Datos LCL Marítimo")
            
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
                if st.button("🔨 Crear Sistema LCL", type="primary"):
                    self.create_lcl_maritime_system()
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
                self.load_existing_lcl_maritime_vectorstore()

    def create_lcl_maritime_system(self):
        """Pipeline de creación del sistema LCL marítimo"""
        
        if not OPENAI_API_KEY:
            st.error("❌ Configura OpenAI API key")
            return

        if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
            st.error("❌ Selecciona archivo Excel y nombre de base de datos")
            return
        
        with st.spinner("🔨 Creando sistema LCL marítimo..."):
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
                
                self.clear_chat_history()
                
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                
                st.success("✅ **Sistema LCL Marítimo creado exitosamente!**")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔧 Detalles del error"):
                    st.code(traceback.format_exc())

    def load_existing_lcl_maritime_vectorstore(self):
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
                
                self.clear_chat_history()
                
                st.success("✅ **Sistema LCL marítimo cargado exitosamente!**")
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
            if hasattr(st.session_state, 'chain'):
                st.success("✅ Sistema Activo")

        # Manejo de ejemplos seleccionados
        if hasattr(st.session_state, 'ejemplo_selected'):
            prompt = st.session_state.ejemplo_selected
            del st.session_state.ejemplo_selected
            
            if hasattr(st.session_state, 'chain'):
                self.get_lcl_maritime_response(prompt)
            else:
                st.warning("⚠️ Crea o carga sistema primero")

        # Mensajes del chat
        if "messages" not in st.session_state:
            self.clear_chat_history()

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
            self.get_lcl_maritime_response(prompt)


def render_lcl_page():
    """Función principal para renderizar la página de LCL marítimo"""
    service = LCLPageService()
    
    # Sidebar
    service.enhanced_sidebar_lcl_maritime()
    
    st.markdown("---")
    
    # Chat interface
    service.render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚢 MSL LCL Marítimo - Sistema de Consulta de Importaciones | POL → POD | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)