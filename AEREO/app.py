import streamlit as st
import os
import glob
from pathlib import Path
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Importar módulos de carga aérea
from .config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, OPENAI_MODELS, 
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, detect_air_freight_query_type,
    extract_airports_from_query, get_airport_region, analyze_route_direction
)
from .core import (
    load_air_freight_documents,
    create_air_freight_chain,
    create_air_freight_retriever,
    validate_air_freight_route,
    analyze_air_freight_sources
)

####################################################################
#            CONFIGURACIÓN STREAMLIT
####################################################################

st.set_page_config(
    page_title="CRAFTTRANSWAY - Carga Aérea",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ CRAFTTRANSWAY - Sistema de Consulta de Tarifas Aéreas")  
st.markdown("*Consulta tarifas de carga aérea entre aeropuertos AOL → AOD*")

####################################################################
#            FUNCIONES PRINCIPALES
####################################################################

def get_air_freight_response(prompt):
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

def display_air_freight_metrics(validation: dict, airport_info: dict, source_docs: list):
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

def display_route_analysis(sources: list, airport_info: dict):
    """Muestra análisis detallado de rutas encontradas"""
    
    if not sources:
        st.warning("⚠️ No se encontraron fuentes")
        return
    
    # Filtrar documentos relevantes
    relevant_docs = []
    if airport_info.get('has_route_pattern'):
        # Buscar ruta específica
        aol_target = airport_info['aol_detected']
        aod_target = airport_info['aod_detected']
        
        for doc in sources:
            if (doc.metadata.get('aol') == aol_target and 
                doc.metadata.get('aod') == aod_target):
                relevant_docs.append(doc)
    
    elif airport_info.get('airports_found'):
        # Buscar documentos que contengan alguno de los aeropuertos
        for doc in sources:
            aol = doc.metadata.get('aol', '')
            aod = doc.metadata.get('aod', '')
            
            if any(airport in [aol, aod] for airport in airport_info['airports_found']):
                relevant_docs.append(doc)
    else:
        relevant_docs = sources[:10]  # Mostrar primeros 10 si es consulta general
    
    if relevant_docs:
        st.write(f"**🔍 Análisis de {len(relevant_docs)} rutas relevantes:**")
        st.markdown("---")
        
        for i, doc in enumerate(relevant_docs[:5], 1):  # Limitar a 5 para no saturar
            st.markdown(f"**Ruta {i}: {doc.metadata.get('aol', 'N/A')} → {doc.metadata.get('aod', 'N/A')}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**AOL:** {doc.metadata.get('aol', 'N/A')}")
                st.write(f"**País Origen:** {doc.metadata.get('pais_origen', 'N/A')}")
                st.write(f"**AOD:** {doc.metadata.get('aod', 'N/A')}")
                st.write(f"**País Destino:** {doc.metadata.get('pais_destino', 'N/A')}")
                st.write(f"**Compañía:** {doc.metadata.get('company', 'N/A')}")
            
            with col2:
                st.write(f"**Airline:** {doc.metadata.get('airline', 'No especificado')}")
                st.write(f"**Tipo Operación:** {doc.metadata.get('operation_type', 'N/A')}")
                st.write(f"**Región Origen:** {doc.metadata.get('aol_region', 'N/A')}")
                st.write(f"**Región Destino:** {doc.metadata.get('aod_region', 'N/A')}")
                st.write(f"**Fila Excel:** {doc.metadata.get('row_number', 'N/A')}")
            
            if i < len(relevant_docs[:5]):  # No agregar línea después del último
                st.markdown("---")
    else:
        # Mostrar análisis general de todas las fuentes
        analysis = analyze_air_freight_sources(sources)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Aeropuertos AOL disponibles:**")
            for aol in sorted(list(analysis.get('aol_airports', set())))[:5]:
                region = get_airport_region(aol)
                st.write(f"🛫 {aol} ({region})")
        
        with col2:
            st.write("**Aeropuertos AOD disponibles:**")
            for aod in sorted(list(analysis.get('aod_airports', set())))[:5]:
                region = get_airport_region(aod)
                st.write(f"🛬 {aod} ({region})")
        
        with col3:
            st.write("**Por compañía:**")
            for company in sorted(list(analysis.get('companies', set()))):
                st.write(f"🏢 {company}")
            
            if analysis.get('airlines'):
                st.write("**Airlines disponibles:**")
                for airline in sorted(list(analysis.get('airlines', set())))[:3]:
                    st.write(f"✈️ {airline}")

def enhanced_sidebar_air_freight():
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
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Crear Sistema", "📂 Cargar", "📊 Estadísticas", "🧪 Ejemplos"])

    with tab1:
        st.markdown("### 📁 Crear Base de Datos de Carga Aérea")
        
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
            if st.button("📁 Crear Sistema Aéreo", type="primary"):
                create_air_freight_system()
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
            load_existing_air_freight_vectorstore()

    with tab3:
        st.markdown("### 📊 Estadísticas del Sistema")
        
        if hasattr(st.session_state, 'vector_store'):
            try:
                collection_count = st.session_state.vector_store._collection.count()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("✈️ Rutas Aéreas", collection_count)
                with col2:
                    st.metric("🤖 Modelo", "gpt-4o")
                with col3:
                    st.metric("🎯 Precisión", "Máxima")
                    
                st.markdown("#### 📈 Estado del Sistema")
                st.success("Sistema de carga aérea activo")
            except:
                st.info("Carga una base de datos para ver estadísticas")
        else:
            st.info("No hay base de datos cargada")

    with tab4:
        st.markdown("### 🧪 Ejemplos de Consultas")
        
        ejemplos = [
            "¿Cuál es la tarifa de MIA a SCL?",
            "¿Qué opciones hay desde Europa a Chile?",
            "¿Cuánto cuesta enviar desde SCL a LIM?",
            "¿Qué airlines vuelan de HKG a SCL?",
            "Tarifas desde Asia a Chile",
            "¿Hay rutas directas desde Madrid?",
            "Comparar precios CRAFT vs TRANSWAY",
            "¿Qué aeropuertos de origen hay disponibles?"
        ]
        
        st.write("**💡 Prueba estas consultas:**")
        for ejemplo in ejemplos:
            if st.button(f"📝 {ejemplo}", key=f"ejemplo_{hash(ejemplo)}"):
                st.session_state.ejemplo_selected = ejemplo

def create_air_freight_system():
    """Pipeline de creación del sistema de carga aérea"""
    
    if not OPENAI_API_KEY:
        st.error("❌ Configura OpenAI API key")
        return

    if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
        st.error("❌ Selecciona archivo Excel y nombre de base de datos")
        return
    
    with st.spinner("📁 Creando sistema de carga aérea..."):
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
            
            # Mostrar resumen de aeropuertos
            with st.expander("🛫 **Aeropuertos Disponibles**"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**AOL (Origen):**")
                    for aol in sorted(analysis['aol_airports']):
                        region = get_airport_region(aol)
                        st.write(f"🛫 {aol} ({region})")
                
                with col2:
                    st.write("**AOD (Destino):**")
                    for aod in sorted(analysis['aod_airports']):
                        region = get_airport_region(aod)
                        st.write(f"🛬 {aod} ({region})")
            
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
            
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ **Sistema de Carga Aérea creado exitosamente!**")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def load_existing_air_freight_vectorstore():
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
            
            clear_chat_history()
            
            st.success("✅ **Sistema de carga aérea cargado exitosamente!**")
            st.info(f"📊 {collection_count} rutas indexadas")
            
        except Exception as e:
            st.error(f"❌ Error cargando: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def air_freight_chatbot():
    """Chatbot principal de carga aérea"""
    enhanced_sidebar_air_freight()
    
    st.markdown("---")
    
    # Header del sistema
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Consultor de Tarifas Aéreas CRAFTTRANSWAY")
        if hasattr(st.session_state, 'chain'):
            st.success("✅ Sistema Activo")
        else:
            st.warning("📁 Crear/Cargar Sistema")

    # Manejo de ejemplos seleccionados
    if hasattr(st.session_state, 'ejemplo_selected'):
        prompt = st.session_state.ejemplo_selected
        del st.session_state.ejemplo_selected
        
        if hasattr(st.session_state, 'chain'):
            get_air_freight_response(prompt)
        else:
            st.warning("⚠️ Crea o carga sistema primero")

    # Mensajes del chat
    if "messages" not in st.session_state:
        clear_chat_history()

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
        get_air_freight_response(prompt)

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
    """Función principal del sistema de carga aérea"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Ejecutar chatbot de carga aérea
    air_freight_chatbot()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    ✈️ CRAFTTRANSWAY - Sistema de Consulta de Tarifas Aéreas | AOL → AOD | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()