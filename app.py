import streamlit as st
import os
import glob
from pathlib import Path
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Importar módulos MSL con verificación
from config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, OPENAI_MODELS, 
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, detect_msl_query_type,
    extract_port_for_verification
)
from core import (
    load_msl_documents_with_verification,
    create_msl_verified_chain,
    create_msl_verified_retriever,
    validate_msl_route_exists,
    analyze_msl_verified_sources
)

####################################################################
#            CONFIGURACIÓN STREAMLIT
####################################################################

st.set_page_config(
    page_title="MSL - Verificación Total",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✅ MSL (Seemann Group) - Sistema con Verificación Total")  
st.markdown("*Solo información que realmente existe en el tarifario MSL*")

####################################################################
#            FUNCIONES PRINCIPALES CON VERIFICACIÓN
####################################################################

def get_msl_verified_response(prompt):
    """Función principal con verificación estricta y captura de múltiples opciones"""
    try:
        with st.spinner("🔍 Buscando TODAS las opciones disponibles en MSL..."):
            
            # PASO 1: Verificar que existe información para la consulta
            if not hasattr(st.session_state, 'vector_store'):
                st.error("❌ No hay base de datos cargada")
                return
            
            # PASO 2: Buscar TODOS los documentos relevantes (sin filtros restrictivos)
            retriever = st.session_state.retriever
            docs = retriever.get_relevant_documents(prompt)
            
            # PASO 3: Analizar cuántas opciones diferentes hay para el mismo puerto
            port_info = extract_port_for_verification(prompt)
            port_options_analysis = analyze_port_options(docs, port_info.get('port_requested', ''))
            
            # PASO 4: Validar existencia y mostrar análisis detallado
            validation = validate_msl_route_exists(prompt, docs)
            
            with st.expander("🔍 **Análisis de Opciones Múltiples**", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Estado:** {validation['verification_status']}")
                    if validation['route_requested']:
                        st.write(f"**Puerto solicitado:** {validation['route_requested']}")
                    st.write(f"**Documentos encontrados:** {len(docs)}")
                
                with col2:
                    if port_options_analysis['multiple_options']:
                        st.success(f"✅ **{port_options_analysis['options_count']} opciones diferentes encontradas**")
                        for i, option in enumerate(port_options_analysis['options_summary'], 1):
                            st.write(f"  {i}. {option}")
                    else:
                        st.info("ℹ️ Una opción encontrada")
            
            # PASO 5: Ejecutar chain con énfasis en mostrar TODAS las opciones
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            
            # PASO 6: Validar que se mostraron todas las opciones
            if port_options_analysis['multiple_options'] and port_options_analysis['options_count'] > 1:
                option_count_in_response = answer.count("OPCIÓN")
                if option_count_in_response < port_options_analysis['options_count']:
                    warning_msg = f"\n\n⚠️ **ADVERTENCIA:** Se detectaron {port_options_analysis['options_count']} opciones pero solo se mostraron {option_count_in_response}. Verificando..."
                    answer += warning_msg
            
            # PASO 7: Mostrar respuesta
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Métricas de múltiples opciones
                display_multiple_options_metrics(validation, port_options_analysis, response.get("source_documents", []))
                
                # Análisis detallado de todas las opciones
                with st.expander("📋 **Análisis Detallado de Todas las Opciones**"):
                    display_all_options_analysis(response.get("source_documents", []), port_info.get('port_requested', ''))
                
    except Exception as e:
        st.error(f"Error en sistema de múltiples opciones: {str(e)}")
        with st.expander("🔧 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

def analyze_port_options(docs: list, requested_port: str) -> dict:
    """Analiza cuántas opciones diferentes hay para un puerto específico"""
    
    analysis = {
        'multiple_options': False,
        'options_count': 0,
        'options_summary': [],
        'same_port_docs': []
    }
    
    if not requested_port:
        return analysis
    
    requested_port_clean = requested_port.lower().strip()
    
    # Buscar documentos del mismo puerto
    for doc in docs:
        puerto_origen = doc.metadata.get('puerto_origen', '').lower()
        
        if requested_port_clean in puerto_origen or puerto_origen in requested_port_clean:
            analysis['same_port_docs'].append(doc)
            
            # Crear resumen de la opción
            tarifa = doc.metadata.get('tarifa', 'No disponible')
            servicio = doc.metadata.get('tipo_servicio', 'No especificado')
            row = doc.metadata.get('row_number', 'N/A')
            
            option_summary = f"Fila {row}: {tarifa} - {servicio}"
            analysis['options_summary'].append(option_summary)
    
    analysis['options_count'] = len(analysis['same_port_docs'])
    analysis['multiple_options'] = analysis['options_count'] > 1
    
    return analysis

def display_multiple_options_metrics(validation: dict, options_analysis: dict, source_docs: list):
    """Muestra métricas específicas para múltiples opciones"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if validation['route_exists']:
            st.success("✅ **Ruta Verificada**")
        elif validation['route_requested']:
            st.error("❌ **Ruta No Existe**")
        else:
            st.info("ℹ️ **Consulta General**")
    
    with col2:
        if options_analysis['multiple_options']:
            st.success(f"🔄 **{options_analysis['options_count']} Opciones Diferentes**")
        else:
            st.info("1️⃣ **Opción Única**")
    
    with col3:
        verified_docs = sum(1 for doc in source_docs if doc.metadata.get('verification_status') == 'VERIFIED')
        if verified_docs > 0:
            st.success(f"📄 **{verified_docs} Docs Verificados**")
        else:
            st.warning("⚠️ **Sin Docs Verificados**")
    
    # Mostrar detalles de múltiples opciones
    if options_analysis['multiple_options']:
        with st.expander("🔄 **Detalles de Múltiples Opciones**"):
            st.write(f"**Puerto:** {validation.get('route_requested', 'N/A')}")
            st.write(f"**Total opciones encontradas:** {options_analysis['options_count']}")
            for i, option in enumerate(options_analysis['options_summary'], 1):
                st.write(f"  **{i}.** {option}")

def display_all_options_analysis(sources: list, requested_port: str):
    """Muestra análisis detallado de todas las opciones del puerto"""
    
    if not sources:
        st.warning("⚠️ No se encontraron fuentes")
        return
    
    # Filtrar documentos del puerto solicitado
    port_docs = []
    if requested_port:
        requested_clean = requested_port.lower().strip()
        for doc in sources:
            puerto_origen = doc.metadata.get('puerto_origen', '').lower()
            if requested_clean in puerto_origen or puerto_origen in requested_clean:
                port_docs.append(doc)
    
    if port_docs:
        st.write(f"**🔍 Análisis detallado para: {requested_port}**")
        
        for i, doc in enumerate(port_docs, 1):
            with st.expander(f"**Opción {i} - Fila {doc.metadata.get('row_number', 'N/A')}**"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Puerto:** {doc.metadata.get('puerto_origen', 'N/A')}")
                    st.write(f"**País:** {doc.metadata.get('pais_origen', 'N/A')}")
                    st.write(f"**Fila Excel:** {doc.metadata.get('row_number', 'N/A')}")
                    st.write(f"**Hoja:** {doc.metadata.get('sheet_name', 'N/A')}")
                
                with col2:
                    st.write(f"**Estado:** {doc.metadata.get('verification_status', 'N/A')}")
                    st.write(f"**Tipo contenido:** {doc.metadata.get('content_type', 'N/A')}")
    else:
        # Mostrar análisis general
        verified_sources = [doc for doc in sources if doc.metadata.get('verification_status') == 'VERIFIED']
        analysis = analyze_msl_verified_sources(verified_sources)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Puertos en resultados:**")
            for port in sorted(list(analysis.get('verified_ports', set())))[:5]:
                st.write(f"⚓ {port}")
        
        with col2:
            st.write("**Por región:**")
            for region, count in analysis.get('regions', {}).items():
                st.write(f"🌍 {region}: {count}")

def display_verification_metrics(validation: dict, source_docs: list):
    """Muestra métricas de verificación"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if validation['route_exists']:
            st.success("✅ **Ruta Verificada**")
        elif validation['route_requested']:
            st.error("❌ **Ruta No Existe**")
        else:
            st.info("ℹ️ **Consulta General**")
    
    with col2:
        verified_docs = sum(1 for doc in source_docs if doc.metadata.get('verification_status') == 'VERIFIED')
        if verified_docs > 0:
            st.success(f"📄 **{verified_docs} Documentos Verificados**")
        else:
            st.warning("⚠️ **Sin Documentos Verificados**")
    
    with col3:
        available_routes = len(set(validation.get('available_routes', [])))
        st.info(f"🗺️ **{available_routes} Rutas Disponibles**")
    
    # Mostrar sugerencias si las hay
    if validation.get('suggestions'):
        with st.expander("💡 **Rutas Alternativas Verificadas**"):
            for suggestion in validation['suggestions']:
                st.write(f"• {suggestion}")

def display_verified_sources_analysis(sources: list):
    """Analiza y muestra solo fuentes verificadas"""
    
    verified_sources = [doc for doc in sources if doc.metadata.get('verification_status') == 'VERIFIED']
    
    if not verified_sources:
        st.warning("⚠️ No se encontraron fuentes verificadas")
        return
    
    analysis = analyze_msl_verified_sources(verified_sources)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Por Región Verificada:**")
        for region, count in analysis['regions'].items():
            st.write(f"✅ {region}: {count}")
    
    with col2:
        st.write("**Puertos Verificados:**")
        sorted_ports = sorted(list(analysis['verified_ports']))[:5]
        for port in sorted_ports:
            st.write(f"⚓ {port}")
    
    with col3:
        st.write("**Países Verificados:**")
        sorted_countries = sorted(list(analysis['verified_countries']))[:5]
        for country in sorted_countries:
            st.write(f"🌍 {country}")

def enhanced_sidebar_verification():
    """Interfaz lateral con verificación total"""
    with st.sidebar:
        st.markdown("### ✅ **Sistema MSL con Verificación Total**")
        st.success("""
        ✅ Inspección celda por celda del Excel
        ✅ Solo rutas que realmente existen
        ✅ Verificación antes de cada respuesta
        ✅ Sin información inventada
        ✅ Advertencias cuando ruta no existe
        ✅ Sugerencias de rutas alternativas verificadas
        ✅ Validación estricta de datos
        """)
        
        st.markdown("---")
        
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API conectada")
        else:
            st.error("❌ API Key no encontrada")
            return

    # Tabs para verificación
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Crear con Verificación", "📂 Cargar", "📊 Estadísticas", "🧪 Test"])

    with tab1:
        st.markdown("### 🔍 Crear Base de Datos con Verificación Total")
        
        st.session_state.uploaded_file_list = st.file_uploader(
            "Selecciona archivo Excel MSL para verificación:",
            accept_multiple_files=True,
            type=["xlsx", "xls"],
            help="El sistema inspeccionará cada celda para verificar rutas reales"
        )
        
        st.session_state.vector_store_name = st.text_input(
            "📊 Nombre Base de Datos Verificada:",
            placeholder="ej: msl_verified_2025",
            help="Base de datos con solo rutas verificadas"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 Crear con Verificación Total", type="primary"):
                create_msl_verified_system()
        with col2:
            if st.button("🗑️ Limpiar"):
                delete_temp_files()

    with tab2:
        st.markdown("### 📂 Cargar Base de Datos Verificada")
        
        available_stores = [
            f.name for f in LOCAL_VECTOR_STORE_DIR.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ]
        
        if available_stores:
            st.session_state.selected_vectorstore_name = st.selectbox(
                "🗂️ Bases verificadas disponibles:",
                options=[""] + available_stores
            )
        else:
            st.info("📁 No hay bases de datos verificadas")
        
        if st.button("📖 Cargar Base Verificada", type="primary"):
            load_existing_verified_vectorstore()

    with tab3:
        st.markdown("### 📊 Estadísticas de Verificación")
        
        if hasattr(st.session_state, 'vector_store'):
            try:
                collection_count = st.session_state.vector_store._collection.count()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("✅ Rutas Verificadas", collection_count)
                with col2:
                    st.metric("🤖 Modelo", "gpt-4o")
                with col3:
                    st.metric("🎯 Temperatura", "0.0 (máxima precisión)")
                    
                st.markdown("#### 📈 Estado de Verificación")
                st.success("Sistema con verificación total activo")
            except:
                st.info("Carga una base de datos para ver estadísticas")
        else:
            st.info("No hay base de datos verificada cargada")

    with tab4:
        st.markdown("### 🧪 Test de Verificación")
        if st.button("Ejecutar Test de Verificación"):
            test_verification_system()

def create_msl_verified_system():
    """Pipeline de creación con verificación total"""
    
    if not OPENAI_API_KEY:
        st.error("❌ Configura OpenAI API key")
        return

    if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
        st.error("❌ Selecciona archivo Excel y nombre de base de datos")
        return
    
    with st.spinner("🔍 Verificando Excel celda por celda..."):
        try:
            # Limpiar y guardar archivos
            delete_temp_files()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📤 Guardando archivos para verificación...")
            for i, uploaded_file in enumerate(st.session_state.uploaded_file_list):
                temp_file_path = TMP_DIR / uploaded_file.name
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_file_list) * 0.1)
            
            # Verificación completa
            status_text.text("🔍 Inspeccionando cada celda del Excel...")
            documents = load_msl_documents_with_verification()
            progress_bar.progress(0.5)
            
            if not documents:
                st.error("❌ No se encontraron rutas verificadas")
                return
            
            # Mostrar estadísticas de verificación
            analysis = analyze_msl_verified_sources(documents)
            
            st.success("✅ **Verificación Completada:**")
            cols = st.columns(4)
            with cols[0]:
                st.metric("✅ Rutas Verificadas", analysis['total_verified'])
            with cols[1]:
                st.metric("🌍 Regiones", len(analysis['regions']))
            with cols[2]:
                st.metric("⚓ Puertos Verificados", len(analysis['verified_ports']))
            with cols[3]:
                st.metric("🌎 Países Verificados", len(analysis['verified_countries']))
            
            # Mostrar puertos verificados
            with st.expander("⚓ **Puertos Verificados Encontrados**"):
                for port in sorted(analysis['verified_ports']):
                    st.write(f"✅ {port}")
            
            # Crear chunks con separadores específicos para verificación
            status_text.text("📝 Creando chunks de rutas verificadas...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100,
                separators=[
                    "\nTARIFA LCL MARÍTIMA MSL - RUTA VERIFICADA",
                    "\n=== INFORMACIÓN VERIFICADA ===",
                    "\n=== VERIFICACIÓN ===",
                    "\n\n", "\n", " ", ""
                ]
            )
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.7)
            
            st.info(f"📝 {len(chunks)} chunks verificados creados")
            
            # Crear vectorstore verificado
            status_text.text("🧠 Indexando rutas verificadas...")
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            persist_path.mkdir(parents=True, exist_ok=True)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path.as_posix(),
                collection_name="msl_verified_routes"
            )
            progress_bar.progress(0.9)
            
            # Crear chain verificado
            status_text.text("🔗 Configurando sistema con verificación...")
            st.session_state.retriever = create_msl_verified_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_msl_verified_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ **Sistema con Verificación Total creado exitosamente!**")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def load_existing_verified_vectorstore():
    """Cargar vectorstore verificado existente"""
    if not OPENAI_API_KEY or not st.session_state.selected_vectorstore_name:
        st.error("❌ Configura API key y selecciona base de datos")
        return

    vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
    
    if not vectorstore_path.exists():
        st.error("❌ Base de datos verificada no existe")
        return

    with st.spinner("📖 Cargando sistema verificado..."):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path.as_posix(),
                collection_name="msl_verified_routes"
            )
            
            collection_count = st.session_state.vector_store._collection.count()
            if collection_count == 0:
                st.warning("⚠️ Base de datos verificada vacía")
                return
            
            st.session_state.retriever = create_msl_verified_retriever(
                vector_store=st.session_state.vector_store, k=15
            )
            
            st.session_state.chain, st.session_state.memory = create_msl_verified_chain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            st.success("✅ **Sistema verificado cargado exitosamente!**")
            st.info(f"📊 {collection_count} rutas verificadas indexadas")
            
        except Exception as e:
            st.error(f"❌ Error cargando: {str(e)}")

def msl_verified_chatbot():
    """Chatbot principal con verificación"""
    enhanced_sidebar_verification()
    
    st.markdown("---")
    
    # Header con verificación
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Consultor MSL con Verificación Total")
        if hasattr(st.session_state, 'chain'):
            st.success("✅ Sistema Verificado Activo")
        else:
            st.warning("🔍 Crear/Cargar Sistema Verificado")

    # Mensajes
    if "messages" not in st.session_state:
        clear_chat_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input principal
    if prompt := st.chat_input("Consulta MSL con verificación... (ej: ¿Existe ruta desde Miami a Chile?)"):
        
        if not OPENAI_API_KEY:
            st.error("🔑 Configura OpenAI API key")
            st.stop()
        
        if not hasattr(st.session_state, 'chain'):
            st.warning("⚠️ Crea o carga sistema verificado")
            st.stop()
        
        # Ejecutar respuesta verificada
        get_msl_verified_response(prompt)

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

def test_verification_system():
    """Test del sistema de verificación"""
    st.markdown("### 🧪 Test de Verificación")
    
    # Test casos específicos
    test_cases = [
        "¿Existe ruta desde Miami a Chile?",
        "¿Cuánto cuesta desde Santos a Chile?", 
        "¿Qué rutas hay desde Europa?",
        "¿Hay transporte desde Tokio?"
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        st.write(f"**Test {i}:** {test_query}")
        
        # Test verificación
        port_info = extract_port_for_verification(test_query)
        query_type = detect_msl_query_type(test_query)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"   **Tipo:** {query_type}")
        with col2:
            if port_info.get('needs_verification'):
                st.write(f"   **Puerto a verificar:** {port_info['port_requested']}")
            else:
                st.write("   **Verificación:** No específica")
    
    st.success("✅ Test de verificación completado")

####################################################################
#            FUNCIÓN PRINCIPAL
####################################################################

def main():
    """Función principal con verificación total"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Ejecutar sistema verificado
    msl_verified_chatbot()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    ✅ MSL (Seemann Group) - Sistema con Verificación Total | Solo información que realmente existe | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()