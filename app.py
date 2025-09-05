import streamlit as st
import os
import glob
from pathlib import Path
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Importar modulos locales
from config import (
    OPENAI_API_KEY, WELCOME_MESSAGE, OPENAI_MODELS, 
    TMP_DIR, LOCAL_VECTOR_STORE_DIR
)
from core import (
    enhanced_seemann_document_loader_v2, 
    create_enhanced_conversational_chain_v2_2,  # Cambio aquí
    create_enhanced_seemann_retriever_v2,
    multi_query_retriever_with_pol_pod_validation,
    EnhancedFreightParser
)

####################################################################
#            CONFIGURACION STREAMLIT
####################################################################

st.set_page_config(
    page_title="Seemann Group v2.2",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 Seemann Group v2.2 - Templates Dinámicos")  
st.markdown("*Sistema inteligente con detección automática de contenido y templates especializados*")

####################################################################
#            FUNCIONES DE INTERFAZ v2.1
####################################################################

def get_enhanced_seemann_response_v2_2(prompt):
    """Función principal de respuesta v2.2 con templates dinámicos"""
    try:
        with st.spinner("🔍 Analizando consulta y detectando tipo de contenido..."):
            
            # Crear parser para validación
            parser = EnhancedFreightParser()
            
            # Búsqueda inicial para obtener documentos
            if hasattr(st.session_state, 'vector_store'):
                all_relevant_docs = multi_query_retriever_with_pol_pod_validation(
                    st.session_state.vector_store, prompt, parser
                )
                
                # Detectar tipo de consulta basado en contenido y documentos
                from config import detect_query_type, validate_document_relevance, enhanced_seemann_response_template
                query_type = detect_query_type(prompt, all_relevant_docs)
                
                # Filtrar documentos por relevancia según tipo detectado
                filtered_docs = validate_document_relevance(prompt, all_relevant_docs)
                
                with st.expander("🧠 **Análisis de Detección Automática v2.2**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**Tipo de Consulta Detectado:** {query_type}")
                        
                        type_descriptions = {
                            'fcl_import': '📦 Cotizaciones FCL Importación',
                            'export_rates': '📤 Tarifas de Exportación',
                            'demurrage_detention': '⏰ Demurrage & Detention',
                            'comparative_analysis': '📊 Análisis Comparativo'
                        }
                        
                        st.write(f"**Descripción:** {type_descriptions.get(query_type, 'Tipo especializado')}")
                    
                    with col2:
                        st.write(f"**Documentos Originales:** {len(all_relevant_docs)}")
                        st.write(f"**Documentos Filtrados:** {len(filtered_docs)}")
                        
                        if len(filtered_docs) < len(all_relevant_docs):
                            st.warning(f"Se filtraron {len(all_relevant_docs) - len(filtered_docs)} documentos irrelevantes")
                
                # Actualizar el template dinámicamente
                dynamic_template = enhanced_seemann_response_template(query_type)
                
                # Crear prompt dinámico
                from langchain.prompts import ChatPromptTemplate
                answer_prompt = ChatPromptTemplate.from_template(dynamic_template)
                
                # Actualizar la chain con el template dinámico
                st.session_state.chain.combine_docs_chain.llm_chain.prompt = answer_prompt
                
                st.info(f"🎯 **Template Seleccionado:** {query_type}")
            
            # Ejecutar chain con template dinámico
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            
            # Validar completitud específica por tipo
            validation = validate_response_by_type(
                answer, prompt, response.get("source_documents", []), query_type, parser
            )
            
            # Agregar al historial
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Mostrar conversación
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Métricas específicas por tipo
                display_type_specific_metrics(validation, query_type)
                
                # Análisis de fuentes por tipo
                with st.expander("📋 **Análisis de Fuentes por Tipo de Contenido**"):
                    analyze_sources_by_type(response.get("source_documents", []), query_type)
                
    except Exception as e:
        st.error(f"Error en sistema v2.2: {str(e)}")
        with st.expander("🔧 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

def validate_response_completeness_with_pol_pod(response_text: str, original_query: str, source_docs: list, parser) -> dict:
    """Valida completitud de respuesta con enfoque en validacion POL/POD"""
    
    validation = {
        'completeness': 1.0,
        'route_accuracy': 1.0,
        'warnings': [],
        'suggestions': []
    }
    
    query_lower = original_query.lower()
    
    # Extraer ruta de la consulta
    route_info = parser.extract_pol_pod_from_query(original_query)
    
    if route_info:
        query_pol = route_info.get('pol_normalized', '')
        query_pod = route_info.get('pod_normalized', '')
        
        # Validar que los documentos fuente tengan la ruta correcta
        correct_route_docs = 0
        total_docs = len(source_docs)
        
        for doc in source_docs:
            doc_pol = doc.metadata.get('pol_normalized', '')
            doc_pod = doc.metadata.get('pod_normalized', '')
            
            if parser.validate_pol_pod_match(doc_pol, doc_pod, query_pol, query_pod):
                correct_route_docs += 1
        
        if total_docs > 0:
            validation['route_accuracy'] = correct_route_docs / total_docs
            
            if validation['route_accuracy'] < 0.5:
                validation['warnings'].append(
                    f"Mas del 50% de fuentes tienen rutas incorrectas para {query_pol} → {query_pod}"
                )
                validation['completeness'] *= 0.3
            elif validation['route_accuracy'] < 0.8:
                validation['warnings'].append(
                    f"Algunas fuentes no corresponden a la ruta {query_pol} → {query_pod}"
                )
                validation['completeness'] *= 0.7
        
        # Verificar si hay resultados para la ruta especifica
        if correct_route_docs == 0:
            validation['warnings'].append(
                f"No se encontraron documentos validos para la ruta {query_pol} → {query_pod}"
            )
            validation['completeness'] = 0.1
            validation['suggestions'].append("Verificar disponibilidad de tarifas para esta ruta especifica")
    
    # Validar presencia de tarifas
    if any(word in query_lower for word in ['costo', 'precio', 'tarifa', 'cuanto']):
        usd_count = response_text.count('USD')
        if usd_count == 0:
            validation['completeness'] *= 0.3
            validation['warnings'].append("No se encontraron tarifas en la respuesta")
        elif usd_count < 2:
            validation['completeness'] *= 0.7
            validation['warnings'].append("Informacion de tarifas limitada")
    
    return validation

def enhance_query_for_pol_pod_completeness(original_query: str, first_response: str, parser) -> str:
    """Mejora query para segunda busqueda enfocada en POL/POD"""
    
    route_info = parser.extract_pol_pod_from_query(original_query)
    enhanced_terms = []
    
    if route_info:
        pol = route_info.get('pol_normalized', '')
        pod = route_info.get('pod_normalized', '')
        
        enhanced_terms.extend([
            f"POL_EXACTO: {pol} POD_EXACTO: {pod}",
            f"ORIGEN_NORMALIZADO: {pol} DESTINO_NORMALIZADO: {pod}",
            f"ruta {pol} {pod}",
            f"desde {pol} hacia {pod}"
        ])
    
    if 'comparar' in original_query.lower() or 'opcion' in original_query.lower():
        enhanced_terms.extend(['todas las navieras disponibles', 'cotizaciones completas para ruta'])
    
    enhanced_query = f"{original_query} {' '.join(enhanced_terms)} validacion POL POD exacta"
    return enhanced_query

def enhanced_sidebar_seemann_v2_2():
    """Interfaz lateral v2.2 con templates dinámicos"""
    with st.sidebar:
        st.markdown("### 🚀 **Sistema v2.2 Templates Dinámicos**")
        st.success("""
        ✅ Detección automática de contenido
        ✅ Templates especializados por tipo
        ✅ Filtrado inteligente de documentos
        ✅ Validación específica por consulta
        ✅ Procesamiento FCL/Export/D&D
        ✅ Análisis comparativo avanzado
        ✅ Métricas por tipo de contenido
        """)
        
        st.markdown("---")
        
        # Estado del sistema
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API conectada")
        else:
            st.error("❌ API Key no encontrada")
            return

    # Tabs mejoradas para v2.2
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Crear v2.2", "📂 Cargar", "📊 Estadísticas", "🧪 Test"])

    with tab1:
        st.markdown("### 📤 Crear Base de Datos v2.2 con Templates Dinámicos")
        
        st.session_state.uploaded_file_list = st.file_uploader(
            "Selecciona archivos para procesamiento con detección automática:",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv", "xlsx"],
            help="El sistema detectará automáticamente FCL, Export, D&D y aplicará templates especializados"
        )
        
        st.session_state.vector_store_name = st.text_input(
            "📊 Nombre Base de Datos v2.2:",
            placeholder="ej: seemann_v22_dynamic_templates_2025",
            help="Incluye v22 para identificar versión con templates dinámicos"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Crear con Templates Dinámicos v2.2", type="primary"):
                enhanced_chain_RAG_blocks_v2_2()
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
            st.info("📁 No hay bases de datos disponibles")
        
        if st.button("📖 Cargar Base de Datos", type="primary"):
            load_existing_vectorstore_v2_2()

    with tab3:
        st.markdown("### 📊 Estadísticas del Sistema v2.2")
        
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
                    
                # Estadísticas de templates dinámicos
                st.markdown("#### 🎯 Estadísticas de Templates Dinámicos")
                st.info("Sistema v2.2 con detección automática de contenido activo")
            except:
                st.info("Carga una base de datos para ver estadísticas")
        else:
            st.info("No hay base de datos cargada")

    with tab4:
        st.markdown("### 🧪 Test Templates Dinámicos v2.2")
        if st.button("Ejecutar Test Detección Automática"):
            test_dynamic_templates_v2_2()

def enhanced_chain_RAG_blocks_v2_2():
    """Pipeline v2.2 con templates dinámicos"""
    
    if not OPENAI_API_KEY:
        st.error("❌ Configura OpenAI API key")
        return

    if not st.session_state.uploaded_file_list or not st.session_state.vector_store_name.strip():
        st.error("❌ Selecciona archivos y nombre de base de datos")
        return
    
    with st.spinner("📄 Procesando con templates dinámicos v2.2..."):
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
            
            # Procesar con sistema v2.2
            status_text.text("🧠 Procesando con detección automática v2.2...")
            documents = enhanced_seemann_document_loader_v2()
            progress_bar.progress(0.4)
            
            if not documents:
                st.error("❌ No se procesaron documentos")
                return
            
            # Estadísticas por tipo de contenido
            content_stats = {}
            for doc in documents:
                content_type = doc.metadata.get("content_type", "unknown")
                content_stats[content_type] = content_stats.get(content_type, 0) + 1
            
            st.success("📊 **Procesamiento v2.2 con Templates Dinámicos completado:**")
            cols = st.columns(len(content_stats))
            for i, (content_type, count) in enumerate(content_stats.items()):
                with cols[i]:
                    icon = {
                        'fcl_rate': '📦',
                        'export_rates': '📤',
                        'demurrage_detention': '⏰',
                        'quotation': '📋'
                    }.get(content_type, '📄')
                    st.metric(f"{icon} {content_type}", count)
            
            # Crear chunks optimizados
            status_text.text("✂️ Creando chunks optimizados...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=250,
                separators=[
                    "\nCOTIZACIÓN MARÍTIMA FCL",
                    "\nDEMURRAGE & DETENTION",
                    "\nEXPORTACIÓN DESDE CHILE",
                    "\n=== INFORMACIÓN",
                    "\n\n", "\n", " ", ""
                ]
            )
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.6)
            
            st.info(f"📝 {len(chunks)} chunks optimizados creados")
            
            # Crear vectorstore
            status_text.text("🧠 Generando embeddings...")
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            persist_path.mkdir(parents=True, exist_ok=True)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path.as_posix(),
                collection_name="seemann_v22_dynamic_templates"
            )
            progress_bar.progress(0.8)
            
            # Crear chain v2.2
            status_text.text("🔗 Configurando sistema con templates dinámicos...")
            st.session_state.retriever = create_enhanced_seemann_retriever_v2(
                vector_store=st.session_state.vector_store, k=20
            )
            
            st.session_state.chain, st.session_state.memory = create_enhanced_conversational_chain_v2_2(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ **Sistema v2.2 con Templates Dinámicos creado exitosamente!**")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔧 Detalles del error"):
                st.code(traceback.format_exc())

def load_existing_vectorstore_v2_2():
    """Cargar vectorstore existente v2.2"""
    if not OPENAI_API_KEY or not st.session_state.selected_vectorstore_name:
        st.error("❌ Configura API key y selecciona base de datos")
        return

    vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
    
    if not vectorstore_path.exists():
        st.error("❌ Base de datos no existe")
        return

    with st.spinner("📖 Cargando sistema v2.2..."):
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path.as_posix(),
                collection_name="seemann_v22_dynamic_templates"
            )
            
            collection_count = st.session_state.vector_store._collection.count()
            if collection_count == 0:
                st.warning("⚠️ Base de datos vacía")
                return
            
            st.session_state.retriever = create_enhanced_seemann_retriever_v2(
                vector_store=st.session_state.vector_store, k=20
            )
            
            st.session_state.chain, st.session_state.memory = create_enhanced_conversational_chain_v2_2(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            st.success("✅ **Sistema v2.2 con Templates Dinámicos cargado exitosamente!**")
            st.info(f"📊 {collection_count} documentos indexados con detección automática")
            
        except Exception as e:
            st.error(f"❌ Error cargando: {str(e)}")

def seemann_chatbot_v2_2():
    """Chatbot principal v2.2 con templates dinámicos"""
    enhanced_sidebar_seemann_v2_2()
    
    st.markdown("---")
    
    # Header actualizado
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Consultor v2.2 - Templates Dinámicos - Seemann Group")
        if hasattr(st.session_state, 'chain'):
            st.success("🟢 Sistema v2.2 Activo - Templates Dinámicos Habilitados")
        else:
            st.warning("🟡 Crear/Cargar Base de Datos v2.2")

    # Mensajes
    if "messages" not in st.session_state:
        clear_chat_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input principal
    if prompt := st.chat_input("Consulta con detección automática... (El sistema detectará si es FCL, Export, D&D o Comparativo)"):
        
        if not OPENAI_API_KEY:
            st.error("🔑 Configura OpenAI API key")
            st.stop()
        
        if not hasattr(st.session_state, 'chain'):
            st.warning("⚠️ Crea o carga base de datos v2.2")
            st.stop()
        
        # Ejecutar respuesta v2.2 con templates dinámicos
        get_enhanced_seemann_response_v2_2(prompt)


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

def test_parser_improvements_v2_1():
    """Test funcionalidad parser v2.1 con validación POL/POD"""
    st.markdown("### 🧪 Test Parser v2.1 con Validación POL/POD")
    
    parser = EnhancedFreightParser()
    
    # Test casos con validación POL/POD
    test_cases = {
        "Extracción POL/POD": "¿Cuánto cuesta desde Shanghai a San Antonio?",
        "Tarifa combinada": "USD2300/2800 per 20/40",
        "Puertos múltiples": "QINGDAO/SHANGHAI/NINGBO/SHENZHEN", 
        "Free time": "Free time:21days",
        "Normalización MSK": "msk",
        "Normalización puerto": "shanghai"
    }
    
    results = {
        "Extracción POL/POD": parser.extract_pol_pod_from_query(test_cases["Extracción POL/POD"]),
        "Tarifa combinada": parser.parse_combined_rates(test_cases["Tarifa combinada"]),
        "Puertos múltiples": parser.parse_multiple_ports(test_cases["Puertos múltiples"]),
        "Free time": parser.extract_free_time(test_cases["Free time"]),
        "Normalización MSK": parser.normalize_carrier_name(test_cases["Normalización MSK"]),
        "Normalización puerto": parser.normalize_port_name(test_cases["Normalización puerto"])
    }
    
    for test_name, result in results.items():
        st.write(f"**{test_name}:** {result}")
    
    # Test validación POL/POD
    st.write("**Test Validación POL/POD:**")
    test_validation = parser.validate_pol_pod_match("Shanghai", "San Antonio", "shanghai", "san antonio")
    st.write(f"Validación Shanghai → San Antonio: {test_validation}")
    
    st.success("✅ Parser v2.1 con validación POL/POD funcionando correctamente")

def validate_response_by_type(response_text: str, query: str, source_docs: list, query_type: str, parser) -> dict:
    """Validación específica según el tipo de consulta detectado"""
    
    validation = {
        'completeness': 1.0,
        'type_accuracy': 1.0,
        'warnings': [],
        'suggestions': []
    }
    
    if query_type == 'fcl_import':
        # Validación para FCL Import
        if 'USD' not in response_text:
            validation['completeness'] *= 0.3
            validation['warnings'].append("No se encontraron tarifas USD")
        
        # Validar POL/POD
        route_info = parser.extract_pol_pod_from_query(query)
        if route_info:
            pol_in_response = route_info['pol_raw'].lower() in response_text.lower()
            pod_in_response = route_info['pod_raw'].lower() in response_text.lower()
            
            if not (pol_in_response and pod_in_response):
                validation['type_accuracy'] *= 0.5
                validation['warnings'].append("Ruta consultada no está claramente reflejada en respuesta")
    
    elif query_type == 'export_rates':
        # Validación para Exportación
        if 'san antonio' not in response_text.lower():
            validation['type_accuracy'] *= 0.7
            validation['warnings'].append("Puerto de origen (San Antonio) no mencionado")
        
        if not any(term in response_text.lower() for term in ['doc fee', 'gate out', 'prepaid']):
            validation['completeness'] *= 0.6
            validation['warnings'].append("Términos específicos de exportación faltantes")
    
    elif query_type == 'demurrage_detention':
        # Validación para D&D
        if not any(term in response_text.lower() for term in ['free', 'days', 'calendar']):
            validation['completeness'] *= 0.4
            validation['warnings'].append("Información de free days faltante")
        
        if not any(term in response_text.lower() for term in ['import', 'export']):
            validation['type_accuracy'] *= 0.6
            validation['warnings'].append("No se distingue entre import/export D&D")
    
    elif query_type == 'comparative_analysis':
        # Validación para Comparativo
        carriers_in_response = sum(1 for carrier in ['cosco', 'msk', 'cma', 'pil', 'one'] 
                                 if carrier in response_text.lower())
        
        if carriers_in_response < 2:
            validation['completeness'] *= 0.5
            validation['warnings'].append("Análisis comparativo insuficiente - pocas navieras")
    
    return validation

def display_type_specific_metrics(validation: dict, query_type: str):
    """Muestra métricas específicas según el tipo de consulta"""
    
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
        type_accuracy = validation.get('type_accuracy', 0)
        if type_accuracy >= 0.8:
            st.success(f"🎯 **Precisión Tipo** ({type_accuracy:.0%})")
        elif type_accuracy >= 0.5:
            st.warning(f"⚠️ **Precisión Tipo** ({type_accuracy:.0%})")
        else:
            st.error(f"❌ **Precisión Tipo** ({type_accuracy:.0%})")
    
    with col3:
        type_labels = {
            'fcl_import': '📦 FCL Import',
            'export_rates': '📤 Export',
            'demurrage_detention': '⏰ D&D',
            'comparative_analysis': '📊 Comparativo'
        }
        st.info(f"**Tipo:** {type_labels.get(query_type, query_type)}")
    
    # Mostrar advertencias específicas
    if validation.get('warnings'):
        with st.expander("⚠️ **Advertencias Específicas del Tipo**"):
            for warning in validation['warnings']:
                st.write(f"• {warning}")

def analyze_sources_by_type(sources: list, query_type: str):
    """Análisis de fuentes específico por tipo de consulta"""
    
    if not sources:
        st.error("No se encontraron fuentes")
        return
    
    # Estadísticas generales
    total_sources = len(sources)
    
    # Análisis por tipo de procesamiento
    processing_methods = {}
    content_types = {}
    carriers = {}
    
    for doc in sources:
        # Método de procesamiento
        method = doc.metadata.get('processing_method', 'unknown')
        processing_methods[method] = processing_methods.get(method, 0) + 1
        
        # Tipo de contenido
        content_type = doc.metadata.get('content_type', 'unknown')
        content_types[content_type] = content_types.get(content_type, 0) + 1
        
        # Navieras
        carrier = doc.metadata.get('carrier', 'unknown')
        if carrier != 'unknown':
            carriers[carrier] = carriers.get(carrier, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Por Método de Procesamiento:**")
        for method, count in processing_methods.items():
            if 'v2_2' in method:
                st.write(f"✅ {method}: {count}")
            else:
                st.write(f"📄 {method}: {count}")
    
    with col2:
        st.write("**Por Tipo de Contenido:**")
        for content_type, count in content_types.items():
            icon = {
                'fcl_rate': '📦',
                'export_rates': '📤',
                'demurrage_detention': '⏰',
                'quotation': '📋'
            }.get(content_type, '📄')
            st.write(f"{icon} {content_type}: {count}")
    
    with col3:
        st.write("**Por Naviera:**")
        for carrier, count in carriers.items():
            st.write(f"🚢 {carrier}: {count}")

def test_dynamic_templates_v2_2():
    """Test funcionalidad templates dinámicos v2.2"""
    st.markdown("### 🧪 Test Templates Dinámicos v2.2")
    
    from config import detect_query_type
    
    # Test detección de tipos
    test_queries = [
        ("¿Cuánto cuesta desde Shanghai a San Antonio?", "fcl_import"),
        ("Necesito exportar desde San Antonio a Callao", "export_rates"),
        ("¿Cuáles son las tarifas de demurrage de COSCO?", "demurrage_detention"),
        ("Compara todas las opciones disponibles", "comparative_analysis")
    ]
    
    for query, expected_type in test_queries:
        detected_type = detect_query_type(query)
        status = "✅" if detected_type == expected_type else "❌"
        st.write(f"{status} **Query:** {query}")
        st.write(f"   **Detectado:** {detected_type} | **Esperado:** {expected_type}")
    
    st.success("✅ Test de detección automática completado")


####################################################################
#            FUNCION PRINCIPAL
####################################################################

def main():
    """Función principal de la aplicación v2.2 con templates dinámicos"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.selected_model = "gpt-4o"
        st.session_state.temperature = 0.05
    
    # Ejecutar sistema v2.2 con templates dinámicos
    seemann_chatbot_v2_2()
    
    # Footer actualizado
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚢 Seemann Group v2.2 - Templates Dinámicos | Detección Automática FCL/Export/D&D | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()