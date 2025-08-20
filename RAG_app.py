import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import re
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Import openai as main LLM service
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Import streamlit
import streamlit as st

####################################################################
#              Config Mejorado
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = "¿Cómo puedo ayudarle hoy? Estoy especializado en consultas de tarifas de shipping."

# Available OpenAI models (priorizando precisión para tarifas)
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
]

# Rutas base
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "vector_stores")

# Asegurar que existan los directorios base al iniciar
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#            MEJORAS: Procesamiento Especializado para CSVs
####################################################################

class ShippingTariffProcessor:
    """Procesador especializado para tarifas de shipping"""
    
    def __init__(self):
        self.port_aliases = {
            'miami': ['miami', 'mia', 'miami usa', 'miami us'],
            'san antonio': ['san antonio', 'san antonio chile', 'valparaiso', 'sap', 'san antonio - chile'],
            'callao': ['callao', 'callao peru', 'lim', 'lima'],
            'guayaquil': ['guayaquil', 'guayaquil ecuador', 'gye'],
            'santos': ['santos', 'santos brasil', 'sao paulo'],
            'cartagena': ['cartagena', 'cartagena colombia', 'ctg'],
            'puerto cabello': ['puerto cabello', 'venezuela', 'pcb'],
            'buenaventura': ['buenaventura', 'colombia', 'bun'],
            'canoas': ['canoas', 'canoas brasil', 'brazil'],
            'rio de janeiro': ['rio de janeiro', 'rio', 'brasil'],
            'curitiba': ['curitiba', 'brasil'],
            'itajai': ['itajai', 'brasil'],
            'manzanillo': ['manzanillo', 'mexico', 'manzanillo mx'],
            'new york': ['new york', 'ny', 'nueva york'],
            'chicago': ['chicago', 'chi'],
            'houston': ['houston', 'hou'],
            'atlanta': ['atlanta', 'atl'],
            'baltimore': ['baltimore', 'bal'],
            'boston': ['boston', 'bos'],
            'detroit': ['detroit', 'det'],
        }
    
    def normalize_port_name(self, port: str) -> str:
        """Normaliza nombres de puertos para mejor matching"""
        if not port:
            return ""
        
        port_lower = port.lower().strip()
        for canonical, aliases in self.port_aliases.items():
            if any(alias in port_lower for alias in aliases):
                return canonical
        return port_lower
    
    def extract_numeric_value(self, value_str: str) -> str:
        """Extrae valores numéricos de strings, manteniendo formato original si es válido"""
        if pd.isna(value_str) or value_str in ['', 'nan', 'NaN']:
            return "no especificado"
        
        value_str = str(value_str).strip()
        # Buscar patrones numéricos
        numeric_match = re.search(r'\d+(?:\.\d+)?', value_str)
        if numeric_match:
            return value_str  # Retornar el string original si contiene números
        return "no especificado"

def enhanced_csv_loader_documents():
    """Cargador mejorado de CSV que crea documentos LangChain estructurados"""
    documents = []
    csv_files = glob.glob(TMP_DIR.as_posix() + "/**/*.csv", recursive=True)
    
    processor = ShippingTariffProcessor()
    
    for csv_file in csv_files:
        try:
            # Leer CSV con múltiples encodings
            df = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.warning(f"No se pudo leer el archivo {csv_file}")
                continue
            
            # CRÍTICO: Limpiar nombres de columnas eliminando espacios
            df.columns = df.columns.str.strip()
            
            # Mapear nombres de columnas que pueden tener variaciones
            column_mapping = {}
            for col in df.columns:
                clean_col = col.strip().upper()
                if 'OF W/M' in clean_col:
                    column_mapping[col] = 'OF_WM'
                elif 'OTHERS' in clean_col and 'W/M' in clean_col:
                    column_mapping[col] = 'OTHERS_WM'
                elif 'BL' in clean_col and len(clean_col.strip()) <= 5:
                    column_mapping[col] = 'BL'
                elif 'SOLAS' in clean_col:
                    column_mapping[col] = 'SOLAS'
                elif 'POL' in clean_col:
                    column_mapping[col] = 'POL'
                elif 'POD' in clean_col:
                    column_mapping[col] = 'POD'
                elif 'SERVICIO' in clean_col or 'VIA' in clean_col:
                    column_mapping[col] = 'SERVICIO_VIA'
                elif 'TT' in clean_col or 'APROX' in clean_col:
                    column_mapping[col] = 'TT_APROX'
            
            # Renombrar columnas
            df = df.rename(columns=column_mapping)
            
            # Debug: mostrar columnas mapeadas
            st.info(f"Columnas procesadas en {Path(csv_file).name}: {list(df.columns)}")
            
            # Procesar cada fila
            for idx, row in df.iterrows():
                # Extraer valores de forma segura con los nuevos nombres
                pol = str(row.get('POL', '')).strip()
                pod = str(row.get('POD', '')).strip()
                servicio = str(row.get('SERVICIO_VIA', '')).strip()
                of_wm = processor.extract_numeric_value(row.get('OF_WM', ''))
                others_wm = processor.extract_numeric_value(row.get('OTHERS_WM', ''))
                bl = processor.extract_numeric_value(row.get('BL', ''))
                solas = processor.extract_numeric_value(row.get('SOLAS', ''))
                tt_aprox = str(row.get('TT_APROX', '')).strip()
                
                # Crear contenido estructurado optimizado para búsqueda
                content = f"""RUTA DE SHIPPING #{idx + 1}

=== INFORMACIÓN PRINCIPAL ===
ORIGEN: {pol}
DESTINO: {pod}
SERVICIO: {servicio}

=== TARIFAS EN USD POR W/M ===
OF W/M: {of_wm}
OTHERS(*) W/M: {others_wm}
BL: {bl}
SOLAS: {solas}

=== TIEMPO Y SERVICIO ===
TIEMPO DE TRÁNSITO: {tt_aprox}
VÍA: {servicio}

=== BÚSQUEDA OPTIMIZADA ===
RUTA: {pol} a {pod}
RUTA NORMALIZADA: {processor.normalize_port_name(pol)} a {processor.normalize_port_name(pod)}
VÍA NORMALIZADA: {processor.normalize_port_name(servicio)}

=== DATOS COMPLETOS ===
{row.to_string()}

PALABRAS CLAVE: {pol.lower()} {pod.lower()} {servicio.lower()} tarifa costo precio shipping marítimo
"""
                
                # Crear documento LangChain
                from langchain.docstore.document import Document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": csv_file,
                        "row_number": idx + 1,
                        "pol": pol,
                        "pod": pod,
                        "pol_normalized": processor.normalize_port_name(pol),
                        "pod_normalized": processor.normalize_port_name(pod),
                        "servicio": servicio,
                        "of_wm": of_wm,
                        "others_wm": others_wm,
                        "bl": bl,
                        "solas": solas,
                        "tt_aprox": tt_aprox,
                        "content_type": "shipping_tariff",
                        "route_key": f"{pol.lower()}_to_{pod.lower()}"
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            st.warning(f"Error procesando {csv_file}: {str(e)}")
    
    return documents

####################################################################
#        Cargador de documentos mejorado
####################################################################

def enhanced_langchain_document_loader():
    """Cargador mejorado que integra CSV especializado con otros documentos"""
    documents = []
    
    # Cargar documentos tradicionales
    try:
        txt_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
        )
        documents.extend(txt_loader.load())
    except:
        pass
    
    try:
        pdf_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        documents.extend(pdf_loader.load())
    except:
        pass
    
    try:
        doc_loader = DirectoryLoader(
            TMP_DIR.as_posix(),
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True,
        )
        documents.extend(doc_loader.load())
    except:
        pass
    
    # Cargar CSVs con procesamiento especializado
    csv_documents = enhanced_csv_loader_documents()
    documents.extend(csv_documents)
    
    return documents

####################################################################
#        Text Splitter Mejorado
####################################################################

def create_enhanced_text_splitter():
    """Text splitter que mantiene integridad de registros de tarifas"""
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Chunks más grandes para mantener contexto completo
        chunk_overlap=200,
        separators=[
            "\nREGISTRO DE TARIFA DE SHIPPING",  # Separador específico para tarifas
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

####################################################################
#        Template de Respuesta Mejorado
####################################################################

def enhanced_answer_template():
    return """Eres un asistente especializado en tarifas de shipping marítimo. 

INSTRUCCIONES CRÍTICAS:
1. Busca información EXACTA en el contexto proporcionado
2. Los datos están estructurados como "ORIGEN: [puerto]" y "DESTINO: [puerto]" 
3. Extrae valores EXACTOS de las secciones "TARIFAS EN USD POR W/M"
4. NUNCA inventes valores

FORMATO DE RESPUESTA OBLIGATORIO:

🚢 **TARIFA PARA RUTA:** [Origen] ➜ [Destino]

💰 **COSTOS POR W/M (por tonelada o m³):**\n
• **OF W/M:** [valor exacto del contexto]\n
• **OTHERS(*) W/M:** [valor exacto del contexto]\n
• **BL:** [valor exacto del contexto]\n
• **SOLAS:** [valor exacto del contexto]\n

📊 **TOTAL VARIABLE W/M:** USD [OF W/M + OTHERS(*) W/M]

⏱️ **TIEMPO DE TRÁNSITO:** [valor del contexto]

🛤️ **SERVICIO/VÍA:** [valor del contexto]

📦 **Cálculo para [X] toneladas:**  
Costo total = (X * (OF W/M + OTHERS(*) W/M)) + BL + SOLAS = [resultado en USD]

PROCESO DE BÚSQUEDA:
1. Identifica los puertos de origen y destino en la pregunta
2. Busca en el contexto la sección que contiene "ORIGEN: [puerto]" y "DESTINO: [puerto]" que coincidan
3. De esa sección, extrae los valores de "TARIFAS EN USD POR W/M"
4. Calcula el costo total multiplicando el valor variable por la cantidad de toneladas/m³, y **SIEMPRE** suma BL y SOLAS (aunque alguno sea cero).
5. Presenta la información en el formato especificado

Si NO encuentras coincidencias exactas, responde:
"❌ No encontré información para la ruta [origen] → [destino] en la base de datos actual."

<context>
{chat_history}

{context}
</context>

Pregunta: {question}

Respuesta:"""

####################################################################
#        Retriever Mejorado
####################################################################

def create_smart_retriever(vector_store, search_type="mmr", k=6):
    """Retriever optimizado para consultas de tarifas"""
    search_kwargs = {
        "k": k,
        "lambda_mult": 0.5,  # Balance entre relevancia y diversidad
        "fetch_k": k * 2  # Obtener más candidatos iniciales
    }
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever

####################################################################
#        Chain Mejorada
####################################################################

def create_enhanced_ConversationalRetrievalChain(retriever, chain_type="stuff"):
    """Chain optimizada para máxima precisión en tarifas"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Basándote en la conversación previa, reformula la pregunta para que sea clara e independiente.

IMPORTANTE: Si la pregunta es sobre tarifas/costos de shipping, mantén EXACTAMENTE los nombres de puertos mencionados.

Historial:
{chat_history}

Pregunta actual: {question}

Pregunta reformulada y clara:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(enhanced_answer_template())
    memory = create_memory()

    # Configuración de LLMs con parámetros optimizados
    standalone_query_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=st.session_state.selected_model,
        temperature=0.0,  # Precisión máxima para reformular
    )
    
    response_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=st.session_state.selected_model,
        temperature=min(0.2, st.session_state.temperature),  # Limitamos temperatura
        model_kwargs={
            "top_p": st.session_state.top_p,
        }
    )

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_llm,
        llm=response_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,  # Cambiar a False para evitar logs excesivos
        return_source_documents=True,
    )

    return chain, memory

####################################################################
#        Validación de Respuestas
####################################################################

def validate_shipping_response(response_text: str, question: str) -> Dict[str, Any]:
    """Valida respuestas de tarifas para detectar inconsistencias"""
    
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'info': []
    }
    
    # Detectar si es consulta de tarifa
    tariff_keywords = ['costo', 'precio', 'tarifa', 'cuánto', 'vale', 'cobran']
    is_tariff_query = any(keyword in question.lower() for keyword in tariff_keywords)
    
    if is_tariff_query:
        # Verificar formato correcto
        required_sections = ['OF W/M:', 'OTHERS(*) W/M:', 'BL:', 'SOLAS:']
        missing_sections = [section for section in required_sections if section not in response_text]
        
        if missing_sections:
            validation_result['warnings'].append(
                f"Faltan secciones: {', '.join(missing_sections)}"
            )
        
        # Verificar presencia de valores numéricos o "no especificado"
        if not any(pattern in response_text for pattern in ['USD', '$', 'no especificado', 'no encontré']):
            validation_result['warnings'].append("Respuesta sin información tarifaria clara")
        
        # Verificar cálculo de total
        if 'TOTAL VARIABLE W/M:' not in response_text and '❌ No encontré' not in response_text:
            validation_result['warnings'].append("Falta cálculo de total variable")
    
    return validation_result

####################################################################
#        Función de Respuesta Mejorada
####################################################################

def get_enhanced_response_from_LLM(prompt):
    """Función mejorada con validación y mejor presentación"""
    try:
        with st.spinner("🔍 Buscando información en la base de datos..."):
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            
            # Validar respuesta
            validation = validate_shipping_response(answer, prompt)
            
            # Agregar al historial
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Mostrar conversación
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Mostrar warnings si los hay
                if validation['warnings']:
                    st.warning("⚠️ **Advertencias detectadas:**")
                    for warning in validation['warnings']:
                        st.write(f"• {warning}")
                    st.write("*Considera revisar la información o reformular la consulta*")
                
                # Mostrar documentos fuente
                with st.expander("📋 **Ver documentos fuente utilizados**"):
                    if response["source_documents"]:
                        for i, doc in enumerate(response["source_documents"], 1):
                            source = doc.metadata.get("source", "Fuente desconocida")
                            row_num = doc.metadata.get("row_number", "")
                            
                            st.write(f"**📄 Documento {i}:** {Path(source).name}")
                            if row_num:
                                st.write(f"**📍 Registro:** #{row_num}")
                            
                            # Mostrar contenido relevante
                            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.code(content_preview, language="text")
                            st.divider()
                    else:
                        st.write("❌ No se encontraron documentos fuente relevantes")
                
    except Exception as e:
        st.error(f"❌ **Error al procesar la consulta:** {str(e)}")
        st.info("💡 **Sugerencias para mejorar tu consulta:**")
        st.write("• Especifica claramente los puertos de origen y destino")
        st.write("• Usa nombres completos de puertos (ej: 'Miami' en lugar de 'MIA')")
        st.write("• Asegúrate de que el vectorstore esté cargado correctamente")

####################################################################
#        Interfaz Mejorada
####################################################################

st.set_page_config(
    page_title="RAG Shipping Tarifas",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 RAG Chatbot Especializado en Tarifas de Shipping")
st.markdown("*Sistema inteligente para consulta de costos y rutas marítimas*")

def enhanced_expander_model_parameters():
    """Configuración de modelo mejorada"""
    with st.expander("⚙️ **Configuración del Modelo**"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.selected_model = st.selectbox(
                "🤖 Modelo OpenAI",
                OPENAI_MODELS,
                help="GPT-4o recomendado para máxima precisión en tarifas"
            )
            
        with col2:
            st.session_state.temperature = st.slider(
                "🌡️ Temperatura",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.1,
                help="Valores bajos = respuestas más precisas y consistentes"
            )
        
        st.session_state.top_p = st.slider(
            "🎯 Top P",
            min_value=0.8,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Control de diversidad en las respuestas"
        )

def enhanced_sidebar_and_documentChooser():
    """Interfaz lateral mejorada"""
    with st.sidebar:
        st.markdown("### 🚀 **Sistema RAG para Shipping**")
        st.caption("Powered by LangChain & OpenAI")
        
        st.markdown("---")
        
        # Status de OpenAI
        st.subheader("🔐 Estado de Conexión")
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API conectada")
            enhanced_expander_model_parameters()
        else:
            st.error("❌ API Key no encontrada")
            st.info("📝 Agrega `OPENAI_API_KEY` a tu archivo `.env`")
            return

    # Tabs para manejo de vectorstore
    tab1, tab2 = st.tabs(["🆕 Crear Vectorstore", "📂 Cargar Vectorstore"])

    # Tab 1: Crear nuevo vectorstore
    with tab1:
        st.markdown("### 📤 Subir Documentos")
        
        st.session_state.uploaded_file_list = st.file_uploader(
            "Selecciona archivos para procesar:",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv"],
            help="Los archivos CSV de tarifas recibirán procesamiento especializado"
        )
        
        st.session_state.vector_store_name = st.text_input(
            "📊 Nombre del Vectorstore:",
            placeholder="ej: tarifas_2024_q1",
            help="Se creará una base de datos con este nombre"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            create_btn = st.button("🚀 Crear Vectorstore", type="primary")
        with col2:
            if st.button("🗑️ Limpiar"):
                st.session_state.uploaded_file_list = None
                st.session_state.vector_store_name = ""
        
        if create_btn:
            enhanced_chain_RAG_blocks()
        
        # Mostrar errores si los hay
        if hasattr(st.session_state, 'error_message') and st.session_state.error_message:
            st.error(st.session_state.error_message)

    # Tab 2: Cargar vectorstore existente
    with tab2:
        st.markdown("### 📖 Cargar Base de Datos Existente")
        
        # Listar vectorstores disponibles
        LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        available_stores = [
            f.name for f in LOCAL_VECTOR_STORE_DIR.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ]
        
        if available_stores:
            st.session_state.selected_vectorstore_name = st.selectbox(
                "🗂️ Vectorstores disponibles:",
                options=[""] + available_stores,
                help=f"Encontrados {len(available_stores)} vectorstore(s)"
            )
        else:
            st.info("📭 No hay vectorstores disponibles. Crea uno primero.")
        
        # Opción de ruta personalizada
        custom_path = st.text_input(
            "📂 O especifica una ruta completa:",
            placeholder="/ruta/completa/a/tu/vectorstore",
            help="Opcional: ruta absoluta a un vectorstore externo"
        )
        
        if st.button("📖 Cargar Vectorstore", type="primary"):
            load_existing_vectorstore(custom_path)

####################################################################
#        Funciones de Procesamiento Mejoradas
####################################################################

def delete_temp_files():
    """Limpieza mejorada de archivos temporales"""
    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        files = glob.glob(TMP_DIR.as_posix() + "/*")
        deleted_count = 0
        
        for f in files:
            try:
                os.remove(f)
                deleted_count += 1
            except:
                pass
        
        if deleted_count > 0:
            st.info(f"🗑️ Limpiados {deleted_count} archivos temporales")
            
    except Exception as e:
        st.warning(f"Error limpiando archivos temporales: {e}")

def enhanced_chain_RAG_blocks():
    """Pipeline mejorado de creación de vectorstore"""
    
    if not OPENAI_API_KEY:
        st.session_state.error_message = "❌ Configura tu OpenAI API key en el archivo .env"
        return

    # Validaciones
    errors = []
    if not st.session_state.uploaded_file_list:
        errors.append("seleccionar archivos para subir")
    if not st.session_state.vector_store_name.strip():
        errors.append("proporcionar un nombre para el vectorstore")
    
    if errors:
        error_msg = "Por favor " + " y ".join(errors) + "."
        st.session_state.error_message = error_msg
        return
    
    st.session_state.error_message = ""
    
    with st.spinner("🔄 Procesando documentos y creando vectorstore..."):
        try:
            # 1. Limpiar directorio temporal
            delete_temp_files()
            
            # 2. Guardar archivos subidos
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📤 Guardando archivos...")
            for i, uploaded_file in enumerate(st.session_state.uploaded_file_list):
                temp_file_path = TMP_DIR / uploaded_file.name
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_file_list) * 0.3)
            
            # 3. Cargar y procesar documentos
            status_text.text("📖 Cargando documentos...")
            documents = enhanced_langchain_document_loader()
            progress_bar.progress(0.5)
            
            if not documents:
                st.error("❌ No se pudieron cargar documentos. Verifica los archivos.")
                return
            
            st.info(f"📊 Cargados {len(documents)} documentos")
            
            # 4. Dividir en chunks
            status_text.text("✂️ Dividiendo documentos en chunks...")
            text_splitter = create_enhanced_text_splitter()
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.7)
            
            st.info(f"📝 Creados {len(chunks)} chunks de texto")
            
            # 5. Crear embeddings y vectorstore
            status_text.text("🧠 Creando embeddings...")
            embeddings = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                model="text-embedding-ada-002"  # Modelo estable y compatible
            )
            
            persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
            persist_path.mkdir(parents=True, exist_ok=True)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path.as_posix(),
                collection_name="shipping_tariffs"
            )
            progress_bar.progress(0.9)
            
            # 6. Crear retriever y chain
            status_text.text("🔗 Configurando sistema de consultas...")
            st.session_state.retriever = create_smart_retriever(
                vector_store=st.session_state.vector_store
            )
            
            st.session_state.chain, st.session_state.memory = create_enhanced_ConversationalRetrievalChain(
                retriever=st.session_state.retriever
            )
            
            # 7. Limpiar historial
            clear_chat_history()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ **Vectorstore '{st.session_state.vector_store_name}' creado exitosamente!**")
            st.balloons()
            
            # Mostrar estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Documentos", len(documents))
            with col2:
                st.metric("📝 Chunks", len(chunks))
            with col3:
                csv_docs = sum(1 for doc in documents if doc.metadata.get("content_type") == "shipping_tariff")
                st.metric("🚢 Registros de Tarifas", csv_docs)
                
        except Exception as e:
            st.error(f"❌ Error creando vectorstore: {str(e)}")
            st.info("💡 Verifica que los archivos sean válidos y que tengas suficiente espacio en disco")

def load_existing_vectorstore(custom_path=""):
    """Carga vectorstore existente con mejor manejo de errores"""
    if not OPENAI_API_KEY:
        st.error("❌ OpenAI API key requerida")
        return

    # Determinar ruta del vectorstore
    vectorstore_path = None
    if custom_path.strip():
        vectorstore_path = Path(custom_path.strip())
    elif hasattr(st.session_state, 'selected_vectorstore_name') and st.session_state.selected_vectorstore_name:
        vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
    
    if not vectorstore_path or not vectorstore_path.exists():
        st.error("❌ Ruta de vectorstore no válida o inexistente")
        return

    with st.spinner("📖 Cargando vectorstore..."):
        try:
            embeddings = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                model="text-embedding-ada-002"  # Modelo estable
            )
            
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path.as_posix(),
                collection_name="shipping_tariffs"
            )
            
            # Verificar que el vectorstore tiene datos
            collection_count = st.session_state.vector_store._collection.count()
            if collection_count == 0:
                st.warning("⚠️ El vectorstore está vacío")
                return
            
            st.session_state.retriever = create_smart_retriever(
                vector_store=st.session_state.vector_store
            )
            
            st.session_state.chain, st.session_state.memory = create_enhanced_ConversationalRetrievalChain(
                retriever=st.session_state.retriever
            )
            
            clear_chat_history()
            
            st.success(f"✅ **Vectorstore cargado exitosamente!**")
            st.info(f"📊 Contiene {collection_count} documentos indexados")
            
        except Exception as e:
            st.error(f"❌ Error cargando vectorstore: {str(e)}")

####################################################################
#        Memoria y Utilidades
####################################################################

def create_memory():
    """Crear memoria de conversación optimizada"""
    return ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question"
    )

def clear_chat_history():
    """Limpiar historial de chat y memoria"""
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]
    if hasattr(st.session_state, 'memory') and st.session_state.memory:
        try:
            st.session_state.memory.clear()
        except:
            pass

####################################################################
#        Función Principal del Chatbot
####################################################################

def enhanced_chatbot():
    """Función principal del chatbot mejorada"""
    enhanced_sidebar_and_documentChooser()
    
    st.markdown("---")
    
    # Header del chat
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.subheader("💬 Chat con tus Datos de Shipping")
    with col2:
        if st.button("🗑️ Limpiar Chat", help="Borra el historial de conversación"):
            clear_chat_history()
            st.rerun()
    with col3:
        # Mostrar estado del sistema
        if hasattr(st.session_state, 'chain'):
            st.success("🟢 Sistema Listo")
        else:
            st.warning("🟡 Cargar Vectorstore")

    # Inicializar mensajes si no existen
    if "messages" not in st.session_state:
        clear_chat_history()

    # Mostrar historial de mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input del usuario
    if prompt := st.chat_input("Pregunta sobre tarifas de shipping... (ej: ¿Cuánto cuesta enviar de Miami a San Antonio?)"):
        
        # Verificar que el sistema esté listo
        if not OPENAI_API_KEY:
            st.error("🔑 Configura tu OpenAI API key para continuar")
            st.stop()
        
        if not hasattr(st.session_state, 'chain'):
            st.warning("⚠️ Primero carga o crea un vectorstore")
            st.stop()
        
        # Procesar consulta
        get_enhanced_response_from_LLM(prompt)

####################################################################
#        Sección de Ayuda y Tips
####################################################################

def show_help_section():
    """Muestra sección de ayuda y ejemplos"""
    with st.expander("❓ **Ayuda y Ejemplos de Uso**"):
        st.markdown("""
        ### 🎯 **Cómo hacer consultas efectivas:**
        
        **✅ Consultas recomendadas:**
        - "¿Cuánto cuesta enviar de Miami a San Antonio?"
        - "Tarifa de Callao a Valparaíso"
        - "Tiempo de tránsito de Miami a Guayaquil"
        - "Precio por tonelada de Santos a Cartagena"
        
        **❌ Evita consultas vagas:**
        - "Cuánto cuesta enviar"
        - "Precios de shipping"
        - "Tarifas generales"
        
        ### 📊 **Información que puedes obtener:**
        - **OF W/M**: Flete oceánico por peso/medida
        - **OTHERS(*) W/M**: Otros costos variables
        - **BL**: Bill of Lading
        - **SOLAS**: Certificación de seguridad
        - **Tiempo de tránsito**: Duración estimada del viaje
        
        ### 🚢 **Puertos soportados (ejemplos):**
        - Miami, San Antonio, Callao, Guayaquil
        - Santos, Cartagena, Puerto Cabello, Buenaventura
        """)

####################################################################
#        Función Principal
####################################################################

if __name__ == "__main__":
    # Configurar página
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        # Configurar valores por defecto
        st.session_state.selected_model = "gpt-4o"
        st.session_state.temperature = 0.1
        st.session_state.top_p = 0.9
        st.session_state.error_message = ""
    
    # Mostrar sección de ayuda
    show_help_section()
    
    # Ejecutar chatbot principal
    enhanced_chatbot()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🚢 RAG Chatbot para Tarifas de Shipping | Powered by LangChain & OpenAI
        </div>
        """, 
        unsafe_allow_html=True
    )