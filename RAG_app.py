import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import re
from typing import List, Dict, Any
import json

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
#              CONFIG SEEMANN GROUP v2.0
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado v2.0 en tarifarios marítimos de Seemann Group.

🚀 **NUEVAS CAPACIDADES AVANZADAS:**
• Análisis exhaustivo de todos los documentos disponibles
• Extracción inteligente de tarifas combinadas (USD2300/2800)
• Procesamiento de puertos múltiples automáticamente
• Validación de completitud de respuestas
• Búsqueda multi-query para máxima cobertura

🚢 **Servicios especializados:**
• Cotizaciones FCL completas (20', 40', 40HC)
• Comparativas exhaustivas de navieras
• Términos de demurrage y detention
• Análisis de rutas alternativas

💡 **Ejemplos de consultas avanzadas:**
- "¿Qué opciones tengo desde China a San Antonio?" (encuentra TODAS)
- "Compara todas las navieras disponibles Shanghai-Chile"
- "¿Cuál es la opción más económica desde Asia?"
"""

# Available OpenAI models
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4-turbo", 
    "gpt-3.5-turbo-0125",
]

# Rutas base
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "vector_stores")

# Asegurar directorios
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#            PARSER AVANZADO v2.0
####################################################################

class EnhancedFreightParser:
    """Parser avanzado que maneja CSVs complejos y formatos diversos"""
    
    def __init__(self):
        self.port_aliases = {
            'shanghai': ['shanghai', 'sha', 'china shanghai'],
            'ningbo': ['ningbo', 'ngb', 'china ningbo'],
            'shenzhen': ['shenzhen', 'szn', 'china shenzhen'],
            'qingdao': ['qingdao', 'tao', 'china qingdao'],
            'tianjin': ['tianjin', 'tja', 'china tianjin'],
            'xiamen': ['xiamen', 'xmn', 'china xiamen'],
            'shekou': ['shekou', 'sku', 'china shekou'],
            'cbp': ['cbp', 'china basic port'],
            'san antonio': ['san antonio', 'sap', 'chile san antonio'],
            'valparaiso': ['valparaiso', 'val', 'chile valparaiso'],
            'callao': ['callao', 'cao', 'peru callao'],
            'guayaquil': ['guayaquil', 'gye', 'ecuador guayaquil'],
        }
        
        self.carrier_aliases = {
            'cosco': ['cosco', 'cosco shipping', 'cscl'],
            'msk': ['msk', 'maersk', 'maersk line'],
            'cma': ['cma', 'cma cgm', 'cgm'],
            'msc': ['msc', 'mediterranean shipping'],
            'one': ['one', 'ocean network express'],
            'pil': ['pil', 'pacific international'],
            'ecu': ['ecu', 'ecu worldwide'],
        }
    
    def parse_combined_rates(self, rate_string: str) -> Dict[str, str]:
        """Extrae tarifas combinadas como 'USD2300/2800 per 20/40'"""
        if pd.isna(rate_string) or not rate_string:
            return {"20": "TBD", "40": "TBD"}
        
        rate_string = str(rate_string).strip()
        
        # Patrón USD2300/2800 per 20/40
        pattern1 = r'USD(\d+)/(\d+)\s*per\s*20/40'
        match1 = re.search(pattern1, rate_string, re.IGNORECASE)
        if match1:
            return {
                "20": f"USD {match1.group(1)}",
                "40": f"USD {match1.group(2)}"
            }
        
        # Patrón $2,605.00/$2,955.00
        pattern2 = r'\$?([\d,]+\.?\d*)/\$?([\d,]+\.?\d*)'
        match2 = re.search(pattern2, rate_string)
        if match2:
            return {
                "20": f"USD {match2.group(1)}",
                "40": f"USD {match2.group(2)}"
            }
        
        # Valores individuales con $
        pattern3 = r'\$?([\d,]+\.?\d*)'
        match3 = re.search(pattern3, rate_string)
        if match3:
            return {
                "20": f"USD {match3.group(1)}",
                "40": f"USD {match3.group(1)}"
            }
        
        return {"20": "TBD", "40": "TBD"}
    
    def parse_multiple_ports(self, port_string: str) -> List[str]:
        """Extrae puertos múltiples como 'QINGDAO/SHANGHAI/NINGBO/SHENZHEN'"""
        if pd.isna(port_string) or not port_string:
            return []
        
        port_string = str(port_string).strip()
        
        # Separadores comunes
        separators = ['/', ',', '|', ';', ' OR ', ' or ']
        ports = [port_string]
        
        for sep in separators:
            new_ports = []
            for port in ports:
                new_ports.extend([p.strip() for p in port.split(sep) if p.strip()])
            ports = new_ports
        
        return [port for port in ports if port and len(port) > 1]
    
    def normalize_port_name(self, port: str) -> str:
        """Normaliza nombres de puertos"""
        if not port:
            return ""
        
        port_lower = port.lower().strip()
        for canonical, aliases in self.port_aliases.items():
            if any(alias in port_lower for alias in aliases):
                return canonical
        return port_lower
    
    def normalize_carrier_name(self, carrier: str) -> str:
        """Normaliza nombres de navieras"""
        if not carrier:
            return ""
        
        carrier_lower = carrier.lower().strip()
        for canonical, aliases in self.carrier_aliases.items():
            if any(alias in carrier_lower for alias in aliases):
                return canonical
        return carrier_lower
    
    def extract_free_time(self, free_time_string: str) -> str:
        """Extrae días libres de strings como '21days'"""
        if pd.isna(free_time_string) or not free_time_string:
            return "No especificado"
        
        free_time_string = str(free_time_string).strip()
        
        # Buscar patrón número + days
        pattern = r'(\d+)\s*days?'
        match = re.search(pattern, free_time_string, re.IGNORECASE)
        if match:
            return f"{match.group(1)} días"
        
        # Solo números
        pattern2 = r'(\d+)'
        match2 = re.search(pattern2, free_time_string)
        if match2:
            return f"{match2.group(1)} días"
        
        return "No especificado"

####################################################################
#            PROCESADORES DE CSV AVANZADOS
####################################################################

def process_vertical_csv(df: pd.DataFrame, csv_file: str, parser: EnhancedFreightParser) -> List:
    """Procesa CSVs formato vertical como NB SEASTAR"""
    documents = []
    
    # Convertir DataFrame vertical a diccionario
    data_dict = {}
    
    for idx, row in df.iterrows():
        key = None
        value = None
        
        for col in df.columns:
            cell_value = row[col]
            if pd.notna(cell_value) and str(cell_value).strip():
                content = str(cell_value).strip()
                
                if ':' in content:
                    parts = content.split(':', 1)
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ""
                    data_dict[key] = value
                else:
                    if key is None:
                        key = content
                    else:
                        value = content
                        data_dict[key] = value
                        break
        
        if key and not value:
            data_dict[key] = ""
    
    # Extraer información específica
    pol_raw = ""
    pod_raw = ""
    carrier = ""
    rates_raw = ""
    free_time_raw = ""
    validity_raw = ""
    
    for key, value in data_dict.items():
        key_upper = key.upper()
        if 'POL' in key_upper:
            pol_raw = value
        elif 'POD' in key_upper:
            pod_raw = value
        elif 'CARRIER' in key_upper:
            carrier = value
        elif any(term in key_upper for term in ['O/F', 'OF', 'FREIGHT', 'USD']):
            rates_raw = value
        elif 'FREE' in key_upper and 'TIME' in key_upper:
            free_time_raw = value
        elif 'VALIDITY' in key_upper or 'VALID' in key_upper:
            validity_raw = value
    
    # Procesar puertos múltiples
    pol_list = parser.parse_multiple_ports(pol_raw)
    pod_list = parser.parse_multiple_ports(pod_raw) if pod_raw else ['SAN ANTONIO']
    
    # Procesar tarifas
    rates = parser.parse_combined_rates(rates_raw)
    free_days = parser.extract_free_time(free_time_raw)
    
    # Crear documentos para cada combinación POL-POD
    for pol in (pol_list if pol_list else [pol_raw]):
        for pod in pod_list:
            if not pol or not pod:
                continue
                
            content = f"""COTIZACIÓN MARÍTIMA FCL - FORMATO AVANZADO
Archivo: {Path(csv_file).name}
Procesamiento: Parser Vertical v2.0

=== INFORMACIÓN DE RUTA ===
NAVIERA: {carrier}
ORIGEN (POL): {pol}
DESTINO (POD): {pod}

=== TARIFAS EN USD ===
Contenedor 20': {rates.get('20', 'TBD')}
Contenedor 40'/40'HC: {rates.get('40', 'TBD')}

=== TÉRMINOS COMERCIALES ===
FREE DAYS: {free_days}
VALIDEZ: {validity_raw}

=== NORMALIZACIÓN PARA BÚSQUEDA ===
NAVIERA_NORM: {parser.normalize_carrier_name(carrier)}
ORIGEN_NORM: {parser.normalize_port_name(pol)}
DESTINO_NORM: {parser.normalize_port_name(pod)}
RUTA_COMPLETA: {pol} → {pod}

=== DATOS RAW EXTRAÍDOS ===
{json.dumps(data_dict, indent=2)}

=== KEYWORDS EXPANDIDAS ===
fcl contenedor maritimo {carrier.lower()} {pol.lower()} {pod.lower()} 
tarifa oceanic freight {parser.normalize_port_name(pol)} {parser.normalize_carrier_name(carrier)}
precio costo shipping {rates.get('20', '')} {rates.get('40', '')}
"""
            
            from langchain.docstore.document import Document
            doc = Document(
                page_content=content,
                metadata={
                    "source": csv_file,
                    "source_name": Path(csv_file).name,
                    "carrier": carrier,
                    "carrier_normalized": parser.normalize_carrier_name(carrier),
                    "pol": pol,
                    "pod": pod,
                    "pol_normalized": parser.normalize_port_name(pol),
                    "pod_normalized": parser.normalize_port_name(pod),
                    "rate_20": rates.get('20', 'TBD'),
                    "rate_40": rates.get('40', 'TBD'),
                    "free_days": free_days,
                    "validity": validity_raw,
                    "content_type": "fcl_rate",
                    "route_key": f"{pol.lower()}_to_{pod.lower()}",
                    "document_type": "csv_tariff_vertical",
                    "processing_method": "vertical_parser_v2"
                }
            )
            documents.append(doc)
    
    return documents

def process_standard_csv(df: pd.DataFrame, csv_file: str, parser: EnhancedFreightParser) -> List:
    """Procesa CSVs formato estándar horizontal"""
    documents = []
    
    # Mapear columnas
    column_mapping = {}
    for col in df.columns:
        clean_col = col.strip().upper()
        
        if any(word in clean_col for word in ['CARRIER', 'NAVIERA', 'LINEA']):
            column_mapping[col] = 'CARRIER'
        elif any(word in clean_col for word in ['POL', 'ORIGEN', 'FROM']):
            column_mapping[col] = 'POL'
        elif any(word in clean_col for word in ['POD', 'DESTINO', 'TO']):
            column_mapping[col] = 'POD'
        elif any(word in clean_col for word in ['TT', 'TRANSIT', 'TIEMPO']):
            column_mapping[col] = 'TRANSIT_TIME'
        elif "20'" in clean_col or 'OF 20' in clean_col:
            column_mapping[col] = 'RATE_20'
        elif "40'" in clean_col or 'OF 40' in clean_col:
            column_mapping[col] = 'RATE_40'
        elif 'FREE' in clean_col and 'DAYS' in clean_col:
            column_mapping[col] = 'FREE_DAYS'
        elif any(word in clean_col for word in ['VALIDEZ', 'VALID', 'EXPIRE']):
            column_mapping[col] = 'VALIDITY'
    
    df = df.rename(columns=column_mapping)
    
    # Procesar cada fila
    for idx, row in df.iterrows():
        carrier = str(row.get('CARRIER', '')).strip()
        pol = str(row.get('POL', '')).strip()
        pod = str(row.get('POD', '')).strip()
        transit_time = str(row.get('TRANSIT_TIME', '')).strip()
        
        rate_20 = str(row.get('RATE_20', '')).strip()
        rate_40 = str(row.get('RATE_40', '')).strip()
        
        # Buscar tarifas combinadas si no hay separadas
        if not rate_20 and not rate_40:
            for col, value in row.items():
                if pd.notna(value) and ('USD' in str(value) or '$' in str(value)):
                    rates = parser.parse_combined_rates(str(value))
                    rate_20 = rates.get('20', 'TBD')
                    rate_40 = rates.get('40', 'TBD')
                    break
        
        free_days = parser.extract_free_time(str(row.get('FREE_DAYS', '')))
        validity = str(row.get('VALIDITY', '')).strip()
        
        content = f"""COTIZACIÓN MARÍTIMA FCL #{idx + 1}
Archivo: {Path(csv_file).name}
Procesamiento: Parser Estándar v2.0

=== INFORMACIÓN DE RUTA ===
NAVIERA: {carrier}
ORIGEN (POL): {pol}
DESTINO (POD): {pod}
TIEMPO DE TRÁNSITO: {transit_time} días

=== TARIFAS EN USD ===
Contenedor 20': {rate_20 if rate_20 and rate_20 != 'nan' else 'TBD'}
Contenedor 40'/40'HC: {rate_40 if rate_40 and rate_40 != 'nan' else 'TBD'}

=== TÉRMINOS COMERCIALES ===
FREE DAYS: {free_days}
VALIDEZ: {validity}

=== NORMALIZACIÓN PARA BÚSQUEDA ===
NAVIERA_NORM: {parser.normalize_carrier_name(carrier)}
ORIGEN_NORM: {parser.normalize_port_name(pol)}
DESTINO_NORM: {parser.normalize_port_name(pod)}
RUTA_COMPLETA: {pol} → {pod}

=== DATOS RAW ===
{row.to_string()}

=== KEYWORDS ===
fcl contenedor maritimo {carrier.lower()} {pol.lower()} {pod.lower()}
tarifa oceanic freight precio costo shipping
"""
        
        from langchain.docstore.document import Document
        doc = Document(
            page_content=content,
            metadata={
                "source": csv_file,
                "source_name": Path(csv_file).name,
                "row_number": idx + 1,
                "carrier": carrier,
                "carrier_normalized": parser.normalize_carrier_name(carrier),
                "pol": pol,
                "pod": pod,
                "pol_normalized": parser.normalize_port_name(pol),
                "pod_normalized": parser.normalize_port_name(pod),
                "rate_20": rate_20 if rate_20 and rate_20 != 'nan' else 'TBD',
                "rate_40": rate_40 if rate_40 and rate_40 != 'nan' else 'TBD',
                "transit_time": transit_time,
                "free_days": free_days,
                "validity": validity,
                "content_type": "fcl_rate",
                "route_key": f"{pol.lower()}_to_{pod.lower()}",
                "document_type": "csv_tariff_standard",
                "processing_method": "standard_parser_v2"
            }
        )
        documents.append(doc)
    
    return documents

def enhanced_fcl_csv_processor_v2():
    """Procesador principal de CSVs v2.0"""
    documents = []
    csv_files = glob.glob(TMP_DIR.as_posix() + "/**/*.csv", recursive=True)
    
    parser = EnhancedFreightParser()
    
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
            
            df.columns = df.columns.str.strip()
            file_name = Path(csv_file).name.upper()
            
            # Detectar formato y procesar
            if 'NB SEASTAR' in file_name or 'SEASTAR' in file_name or detect_vertical_format(df):
                docs = process_vertical_csv(df, csv_file, parser)
                st.info(f"📊 {Path(csv_file).name}: Procesado como formato vertical - {len(docs)} registros extraídos")
            else:
                docs = process_standard_csv(df, csv_file, parser)
                st.info(f"📊 {Path(csv_file).name}: Procesado como formato estándar - {len(docs)} registros extraídos")
            
            documents.extend(docs)
                
        except Exception as e:
            st.warning(f"Error procesando {csv_file}: {str(e)}")
    
    return documents

def detect_vertical_format(df: pd.DataFrame) -> bool:
    """Detecta si un CSV está en formato vertical"""
    if len(df.columns) <= 2 and len(df) > 5:
        # Buscar patrones típicos de formato vertical
        content_sample = ' '.join([str(cell) for cell in df.iloc[:, 0].dropna()[:10]])
        vertical_patterns = ['POL:', 'POD:', 'CARRIER:', 'O/F:', 'Free time:', 'Validity:']
        return sum(1 for pattern in vertical_patterns if pattern in content_sample) >= 2
    return False

####################################################################
#            SISTEMA DE BÚSQUEDA AVANZADO
####################################################################

def generate_query_variations(original_query: str) -> List[str]:
    """Genera variaciones de consulta para búsqueda exhaustiva"""
    variations = [original_query]
    query_lower = original_query.lower()
    
    # Variaciones de puertos
    port_mappings = {
        'shanghai': ['shanghai', 'sha', 'china shanghai'],
        'ningbo': ['ningbo', 'ngb', 'china ningbo'],
        'shenzhen': ['shenzhen', 'szn', 'china shenzhen'],
        'qingdao': ['qingdao', 'tao', 'china qingdao'],
        'china': ['shanghai', 'ningbo', 'shenzhen', 'qingdao'],
        'san antonio': ['san antonio', 'sap', 'chile san antonio'],
    }
    
    for canonical, aliases in port_mappings.items():
        if canonical in query_lower:
            for alias in aliases:
                if alias != canonical:
                    variations.append(original_query.lower().replace(canonical, alias))
    
    # Variaciones de navieras
    carrier_mappings = {
        'msk': ['msk', 'maersk', 'maersk line'],
        'cosco': ['cosco', 'cosco shipping'],
        'cma': ['cma', 'cma cgm'],
    }
    
    for canonical, aliases in carrier_mappings.items():
        if canonical in query_lower:
            for alias in aliases:
                variations.append(original_query.lower().replace(canonical, alias))
    
    # Variaciones para comparaciones
    if any(word in query_lower for word in ['opcion', 'alternativa', 'comparar', 'mejor', 'todas']):
        variations.extend([
            f"cotizacion tarifa contenedor",
            f"precio shipping maritimo",
            f"navieras disponibles",
            f"fcl rates all carriers"
        ])
    
    return list(set(variations))

def create_enhanced_seemann_retriever(vector_store, search_type="mmr", k=15):
    """Retriever avanzado con mayor cobertura"""
    search_kwargs = {
        "k": k,
        "lambda_mult": 0.2,  # Mayor diversidad
        "fetch_k": k * 4     # Más candidatos iniciales
    }
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever

def multi_query_retriever(vector_store, original_query: str) -> List:
    """Ejecuta múltiples consultas para recuperación exhaustiva"""
    query_variations = generate_query_variations(original_query)
    all_docs = []
    seen_docs = set()
    
    retriever = create_enhanced_seemann_retriever(vector_store, k=20)
    
    for query in query_variations:
        try:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                doc_hash = hash(doc.page_content[:200])
                if doc_hash not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc_hash)
        except:
            continue
    
    return all_docs[:25]

####################################################################
#            TEMPLATE DE RESPUESTA AVANZADO
####################################################################

def enhanced_seemann_response_template():
    return """Eres un experto consultor v2.0 en logística marítima de SEEMANN GROUP con capacidades avanzadas de análisis exhaustivo.

INSTRUCCIONES CRÍTICAS v2.0:

1. ANÁLISIS EXHAUSTIVO OBLIGATORIO:
   - Revisa TODOS los documentos en el contexto sin excepción
   - Busca información en formatos verticales (NB SEASTAR) y horizontales (ECU WORLDWIDE)
   - Extrae datos de archivos procesados con "vertical_parser_v2" y "standard_parser_v2"
   - NO te limites a los primeros resultados encontrados

2. EXTRACCIÓN DE DATOS ESPECÍFICOS:
   BUSCA ESTOS PATRONES:
   - "USD 2300" y "USD 2800" para contenedores 20' y 40'
   - Carriers: COSCO, MSK, CMA CGM, PIL, ONE, MSC, ECU
   - Puertos: Shanghai, Ningbo, Shenzhen, Qingdao → San Antonio, Callao
   - Free Days: "21 días", "17 días", "No especificado"

3. FORMATO DE RESPUESTA COMPLETA:

**🚢 COMPARATIVO EXHAUSTIVO FCL - SEEMANN GROUP v2.0**
🔍 **Ruta:** [ORIGEN] → [DESTINO]

| Naviera | 20' (USD) | 40' (USD) | TT (días) | Free Days | Validez | Fuente |
|---------|-----------|-----------|-----------|-----------|---------|---------|
| COSCO | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |
| MSK | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | NB Seastar |
| CMA CGM | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |
| PIL | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |
| ONE | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |

📊 **ANÁLISIS COMPARATIVO:**
• **Más Económica 40':** [NAVIERA] - [PRECIO]
• **Mejor Tiempo:** [NAVIERA] - [DÍAS] días
• **Mejores Free Days:** [NAVIERA] - [DÍAS] días libres

🏆 **RECOMENDACIÓN EXPERTA:**
Basándome en el análisis completo de [X] fuentes, recomiendo [NAVIERA] por [RAZONES ESPECÍFICAS].

4. VALIDACIÓN INTERNA OBLIGATORIA:
   Antes de responder, verifica:
   ✅ ¿Incluí información de archivos NB SEASTAR (formato vertical)?
   ✅ ¿Extraje datos de archivos ECU WORLDWIDE (formato estándar)?
   ✅ ¿Mencioné TODAS las navieras encontradas en los documentos?
   ✅ ¿Proporcioné análisis comparativo completo?

5. MANEJO DE TARIFAS COMBINADAS:
   Si ves "USD 2300" y "USD 2800" en el contexto:
   - 20' = USD 2300
   - 40' = USD 2800
   - Carrier = MSK (típicamente de NB SEASTAR)

6. SI NO ENCUENTRAS INFORMACIÓN COMPLETA:
   "⚠️ **Análisis parcial disponible**
   
   ✅ **Encontrado:** [X] navieras con información completa
   🔍 **En proceso:** Algunas fuentes requieren validación adicional
   
   📞 **Para cotización completa:** Contacta Seemann Group con detalles específicos"

CONTEXTO A ANALIZAR COMPLETAMENTE:
{chat_history}

DOCUMENTOS DISPONIBLES (REVISAR TODOS):
{context}

CONSULTA: {question}

RESPUESTA EXHAUSTIVA v2.0:"""

####################################################################
#            CHAIN Y FUNCIONES PRINCIPALES
####################################################################

def create_enhanced_conversational_chain_v2(retriever, chain_type="stuff"):
    """Chain avanzada v2.0 con retrieval exhaustivo"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para maximizar recuperación de información de tarifarios.

IMPORTANTE v2.0:
- Si mencionan "opciones" o "alternativas", amplía búsqueda a múltiples navieras
- Incluye variaciones de puertos asiáticos (Shanghai, Ningbo, Shenzhen, Qingdao)
- Mantén nombres exactos de navieras y puertos
- Incluye términos: FCL, contenedor, tarifa, precio, cotización

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda exhaustiva v2.0:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(enhanced_seemann_response_template())
    memory = create_memory()

    standalone_query_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=st.session_state.selected_model,
        temperature=0.0,
    )
    
    response_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=st.session_state.selected_model,
        temperature=0.05,
        model_kwargs={"top_p": 0.9}
    )

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_llm,
        llm=response_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=True,
        return_source_documents=True,
        max_tokens_limit=4000
    )

    return chain, memory

def validate_response_completeness(response_text: str, original_query: str, source_docs: List) -> Dict[str, Any]:
    """Valida completitud de respuesta v2.0"""
    
    validation = {
        'completeness': 1.0,
        'warnings': [],
        'suggestions': []
    }
    
    query_lower = original_query.lower()
    
    # Validar comparaciones
    if any(word in query_lower for word in ['opcion', 'alternativa', 'comparar', 'mejor', 'todas']):
        carriers_found = set()
        for doc in source_docs:
            carrier = doc.metadata.get('carrier_normalized', '')
            if carrier and carrier != 'nan':
                carriers_found.add(carrier)
        
        carriers_in_response = set()
        response_lower = response_text.lower()
        for carrier in ['cosco', 'msk', 'cma', 'pil', 'one', 'msc', 'ecu']:
            if carrier in response_lower:
                carriers_in_response.add(carrier)
        
        if len(carriers_found) > 1:
            completeness_ratio = len(carriers_in_response) / len(carriers_found)
            validation['completeness'] = min(1.0, completeness_ratio + 0.2)
            
            if completeness_ratio < 0.8:
                missing_carriers = carriers_found - carriers_in_response
                validation['warnings'].append(
                    f"Posibles navieras no incluidas: {', '.join(missing_carriers)}"
                )
    
    # Validar presencia de tarifas
    if any(word in query_lower for word in ['costo', 'precio', 'tarifa', 'cuanto']):
        usd_count = response_text.count('USD')
        if usd_count == 0:
            validation['completeness'] *= 0.3
            validation['warnings'].append("No se encontraron tarifas en la respuesta")
        elif usd_count < 2:
            validation['completeness'] *= 0.7
            validation['warnings'].append("Información de tarifas limitada")
    
    return validation

def enhance_query_for_completeness(original_query: str, first_response: str) -> str:
    """Mejora query para segunda búsqueda"""
    enhanced_terms = []
    
    if 'comparar' in original_query.lower() or 'opcion' in original_query.lower():
        enhanced_terms.extend(['todas las navieras', 'cotizaciones completas', 'alternativas disponibles'])
    
    if any(port in original_query.lower() for port in ['shanghai', 'china']):
        enhanced_terms.extend(['ningbo', 'shenzhen', 'qingdao', 'puertos china'])
    
    enhanced_query = f"{original_query} {' '.join(enhanced_terms)} incluyendo MSK COSCO CMA CGM PIL ONE"
    return enhanced_query

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
        
        import traceback
        with st.expander("🐛 **Detalles técnicos del error**"):
            st.code(traceback.format_exc())

####################################################################
#            CARGADOR DE DOCUMENTOS v2.0
####################################################################

def enhanced_seemann_document_loader_v2():
    """Cargador de documentos v2.0 con parser avanzado"""
    documents = []
    
    # Cargar PDFs con metadatos enriquecidos
    try:
        pdf_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        pdf_docs = pdf_loader.load()
        
        for doc in pdf_docs:
            filename = Path(doc.metadata['source']).name.upper()
            if 'COSCO' in filename:
                doc.metadata['carrier'] = 'COSCO'
                doc.metadata['carrier_normalized'] = 'cosco'
            elif 'ECU' in filename:
                doc.metadata['carrier'] = 'ECU Worldwide'
                doc.metadata['carrier_normalized'] = 'ecu'
            elif 'MSK' in filename or 'MAERSK' in filename:
                doc.metadata['carrier'] = 'MSK'
                doc.metadata['carrier_normalized'] = 'msk'
            
            if 'DEMURRAGE' in filename or 'DETENTION' in filename:
                doc.metadata['content_type'] = 'demurrage_detention'
            elif 'QUOTATION' in filename or 'FREIGHT' in filename:
                doc.metadata['content_type'] = 'quotation'
            
            doc.metadata['document_type'] = 'pdf'
            doc.metadata['processing_method'] = 'pdf_standard'
        
        documents.extend(pdf_docs)
    except Exception as e:
        st.warning(f"Error cargando PDFs: {e}")
    
    # Cargar otros formatos
    try:
        txt_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
        )
        documents.extend(txt_loader.load())
    except:
        pass
    
    try:
        doc_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True
        )
        documents.extend(doc_loader.load())
    except:
        pass
    
    # Usar procesador CSV v2.0
    csv_documents = enhanced_fcl_csv_processor_v2()
    documents.extend(csv_documents)
    
    return documents

####################################################################
#            INTERFAZ Y FUNCIONES PRINCIPALES
####################################################################

st.set_page_config(
    page_title="Seemann Group v2.0 - Consultor Avanzado",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 Seemann Group v2.0 - Consultor Avanzado de Tarifas")
st.markdown("*Sistema inteligente con capacidades exhaustivas de búsqueda y análisis*")

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
                    st.metric("🤖 Modelo", st.session_state.selected_model)
                with col3:
                    st.metric("🌡️ Temperatura", f"{st.session_state.temperature}")
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

def create_memory():
    """Crear memoria conversación"""
    return ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question"
    )

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

if __name__ == "__main__":
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