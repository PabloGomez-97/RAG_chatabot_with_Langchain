import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from config import (
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, OPENAI_API_KEY,
    get_msl_response_template, PORT_ALIASES, COUNTRY_ALIASES,
    extract_ports_from_query, normalize_port_name, detect_msl_query_type
)

####################################################################
#            PROCESADOR MSL EXCEL
####################################################################

class MSLExcelProcessor:
    """Procesador especializado para archivos Excel MSL"""
    
    def __init__(self):
        self.port_aliases = PORT_ALIASES
        self.country_aliases = COUNTRY_ALIASES
        
    def detect_header_row(self, df: pd.DataFrame, sheet_name: str) -> int:
        """Detecta fila de encabezados según estructura MSL"""
        # Basado en el análisis: AMERICA usa fila 4, otras hojas fila 5
        if sheet_name.upper() == 'AMERICA':
            return 3  # Fila 4 en Excel (0-indexed)
        else:
            return 4  # Fila 5 en Excel (0-indexed)
    
    def normalize_currency_value(self, value: str) -> str:
        """Normaliza valores de moneda MSL"""
        if pd.isna(value) or not value:
            return "TBD"
        
        value_str = str(value).strip()
        
        # Extraer valor numérico
        import re
        number_pattern = r'([\d,]+\.?\d*)'
        match = re.search(number_pattern, value_str)
        
        if match:
            number = match.group(1)
            
            # Detectar moneda
            if 'EUR' in value_str.upper():
                return f"EUR {number}"
            else:
                return f"USD {number}"
        
        return value_str
    
    def extract_time_value(self, time_str: str) -> str:
        """Extrae valor de tiempo de tránsito"""
        if pd.isna(time_str) or not time_str:
            return "No especificado"
        
        time_str = str(time_str).strip()
        
        # Buscar números seguidos de "días" o similar
        patterns = [
            r'(\d+)\s*días?',
            r'(\d+)\s*days?',
            r'(\d+)\s*d',
            r'(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, time_str, re.IGNORECASE)
            if match:
                return f"{match.group(1)} días"
        
        return time_str
    
    def process_excel_file(self, file_path: str) -> List[Document]:
        """Procesa archivo Excel MSL completo"""
        documents = []
        
        try:
            xl_file = pd.ExcelFile(file_path)
            
            for sheet_name in xl_file.sheet_names:
                print(f"[MSL] Procesando hoja: {sheet_name}")
                sheet_docs = self.process_excel_sheet(file_path, sheet_name)
                documents.extend(sheet_docs)
                print(f"[MSL] Hoja {sheet_name}: {len(sheet_docs)} documentos")
        
        except Exception as e:
            print(f"[MSL] Error procesando archivo: {str(e)}")
        
        return documents
    
    def process_excel_sheet(self, file_path: str, sheet_name: str) -> List[Document]:
        """Procesa una hoja específica del Excel MSL"""
        documents = []
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            if df.empty:
                return documents
            
            # Detectar fila de encabezados según estructura MSL
            header_row = self.detect_header_row(df, sheet_name)
            
            # Tomar encabezados
            headers = []
            for col in range(len(df.columns)):
                header_val = self._safe_get_cell(df, header_row, col)
                headers.append(header_val)
            
            print(f"[MSL] Encabezados {sheet_name}: {[h for h in headers if h]}")
            
            # Mapear columnas MSL estándar
            column_mapping = self._map_msl_columns(headers)
            
            # Procesar cada fila de datos
            for row_idx in range(header_row + 1, len(df)):
                doc = self._create_msl_document_from_row(
                    df, row_idx, column_mapping, headers, sheet_name, file_path, len(documents)
                )
                if doc:
                    documents.append(doc)
        
        except Exception as e:
            print(f"[MSL] Error procesando hoja {sheet_name}: {str(e)}")
        
        return documents
    
    def _map_msl_columns(self, headers: List) -> Dict[str, int]:
        """Mapea columnas MSL estándar"""
        mapping = {}
        
        for idx, header in enumerate(headers):
            if pd.isna(header) or not header:
                continue
                
            header_upper = str(header).upper().strip()
            
            # Mapeo específico MSL según análisis del archivo
            if 'POL' in header_upper:
                mapping['puerto_origen'] = idx
            elif 'PAIS' in header_upper or 'PAÍS' in header_upper:
                mapping['pais'] = idx
            elif ('TON' in header_upper and 'M3' in header_upper) or ('TON / M3' in header_upper):
                mapping['tarifa'] = idx
            elif 'MINIMO' in header_upper or 'MÍNIMO' in header_upper:
                mapping['minimo'] = idx
            elif 'T / T' in header_upper or 'APROX' in header_upper:
                mapping['transito'] = idx
            elif 'FREC' in header_upper:
                mapping['frecuencia'] = idx
            elif 'OTROS' in header_upper:
                mapping['otros'] = idx
            elif 'SERVICIO' in header_upper:
                mapping['servicio'] = idx
            elif 'AGENTE' in header_upper:
                mapping['agente'] = idx
            elif 'OBSERVACIONES' in header_upper:
                mapping['observaciones'] = idx
        
        return mapping
    
    def _create_msl_document_from_row(self, df, row_idx: int, mapping: Dict, headers: List, 
                                     sheet_name: str, file_path: str, doc_count: int) -> Document:
        """Crea un documento MSL a partir de una fila de datos"""
        
        # Extraer datos MSL
        puerto_origen = self._safe_get_cell(df, row_idx, mapping.get('puerto_origen', 0))
        pais = self._safe_get_cell(df, row_idx, mapping.get('pais', 1))
        tarifa = self._safe_get_cell(df, row_idx, mapping.get('tarifa', 3))
        minimo = self._safe_get_cell(df, row_idx, mapping.get('minimo', 4))
        transito = self._safe_get_cell(df, row_idx, mapping.get('transito', 5))
        frecuencia = self._safe_get_cell(df, row_idx, mapping.get('frecuencia', 6))
        otros = self._safe_get_cell(df, row_idx, mapping.get('otros', 7))
        servicio = self._safe_get_cell(df, row_idx, mapping.get('servicio', 8))
        agente = self._safe_get_cell(df, row_idx, mapping.get('agente', 9))
        observaciones = self._safe_get_cell(df, row_idx, mapping.get('observaciones', 10))
        
        # Validar que tiene datos mínimos
        if not puerto_origen or not pais or puerto_origen.strip() == '' or pais.strip() == '':
            return None
        
        # Normalizar datos
        tarifa_norm = self.normalize_currency_value(tarifa)
        minimo_norm = self.normalize_currency_value(minimo)
        transito_norm = self.extract_time_value(transito)
        puerto_normalizado = normalize_port_name(puerto_origen)
        
        # Crear contenido del documento MSL
        content = f"""TARIFA LCL MARÍTIMA - MSL (SEEMANN GROUP)
Archivo: {Path(file_path).name}
Hoja: {sheet_name}
Registro #{doc_count + 1}

=== INFORMACIÓN DE RUTA MSL ===
PUERTO ORIGEN: {puerto_origen}
PUERTO_NORMALIZADO: {puerto_normalizado}
PAÍS ORIGEN: {pais}
DESTINO: Chile (San Antonio/Valparaíso) - IMPLÍCITO MSL
REGIÓN: {sheet_name}

=== TARIFAS MSL ===
TARIFA POR TON/M³: {tarifa_norm}
TARIFA MÍNIMA: {minimo_norm}
TIEMPO TRÁNSITO: {transito_norm}
FRECUENCIA: {frecuencia if frecuencia else 'No especificado'}

=== INFORMACIÓN DEL SERVICIO MSL ===
TIPO SERVICIO: {servicio if servicio else 'No especificado'}
AGENTE LOCAL: {agente if agente else 'No especificado'}

=== COSTOS ADICIONALES MSL ===
OTROS COSTOS: {otros if otros else 'No especificado'}

=== OBSERVACIONES ESPECIALES MSL ===
{observaciones if observaciones else 'Sin observaciones especiales'}

=== TÉRMINOS DE BÚSQUEDA ===
msl seemann lcl marítimo {puerto_normalizado} {pais.lower()} chile
tarifa costo precio {puerto_origen.lower()} chile
transporte marítimo menos contenedor completo
envío desde {puerto_origen.lower()} hacia chile
"""
        
        metadata = {
            "source": file_path,
            "source_name": Path(file_path).name,
            "sheet_name": sheet_name,
            "row_number": row_idx + 1,
            "puerto_origen": puerto_origen,
            "puerto_normalizado": puerto_normalizado,
            "pais_origen": pais,
            "destino": "Chile",
            "tarifa_ton_m3": tarifa_norm,
            "tarifa_minima": minimo_norm,
            "tiempo_transito": transito_norm,
            "frecuencia": frecuencia or "No especificado",
            "tipo_servicio": servicio or "No especificado",
            "agente_local": agente or "No especificado",
            "costos_adicionales": otros or "No especificado",
            "observaciones": observaciones or "Sin observaciones",
            "content_type": "msl_lcl_rate",
            "document_type": "excel_tariff_msl",
            "route_key": f"{puerto_normalizado}_to_chile",
            "search_terms": f"msl {puerto_normalizado} {pais.lower()} chile"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _safe_get_cell(self, df, row: int, col: int) -> str:
        """Obtiene valor de celda de forma segura"""
        try:
            if row >= len(df) or col >= len(df.columns):
                return ""
            value = df.iloc[row, col]
            if pd.isna(value):
                return ""
            return str(value).strip()
        except:
            return ""

####################################################################
#            CARGADOR DE DOCUMENTOS MSL
####################################################################

def load_msl_excel_documents() -> List[Document]:
    """Carga y procesa archivos Excel MSL"""
    documents = []
    
    print("[MSL] Iniciando carga de documentos MSL...")
    
    # Buscar archivos Excel
    excel_files = list(TMP_DIR.glob("**/*.xlsx")) + list(TMP_DIR.glob("**/*.xls"))
    
    print(f"[MSL] Archivos encontrados: {[f.name for f in excel_files]}")
    
    if not excel_files:
        print("[MSL] No se encontraron archivos Excel")
        return documents
    
    processor = MSLExcelProcessor()
    
    for excel_file in excel_files:
        try:
            print(f"[MSL] === PROCESANDO: {excel_file.name} ===")
            file_docs = processor.process_excel_file(str(excel_file))
            documents.extend(file_docs)
            print(f"[MSL] Archivo {excel_file.name} generó {len(file_docs)} documentos")
            
        except Exception as e:
            print(f"[MSL] Error procesando {excel_file.name}: {str(e)}")
            continue
    
    print(f"[MSL] === RESUMEN ===")
    print(f"[MSL] Total documentos: {len(documents)}")
    
    return documents

####################################################################
#            SISTEMA DE BÚSQUEDA MSL
####################################################################

def create_msl_retriever(vector_store, k=15):
    """Crea retriever especializado para consultas MSL"""
    search_kwargs = {
        "k": k,
        "lambda_mult": 0.2,  # Balance entre precisión y diversidad
        "fetch_k": k * 2     # Candidatos para filtrar
    }
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    return retriever

def generate_msl_query_variations(original_query: str) -> List[str]:
    """Genera variaciones de consulta para búsqueda MSL"""
    variations = [original_query]
    query_lower = original_query.lower()
    
    # Extraer información de puertos
    route_info = extract_ports_from_query(original_query)
    
    if route_info.get('has_route'):
        origin = route_info.get('origin_normalized', '')
        
        if origin:
            variations.extend([
                f"MSL {origin}",
                f"tarifa {origin}",
                f"puerto {origin}",
                f"PUERTO_NORMALIZADO: {origin}",
                f"desde {origin}",
                f"Seemann {origin}",
                origin
            ])
    
    # Agregar términos MSL específicos
    variations.extend([
        f"{query_lower} msl",
        f"{query_lower} seemann",
        f"{query_lower} lcl",
        f"{query_lower} marítimo",
        f"{query_lower} ton m3"
    ])
    
    return list(set(variations))

def multi_query_msl_retriever(vector_store, original_query: str) -> List[Document]:
    """Ejecuta búsqueda multi-query para MSL"""
    
    query_variations = generate_msl_query_variations(original_query)
    all_docs = []
    seen_docs = set()
    
    retriever = create_msl_retriever(vector_store, k=20)
    
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
    
    return all_docs[:25]  # Limitar resultados

####################################################################
#            CHAIN CONVERSACIONAL MSL
####################################################################

def create_msl_conversational_chain(retriever):
    """Crea chain conversacional especializado MSL"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para maximizar recuperación de tarifas LCL MSL.

IMPORTANTE - BÚSQUEDA MSL:
- Mantener nombres exactos de puertos y países
- Incluir términos: MSL, Seemann, LCL, marítimo, tarifa, puerto
- Recordar que destino siempre es Chile en MSL
- Buscar estructura: TON/M3 USD

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda MSL:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(get_msl_response_template())
    
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question"
    )

    standalone_query_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o",
        temperature=0.0,
    )
    
    response_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o",
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
        chain_type="stuff",
        verbose=True,
        return_source_documents=True,
        max_tokens_limit=4000
    )

    return chain, memory

####################################################################
#            VALIDACIÓN Y ANÁLISIS MSL
####################################################################

def validate_msl_response(response_text: str, original_query: str, source_docs: List) -> Dict[str, Any]:
    """Valida completitud de respuesta MSL"""
    
    validation = {
        'completeness': 1.0,
        'route_accuracy': 1.0,
        'warnings': [],
        'suggestions': []
    }
    
    query_lower = original_query.lower()
    
    # Validar presencia de información MSL obligatoria
    required_info = [
        ('puerto', ['puerto', 'origen']),
        ('tarifa', ['usd', 'eur', 'ton', 'm3']),
        ('tiempo', ['días', 'día', 'tiempo']),
        ('servicio', ['servicio', 'directo', 'vía'])
    ]
    
    missing_info = []
    for info_type, keywords in required_info:
        if not any(keyword in response_text.lower() for keyword in keywords):
            missing_info.append(info_type)
    
    if missing_info:
        validation['completeness'] *= (1 - len(missing_info) * 0.2)
        validation['warnings'].append(f"Información faltante: {', '.join(missing_info)}")
    
    # Validar ruta específica si se consultó
    route_info = extract_ports_from_query(original_query)
    if route_info.get('has_route'):
        origin = route_info.get('origin_normalized', '')
        
        # Verificar que el puerto origen aparezca en la respuesta
        if origin and origin not in response_text.lower():
            validation['route_accuracy'] *= 0.5
            validation['warnings'].append(f"Puerto origen {origin} no claramente reflejado")
    
    # Validar presencia de costos adicionales
    if 'costo' in query_lower or 'precio' in query_lower:
        if 'otros' not in response_text.lower() and 'adicional' not in response_text.lower():
            validation['suggestions'].append("Considerar mostrar costos adicionales (DDT, VGM, etc.)")
    
    return validation

def analyze_msl_sources(sources: List[Document]) -> Dict[str, Any]:
    """Analiza fuentes de documentos MSL"""
    
    if not sources:
        return {"total": 0, "regions": {}, "ports": {}}
    
    analysis = {
        "total": len(sources),
        "regions": {},
        "ports": {},
        "countries": {},
        "services": {}
    }
    
    for doc in sources:
        # Análisis por región
        region = doc.metadata.get('sheet_name', 'Desconocido')
        analysis['regions'][region] = analysis['regions'].get(region, 0) + 1
        
        # Análisis por puerto
        puerto = doc.metadata.get('puerto_normalizado', 'Desconocido')
        analysis['ports'][puerto] = analysis['ports'].get(puerto, 0) + 1
        
        # Análisis por país
        pais = doc.metadata.get('pais_origen', 'Desconocido')
        analysis['countries'][pais] = analysis['countries'].get(pais, 0) + 1
        
        # Análisis por tipo de servicio
        servicio = doc.metadata.get('tipo_servicio', 'Desconocido')
        analysis['services'][servicio] = analysis['services'].get(servicio, 0) + 1
    
    return analysis