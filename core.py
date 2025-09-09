import glob
import pandas as pd
from pathlib import Path
import re
import json
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from config import (
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, OPENAI_API_KEY,
    get_lcl_response_template, PORT_ALIASES, COUNTRY_ALIASES,
    extract_ports_from_query, normalize_port_name, detect_lcl_query_type
)

####################################################################
#            PROCESADOR ESPECIALIZADO EXCEL LCL v1.0
####################################################################

class LCLExcelProcessor:
    """Procesador especializado para archivos Excel de tarifas LCL marítimas"""
    
    def __init__(self):
        self.port_aliases = PORT_ALIASES
        self.country_aliases = COUNTRY_ALIASES
        
    def detect_header_row(self, df: pd.DataFrame) -> int:
        """Detecta automáticamente la fila de encabezados"""
        for row_idx in range(min(10, len(df))):
            row_content = ' '.join([str(cell) for cell in df.iloc[row_idx] if pd.notna(cell)])
            
            # Buscar patrones típicos de encabezados LCL
            header_patterns = ['PUERTO CARGA', 'PAIS', 'TON', 'M3', 'MINIMO', 'FREC', 'SERVICIO']
            matches = sum(1 for pattern in header_patterns if pattern in row_content.upper())
            
            if matches >= 4:  # Si encontramos al menos 4 patrones típicos
                return row_idx
        
        return 4  # Default basado en análisis previo
    
    def normalize_currency_value(self, value: str) -> str:
        """Normaliza valores de moneda"""
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
    
    def process_excel_sheet(self, file_path: str, sheet_name: str) -> List[Document]:
        """Procesa una hoja específica del Excel"""
        documents = []
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                return documents
            
            # Detectar fila de encabezados
            header_row = self.detect_header_row(df)
            
            # Tomar encabezados y datos
            headers = df.iloc[header_row].tolist()
            data_rows = df.iloc[header_row + 1:].reset_index(drop=True)
            
            # Mapear columnas estándar
            column_mapping = self._map_columns(headers)
            
            # Procesar cada fila de datos
            for idx, row in data_rows.iterrows():
                doc = self._create_document_from_row(
                    row, column_mapping, headers, sheet_name, file_path, idx
                )
                if doc:
                    documents.append(doc)
        
        except Exception as e:
            print(f"Error procesando hoja {sheet_name}: {str(e)}")
        
        return documents
    
    def _map_columns(self, headers: List) -> Dict[str, int]:
        """Mapea columnas a índices estándar"""
        mapping = {}
        
        for idx, header in enumerate(headers):
            if pd.isna(header):
                continue
                
            header_upper = str(header).upper().strip()
            
            if 'PUERTO' in header_upper and 'CARGA' in header_upper:
                mapping['puerto_carga'] = idx
            elif 'PAIS' in header_upper or 'PAÍS' in header_upper:
                mapping['pais'] = idx
            elif ('TON' in header_upper or 'M3' in header_upper) and 'USD' in header_upper:
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
    
    def _create_document_from_row(self, row, mapping: Dict, headers: List, 
                                  sheet_name: str, file_path: str, row_idx: int) -> Document:
        """Crea un documento a partir de una fila de datos"""
        
        # Extraer datos básicos
        puerto_carga = self._get_cell_value(row, mapping.get('puerto_carga'), headers)
        pais = self._get_cell_value(row, mapping.get('pais'), headers)
        tarifa = self._get_cell_value(row, mapping.get('tarifa'), headers)
        minimo = self._get_cell_value(row, mapping.get('minimo'), headers)
        transito = self._get_cell_value(row, mapping.get('transito'), headers)
        frecuencia = self._get_cell_value(row, mapping.get('frecuencia'), headers)
        otros = self._get_cell_value(row, mapping.get('otros'), headers)
        servicio = self._get_cell_value(row, mapping.get('servicio'), headers)
        agente = self._get_cell_value(row, mapping.get('agente'), headers)
        observaciones = self._get_cell_value(row, mapping.get('observaciones'), headers)
        
        # Validar que tiene datos mínimos
        if not puerto_carga or not pais or puerto_carga.strip() == '' or pais.strip() == '':
            return None
        
        # Normalizar datos
        tarifa_norm = self.normalize_currency_value(tarifa)
        minimo_norm = self.normalize_currency_value(minimo)
        transito_norm = self.extract_time_value(transito)
        puerto_normalizado = normalize_port_name(puerto_carga)
        
        # Determinar región destino (asumiendo que la región de la hoja es el destino)
        region_destino = sheet_name.lower()
        if region_destino == 'america':
            destino_region = 'América del Sur'
        elif region_destino == 'norteamerica':
            destino_region = 'América del Norte'
        elif region_destino == 'europa':
            destino_region = 'Europa'
        elif region_destino == 'asia':
            destino_region = 'Asia-Pacífico'
        else:
            destino_region = 'Chile'  # Default
        
        # Crear contenido del documento
        content = f"""TARIFA LCL MARÍTIMA - {sheet_name.upper()}
Archivo: {Path(file_path).name}
Registro #{row_idx + 1}

=== INFORMACIÓN DE RUTA ===
PUERTO ORIGEN: {puerto_carga}
PUERTO_NORMALIZADO: {puerto_normalizado}
PAÍS ORIGEN: {pais}
REGIÓN DESTINO: {destino_region}
RUTA: {puerto_carga} → Chile

=== TARIFAS Y COSTOS ===
TARIFA POR TON/M³: {tarifa_norm}
TARIFA MÍNIMA: {minimo_norm}
TIEMPO TRÁNSITO: {transito_norm}
FRECUENCIA: {frecuencia if frecuencia else 'No especificado'}

=== INFORMACIÓN DEL SERVICIO ===
TIPO SERVICIO: {servicio if servicio else 'No especificado'}
AGENTE LOCAL: {agente if agente else 'No especificado'}

=== COSTOS ADICIONALES ===
OTROS COSTOS: {otros if otros else 'No especificado'}

=== OBSERVACIONES ESPECIALES ===
{observaciones if observaciones else 'Sin observaciones especiales'}

=== TÉRMINOS DE BÚSQUEDA ===
lcl maritimo {puerto_normalizado} {pais.lower()} {region_destino.lower()}
tarifa costo precio {puerto_carga.lower()} chile
transporte marítimo menos contenedor completo
envío desde {puerto_carga.lower()} hacia chile
"""
        
        metadata = {
            "source": file_path,
            "source_name": Path(file_path).name,
            "sheet_name": sheet_name,
            "row_number": row_idx + 1,
            "puerto_origen": puerto_carga,
            "puerto_normalizado": puerto_normalizado,
            "pais_origen": pais,
            "region_destino": destino_region,
            "tarifa_ton_m3": tarifa_norm,
            "tarifa_minima": minimo_norm,
            "tiempo_transito": transito_norm,
            "frecuencia": frecuencia or "No especificado",
            "tipo_servicio": servicio or "No especificado",
            "agente_local": agente or "No especificado",
            "costos_adicionales": otros or "No especificado",
            "observaciones": observaciones or "Sin observaciones",
            "content_type": "lcl_rate",
            "document_type": "excel_tariff_lcl",
            "route_key": f"{puerto_normalizado}_to_chile",
            "search_terms": f"{puerto_normalizado} {pais.lower()} {region_destino.lower()}"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _get_cell_value(self, row, col_idx: int, headers: List) -> str:
        """Obtiene valor de celda de forma segura"""
        if col_idx is None or col_idx >= len(row):
            return ""
        
        value = row.iloc[col_idx] if hasattr(row, 'iloc') else row[col_idx]
        
        if pd.isna(value):
            return ""
        
        return str(value).strip()

####################################################################
#            CARGADOR DE DOCUMENTOS LCL v1.0
####################################################################

def load_lcl_excel_documents() -> List[Document]:
    """Carga y procesa archivos Excel LCL con detección multi-empresa
    
    Explicación del flujo:
    1. Busca archivos Excel en directorio temporal
    2. Para cada archivo, detecta qué empresa es (MSL vs PLUSCARGO)
    3. Usa el procesador específico según empresa detectada
    4. Combina todos los documentos en una sola base de datos
    """
    documents = []
    
    print("[DEBUG MULTI] Iniciando carga multi-empresa...")
    
    # Buscar archivos Excel
    excel_files = list(TMP_DIR.glob("**/*.xlsx")) + list(TMP_DIR.glob("**/*.xls"))
    
    print(f"[DEBUG MULTI] Archivos encontrados: {[f.name for f in excel_files]}")
    
    if not excel_files:
        print("[DEBUG MULTI] No se encontraron archivos Excel")
        return documents
    
    # Importar nuevas funciones de config
    from config import detect_company_from_excel, get_company_column_mapping
    
    for excel_file in excel_files:
        try:
            print(f"[DEBUG MULTI] === PROCESANDO: {excel_file.name} ===")
            
            # PASO 1: Detectar empresa automáticamente
            company_info = detect_company_from_excel(str(excel_file))
            company = company_info.get("company", "UNKNOWN")
            
            print(f"[DEBUG MULTI] Empresa detectada: {company}")
            print(f"[DEBUG MULTI] Info: {company_info}")
            
            # PASO 2: Procesar según empresa detectada
            if company == "MSL":
                file_docs = process_msl_excel(str(excel_file), company_info)
            elif company == "PLUSCARGO":
                file_docs = process_pluscargo_excel(str(excel_file), company_info)
            elif company == "ECU":
                file_docs = process_ecu_excel(str(excel_file), company_info)
            else:
                print(f"[DEBUG MULTI] Empresa {company} no reconocida, usando procesador genérico")
                file_docs = process_generic_excel(str(excel_file), company_info)
            
            print(f"[DEBUG MULTI] Archivo {excel_file.name} generó {len(file_docs)} documentos")
            documents.extend(file_docs)
            
        except Exception as e:
            print(f"[DEBUG MULTI] Error procesando {excel_file.name}: {str(e)}")
            continue
    
    print(f"[DEBUG MULTI] === RESUMEN MULTI-EMPRESA ===")
    print(f"[DEBUG MULTI] Total documentos: {len(documents)}")
    
    # Análisis por empresa
    companies_found = {}
    for doc in documents:
        company = doc.metadata.get('company', 'UNKNOWN')
        companies_found[company] = companies_found.get(company, 0) + 1
    
    for company, count in companies_found.items():
        print(f"[DEBUG MULTI] {company}: {count} documentos")
    
    return documents

####################################################################
#            SISTEMA DE BÚSQUEDA LCL v1.0
####################################################################

def create_lcl_retriever(vector_store, k=15):
    """Crea retriever especializado para consultas LCL"""
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

def generate_lcl_query_variations(original_query: str) -> List[str]:
    """Genera variaciones de consulta para búsqueda LCL"""
    variations = [original_query]
    query_lower = original_query.lower()
    
    # Extraer información de puertos
    route_info = extract_ports_from_query(original_query)
    
    if route_info.get('has_route'):
        origin = route_info.get('origin_normalized', '')
        destination = route_info.get('destination_normalized', '')
        
        if origin:
            variations.extend([
                f"LCL {origin}",
                f"tarifa {origin}",
                f"puerto {origin}",
                f"PUERTO_NORMALIZADO: {origin}",
                f"desde {origin}",
                origin
            ])
    
    # Agregar términos LCL específicos
    variations.extend([
        f"{query_lower} lcl",
        f"{query_lower} maritimo",
        f"{query_lower} ton m3",
        f"{query_lower} carga suelta"
    ])
    
    return list(set(variations))

def multi_query_lcl_retriever(vector_store, original_query: str) -> List[Document]:
    """Ejecuta búsqueda multi-query para LCL"""
    
    query_variations = generate_lcl_query_variations(original_query)
    all_docs = []
    seen_docs = set()
    
    retriever = create_lcl_retriever(vector_store, k=20)
    
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
#            CHAIN CONVERSACIONAL LCL v1.0
####################################################################

def create_lcl_conversational_chain(retriever):
    """Crea chain conversacional con template multi-empresa
    
    Explicación:
    - Usa el nuevo template que maneja MSL y PLUSCARGO
    - Adapta respuestas según empresas detectadas en documentos
    """
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para maximizar recuperación de tarifas LCL de múltiples empresas.

IMPORTANTE - BÚSQUEDA MULTI-EMPRESA:
- Mantener nombres exactos de puertos y países
- Incluir términos: LCL, marítimo, tarifa, puerto
- Preservar preferencias de empresa (MSL, PLUSCARGO)
- Buscar en ambas estructuras: TON/M3 (MSL) y CBM/W/M (PLUSCARGO)

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda multi-empresa:""",
    )

    # CAMBIO PRINCIPAL: Usar template multi-empresa
    from config import get_multi_company_lcl_template
    answer_prompt = ChatPromptTemplate.from_template(get_multi_company_lcl_template())
    
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
#            VALIDACIÓN Y ANÁLISIS LCL v1.0
####################################################################

def validate_lcl_response(response_text: str, original_query: str, source_docs: List) -> Dict[str, Any]:
    """Valida completitud de respuesta LCL"""
    
    validation = {
        'completeness': 1.0,
        'route_accuracy': 1.0,
        'warnings': [],
        'suggestions': []
    }
    
    query_lower = original_query.lower()
    
    # Validar presencia de información LCL obligatoria
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

def analyze_lcl_sources(sources: List[Document]) -> Dict[str, Any]:
    """Analiza fuentes de documentos LCL"""
    
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
        region = doc.metadata.get('region_destino', 'Desconocido')
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

def process_msl_excel(file_path: str, company_info: dict) -> List[Document]:
    """Procesa Excel con estructura MSL
    
    Explicación:
    - Usa la estructura MSL detectada (fila encabezados, columnas esperadas)
    - Destino siempre es Chile (implícito)
    - Procesa PUERTO CARGA, PAIS, TON/M3, etc.
    """
    documents = []
    
    try:
        xl_file = pd.ExcelFile(file_path)
        
        for sheet_name in xl_file.sheet_names:
            print(f"[DEBUG MSL] Procesando hoja: {sheet_name}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # MSL: fila encabezados varía según hoja
            header_row = 3 if sheet_name.upper() == 'AMERICA' else 4
            
            # Procesar con estructura MSL
            sheet_docs = process_msl_sheet(df, sheet_name, header_row, file_path)
            documents.extend(sheet_docs)
            
    except Exception as e:
        print(f"[DEBUG MSL] Error: {e}")
    
    return documents

def process_pluscargo_excel(file_path: str, company_info: dict) -> List[Document]:
    """Procesa Excel con estructura PLUSCARGO
    
    Explicación:
    - Usa estructura PLUSCARGO (fila 10 para encabezados)
    - Destino explícito en columna POD
    - Procesa País, POL, POD, CBM, etc.
    """
    documents = []
    
    try:
        xl_file = pd.ExcelFile(file_path)
        
        for sheet_name in xl_file.sheet_names:
            print(f"[DEBUG PLUSCARGO] Procesando hoja: {sheet_name}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # PLUSCARGO: siempre fila 10 para encabezados
            header_row = 9  # Fila 10 en Excel
            
            # Procesar con estructura PLUSCARGO
            sheet_docs = process_pluscargo_sheet(df, sheet_name, header_row, file_path)
            documents.extend(sheet_docs)
            
    except Exception as e:
        print(f"[DEBUG PLUSCARGO] Error: {e}")
    
    return documents

def process_msl_sheet(df: pd.DataFrame, sheet_name: str, header_row: int, file_path: str) -> List[Document]:
    """Procesa hoja con estructura específica MSL"""
    documents = []
    
    # Importar mapeo MSL
    from config import get_company_column_mapping
    msl_config = get_company_column_mapping("MSL")
    
    # Extraer encabezados
    headers = []
    for col in range(len(df.columns)):
        header_val = safe_get_cell(df, header_row, col)
        headers.append(header_val)
    
    print(f"[DEBUG MSL] Encabezados {sheet_name}: {headers}")
    
    # Mapear columnas MSL
    column_mapping = map_columns_flexible(headers, msl_config["column_mapping"])
    print(f"[DEBUG MSL] Mapeo: {column_mapping}")
    
    # Procesar filas de datos MSL
    for row_idx in range(header_row + 1, len(df)):
        try:
            doc = create_msl_document(df, row_idx, column_mapping, sheet_name, file_path, len(documents))
            if doc:
                documents.append(doc)
        except Exception as e:
            continue
    
    return documents

def process_pluscargo_sheet(df: pd.DataFrame, sheet_name: str, header_row: int, file_path: str) -> List[Document]:
    """Procesa hoja con estructura específica PLUSCARGO"""
    documents = []
    
    # Importar mapeo PLUSCARGO
    from config import get_company_column_mapping
    pluscargo_config = get_company_column_mapping("PLUSCARGO")
    
    # Extraer encabezados
    headers = []
    for col in range(len(df.columns)):
        header_val = safe_get_cell(df, header_row, col)
        headers.append(header_val)
    
    print(f"[DEBUG PLUSCARGO] Encabezados {sheet_name}: {headers}")
    
    # Mapear columnas PLUSCARGO
    column_mapping = map_columns_flexible(headers, pluscargo_config["column_mapping"])
    print(f"[DEBUG PLUSCARGO] Mapeo: {column_mapping}")
    
    # Procesar filas de datos PLUSCARGO
    for row_idx in range(header_row + 1, len(df)):
        try:
            doc = create_pluscargo_document(df, row_idx, column_mapping, sheet_name, file_path, len(documents))
            if doc:
                documents.append(doc)
        except Exception as e:
            continue
    
    return documents

def create_msl_document(df: pd.DataFrame, row_idx: int, mapping: dict, 
                       sheet_name: str, file_path: str, doc_count: int) -> Document:
    """Crea documento específico para MSL"""
    
    # Extraer datos con mapeo MSL
    puerto_carga = safe_get_cell(df, row_idx, mapping.get('puerto_origen', 0))
    pais = safe_get_cell(df, row_idx, mapping.get('pais', 1))
    tarifa = safe_get_cell(df, row_idx, mapping.get('tarifa', 2))
    minimo = safe_get_cell(df, row_idx, mapping.get('minimo', 3))
    transito = safe_get_cell(df, row_idx, mapping.get('transito', 4))
    frecuencia = safe_get_cell(df, row_idx, mapping.get('frecuencia', 5))
    otros = safe_get_cell(df, row_idx, mapping.get('otros', 6))
    servicio = safe_get_cell(df, row_idx, mapping.get('servicio', 7))
    agente = safe_get_cell(df, row_idx, mapping.get('agente', 8))
    observaciones = safe_get_cell(df, row_idx, mapping.get('observaciones', 9))
    
    # Validación MSL
    if not puerto_carga and not pais:
        return None
    
    puerto_carga = puerto_carga or f"Puerto en {pais}"
    pais = pais or "País no especificado"
    
    # Normalizar datos
    from config import normalize_port_name
    puerto_normalizado = normalize_port_name(puerto_carga)
    tarifa_norm = normalize_currency_value(tarifa)
    minimo_norm = normalize_currency_value(minimo)
    
    # Crear contenido MSL específico
    content = f"""TARIFA LCL MARÍTIMA - MSL (SEEMANN GROUP)
Empresa: MSL
Archivo: {Path(file_path).name}
Hoja: {sheet_name}
Registro #{doc_count + 1}

=== INFORMACIÓN MSL ===
EMPRESA: MSL (Seemann Group)
PUERTO ORIGEN: {puerto_carga}
PUERTO_NORMALIZADO: {puerto_normalizado}
PAÍS ORIGEN: {pais}
DESTINO: Chile (San Antonio/Valparaíso) - IMPLÍCITO
REGIÓN: {sheet_name}

=== TARIFAS MSL ===
TARIFA TON/M³: {tarifa_norm}
TARIFA MÍNIMA: {minimo_norm}
TIEMPO TRÁNSITO: {extract_time_value(transito)}
FRECUENCIA: {frecuencia or 'No especificado'}

=== SERVICIO MSL ===
TIPO SERVICIO: {servicio or 'No especificado'}
AGENTE LOCAL: {agente or 'No especificado'}
COSTOS ADICIONALES: {otros or 'No especificado'}
OBSERVACIONES: {observaciones or 'Sin observaciones'}

=== TÉRMINOS BÚSQUEDA ===
lcl msl seemann {puerto_normalizado} {pais.lower()} chile
"""
    
    metadata = {
        "company": "MSL",
        "source": file_path,
        "sheet_name": sheet_name,
        "puerto_origen": puerto_carga,
        "puerto_normalizado": puerto_normalizado,
        "pais_origen": pais,
        "destino": "Chile",
        "tarifa": tarifa_norm,
        "agente": agente or "No especificado",
        "content_type": "lcl_rate_msl"
    }
    
    return Document(page_content=content, metadata=metadata)

def create_pluscargo_document(df: pd.DataFrame, row_idx: int, mapping: dict,
                             sheet_name: str, file_path: str, doc_count: int) -> Document:
    """Crea documento específico para PLUSCARGO"""
    
    # Extraer datos con mapeo PLUSCARGO
    pais = safe_get_cell(df, row_idx, mapping.get('pais', 0))
    pol = safe_get_cell(df, row_idx, mapping.get('puerto_origen', 1))
    pod = safe_get_cell(df, row_idx, mapping.get('puerto_destino', 2))
    tarifa = safe_get_cell(df, row_idx, mapping.get('tarifa', 3))
    minimo = safe_get_cell(df, row_idx, mapping.get('minimo', 4))
    frecuencia = safe_get_cell(df, row_idx, mapping.get('frecuencia', 5))
    servicio = safe_get_cell(df, row_idx, mapping.get('servicio', 6))
    transito = safe_get_cell(df, row_idx, mapping.get('transito', 7))
    modo = safe_get_cell(df, row_idx, mapping.get('modo', 8))
    agente = safe_get_cell(df, row_idx, mapping.get('agente', 9))
    bl_fee = safe_get_cell(df, row_idx, mapping.get('bl_fee', 4))
    
    # Validación PLUSCARGO
    if not pol and not pais:
        return None
    
    pol = pol or f"Puerto en {pais}"
    pais = pais or "País no especificado"
    pod = pod or "San Antonio/Valparaíso"
    
    # Normalizar datos
    from config import normalize_port_name
    pol_normalizado = normalize_port_name(pol)
    tarifa_norm = normalize_currency_value(tarifa)
    
    # Crear contenido PLUSCARGO específico
    content = f"""TARIFA LCL MARÍTIMA - PLUSCARGO
Empresa: PLUSCARGO
Archivo: {Path(file_path).name}
Hoja: {sheet_name}
Registro #{doc_count + 1}

=== INFORMACIÓN PLUSCARGO ===
EMPRESA: PLUSCARGO
PUERTO ORIGEN (POL): {pol}
POL_NORMALIZADO: {pol_normalizado}
PAÍS ORIGEN: {pais}
PUERTO DESTINO (POD): {pod}
REGIÓN: {sheet_name}

=== TARIFAS PLUSCARGO ===
TARIFA CBM/W/M: {tarifa_norm}
TARIFA MÍNIMA: {normalize_currency_value(minimo)}
BL FEE: {normalize_currency_value(bl_fee) if bl_fee else 'No aplica'}
TIEMPO TRÁNSITO: {extract_time_value(transito)}
FRECUENCIA: {frecuencia or 'No especificado'}

=== SERVICIO PLUSCARGO ===
TIPO SERVICIO: {servicio or 'No especificado'}
MODO TRANSPORTE: {modo or 'No especificado'}
AGENTE LOCAL: {agente or 'No especificado'}

=== TÉRMINOS BÚSQUEDA ===
lcl pluscargo {pol_normalizado} {pais.lower()} {pod.lower()}
"""
    
    metadata = {
        "company": "PLUSCARGO",
        "source": file_path,
        "sheet_name": sheet_name,
        "puerto_origen": pol,
        "puerto_destino": pod,
        "pais_origen": pais,
        "tarifa": tarifa_norm,
        "agente": agente or "No especificado",
        "content_type": "lcl_rate_pluscargo"
    }
    
    return Document(page_content=content, metadata=metadata)

def process_ecu_excel(file_path: str, company_info: dict) -> List[Document]:
    """Procesa Excel con estructura ECU"""
    documents = []
    
    try:
        xl_file = pd.ExcelFile(file_path)
        
        for sheet_name in xl_file.sheet_names:
            print(f"[DEBUG ECU] Procesando hoja: {sheet_name}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # ECU: siempre fila 2 para encabezados
            header_row = 1  # Fila 2 en Excel
            
            # Procesar con estructura ECU
            sheet_docs = process_ecu_sheet(df, sheet_name, header_row, file_path)
            documents.extend(sheet_docs)
            
    except Exception as e:
        print(f"[DEBUG ECU] Error: {e}")
    
    return documents

def process_ecu_sheet(df: pd.DataFrame, sheet_name: str, header_row: int, file_path: str) -> List[Document]:
    """Procesa hoja con estructura específica ECU"""
    documents = []
    
    # Importar mapeo ECU
    from config import get_company_column_mapping
    ecu_config = get_company_column_mapping("ECU")
    
    # Extraer encabezados
    headers = []
    for col in range(len(df.columns)):
        header_val = safe_get_cell(df, header_row, col)
        headers.append(header_val)
    
    print(f"[DEBUG ECU] Encabezados {sheet_name}: {headers}")
    
    # Mapear columnas ECU
    column_mapping = map_columns_flexible(headers, ecu_config["column_mapping"])
    print(f"[DEBUG ECU] Mapeo: {column_mapping}")
    
    # Procesar filas de datos ECU
    for row_idx in range(header_row + 1, len(df)):
        try:
            doc = create_ecu_document(df, row_idx, column_mapping, sheet_name, file_path, len(documents))
            if doc:
                documents.append(doc)
        except Exception as e:
            continue
    
    return documents

def create_ecu_document(df: pd.DataFrame, row_idx: int, mapping: dict,
                       sheet_name: str, file_path: str, doc_count: int) -> Document:
    """Crea documento específico para ECU"""
    
    # Extraer datos con mapeo ECU
    region = safe_get_cell(df, row_idx, mapping.get('region', 0))
    pais = safe_get_cell(df, row_idx, mapping.get('pais', 1))
    first_leg_pol = safe_get_cell(df, row_idx, mapping.get('first_leg_pol', 2))
    pol = safe_get_cell(df, row_idx, mapping.get('puerto_origen', 3))
    ruta = safe_get_cell(df, row_idx, mapping.get('ruta', 4))
    pod = safe_get_cell(df, row_idx, mapping.get('puerto_destino', 5))
    servicio = safe_get_cell(df, row_idx, mapping.get('servicio', 6))
    moneda = safe_get_cell(df, row_idx, mapping.get('moneda', 7))
    tarifa = safe_get_cell(df, row_idx, mapping.get('tarifa', 8))
    bl_fee = safe_get_cell(df, row_idx, mapping.get('bl_fee', 9))
    transito = safe_get_cell(df, row_idx, mapping.get('transito', 10))
    validez = safe_get_cell(df, row_idx, mapping.get('validez', 11))
    
    # Validación ECU
    if not first_leg_pol and not pol and not pais:
        return None
    
    # Normalizar datos ECU
    puerto_origen = first_leg_pol or pol or f"Puerto en {pais}"
    pais = pais or "País no especificado"
    pod = pod or "SAI/VAP"
    
    # Normalizar tarifa con moneda específica ECU
    tarifa_norm = normalize_ecu_currency(tarifa, moneda)
    
    from config import normalize_port_name
    puerto_normalizado = normalize_port_name(puerto_origen)
    
    # Crear contenido ECU específico
    content = f"""TARIFA LCL MARÍTIMA - ECU WORLDWIDE
Empresa: ECU Worldwide
Archivo: {Path(file_path).name}
Hoja: {sheet_name}
Registro #{doc_count + 1}

=== INFORMACIÓN ECU ===
EMPRESA: ECU Worldwide
REGIÓN: {region}
PAÍS ORIGEN: {pais}
FIRST LEG POL: {first_leg_pol}
PUERTO ORIGEN (POL): {pol}
PUERTO_NORMALIZADO: {puerto_normalizado}
RUTA: {ruta}
PUERTO DESTINO (POD): {pod}

=== TARIFAS ECU ===
MONEDA: {moneda}
TARIFA TON/M³ (01-15 CBM): {tarifa_norm}
BL FEE: {normalize_ecu_currency(bl_fee, moneda) if bl_fee else 'No aplica'}
TIEMPO TRÁNSITO: {extract_time_value(transito)}
VALIDEZ: {validez or 'No especificado'}

=== SERVICIO ECU ===
TIPO SERVICIO: {servicio or 'ECU CONSOL'}
RUTA DETALLADA: {first_leg_pol} → {ruta} → {pod}

=== TÉRMINOS BÚSQUEDA ===
lcl ecu worldwide {puerto_normalizado} {pais.lower()} {pod.lower()}
"""
    
    metadata = {
        "company": "ECU",
        "source": file_path,
        "sheet_name": sheet_name,
        "region": region,
        "puerto_origen": puerto_origen,
        "puerto_destino": pod,
        "pais_origen": pais,
        "first_leg_pol": first_leg_pol,
        "pol_code": pol,
        "tarifa": tarifa_norm,
        "moneda": moneda,
        "content_type": "lcl_rate_ecu"
    }
    
    return Document(page_content=content, metadata=metadata)

def normalize_ecu_currency(value: str, currency: str) -> str:
    """Normaliza moneda específicamente para ECU con moneda explícita"""
    if not value or pd.isna(value) or str(value).strip() == '':
        return "TBD"
    
    value_str = str(value).strip()
    currency_str = str(currency).strip().upper() if currency else "USD"
    
    import re
    number_pattern = r'([\d,]+\.?\d*)'
    match = re.search(number_pattern, value_str)
    
    if match:
        number = match.group(1)
        return f"{currency_str} {number}"
    
    return f"{currency_str} {value_str}" if value_str else "TBD"

# Funciones auxiliares (agregar también):

def msl_query_retriever(vector_store, original_query: str) -> List[Document]:
    """Búsqueda optimizada para MSL cuando es explícitamente solicitado"""
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20})
    
    # Generar consultas optimizadas para MSL
    search_queries = [
        original_query,
        f"{original_query} MSL",
        f"{original_query} Seemann",
        original_query.replace('MSL', '').replace('Seemann', '').strip()
    ]
    
    all_docs = []
    seen_docs = set()
    
    for query in search_queries:
        try:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                doc_hash = hash(doc.page_content[:200])
                if doc_hash not in seen_docs:
                    all_docs.append(doc)
                    seen_docs.add(doc_hash)
        except:
            continue
    
    return all_docs[:15]

def safe_get_cell(df: pd.DataFrame, row: int, col: int) -> str:
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

def map_columns_flexible(headers: List[str], mapping_rules: dict) -> dict:
    """Mapea columnas de forma flexible"""
    mapping = {}
    
    for field, possible_names in mapping_rules.items():
        for idx, header in enumerate(headers):
            if header and any(name.upper() in header.upper() for name in possible_names):
                mapping[field] = idx
                break
    
    return mapping

def normalize_currency_value(value: str) -> str:
    """Normaliza valores de moneda con detección mejorada EUR/USD"""
    if not value or pd.isna(value) or str(value).strip() == '':
        return "TBD"
    
    value_str = str(value).strip()
    
    import re
    number_pattern = r'([\d,]+\.?\d*)'
    match = re.search(number_pattern, value_str)
    
    if match:
        number = match.group(1)
        
        # DETECCIÓN MEJORADA DE MONEDA
        value_upper = value_str.upper()
        
        # Buscar indicadores explícitos de EUR
        if any(indicator in value_upper for indicator in ['EUR', '€', 'EURO']):
            return f"EUR {number}"
        
        # Buscar indicadores explícitos de USD
        elif any(indicator in value_upper for indicator in ['USD', '$', 'DOLLAR']):
            return f"USD {number}"
        
        # Si no hay indicador claro, mantener valor original SIN asumir moneda
        else:
            return f"{number} (moneda no especificada)"
    
    return value_str if value_str else "TBD"

def extract_time_value(time_str: str) -> str:
    """Extrae valor de tiempo de tránsito"""
    if not time_str or pd.isna(time_str) or str(time_str).strip() == '':
        return "No especificado"
    
    time_str = str(time_str).strip()
    
    import re
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

def process_generic_excel(file_path: str, company_info: dict) -> List[Document]:
    """Procesador genérico para empresas no reconocidas"""
    documents = []
    
    try:
        xl_file = pd.ExcelFile(file_path)
        
        for sheet_name in xl_file.sheet_names:
            print(f"[DEBUG GENERIC] Procesando hoja: {sheet_name}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # Buscar encabezados dinámicamente
            header_row = find_header_row_generic(df)
            
            if header_row >= 0:
                sheet_docs = process_generic_sheet(df, sheet_name, header_row, file_path)
                documents.extend(sheet_docs)
                
    except Exception as e:
        print(f"[DEBUG GENERIC] Error: {e}")
    
    return documents

def find_header_row_generic(df: pd.DataFrame) -> int:
    """Encuentra fila de encabezados de forma genérica"""
    for row in range(min(15, len(df))):
        row_content = ' '.join([str(cell) for cell in df.iloc[row] if pd.notna(cell)])
        
        # Buscar patrones típicos
        patterns = ['PUERTO', 'POL', 'PAIS', 'TARIFA', 'USD', 'ORIGEN']
        matches = sum(1 for pattern in patterns if pattern.upper() in row_content.upper())
        
        if matches >= 2:
            return row
    
    return -1

def process_generic_sheet(df: pd.DataFrame, sheet_name: str, header_row: int, file_path: str) -> List[Document]:
    """Procesa hoja con estructura genérica"""
    documents = []
    
    # Usar mapeo genérico
    from config import get_company_column_mapping
    generic_config = get_company_column_mapping("GENERIC")
    
    headers = []
    for col in range(len(df.columns)):
        header_val = safe_get_cell(df, header_row, col)
        headers.append(header_val)
    
    column_mapping = map_columns_flexible(headers, generic_config["column_mapping"])
    
    for row_idx in range(header_row + 1, len(df)):
        try:
            doc = create_generic_document(df, row_idx, column_mapping, sheet_name, file_path, len(documents))
            if doc:
                documents.append(doc)
        except Exception as e:
            continue
    
    return documents

def create_generic_document(df: pd.DataFrame, row_idx: int, mapping: dict,
                           sheet_name: str, file_path: str, doc_count: int) -> Document:
    """Crea documento genérico"""
    
    puerto_origen = safe_get_cell(df, row_idx, mapping.get('puerto_origen', 0))
    pais = safe_get_cell(df, row_idx, mapping.get('pais', 0))
    tarifa = safe_get_cell(df, row_idx, mapping.get('tarifa', 1))
    frecuencia = safe_get_cell(df, row_idx, mapping.get('frecuencia', 2))
    servicio = safe_get_cell(df, row_idx, mapping.get('servicio', 3))
    
    if not puerto_origen and not pais:
        return None
    
    content = f"""TARIFA LCL MARÍTIMA - EMPRESA DESCONOCIDA
Archivo: {Path(file_path).name}
Hoja: {sheet_name}
Registro #{doc_count + 1}

=== INFORMACIÓN BÁSICA ===
PUERTO ORIGEN: {puerto_origen or 'No especificado'}
PAÍS: {pais or 'No especificado'}
TARIFA: {normalize_currency_value(tarifa)}
FRECUENCIA: {frecuencia or 'No especificado'}
SERVICIO: {servicio or 'No especificado'}

=== NOTA ===
Empresa no identificada automáticamente.
Estructura procesada de forma genérica.
"""
    
    metadata = {
        "company": "UNKNOWN",
        "source": file_path,
        "sheet_name": sheet_name,
        "puerto_origen": puerto_origen or pais,
        "content_type": "lcl_rate_generic"
    }
    
    return Document(page_content=content, metadata=metadata)