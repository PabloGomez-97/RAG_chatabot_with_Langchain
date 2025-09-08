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
    """Carga y procesa archivos Excel de tarifas LCL"""
    documents = []
    processor = LCLExcelProcessor()
    
    # Buscar archivos Excel en directorio temporal
    excel_files = list(TMP_DIR.glob("**/*.xlsx")) + list(TMP_DIR.glob("**/*.xls"))
    
    for excel_file in excel_files:
        try:
            # Leer todas las hojas del archivo
            xl_file = pd.ExcelFile(excel_file)
            
            for sheet_name in xl_file.sheet_names:
                print(f"Procesando hoja: {sheet_name} de {excel_file.name}")
                
                sheet_docs = processor.process_excel_sheet(
                    str(excel_file), sheet_name
                )
                documents.extend(sheet_docs)
                
        except Exception as e:
            print(f"Error procesando {excel_file.name}: {str(e)}")
            continue
    
    print(f"Total documentos LCL procesados: {len(documents)}")
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
    """Crea chain conversacional especializada para LCL"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para maximizar recuperación de tarifas LCL marítimas.

IMPORTANTE - BÚSQUEDA LCL:
- Mantener nombres exactos de puertos y países
- Incluir variaciones de puertos (Shanghai, Ningbo, etc.)
- Agregar términos: LCL, marítimo, tarifa, puerto
- Preservar información de origen y destino

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda LCL:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(get_lcl_response_template())
    
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