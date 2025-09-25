import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Any, Set
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from backend.aereo.config import (
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, OPENAI_API_KEY,
    get_air_freight_response_template, detect_air_freight_query_type, 
    extract_airports_from_query, get_airport_region, analyze_route_direction
)

####################################################################
#            INSPECTOR EXCEL CARGA AÉREA - CRAFTTRANSWAY
####################################################################

class AirFreightExcelInspector:
    """Inspector especializado para archivos Excel de carga aérea"""
    
    def __init__(self):
        self.verified_routes = set()
        self.all_airports_aol = set()
        self.all_airports_aod = set()
        self.companies = set()
        self.airlines = set()
        self.raw_data = []
        
    def inspect_air_freight_excel(self, file_path: str) -> Dict[str, Any]:
        """Inspección completa del Excel de carga aérea"""
        print(f"\n[AIR FREIGHT] === INSPECCIÓN DE {Path(file_path).name} ===")
        
        inspection_result = {
            'file_path': file_path,
            'sheets_found': [],
            'total_routes': 0,
            'aol_airports': set(),
            'aod_airports': set(),
            'companies': set(),
            'airlines': set(),
            'currencies': set(),
            'routes_data': [],
            'inspection_log': []
        }
        
        try:
            xl_file = pd.ExcelFile(file_path)
            inspection_result['sheets_found'] = xl_file.sheet_names
            
            print(f"[AIR FREIGHT] Hojas encontradas: {xl_file.sheet_names}")
            
            for sheet_name in xl_file.sheet_names:
                sheet_data = self._inspect_air_freight_sheet(file_path, sheet_name)
                inspection_result['total_routes'] += sheet_data['routes_count']
                inspection_result['aol_airports'].update(sheet_data['aol_airports'])
                inspection_result['aod_airports'].update(sheet_data['aod_airports'])
                inspection_result['companies'].update(sheet_data['companies'])
                inspection_result['airlines'].update(sheet_data['airlines'])
                inspection_result['currencies'].update(sheet_data['currencies'])
                inspection_result['routes_data'].extend(sheet_data['routes_data'])
                
        except Exception as e:
            error_msg = f"[AIR FREIGHT] Error inspeccionando archivo: {str(e)}"
            print(error_msg)
            inspection_result['inspection_log'].append(error_msg)
        
        # Resumen final
        print(f"\n[AIR FREIGHT] === RESUMEN DE INSPECCIÓN ===")
        print(f"[AIR FREIGHT] Total rutas encontradas: {inspection_result['total_routes']}")
        print(f"[AIR FREIGHT] Aeropuertos AOL: {sorted(list(inspection_result['aol_airports']))}")
        print(f"[AIR FREIGHT] Aeropuertos AOD: {sorted(list(inspection_result['aod_airports']))}")
        print(f"[AIR FREIGHT] Compañías: {sorted(list(inspection_result['companies']))}")
        
        return inspection_result
    
    def _inspect_air_freight_sheet(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """Inspección de una hoja específica"""
        print(f"\n[AIR FREIGHT] --- Inspeccionando hoja: {sheet_name} ---")
        
        sheet_result = {
            'sheet_name': sheet_name,
            'routes_count': 0,
            'aol_airports': set(),
            'aod_airports': set(),
            'companies': set(),
            'airlines': set(),
            'currencies': set(),
            'routes_data': []
        }
        
        try:
            # Leer la hoja completa
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                print(f"[AIR FREIGHT] Hoja {sheet_name} está vacía")
                return sheet_result
            
            print(f"[AIR FREIGHT] Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
            print(f"[AIR FREIGHT] Columnas: {list(df.columns)}")
            
            # Mapear columnas basado en los encabezados conocidos
            column_mapping = self._map_air_freight_columns(df.columns)
            print(f"[AIR FREIGHT] Mapeo de columnas: {column_mapping}")
            
            # Procesar cada fila de datos
            for index, row in df.iterrows():
                route_data = self._process_air_freight_row(row, column_mapping, index + 1, sheet_name)
                
                if route_data['is_valid']:
                    sheet_result['routes_count'] += 1
                    sheet_result['aol_airports'].add(route_data['aol'])
                    sheet_result['aod_airports'].add(route_data['aod'])
                    sheet_result['companies'].add(route_data['company'])
                    if route_data['airline']:
                        sheet_result['airlines'].add(route_data['airline'])
                    
                    # Extraer monedas
                    if route_data['min_currency']:
                        sheet_result['currencies'].add(route_data['min_currency'])
                    if route_data['flat_currency']:
                        sheet_result['currencies'].add(route_data['flat_currency'])
                    
                    sheet_result['routes_data'].append(route_data)
            
            print(f"[AIR FREIGHT] Rutas válidas en {sheet_name}: {sheet_result['routes_count']}")
            
        except Exception as e:
            print(f"[AIR FREIGHT] Error inspeccionando hoja {sheet_name}: {str(e)}")
        
        return sheet_result
    
    def _map_air_freight_columns(self, columns: List[str]) -> Dict[str, str]:
        """Mapea las columnas del Excel basado en los encabezados conocidos"""
        mapping = {}
        
        for col in columns:
            col_upper = str(col).upper().strip()
            
            if col_upper == 'AOL':
                mapping['aol'] = col
            elif col_upper == 'PAIS ORIGEN':
                mapping['pais_origen'] = col
            elif col_upper == 'AOD':
                mapping['aod'] = col
            elif col_upper == 'PAIS DESTINO':
                mapping['pais_destino'] = col
            elif col_upper == 'SERVICIO/AIRLINE':
                mapping['airline'] = col
            elif col_upper == 'MIN':
                mapping['min'] = col
            elif col_upper == 'FLAT/KG':
                mapping['flat_kg'] = col
            elif col_upper == 'SALIDAS':
                mapping['salidas'] = col
            elif col_upper == 'OTROS':
                mapping['otros'] = col
            elif col_upper == 'COMPANY':
                mapping['company'] = col
        
        return mapping
    
    def _process_air_freight_row(self, row: pd.Series, column_mapping: Dict[str, str], 
                                row_number: int, sheet_name: str) -> Dict[str, Any]:
        """Procesa una fila individual del Excel"""
        
        route_data = {
            'is_valid': False,
            'row_number': row_number,
            'sheet_name': sheet_name,
            'aol': '',
            'pais_origen': '',
            'aod': '',
            'pais_destino': '',
            'airline': '',
            'min': '',
            'flat_kg': '',
            'salidas': '',
            'otros': '',
            'company': '',
            'min_currency': '',
            'flat_currency': '',
            'route_key': ''
        }
        
        try:
            # Extraer datos usando el mapeo
            aol = self._safe_get_value(row, column_mapping.get('aol', ''))
            pais_origen = self._safe_get_value(row, column_mapping.get('pais_origen', ''))
            aod = self._safe_get_value(row, column_mapping.get('aod', ''))
            pais_destino = self._safe_get_value(row, column_mapping.get('pais_destino', ''))
            airline = self._safe_get_value(row, column_mapping.get('airline', ''))
            min_value = self._safe_get_value(row, column_mapping.get('min', ''))
            flat_kg = self._safe_get_value(row, column_mapping.get('flat_kg', ''))
            salidas = self._safe_get_value(row, column_mapping.get('salidas', ''))
            otros = self._safe_get_value(row, column_mapping.get('otros', ''))
            company = self._safe_get_value(row, column_mapping.get('company', ''))
            
            # Validar que tiene datos mínimos necesarios
            if aol and aod and pais_origen and pais_destino:
                route_data['is_valid'] = True
                route_data['aol'] = aol
                route_data['pais_origen'] = pais_origen
                route_data['aod'] = aod
                route_data['pais_destino'] = pais_destino
                route_data['airline'] = airline
                route_data['min'] = min_value
                route_data['flat_kg'] = flat_kg
                route_data['salidas'] = salidas
                route_data['otros'] = otros
                route_data['company'] = company
                route_data['route_key'] = f"{aol}_{aod}"
                
                # Extraer monedas
                route_data['min_currency'] = self._extract_currency(min_value)
                route_data['flat_currency'] = self._extract_currency(flat_kg)
                
        except Exception as e:
            print(f"[AIR FREIGHT] Error procesando fila {row_number}: {str(e)}")
        
        return route_data
    
    def _safe_get_value(self, row: pd.Series, column: str) -> str:
        """Obtiene valor de forma segura"""
        if not column or column not in row:
            return ""
        
        value = row[column]
        if pd.isna(value) or value is None:
            return ""
        
        return str(value).strip()
    
    def _extract_currency(self, value: str) -> str:
        """Extrae la moneda de un valor monetario"""
        if not value:
            return ""
        
        value_upper = str(value).upper()
        
        if 'USD' in value_upper:
            return 'USD'
        elif 'EUR' in value_upper:
            return 'EUR'
        elif 'GBP' in value_upper:
            return 'GBP'
        else:
            return ""

####################################################################
#            PROCESADOR DE CARGA AÉREA
####################################################################

class AirFreightProcessor:
    """Procesador que crea documentos de rutas aéreas verificadas"""
    
    def __init__(self):
        self.inspector = AirFreightExcelInspector()
        
    def process_air_freight_excel(self, file_path: str) -> List[Document]:
        """Procesa Excel de carga aérea y crea documentos"""
        
        print(f"\n[PROCESSOR] === PROCESANDO CARGA AÉREA: {file_path} ===")
        
        # Inspección completa
        inspection = self.inspector.inspect_air_freight_excel(file_path)
        
        if inspection['total_routes'] == 0:
            print(f"[PROCESSOR] No se encontraron rutas válidas en {Path(file_path).name}")
            return []
        
        print(f"[PROCESSOR] {inspection['total_routes']} rutas encontradas")
        
        # Crear documentos
        documents = []
        
        for route_data in inspection['routes_data']:
            doc = self._create_air_freight_document(route_data, file_path)
            if doc:
                documents.append(doc)
        
        print(f"[PROCESSOR] {len(documents)} documentos creados")
        
        return documents
    
    def _create_air_freight_document(self, route_data: Dict, file_path: str) -> Document:
        """Crea documento de una ruta aérea"""
        
        # Información básica
        aol = route_data['aol']
        aod = route_data['aod']
        pais_origen = route_data['pais_origen']
        pais_destino = route_data['pais_destino']
        company = route_data['company']
        
        # Determinar tipo de operación
        operation_type = analyze_route_direction(aol, aod)
        aol_region = get_airport_region(aol)
        aod_region = get_airport_region(aod)
        
        # Formatear tarifas
        min_formatted = self._format_tariff(route_data['min'])
        flat_formatted = self._format_tariff(route_data['flat_kg'])
        
        # Crear contenido del documento
        content = f"""TARIFA CARGA AÉREA CRAFTTRANSWAY - RUTA VERIFICADA
Archivo: {Path(file_path).name}
Fila: {route_data['row_number']}
Estado: VERIFICADO ✅

=== INFORMACIÓN DE RUTA ===
AEROPUERTO ORIGEN (AOL): {aol}
PAÍS ORIGEN: {pais_origen}
REGIÓN ORIGEN: {aol_region}
AEROPUERTO DESTINO (AOD): {aod}
PAÍS DESTINO: {pais_destino}
REGIÓN DESTINO: {aod_region}
TIPO OPERACIÓN: {operation_type}
COMPAÑÍA: {company}

=== TARIFAS VERIFICADAS ===
MÍNIMO POR ENVÍO: {min_formatted}
TARIFA POR KG: {flat_formatted}
MONEDA MÍNIMO: {route_data['min_currency']}
MONEDA FLAT/KG: {route_data['flat_currency']}

=== SERVICIO ===
AIRLINE/CÓDIGO: {route_data['airline'] or 'No especificado'}
FRECUENCIA SALIDAS: {route_data['salidas'] or 'No especificado'}
INFORMACIÓN ADICIONAL: {route_data['otros'] or 'Sin información adicional'}

=== RUTA DETALLADA ===
🛫 {aol} ({pais_origen}) → 🛬 {aod} ({pais_destino})
💼 Operadora: {company}
✈️ Airline: {route_data['airline'] or 'Sin especificar'}
💰 Desde {min_formatted} mínimo
📊 {flat_formatted} por kilogramo

=== VERIFICACIÓN ===
✅ Ruta confirmada en tarifario CRAFTTRANSWAY
✅ Datos extraídos directamente del Excel
✅ Aeropuerto origen verificado: {aol}
✅ Aeropuerto destino verificado: {aod}
✅ Compañía verificada: {company}
"""
        
        metadata = {
            "source": file_path,
            "aol": aol,
            "aod": aod,
            "pais_origen": pais_origen,
            "pais_destino": pais_destino,
            "company": company,
            "airline": route_data['airline'],
            "row_number": route_data['row_number'],
            "route_key": route_data['route_key'],
            "operation_type": operation_type,
            "aol_region": aol_region,
            "aod_region": aod_region,
            "min_currency": route_data['min_currency'],
            "flat_currency": route_data['flat_currency'],
            "verification_status": "VERIFIED",
            "content_type": "air_freight_route"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _format_tariff(self, value: str) -> str:
        """Formatea una tarifa para mostrar"""
        if not value or str(value).strip() == '':
            return "No disponible"
        
        return str(value).strip()

####################################################################
#            FUNCIÓN DE CARGA DE DOCUMENTOS
####################################################################

def load_air_freight_documents() -> List[Document]:
    """Carga documentos de carga aérea"""
    
    print("\n[AIR FREIGHT] === CARGA DE DOCUMENTOS CARGA AÉREA ===")
    
    # Buscar archivos Excel
    excel_files = list(TMP_DIR.glob("**/*.xlsx")) + list(TMP_DIR.glob("**/*.xls"))
    
    if not excel_files:
        print("[AIR FREIGHT] No se encontraron archivos Excel")
        return []
    
    print(f"[AIR FREIGHT] Archivos encontrados: {[f.name for f in excel_files]}")
    
    processor = AirFreightProcessor()
    all_documents = []
    
    for excel_file in excel_files:
        try:
            print(f"\n[AIR FREIGHT] === PROCESANDO: {excel_file.name} ===")
            file_docs = processor.process_air_freight_excel(str(excel_file))
            all_documents.extend(file_docs)
            
        except Exception as e:
            print(f"[AIR FREIGHT] Error procesando {excel_file.name}: {str(e)}")
            continue
    
    print(f"\n[AIR FREIGHT] === RESUMEN FINAL ===")
    print(f"[AIR FREIGHT] Total documentos: {len(all_documents)}")
    
    # Analizar rutas únicas
    unique_routes = set()
    aol_airports = set()
    aod_airports = set()
    companies = set()
    
    for doc in all_documents:
        route_key = doc.metadata.get('route_key', '')
        if route_key:
            unique_routes.add(route_key)
        
        aol = doc.metadata.get('aol', '')
        if aol:
            aol_airports.add(aol)
            
        aod = doc.metadata.get('aod', '')
        if aod:
            aod_airports.add(aod)
            
        company = doc.metadata.get('company', '')
        if company:
            companies.add(company)
    
    print(f"[AIR FREIGHT] Rutas únicas: {len(unique_routes)}")
    print(f"[AIR FREIGHT] Aeropuertos AOL: {sorted(list(aol_airports))}")
    print(f"[AIR FREIGHT] Aeropuertos AOD: {sorted(list(aod_airports))}")
    print(f"[AIR FREIGHT] Compañías: {sorted(list(companies))}")
    
    return all_documents

####################################################################
#            RETRIEVER Y CHAIN
####################################################################

def create_air_freight_retriever(vector_store, k=20):
    """Crea retriever para documentos de carga aérea"""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        }
    )

def create_air_freight_chain(retriever):
    """Crea chain conversacional para carga aérea"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para buscar información específica de tarifas de carga aérea.

ENFOQUE CARGA AÉREA:
- Buscar rutas específicas entre aeropuertos (AOL → AOD)
- Identificar códigos de aeropuertos si están presentes
- Verificar disponibilidad de rutas
- Buscar información de tarifas, airlines y servicios

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(get_air_freight_response_template())
    
    doc_prompt = PromptTemplate(
        input_variables=["page_content", "aol", "aod", "company", "airline", "row_number"],
        template=(
            "{page_content}\n\n"
            "[METADATOS]\n"
            "AOL: {aol}\n"
            "AOD: {aod}\n"
            "Company: {company}\n"
            "Airline: {airline}\n"
            "Fila: {row_number}\n"
        )
    )
    
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
        temperature=0.0,
        model_kwargs={"top_p": 0.9}
    )

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt, "document_prompt": doc_prompt},
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
#            VALIDACIÓN Y ANÁLISIS
####################################################################

def validate_air_freight_route(query: str, documents: List[Document]) -> Dict[str, Any]:
    """Valida que la ruta aérea consultada existe"""
    
    validation = {
        'route_exists': False,
        'airports_requested': [],
        'aol_requested': None,
        'aod_requested': None,
        'available_routes': [],
        'suggestions': [],
        'verification_status': 'NOT_FOUND'
    }
    
    # Extraer aeropuertos de la consulta
    airport_info = extract_airports_from_query(query)
    validation['airports_requested'] = airport_info['airports_found']
    validation['aol_requested'] = airport_info['aol_detected']
    validation['aod_requested'] = airport_info['aod_detected']
    
    if not airport_info['needs_verification']:
        validation['verification_status'] = 'NO_SPECIFIC_ROUTE'
        return validation
    
    # Verificar en documentos
    for doc in documents:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            aol = doc.metadata.get('aol', '')
            aod = doc.metadata.get('aod', '')
            route_key = f"{aol}_{aod}"
            
            validation['available_routes'].append(route_key)
            
            # Verificar coincidencia exacta de ruta
            if airport_info['has_route_pattern']:
                if (aol == airport_info['aol_detected'] and 
                    aod == airport_info['aod_detected']):
                    validation['route_exists'] = True
                    validation['verification_status'] = 'EXACT_ROUTE_FOUND'
            
            # Verificar aeropuertos individuales
            elif airport_info['airports_found']:
                for airport in airport_info['airports_found']:
                    if airport in [aol, aod]:
                        validation['route_exists'] = True
                        validation['verification_status'] = 'AIRPORT_FOUND'
    
    # Generar sugerencias si no se encuentra la ruta exacta
    if not validation['route_exists'] and validation['available_routes']:
        # Buscar rutas similares
        if airport_info['aol_detected']:
            similar_routes = [route for route in validation['available_routes'] 
                            if route.startswith(airport_info['aol_detected'])]
            validation['suggestions'].extend(similar_routes[:5])
        
        if airport_info['aod_detected']:
            similar_routes = [route for route in validation['available_routes'] 
                            if route.endswith(airport_info['aod_detected'])]
            validation['suggestions'].extend(similar_routes[:5])
    
    return validation

def analyze_air_freight_sources(sources: List[Document]) -> Dict[str, Any]:
    """Analiza fuentes de carga aérea"""
    
    analysis = {
        "total_routes": 0,
        "aol_airports": set(),
        "aod_airports": set(),
        "companies": set(),
        "airlines": set(),
        "currencies": set(),
        "regions_aol": {},
        "regions_aod": {},
        "operation_types": {}
    }
    
    for doc in sources:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            analysis['total_routes'] += 1
            
            # Aeropuertos
            aol = doc.metadata.get('aol', '')
            aod = doc.metadata.get('aod', '')
            if aol:
                analysis['aol_airports'].add(aol)
                # Contar por región AOL
                region = doc.metadata.get('aol_region', '')
                analysis['regions_aol'][region] = analysis['regions_aol'].get(region, 0) + 1
            
            if aod:
                analysis['aod_airports'].add(aod)
                # Contar por región AOD
                region = doc.metadata.get('aod_region', '')
                analysis['regions_aod'][region] = analysis['regions_aod'].get(region, 0) + 1
            
            # Compañías
            company = doc.metadata.get('company', '')
            if company:
                analysis['companies'].add(company)
            
            # Airlines
            airline = doc.metadata.get('airline', '')
            if airline:
                analysis['airlines'].add(airline)
            
            # Monedas
            min_currency = doc.metadata.get('min_currency', '')
            flat_currency = doc.metadata.get('flat_currency', '')
            if min_currency:
                analysis['currencies'].add(min_currency)
            if flat_currency:
                analysis['currencies'].add(flat_currency)
            
            # Tipos de operación
            op_type = doc.metadata.get('operation_type', '')
            if op_type:
                analysis['operation_types'][op_type] = analysis['operation_types'].get(op_type, 0) + 1
    
    return analysis