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

from config import (
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, OPENAI_API_KEY,
    get_msl_response_template, detect_msl_query_type, extract_port_for_verification
)

####################################################################
#            INSPECTOR EXCEL MSL - VERIFICACIÓN TOTAL
####################################################################

class MSLExcelInspector:
    """Inspector que verifica cada celda del Excel MSL antes de procesar"""
    
    def __init__(self):
        self.verified_routes = set()  # Rutas que realmente existen
        self.all_ports = set()        # Puertos que realmente existen
        self.raw_data = []           # Datos crudos para verificación
        
    def full_excel_inspection(self, file_path: str) -> Dict[str, Any]:
        """Inspección completa del Excel MSL celda por celda"""
        print(f"\n[INSPECTOR] === INSPECCIÓN TOTAL DE {Path(file_path).name} ===")
        
        inspection_result = {
            'file_path': file_path,
            'sheets_found': [],
            'total_data_rows': 0,
            'verified_routes': set(),
            'all_ports': set(),
            'regions': {},
            'inspection_log': []
        }
        
        try:
            xl_file = pd.ExcelFile(file_path)
            inspection_result['sheets_found'] = xl_file.sheet_names
            
            print(f"[INSPECTOR] Hojas encontradas: {xl_file.sheet_names}")
            
            for sheet_name in xl_file.sheet_names:
                sheet_inspection = self._inspect_sheet_completely(file_path, sheet_name)
                inspection_result['regions'][sheet_name] = sheet_inspection
                inspection_result['total_data_rows'] += sheet_inspection['data_rows_count']
                inspection_result['verified_routes'].update(sheet_inspection['verified_routes'])
                inspection_result['all_ports'].update(sheet_inspection['ports_found'])
                
        except Exception as e:
            error_msg = f"[INSPECTOR] Error inspeccionando archivo: {str(e)}"
            print(error_msg)
            inspection_result['inspection_log'].append(error_msg)
        
        # Resumen final de inspección
        print(f"\n[INSPECTOR] === RESUMEN DE INSPECCIÓN ===")
        print(f"[INSPECTOR] Total rutas verificadas: {len(inspection_result['verified_routes'])}")
        print(f"[INSPECTOR] Total puertos encontrados: {len(inspection_result['all_ports'])}")
        print(f"[INSPECTOR] Puertos disponibles: {sorted(list(inspection_result['all_ports']))}")
        
        return inspection_result
    
    def _inspect_sheet_completely(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """Inspección completa de una hoja"""
        print(f"\n[INSPECTOR] --- Inspeccionando hoja: {sheet_name} ---")
        
        sheet_result = {
            'sheet_name': sheet_name,
            'header_row_found': -1,
            'data_rows_count': 0,
            'verified_routes': set(),
            'ports_found': set(),
            'countries_found': set(),
            'columns_detected': {},
            'raw_entries': []
        }
        
        try:
            # Leer toda la hoja sin asumir estructura
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            if df.empty:
                print(f"[INSPECTOR] Hoja {sheet_name} está vacía")
                return sheet_result
            
            print(f"[INSPECTOR] Dimensiones de {sheet_name}: {len(df)} filas x {len(df.columns)} columnas")
            
            # PASO 1: Encontrar fila de encabezados inspeccionando celda por celda
            header_row = self._find_header_row_by_inspection(df, sheet_name)
            sheet_result['header_row_found'] = header_row
            
            if header_row == -1:
                print(f"[INSPECTOR] No se encontró fila de encabezados válida en {sheet_name}")
                return sheet_result
            
            # PASO 2: Extraer y mapear columnas reales
            headers = self._extract_real_headers(df, header_row)
            column_mapping = self._map_columns_by_content(headers)
            sheet_result['columns_detected'] = column_mapping
            
            print(f"[INSPECTOR] Encabezados detectados: {[h for h in headers if h and str(h).strip()]}")
            print(f"[INSPECTOR] Mapeo de columnas: {column_mapping}")
            
            # PASO 3: Verificar cada fila de datos
            for row_idx in range(header_row + 1, len(df)):
                row_verification = self._verify_row_completely(df, row_idx, column_mapping, sheet_name)
                
                if row_verification['is_valid_route']:
                    sheet_result['verified_routes'].add(row_verification['route_key'])
                    sheet_result['ports_found'].add(row_verification['port_origin'])
                    sheet_result['countries_found'].add(row_verification['country'])
                    sheet_result['raw_entries'].append(row_verification)
                    sheet_result['data_rows_count'] += 1
            
            print(f"[INSPECTOR] Hoja {sheet_name}: {sheet_result['data_rows_count']} rutas válidas verificadas")
            print(f"[INSPECTOR] Puertos en {sheet_name}: {sorted(list(sheet_result['ports_found']))}")
            
        except Exception as e:
            print(f"[INSPECTOR] Error inspeccionando hoja {sheet_name}: {str(e)}")
        
        return sheet_result
    
    def _find_header_row_by_inspection(self, df: pd.DataFrame, sheet_name: str) -> int:
        """Encuentra fila de encabezados inspeccionando contenido real"""
        
        # Palabras clave que indican fila de encabezados MSL
        header_keywords = ['POL', 'PAIS', 'PAÍS', 'PUERTO', 'TON', 'M3', 'MINIMO', 'MÍNIMO', 'FREC', 'SERVICIO', 'AGENTE', "COMPANY"]
        
        for row_idx in range(min(15, len(df))):
            row_content = []
            keyword_matches = 0
            
            # Inspeccionar cada celda de la fila
            for col_idx in range(len(df.columns)):
                cell_value = self._get_safe_cell_value(df, row_idx, col_idx)
                if cell_value:
                    row_content.append(cell_value)
                    # Contar coincidencias con palabras clave
                    for keyword in header_keywords:
                        if keyword.upper() in str(cell_value).upper():
                            keyword_matches += 1
            
            print(f"[INSPECTOR] Fila {row_idx + 1}: {keyword_matches} keywords, contenido: {row_content[:5]}")
            
            # Si encontramos suficientes palabras clave, es la fila de encabezados
            if keyword_matches >= 3:
                print(f"[INSPECTOR] Fila de encabezados detectada en: {row_idx + 1}")
                return row_idx
        
        print(f"[INSPECTOR] No se detectó fila de encabezados válida")
        return -1
    
    def _extract_real_headers(self, df: pd.DataFrame, header_row: int) -> List[str]:
        """Extrae encabezados reales de la fila detectada"""
        headers = []
        
        for col_idx in range(len(df.columns)):
            header_value = self._get_safe_cell_value(df, header_row, col_idx)
            headers.append(header_value if header_value else "")
        
        return headers
    
    def _map_columns_by_content(self, headers: List[str]) -> Dict[str, int]:
        """Mapea columnas basándose en contenido real de encabezados"""
        mapping = {}
        
        for idx, header in enumerate(headers):
            if not header:
                continue
                
            header_upper = str(header).upper().strip()
            
            # Mapeo específico basado en inspección real
            if 'POL' in header_upper or ('PUERTO' in header_upper and 'CARGA' in header_upper):
                mapping['puerto_origen'] = idx
            elif 'PAIS' in header_upper or 'PAÍS' in header_upper:
                mapping['pais'] = idx
            elif 'TON' in header_upper and 'M3' in header_upper:
                mapping['tarifa'] = idx
            elif 'MINIMO' in header_upper or 'MÍNIMO' in header_upper:
                mapping['minimo'] = idx
            elif 'T / T' in header_upper or ('APROX' in header_upper and 'T' in header_upper):
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
            elif 'COMPANY' in header_upper:
                mapping['company'] = idx
        
        return mapping
    
    def _verify_row_completely(self, df: pd.DataFrame, row_idx: int, 
                               column_mapping: Dict[str, int], sheet_name: str) -> Dict[str, Any]:
        """Verifica completamente una fila de datos"""
        
        verification = {
            'row_index': row_idx,
            'is_valid_route': False,
            'port_origin': '',
            'country': '',
            'route_key': '',
            'raw_data': {},
            'verification_log': []
        }
        
        try:
            # Extraer datos de la fila usando mapeo real
            puerto_origen = self._get_safe_cell_value(df, row_idx, column_mapping.get('puerto_origen', 0))
            pais = self._get_safe_cell_value(df, row_idx, column_mapping.get('pais', 1))
            tarifa = self._get_safe_cell_value(df, row_idx, column_mapping.get('tarifa', 2))
            minimo = self._get_safe_cell_value(df, row_idx, column_mapping.get('minimo', 3))
            transito = self._get_safe_cell_value(df, row_idx, column_mapping.get('transito', 4))
            frecuencia = self._get_safe_cell_value(df, row_idx, column_mapping.get('frecuencia', 5))
            otros = self._get_safe_cell_value(df, row_idx, column_mapping.get('otros', 6))
            servicio = self._get_safe_cell_value(df, row_idx, column_mapping.get('servicio', 7))
            agente = self._get_safe_cell_value(df, row_idx, column_mapping.get('agente', 8))
            observaciones = self._get_safe_cell_value(df, row_idx, column_mapping.get('observaciones', 9))
            company = self._get_safe_cell_value(df, row_idx, column_mapping.get('company', -1))
            
            # VERIFICACIÓN ESTRICTA: Solo acepta filas con datos mínimos válidos
            if puerto_origen and pais and puerto_origen.strip() and pais.strip():
                # Es una ruta válida
                verification['is_valid_route'] = True
                verification['port_origin'] = puerto_origen.strip()
                verification['country'] = pais.strip()
                verification['route_key'] = f"{puerto_origen.strip().lower()}_to_chile"
                
                verification['raw_data'] = {
                    'puerto_origen': puerto_origen,
                    'pais': pais,
                    'tarifa': tarifa,
                    'minimo': minimo,
                    'transito': transito,
                    'frecuencia': frecuencia,
                    'otros': otros,
                    'servicio': servicio,
                    'agente': agente,
                    'observaciones': observaciones,
                    'company': company,
                    'sheet_name': sheet_name,
                    'row_number': row_idx + 1
                }
                
                verification['verification_log'].append(f"✅ Ruta válida: {puerto_origen} ({pais}) → Chile")
            else:
                verification['verification_log'].append(f"❌ Fila inválida: puerto='{puerto_origen}', país='{pais}'")
                
        except Exception as e:
            verification['verification_log'].append(f"❌ Error verificando fila {row_idx + 1}: {str(e)}")
        
        return verification
    
    def _get_safe_cell_value(self, df: pd.DataFrame, row: int, col: int) -> str:
        """Obtiene valor de celda de forma segura"""
        try:
            if row >= len(df) or col < 0 or col >= len(df.columns):
                return ""
            
            value = df.iloc[row, col]
            
            if pd.isna(value) or value is None:
                return ""
            
            return str(value).strip()
        except Exception:
            return ""

####################################################################
#            PROCESADOR MSL CON VERIFICACIÓN
####################################################################

class MSLVerifiedProcessor:
    """Procesador que solo crea documentos de rutas verificadas"""
    
    def __init__(self):
        self.inspector = MSLExcelInspector()
        
    def process_excel_with_verification(self, file_path: str) -> List[Document]:
        """Procesa Excel MSL solo después de verificación completa"""
        
        print(f"\n[PROCESADOR] === PROCESAMIENTO CON VERIFICACIÓN ===")
        
        # PASO 1: Inspección completa
        inspection = self.inspector.full_excel_inspection(file_path)
        
        if inspection['total_data_rows'] == 0:
            print(f"[PROCESADOR] ❌ No se encontraron rutas válidas en {Path(file_path).name}")
            return []
        
        print(f"[PROCESADOR] ✅ {inspection['total_data_rows']} rutas verificadas encontradas")
        
        # PASO 2: Crear documentos solo de rutas verificadas
        documents = []
        
        for sheet_name, sheet_data in inspection['regions'].items():
            for verified_entry in sheet_data['raw_entries']:
                doc = self._create_verified_document(verified_entry, file_path)
                if doc:
                    documents.append(doc)
        
        print(f"[PROCESADOR] ✅ {len(documents)} documentos creados de rutas verificadas")
        
        return documents
    
    def _create_verified_document(self, verified_entry: Dict, file_path: str) -> Document:
        """Crea documento solo de entrada verificada"""
        
        raw_data = verified_entry['raw_data']
        
        # Datos verificados
        puerto_origen = raw_data['puerto_origen']
        pais = raw_data['pais']
        sheet_name = raw_data['sheet_name']
        company_value = raw_data.get('company') or 'No especificado'
        
        # Normalizar datos solo si existen
        tarifa_text = self._safe_format_currency(raw_data.get('tarifa', ''))
        minimo_text = self._safe_format_currency(raw_data.get('minimo', ''))
        transito_text = self._safe_format_time(raw_data.get('transito', ''))
        
        # Crear contenido del documento verificado
        content = f"""TARIFA LCL MARÍTIMA MSL - RUTA VERIFICADA
Archivo: {Path(file_path).name}
Región: {sheet_name}
Fila: {raw_data['row_number']}
Estado: VERIFICADO ✅

=== INFORMACIÓN VERIFICADA ===
PUERTO ORIGEN: {puerto_origen}
PAÍS ORIGEN: {pais}
DESTINO: Chile (San Antonio/Valparaíso)
REGIÓN MSL: {sheet_name}
COMPANY: {raw_data.get('company', 'No especificado')}

=== TARIFAS VERIFICADAS ===
TARIFA TON/M³: {tarifa_text}
MÍNIMO: {minimo_text}
TIEMPO TRÁNSITO: {transito_text}
FRECUENCIA: {raw_data.get('frecuencia', 'No especificado')}

=== SERVICIO VERIFICADO ===
TIPO SERVICIO: {raw_data.get('servicio', 'No especificado')}
AGENTE LOCAL: {raw_data.get('agente', 'No especificado')}
COSTOS ADICIONALES: {raw_data.get('otros', 'No especificado')}
COMPANIA LOCAL: {raw_data.get('company', 'No especificado')}

=== OBSERVACIONES VERIFICADAS ===
{raw_data.get('observaciones', 'Sin observaciones')}

=== VERIFICACIÓN ===
✅ Ruta confirmada en tarifario MSL
✅ Datos extraídos directamente del Excel
✅ Puerto origen: {puerto_origen}
✅ Destino confirmado: Chile
"""
        
        metadata = {
            "source": file_path,
            "sheet_name": sheet_name,
            "puerto_origen": puerto_origen,
            "pais_origen": pais,
            "company": raw_data.get('company', 'No especificado'),  # ← NUEVO: Agregar company a metadata
            "row_number": raw_data['row_number'],
            "verification_status": "VERIFIED",
            "route_exists": True,
            "content_type": "msl_verified_route"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _safe_format_currency(self, value: str) -> str:
        """Formato seguro de moneda"""
        if not value or str(value).strip() == '':
            return "No disponible"
        
        value_str = str(value).strip()
        if 'EUR' in value_str.upper():
            return value_str
        elif 'USD' in value_str.upper():
            return value_str
        elif any(char.isdigit() for char in value_str):
            return f"USD {value_str}"
        else:
            return "No disponible"
    
    def _safe_format_time(self, value: str) -> str:
        """Formato seguro de tiempo"""
        if not value or str(value).strip() == '':
            return "No especificado"
        
        value_str = str(value).strip()
        if 'día' in value_str.lower() or 'day' in value_str.lower():
            return value_str
        elif any(char.isdigit() for char in value_str):
            return f"{value_str} días"
        else:
            return value_str

####################################################################
#            FUNCIÓN DE CARGA CON VERIFICACIÓN
####################################################################

def load_msl_documents_with_verification() -> List[Document]:
    """Carga documentos MSL con verificación total"""
    
    print("\n[MSL] === CARGA CON VERIFICACIÓN TOTAL ===")
    
    # Buscar archivos Excel
    excel_files = list(TMP_DIR.glob("**/*.xlsx")) + list(TMP_DIR.glob("**/*.xls"))
    
    if not excel_files:
        print("[MSL] ❌ No se encontraron archivos Excel")
        return []
    
    print(f"[MSL] Archivos encontrados: {[f.name for f in excel_files]}")
    
    processor = MSLVerifiedProcessor()
    all_documents = []
    
    for excel_file in excel_files:
        try:
            print(f"\n[MSL] === PROCESANDO CON VERIFICACIÓN: {excel_file.name} ===")
            file_docs = processor.process_excel_with_verification(str(excel_file))
            all_documents.extend(file_docs)
            
        except Exception as e:
            print(f"[MSL] ❌ Error procesando {excel_file.name}: {str(e)}")
            continue
    
    print(f"\n[MSL] === RESUMEN FINAL ===")
    print(f"[MSL] Total documentos verificados: {len(all_documents)}")
    
    # Verificar rutas únicas
    unique_routes = set()
    for doc in all_documents:
        puerto = doc.metadata.get('puerto_origen', '')
        if puerto:
            unique_routes.add(puerto)
    
    print(f"[MSL] Rutas únicas verificadas: {len(unique_routes)}")
    print(f"[MSL] Puertos disponibles: {sorted(list(unique_routes))}")
    
    return all_documents

####################################################################
#            RETRIEVER Y CHAIN CON VERIFICACIÓN
####################################################################

def create_msl_verified_retriever(vector_store, k=30):
    """Crea retriever para documentos verificados - optimizado para múltiples opciones"""
    return vector_store.as_retriever(
        search_type="similarity",  # Usar similarity en lugar de MMR
        search_kwargs={
            "k": k,  # Aumentar k para capturar más opciones
        }
    )

def create_msl_verified_chain(retriever):
    from langchain.prompts import PromptTemplate
    """Crea chain conversacional con verificación estricta"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para buscar SOLO información que realmente existe en el tarifario MSL.

VERIFICACIÓN ESTRICTA:
- Solo buscar rutas que realmente existen
- No asumir información que no esté documentada
- Verificar disponibilidad antes de responder

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para verificación:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(get_msl_response_template())
    
    doc_prompt = PromptTemplate(
        input_variables=["page_content","company","sheet_name","row_number","puerto_origen","pais_origen"],
        template=(
            "{page_content}\n\n"
            "[METADATOS]\n"
            "Company: {company}\n"
            "Hoja: {sheet_name}\n"
            "Fila: {row_number}\n"
            "Puerto: {puerto_origen}\n"
            "País: {pais_origen}\n"
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
        temperature=0.0,  # Temperatura 0 para máxima precisión
        model_kwargs={"top_p": 0.9}
    )

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt, "document_prompt": doc_prompt,},
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
#            VALIDACIÓN ESTRICTA
####################################################################

def validate_msl_route_exists(query: str, documents: List[Document]) -> Dict[str, Any]:
    """Valida que la ruta consultada realmente existe"""
    
    validation = {
        'route_exists': False,
        'route_requested': '',
        'available_routes': [],
        'verification_status': 'NOT_FOUND',
        'suggestions': []
    }
    
    # Extraer puerto solicitado
    port_info = extract_port_for_verification(query)
    
    if not port_info.get('needs_verification'):
        validation['verification_status'] = 'NO_SPECIFIC_ROUTE'
        return validation
    
    port_requested = port_info['port_requested'].lower()
    validation['route_requested'] = port_requested
    
    # Verificar en documentos
    for doc in documents:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            puerto_origen = doc.metadata.get('puerto_origen', '').lower()
            
            if puerto_origen:
                validation['available_routes'].append(puerto_origen)
                
                # Verificar coincidencia exacta o parcial
                if port_requested in puerto_origen or puerto_origen in port_requested:
                    validation['route_exists'] = True
                    validation['verification_status'] = 'VERIFIED_EXISTS'
    
    # Si no existe, sugerir alternativas
    if not validation['route_exists'] and validation['available_routes']:
        # Buscar rutas similares
        similar_routes = []
        for route in set(validation['available_routes']):
            if any(word in route for word in port_requested.split()):
                similar_routes.append(route)
        
        validation['suggestions'] = similar_routes[:5]
        validation['verification_status'] = 'NOT_FOUND_WITH_SUGGESTIONS'
    
    return validation

def analyze_msl_verified_sources(sources: List[Document]) -> Dict[str, Any]:
    """Analiza fuentes verificadas MSL"""
    
    analysis = {
        "total_verified": 0,
        "regions": {},
        "verified_ports": set(),
        "verified_countries": set(),
        "companies": {}  # ← NUEVO: Agregar análisis por company
    }
    
    for doc in sources:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            analysis['total_verified'] += 1
            
            # Por región
            region = doc.metadata.get('sheet_name', 'Desconocido')
            analysis['regions'][region] = analysis['regions'].get(region, 0) + 1
            
            # Puertos verificados
            puerto = doc.metadata.get('puerto_origen', '')
            if puerto:
                analysis['verified_ports'].add(puerto)
            
            # Países verificados
            pais = doc.metadata.get('pais_origen', '')
            if pais:
                analysis['verified_countries'].add(pais)
            
            # ← NUEVO: Companies verificadas
            company = doc.metadata.get('company', 'No especificado')
            if company:
                analysis['companies'][company] = analysis['companies'].get(company, 0) + 1
    
    return analysis