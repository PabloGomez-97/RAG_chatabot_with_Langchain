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


from .config import (
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, OPENAI_API_KEY,
    get_lcl_maritime_response_template, detect_lcl_maritime_query_type, 
    extract_lcl_ports_from_query, get_lcl_port_region, analyze_lcl_route_direction,
    extract_region_from_query, normalize_port_name, matches_pod
)

####################################################################
#            INSPECTOR EXCEL LCL MARÍTIMO MULTI-REGIONAL
####################################################################

class LCLMaritimeExcelInspector:
    """Inspector especializado para archivos Excel de LCL marítimo multi-regional"""
    
    def __init__(self):
        self.verified_routes = set()
        self.all_ports_pol = set()
        self.all_ports_pod = set()
        self.companies = set()
        self.agents = set()
        self.raw_data = []
        
    def inspect_lcl_maritime_excel(self, file_path: str) -> Dict[str, Any]:
        """Inspección completa del Excel de LCL marítimo multi-regional"""
        print(f"\n[LCL MARITIME] === INSPECCIÓN DE {Path(file_path).name} ===")
        
        inspection_result = {
            'file_path': file_path,
            'sheets_found': [],
            'total_routes': 0,
            'regional_data': {},
            'pol_ports': set(),
            'pod_ports': set(),
            'companies': set(),
            'agents': set(),
            'currencies': set(),
            'routes_data': [],
            'inspection_log': []
        }
        
        try:
            xl_file = pd.ExcelFile(file_path)
            inspection_result['sheets_found'] = xl_file.sheet_names
            
            print(f"[LCL MARITIME] Hojas encontradas: {xl_file.sheet_names}")
            
            # Procesar cada hoja regional
            for sheet_name in xl_file.sheet_names:
                sheet_data = self._inspect_lcl_maritime_sheet(file_path, sheet_name)
                inspection_result['regional_data'][sheet_name] = sheet_data
                inspection_result['total_routes'] += sheet_data['routes_count']
                inspection_result['pol_ports'].update(sheet_data['pol_ports'])
                inspection_result['pod_ports'].update(sheet_data['pod_ports'])
                inspection_result['companies'].update(sheet_data['companies'])
                inspection_result['agents'].update(sheet_data['agents'])
                inspection_result['currencies'].update(sheet_data['currencies'])
                inspection_result['routes_data'].extend(sheet_data['routes_data'])
                
        except Exception as e:
            error_msg = f"[LCL MARITIME] Error inspeccionando archivo: {str(e)}"
            print(error_msg)
            inspection_result['inspection_log'].append(error_msg)
        
        # Resumen final
        print(f"\n[LCL MARITIME] === RESUMEN DE INSPECCIÓN ===")
        print(f"[LCL MARITIME] Total rutas encontradas: {inspection_result['total_routes']}")
        print(f"[LCL MARITIME] Puertos POL: {len(inspection_result['pol_ports'])} únicos")
        print(f"[LCL MARITIME] Puertos POD: {len(inspection_result['pod_ports'])} únicos")
        print(f"[LCL MARITIME] Compañías: {sorted(list(inspection_result['companies']))}")
        print(f"[LCL MARITIME] Distribución regional:")
        for region, data in inspection_result['regional_data'].items():
            print(f"  - {region}: {data['routes_count']} rutas")
        
        return inspection_result
    
    def _inspect_lcl_maritime_sheet(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """Inspección de una hoja regional específica"""
        print(f"\n[LCL MARITIME] --- Inspeccionando región: {sheet_name} ---")
        
        sheet_result = {
            'sheet_name': sheet_name,
            'routes_count': 0,
            'pol_ports': set(),
            'pod_ports': set(),
            'countries': set(),
            'companies': set(),
            'agents': set(),
            'services': set(),
            'currencies': set(),
            'routes_data': []
        }
        
        try:
            # Leer la hoja completa
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                print(f"[LCL MARITIME] Región {sheet_name} está vacía")
                return sheet_result
            
            print(f"[LCL MARITIME] Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
            print(f"[LCL MARITIME] Columnas: {list(df.columns)}")
            
            # Mapear columnas basado en los encabezados conocidos
            column_mapping = self._map_lcl_maritime_columns(df.columns)
            print(f"[LCL MARITIME] Mapeo de columnas: {column_mapping}")
            
            # Procesar cada fila de datos
            for index, row in df.iterrows():
                route_data = self._process_lcl_maritime_row(row, column_mapping, index + 1, sheet_name)
                
                if route_data['is_valid']:
                    sheet_result['routes_count'] += 1
                    sheet_result['pol_ports'].add(route_data['pol'])
                    sheet_result['pod_ports'].add(route_data['pod'])
                    sheet_result['countries'].add(route_data['country'])
                    sheet_result['companies'].add(route_data['company'])
                    if route_data['agent']:
                        sheet_result['agents'].add(route_data['agent'])
                    if route_data['service']:
                        sheet_result['services'].add(route_data['service'])
                    
                    # Detectar monedas
                    if route_data['ton_m3_currency']:
                        sheet_result['currencies'].add(route_data['ton_m3_currency'])
                    if route_data['minimo_currency']:
                        sheet_result['currencies'].add(route_data['minimo_currency'])
                    
                    sheet_result['routes_data'].append(route_data)
            
            print(f"[LCL MARITIME] Rutas válidas en {sheet_name}: {sheet_result['routes_count']}")
            
        except Exception as e:
            print(f"[LCL MARITIME] Error inspeccionando región {sheet_name}: {str(e)}")
        
        return sheet_result
    
    def _map_lcl_maritime_columns(self, columns: List[str]) -> Dict[str, str]:
        """Mapea las columnas del Excel basado en los encabezados conocidos"""
        mapping = {}
        
        for col in columns:
            col_upper = str(col).upper().strip()
            
            if col_upper == 'POL':
                mapping['pol'] = col
            elif col_upper == 'PAIS':
                mapping['country'] = col
            elif col_upper == 'POD':
                mapping['pod'] = col
            elif col_upper == 'TON / M3 USD/EUR':
                mapping['ton_m3'] = col
            elif col_upper == 'MINIMO':
                mapping['minimo'] = col
            elif col_upper == 'T / T APROX.':
                mapping['transit_time'] = col
            elif col_upper == 'FREC.':
                mapping['frequency'] = col
            elif col_upper == 'OTROS':
                mapping['others'] = col
            elif col_upper == 'SERVICIO':
                mapping['service'] = col
            elif col_upper == 'AGENTE':
                mapping['agent'] = col
            elif col_upper == 'COMPANY':
                mapping['company'] = col
        
        return mapping
    
    def _process_lcl_maritime_row(self, row: pd.Series, column_mapping: Dict[str, str], 
                                 row_number: int, sheet_name: str) -> Dict[str, Any]:
        """Procesa una fila individual del Excel"""
        
        route_data = {
            'is_valid': False,
            'row_number': row_number,
            'sheet_name': sheet_name,
            'region': sheet_name,
            'pol': '',
            'country': '',
            'pod': '',
            'ton_m3': '',
            'minimo': '',
            'transit_time': '',
            'frequency': '',
            'others': '',
            'service': '',
            'agent': '',
            'company': '',
            'ton_m3_currency': '',
            'minimo_currency': '',
            'route_key': ''
        }
        
        try:
            # Extraer datos usando el mapeo
            pol = self._safe_get_value(row, column_mapping.get('pol', ''))
            country = self._safe_get_value(row, column_mapping.get('country', ''))
            pod = self._safe_get_value(row, column_mapping.get('pod', ''))
            ton_m3 = self._safe_get_value(row, column_mapping.get('ton_m3', ''))
            minimo = self._safe_get_value(row, column_mapping.get('minimo', ''))
            transit_time = self._safe_get_value(row, column_mapping.get('transit_time', ''))
            frequency = self._safe_get_value(row, column_mapping.get('frequency', ''))
            others = self._safe_get_value(row, column_mapping.get('others', ''))
            service = self._safe_get_value(row, column_mapping.get('service', ''))
            agent = self._safe_get_value(row, column_mapping.get('agent', ''))
            company = self._safe_get_value(row, column_mapping.get('company', ''))
            pol = normalize_port_name(pol, "pol")
            pod = normalize_port_name(pod, "pod")
            country = (country or "").strip()
            
            
            # Validar que tiene datos mínimos necesarios
            if pol and pod and country:
                route_data['is_valid'] = True
                route_data['pol'] = pol
                route_data['country'] = country
                route_data['pod'] = pod
                route_data['ton_m3'] = ton_m3
                route_data['minimo'] = minimo
                route_data['transit_time'] = transit_time
                route_data['frequency'] = frequency
                route_data['others'] = others
                route_data['service'] = service
                route_data['agent'] = agent
                route_data['company'] = company
                route_data['route_key'] = f"{pol}_{pod}_{sheet_name}"

                
                # Extraer monedas
                route_data['ton_m3_currency'] = self._extract_currency(ton_m3)
                route_data['minimo_currency'] = self._extract_currency(minimo)
                
        except Exception as e:
            print(f"[LCL MARITIME] Error procesando fila {row_number}: {str(e)}")
        
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
#            PROCESADOR DE LCL MARÍTIMO
####################################################################

class LCLMaritimeProcessor:
    """Procesador que crea documentos de rutas LCL marítimas verificadas"""
    
    def __init__(self):
        self.inspector = LCLMaritimeExcelInspector()
        
    def process_lcl_maritime_excel(self, file_path: str) -> List[Document]:
        """Procesa Excel de LCL marítimo y crea documentos"""
        
        print(f"\n[PROCESSOR] === PROCESANDO LCL MARÍTIMO: {file_path} ===")
        
        # Inspección completa
        inspection = self.inspector.inspect_lcl_maritime_excel(file_path)
        
        if inspection['total_routes'] == 0:
            print(f"[PROCESSOR] No se encontraron rutas válidas en {Path(file_path).name}")
            return []
        
        print(f"[PROCESSOR] {inspection['total_routes']} rutas encontradas")
        
        # Crear documentos
        documents = []
        
        for route_data in inspection['routes_data']:
            doc = self._create_lcl_maritime_document(route_data, file_path)
            if doc:
                documents.append(doc)
        
        print(f"[PROCESSOR] {len(documents)} documentos creados")
        
        return documents
    
    def _create_lcl_maritime_document(self, route_data: Dict, file_path: str) -> Document:
        """Crea documento de una ruta LCL marítima"""
        
        # Información básica
        pol = route_data['pol']
        pod = route_data['pod']
        country = route_data['country']
        region = route_data['region']
        company = route_data['company']
        
        # Determinar tipo de operación
        operation_type = analyze_lcl_route_direction(pol, pod)
        pol_region = get_lcl_port_region(pol)
        pod_region = get_lcl_port_region(pod)
        
        # Formatear tarifas
        ton_m3_formatted = self._format_tariff(route_data['ton_m3'])
        minimo_formatted = self._format_tariff(route_data['minimo'])
        
        # Crear contenido del documento
        content = f"""TARIFA LCL MARÍTIMO MSL - RUTA VERIFICADA
Archivo: {Path(file_path).name}
Región: {region}
Fila: {route_data['row_number']}
Estado: VERIFICADO ✅

=== INFORMACIÓN DE RUTA ===
PUERTO ORIGEN (POL): {pol}
PAÍS ORIGEN: {country}
REGIÓN ORIGEN: {pol_region}
PUERTO DESTINO (POD): {pod}
REGIÓN DESTINO: {pod_region}
TIPO OPERACIÓN: {operation_type}
COMPAÑÍA: {company}

=== TARIFAS VERIFICADAS ===
TARIFA TON/M3: {ton_m3_formatted}
MÍNIMO: {minimo_formatted}
MONEDA TON/M3: {route_data['ton_m3_currency']}
MONEDA MÍNIMO: {route_data['minimo_currency']}

=== SERVICIO ===
TIEMPO TRÁNSITO: {route_data['transit_time'] or 'No especificado'}
FRECUENCIA: {route_data['frequency'] or 'No especificado'}
TIPO SERVICIO: {route_data['service'] or 'No especificado'}
AGENTE LOCAL: {route_data['agent'] or 'No especificado'}
COSTOS ADICIONALES: {route_data['others'] or 'No especificado'}

=== RUTA DETALLADA ===
🚢 {pol} ({country}) → 🏢 {pod} ({pod_region})
📍 Región: {region}
💼 Operadora: {company}
🚢 Servicio: {route_data['service'] or 'Sin especificar'}
💰 Desde {ton_m3_formatted} por TON/M3
📊 Mínimo {minimo_formatted}
⏰ Tránsito: {route_data['transit_time'] or 'No especificado'}
🔄 Frecuencia: {route_data['frequency'] or 'No especificado'}

=== VERIFICACIÓN ===
✅ Ruta confirmada en tarifario LCL MSL
✅ Datos extraídos directamente del Excel
✅ Puerto origen verificado: {pol}
✅ Puerto destino verificado: {pod}
✅ Región verificada: {region}
✅ Compañía verificada: {company}
"""
        
        metadata = {
            "source": file_path,
            "region": region,
            "pol": pol,
            "country": country,
            "pod": pod,
            "company": company,
            "agent": route_data['agent'],
            "service": route_data['service'],
            "row_number": route_data['row_number'],
            "route_key": route_data['route_key'],
            "operation_type": operation_type,
            "pol_region": pol_region,
            "pod_region": pod_region,
            "ton_m3_currency": route_data['ton_m3_currency'],
            "minimo_currency": route_data['minimo_currency'],
            "transit_time": route_data['transit_time'],
            "frequency": route_data['frequency'],
            "verification_status": "VERIFIED",
            "content_type": "lcl_maritime_route"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _format_tariff(self, value: str) -> str:
        """Formatea una tarifa para mostrar"""
        if not value or str(value).strip() == '' or str(value).lower() == 'nan':
            return "No disponible"
        
        return str(value).strip()

####################################################################
#            FUNCIÓN DE CARGA DE DOCUMENTOS LCL
####################################################################

def load_lcl_maritime_documents() -> List[Document]:
    """Carga documentos de LCL marítimo"""
    
    print("\n[LCL MARITIME] === CARGA DE DOCUMENTOS LCL MARÍTIMO ===")
    
    # Buscar archivos Excel
    excel_files = list(TMP_DIR.glob("**/*.xlsx")) + list(TMP_DIR.glob("**/*.xls"))
    
    if not excel_files:
        print("[LCL MARITIME] No se encontraron archivos Excel")
        return []
    
    print(f"[LCL MARITIME] Archivos encontrados: {[f.name for f in excel_files]}")
    
    processor = LCLMaritimeProcessor()
    all_documents = []
    
    for excel_file in excel_files:
        try:
            print(f"\n[LCL MARITIME] === PROCESANDO: {excel_file.name} ===")
            file_docs = processor.process_lcl_maritime_excel(str(excel_file))
            all_documents.extend(file_docs)
            
        except Exception as e:
            print(f"[LCL MARITIME] Error procesando {excel_file.name}: {str(e)}")
            continue
    
    print(f"\n[LCL MARITIME] === RESUMEN FINAL ===")
    print(f"[LCL MARITIME] Total documentos: {len(all_documents)}")
    
    # Analizar rutas únicas por región
    regional_stats = {}
    pol_ports = set()
    pod_ports = set()
    companies = set()
    agents = set()
    
    for doc in all_documents:
        region = doc.metadata.get('region', 'Unknown')
        if region not in regional_stats:
            regional_stats[region] = 0
        regional_stats[region] += 1
        
        pol = doc.metadata.get('pol', '')
        if pol:
            pol_ports.add(pol)
            
        pod = doc.metadata.get('pod', '')
        if pod:
            pod_ports.add(pod)
            
        company = doc.metadata.get('company', '')
        if company:
            companies.add(company)
            
        agent = doc.metadata.get('agent', '')
        if agent:
            agents.add(agent)
    
    print(f"[LCL MARITIME] Distribución regional:")
    for region, count in regional_stats.items():
        print(f"  - {region}: {count} rutas")
    print(f"[LCL MARITIME] Puertos POL únicos: {len(pol_ports)}")
    print(f"[LCL MARITIME] Puertos POD únicos: {len(pod_ports)}")
    print(f"[LCL MARITIME] Compañías: {sorted(list(companies))}")
    print(f"[LCL MARITIME] Agentes únicos: {len(agents)}")
    
    return all_documents

####################################################################
#            RETRIEVER Y CHAIN
####################################################################

def create_lcl_maritime_retriever(vector_store, k=25):
    """Crea retriever para documentos de LCL marítimo"""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        }
    )

def create_lcl_maritime_chain(retriever):
    """Crea chain conversacional para LCL marítimo"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para buscar información específica de tarifas LCL marítimas.

ENFOQUE LCL MARÍTIMO:
- Buscar rutas específicas entre puertos (POL → POD)
- Identificar regiones si están presentes (América, Europa, Norteamérica, Asia)
- Verificar disponibilidad de rutas
- Buscar información de tarifas TON/M3, agentes y tiempos de tránsito

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(get_lcl_maritime_response_template())
    
    doc_prompt = PromptTemplate(
        input_variables=["page_content", "region", "pol", "pod", "company", "agent", "row_number"],
        template=(
            "{page_content}\n\n"
            "[METADATOS]\n"
            "Región: {region}\n"
            "POL: {pol}\n"
            "POD: {pod}\n"
            "Company: {company}\n"
            "Agente: {agent}\n"
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

def validate_lcl_maritime_route(query: str, documents: List[Document]) -> Dict[str, Any]:
    """Valida que la ruta LCL marítima consultada existe"""
    
    validation = {
        'route_exists': False,
        'ports_requested': [],
        'pol_requested': None,
        'pod_requested': None,
        'region_requested': None,
        'available_routes': [],
        'suggestions': [],
        'verification_status': 'NOT_FOUND'
    }
    
    # Extraer puertos de la consulta
    port_info = extract_lcl_ports_from_query(query)
    requested_pol = normalize_port_name(port_info['pol_detected'], "pol") if port_info['pol_detected'] else None
    requested_pod = normalize_port_name(port_info['pod_detected'], "pod") if port_info['pod_detected'] else None
    ports_found_norm = [normalize_port_name(p, "any") for p in port_info['ports_found']]

    validation['ports_requested'] = port_info['ports_found']
    validation['pol_requested'] = port_info['pol_detected']
    validation['pod_requested'] = port_info['pod_detected']
    validation['region_requested'] = extract_region_from_query(query)
    
    # Verificar en documentos
    for doc in documents:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            pol = doc.metadata.get('pol', '')
            pod = doc.metadata.get('pod', '')
            pol_doc = normalize_port_name(doc.metadata.get('pol', ''), "pol")
            pod_doc = normalize_port_name(doc.metadata.get('pod', ''), "pod")
            region = doc.metadata.get('region', '')
            route_key = f"{pol_doc}_{pod_doc}_{region}"
            validation['available_routes'].append(route_key)
            
            # Verificar coincidencia exacta de ruta
            if port_info['has_route_pattern']:
                if pol_doc == (requested_pol or "") and matches_pod((requested_pod or ""), pod_doc):
                    validation['route_exists'] = True
                    validation['verification_status'] = 'EXACT_ROUTE_FOUND'
            
            # Verificar puertos individuales
            elif ports_found_norm:
                for p in ports_found_norm:
                    # Si p parece un POD chileno, aplicar la regla de POD
                    if p in {"SAN ANTONIO", "VALPARAISO", "SAI/VAP"}:
                        if matches_pod(p, pod_doc):
                            validation['route_exists'] = True
                            validation['verification_status'] = 'PORT_FOUND'
                    else:
                        # p puede ser un POL u otro puerto; igualdad normalizada
                        if p in (pol_doc, pod_doc):
                            validation['route_exists'] = True
                            validation['verification_status'] = 'PORT_FOUND'
    
    # Generar sugerencias si no se encuentra la ruta exacta
    if not validation['route_exists'] and validation['available_routes']:
        if requested_pol:
            validation['suggestions'].extend([rk for rk in validation['available_routes'] if rk.startswith(f"{requested_pol}_")][:5])
        if requested_pod:
            validation['suggestions'].extend([rk for rk in validation['available_routes'] if rk.split('_')[1] == requested_pod][:5])
    
    return validation

def analyze_lcl_maritime_sources(sources: List[Document]) -> Dict[str, Any]:
    """Analiza fuentes de LCL marítimo"""
    
    analysis = {
        "total_routes": 0,
        "regional_distribution": {},
        "pol_ports": set(),
        "pod_ports": set(),
        "companies": set(),
        "agents": set(),
        "countries": set(),
        "currencies": set(),
        "operation_types": {}
    }
    
    for doc in sources:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            analysis['total_routes'] += 1
            
            # Distribución regional
            region = doc.metadata.get('region', 'Unknown')
            analysis['regional_distribution'][region] = analysis['regional_distribution'].get(region, 0) + 1
            
            # Puertos
            pol = doc.metadata.get('pol', '')
            pod = doc.metadata.get('pod', '')
            if pol:
                analysis['pol_ports'].add(pol)
            if pod:
                analysis['pod_ports'].add(pod)
            
            # Compañías y agentes
            company = doc.metadata.get('company', '')
            if company:
                analysis['companies'].add(company)
            
            agent = doc.metadata.get('agent', '')
            if agent:
                analysis['agents'].add(agent)
            
            # Países
            country = doc.metadata.get('country', '')
            if country:
                analysis['countries'].add(country)
            
            # Monedas
            ton_m3_currency = doc.metadata.get('ton_m3_currency', '')
            minimo_currency = doc.metadata.get('minimo_currency', '')
            if ton_m3_currency:
                analysis['currencies'].add(ton_m3_currency)
            if minimo_currency:
                analysis['currencies'].add(minimo_currency)
            
            # Tipos de operación
            op_type = doc.metadata.get('operation_type', '')
            if op_type:
                analysis['operation_types'][op_type] = analysis['operation_types'].get(op_type, 0) + 1
    
    return analysis