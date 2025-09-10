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
    get_maritime_fcl_response_template, detect_maritime_fcl_query_type, 
    extract_ports_from_query, get_port_region, analyze_maritime_route_direction,
    extract_container_type_from_query
)

####################################################################
#            INSPECTOR EXCEL FCL MARÍTIMO
####################################################################

class MaritimeFCLExcelInspector:
    """Inspector especializado para archivos Excel de FCL marítimo"""
    
    def __init__(self):
        self.verified_routes = set()
        self.all_ports_pol = set()
        self.all_ports_pod = set()
        self.companies = set()
        self.carriers = set()
        self.raw_data = []
        
    def inspect_maritime_fcl_excel(self, file_path: str) -> Dict[str, Any]:
        """Inspección completa del Excel de FCL marítimo"""
        print(f"\n[MARITIME FCL] === INSPECCIÓN DE {Path(file_path).name} ===")
        
        inspection_result = {
            'file_path': file_path,
            'sheets_found': [],
            'total_routes': 0,
            'pol_ports': set(),
            'pod_ports': set(),
            'companies': set(),
            'carriers': set(),
            'container_types': set(),
            'routes_data': [],
            'inspection_log': []
        }
        
        try:
            xl_file = pd.ExcelFile(file_path)
            inspection_result['sheets_found'] = xl_file.sheet_names
            
            print(f"[MARITIME FCL] Hojas encontradas: {xl_file.sheet_names}")
            
            for sheet_name in xl_file.sheet_names:
                sheet_data = self._inspect_maritime_fcl_sheet(file_path, sheet_name)
                inspection_result['total_routes'] += sheet_data['routes_count']
                inspection_result['pol_ports'].update(sheet_data['pol_ports'])
                inspection_result['pod_ports'].update(sheet_data['pod_ports'])
                inspection_result['companies'].update(sheet_data['companies'])
                inspection_result['carriers'].update(sheet_data['carriers'])
                inspection_result['container_types'].update(sheet_data['container_types'])
                inspection_result['routes_data'].extend(sheet_data['routes_data'])
                
        except Exception as e:
            error_msg = f"[MARITIME FCL] Error inspeccionando archivo: {str(e)}"
            print(error_msg)
            inspection_result['inspection_log'].append(error_msg)
        
        # Resumen final
        print(f"\n[MARITIME FCL] === RESUMEN DE INSPECCIÓN ===")
        print(f"[MARITIME FCL] Total rutas encontradas: {inspection_result['total_routes']}")
        print(f"[MARITIME FCL] Puertos POL: {sorted(list(inspection_result['pol_ports']))}")
        print(f"[MARITIME FCL] Puertos POD: {sorted(list(inspection_result['pod_ports']))}")
        print(f"[MARITIME FCL] Compañías: {sorted(list(inspection_result['companies']))}")
        
        return inspection_result
    
    def _inspect_maritime_fcl_sheet(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """Inspección de una hoja específica"""
        print(f"\n[MARITIME FCL] --- Inspeccionando hoja: {sheet_name} ---")
        
        sheet_result = {
            'sheet_name': sheet_name,
            'routes_count': 0,
            'pol_ports': set(),
            'pod_ports': set(),
            'companies': set(),
            'carriers': set(),
            'container_types': set(),
            'routes_data': []
        }
        
        try:
            # Leer la hoja completa
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                print(f"[MARITIME FCL] Hoja {sheet_name} está vacía")
                return sheet_result
            
            print(f"[MARITIME FCL] Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
            print(f"[MARITIME FCL] Columnas: {list(df.columns)}")
            
            # Mapear columnas basado en los encabezados conocidos
            column_mapping = self._map_maritime_fcl_columns(df.columns)
            print(f"[MARITIME FCL] Mapeo de columnas: {column_mapping}")
            
            # Procesar cada fila de datos
            for index, row in df.iterrows():
                route_data = self._process_maritime_fcl_row(row, column_mapping, index + 1, sheet_name)
                
                if route_data['is_valid']:
                    sheet_result['routes_count'] += 1
                    sheet_result['pol_ports'].add(route_data['pol'])
                    sheet_result['pod_ports'].add(route_data['pod'])
                    sheet_result['companies'].add(route_data['company'])
                    if route_data['carrier']:
                        sheet_result['carriers'].add(route_data['carrier'])
                    
                    # Agregar tipos de contenedores disponibles
                    if route_data['container_20gp']:
                        sheet_result['container_types'].add('20GP')
                    if route_data['container_40gp']:
                        sheet_result['container_types'].add('40GP')
                    if route_data['container_40hq']:
                        sheet_result['container_types'].add('40HQ')
                    if route_data['container_40nor']:
                        sheet_result['container_types'].add('40NOR')
                    
                    sheet_result['routes_data'].append(route_data)
            
            print(f"[MARITIME FCL] Rutas válidas en {sheet_name}: {sheet_result['routes_count']}")
            
        except Exception as e:
            print(f"[MARITIME FCL] Error inspeccionando hoja {sheet_name}: {str(e)}")
        
        return sheet_result
    
    def _map_maritime_fcl_columns(self, columns: List[str]) -> Dict[str, str]:
        """Mapea las columnas del Excel basado en los encabezados conocidos"""
        mapping = {}
        
        for col in columns:
            col_upper = str(col).upper().strip()
            
            if col_upper == 'POL':
                mapping['pol'] = col
            elif col_upper == 'POD':
                mapping['pod'] = col
            elif col_upper == 'CARRIER/SERVICIO':
                mapping['carrier'] = col
            elif col_upper == '20GP (USD)':
                mapping['container_20gp'] = col
            elif col_upper == '40GP (USD)':
                mapping['container_40gp'] = col
            elif col_upper == '40HQ (USD)':
                mapping['container_40hq'] = col
            elif col_upper == '40NOR (USD)':
                mapping['container_40nor'] = col
            elif col_upper == 'FREE TIME':
                mapping['free_time'] = col
            elif col_upper == 'OTHER':
                mapping['other'] = col
            elif col_upper == 'COMPANY':
                mapping['company'] = col
        
        return mapping
    
    def _process_maritime_fcl_row(self, row: pd.Series, column_mapping: Dict[str, str], 
                                 row_number: int, sheet_name: str) -> Dict[str, Any]:
        """Procesa una fila individual del Excel"""
        
        route_data = {
            'is_valid': False,
            'row_number': row_number,
            'sheet_name': sheet_name,
            'pol': '',
            'pod': '',
            'carrier': '',
            'container_20gp': '',
            'container_40gp': '',
            'container_40hq': '',
            'container_40nor': '',
            'free_time': '',
            'other': '',
            'company': '',
            'route_key': ''
        }
        
        try:
            # Extraer datos usando el mapeo
            pol = self._safe_get_value(row, column_mapping.get('pol', ''))
            pod = self._safe_get_value(row, column_mapping.get('pod', ''))
            carrier = self._safe_get_value(row, column_mapping.get('carrier', ''))
            container_20gp = self._safe_get_value(row, column_mapping.get('container_20gp', ''))
            container_40gp = self._safe_get_value(row, column_mapping.get('container_40gp', ''))
            container_40hq = self._safe_get_value(row, column_mapping.get('container_40hq', ''))
            container_40nor = self._safe_get_value(row, column_mapping.get('container_40nor', ''))
            free_time = self._safe_get_value(row, column_mapping.get('free_time', ''))
            other = self._safe_get_value(row, column_mapping.get('other', ''))
            company = self._safe_get_value(row, column_mapping.get('company', ''))
            pol = pol.strip().upper() if pol else ""
            pod = pod.strip().upper() if pod else ""
            
            # Validar que tiene datos mínimos necesarios
            if pol and pod:
                route_data['is_valid'] = True
                route_data['pol'] = pol
                route_data['pod'] = pod
                route_data['carrier'] = carrier
                route_data['container_20gp'] = container_20gp
                route_data['container_40gp'] = container_40gp
                route_data['container_40hq'] = container_40hq
                route_data['container_40nor'] = container_40nor
                route_data['free_time'] = free_time
                route_data['other'] = other
                route_data['company'] = company
                route_data['route_key'] = f"{pol}_{pod}"
                
        except Exception as e:
            print(f"[MARITIME FCL] Error procesando fila {row_number}: {str(e)}")
        
        return route_data
    
    def _safe_get_value(self, row: pd.Series, column: str) -> str:
        """Obtiene valor de forma segura"""
        if not column or column not in row:
            return ""
        
        value = row[column]
        if pd.isna(value) or value is None:
            return ""
        
        return str(value).strip()

####################################################################
#            PROCESADOR DE FCL MARÍTIMO
####################################################################

class MaritimeFCLProcessor:
    """Procesador que crea documentos de rutas FCL marítimas verificadas"""
    
    def __init__(self):
        self.inspector = MaritimeFCLExcelInspector()
        
    def process_maritime_fcl_excel(self, file_path: str) -> List[Document]:
        """Procesa Excel de FCL marítimo y crea documentos"""
        
        print(f"\n[PROCESSOR] === PROCESANDO FCL MARÍTIMO: {file_path} ===")
        
        # Inspección completa
        inspection = self.inspector.inspect_maritime_fcl_excel(file_path)
        
        if inspection['total_routes'] == 0:
            print(f"[PROCESSOR] No se encontraron rutas válidas en {Path(file_path).name}")
            return []
        
        print(f"[PROCESSOR] {inspection['total_routes']} rutas encontradas")
        
        # Crear documentos
        documents = []
        
        for route_data in inspection['routes_data']:
            doc = self._create_maritime_fcl_document(route_data, file_path)
            if doc:
                documents.append(doc)
        
        print(f"[PROCESSOR] {len(documents)} documentos creados")
        
        return documents
    
    def _create_maritime_fcl_document(self, route_data: Dict, file_path: str) -> Document:
        """Crea documento de una ruta FCL marítima"""
        
        # Información básica
        pol = route_data['pol']
        pod = route_data['pod']
        carrier = route_data['carrier']
        company = route_data['company']
        
        # Determinar tipo de operación
        operation_type = analyze_maritime_route_direction(pol, pod)
        pol_region = get_port_region(pol)
        pod_region = get_port_region(pod)
        
        # Formatear tarifas de contenedores
        container_20gp = self._format_container_rate(route_data['container_20gp'])
        container_40gp = self._format_container_rate(route_data['container_40gp'])
        container_40hq = self._format_container_rate(route_data['container_40hq'])
        container_40nor = self._format_container_rate(route_data['container_40nor'])
        
        # Crear contenido del documento
        content = f"""TARIFA FCL MARÍTIMO - RUTA VERIFICADA
Archivo: {Path(file_path).name}
Fila: {route_data['row_number']}
Estado: VERIFICADO ✅

=== INFORMACIÓN DE RUTA ===
PUERTO ORIGEN (POL): {pol}
REGIÓN ORIGEN: {pol_region}
PUERTO DESTINO (POD): {pod}
REGIÓN DESTINO: {pod_region}
TIPO OPERACIÓN: {operation_type}
CARRIER/SERVICIO: {carrier or 'No especificado'}
COMPAÑÍA: {company}

=== TARIFAS POR CONTENEDOR ===
CONTENEDOR 20GP: {container_20gp}
CONTENEDOR 40GP: {container_40gp}
CONTENEDOR 40HQ: {container_40hq}
CONTENEDOR 40NOR: {container_40nor}

=== CONDICIONES DE SERVICIO ===
FREE TIME: {route_data['free_time'] or 'No especificado'}
INFORMACIÓN ADICIONAL: {route_data['other'] or 'Sin información adicional'}

=== RUTA DETALLADA ===
🚢 {pol} ({pol_region}) → 🏢 {pod} ({pod_region})
💼 Operadora: {company}
🚢 Carrier: {carrier or 'Sin especificar'}
📦 Contenedores disponibles: {self._get_available_containers(route_data)}
⏰ Free time: {route_data['free_time'] or 'No especificado'}

=== VERIFICACIÓN ===
✅ Ruta confirmada en tarifario FCL marítimo
✅ Datos extraídos directamente del Excel
✅ Puerto origen verificado: {pol}
✅ Puerto destino verificado: {pod}
✅ Compañía verificada: {company}
"""
        
        metadata = {
            "source": file_path,
            "pol": pol,
            "pod": pod,
            "carrier": carrier,
            "company": company,
            "row_number": route_data['row_number'],
            "route_key": route_data['route_key'],
            "operation_type": operation_type,
            "pol_region": pol_region,
            "pod_region": pod_region,
            "container_20gp": route_data['container_20gp'],
            "container_40gp": route_data['container_40gp'],
            "container_40hq": route_data['container_40hq'],
            "container_40nor": route_data['container_40nor'],
            "free_time": route_data['free_time'],
            "verification_status": "VERIFIED",
            "content_type": "maritime_fcl_route"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _get_available_containers(self, route_data: Dict) -> str:
        """Obtiene lista de contenedores disponibles"""
        containers = []
        if route_data['container_20gp']:
            containers.append('20GP')
        if route_data['container_40gp']:
            containers.append('40GP')
        if route_data['container_40hq']:
            containers.append('40HQ')
        if route_data['container_40nor']:
            containers.append('40NOR')
        
        return ', '.join(containers) if containers else 'Ninguno especificado'

    def _format_container_rate(self, value: str) -> str:
        """Formatea una tarifa de contenedor para mostrar"""
        if not value or str(value).strip() == '' or str(value).lower() == 'nan':
            return "No disponible"
        
        return str(value).strip()

####################################################################
#            FUNCIÓN DE CARGA DE DOCUMENTOS FCL
####################################################################

def load_maritime_fcl_documents() -> List[Document]:
    """Carga documentos de FCL marítimo"""
    
    print("\n[MARITIME FCL] === CARGA DE DOCUMENTOS FCL MARÍTIMO ===")
    
    # Buscar archivos Excel
    excel_files = list(TMP_DIR.glob("**/*.xlsx")) + list(TMP_DIR.glob("**/*.xls"))
    
    if not excel_files:
        print("[MARITIME FCL] No se encontraron archivos Excel")
        return []
    
    print(f"[MARITIME FCL] Archivos encontrados: {[f.name for f in excel_files]}")
    
    processor = MaritimeFCLProcessor()
    all_documents = []
    
    for excel_file in excel_files:
        try:
            print(f"\n[MARITIME FCL] === PROCESANDO: {excel_file.name} ===")
            file_docs = processor.process_maritime_fcl_excel(str(excel_file))
            all_documents.extend(file_docs)
            
        except Exception as e:
            print(f"[MARITIME FCL] Error procesando {excel_file.name}: {str(e)}")
            continue
    
    print(f"\n[MARITIME FCL] === RESUMEN FINAL ===")
    print(f"[MARITIME FCL] Total documentos: {len(all_documents)}")
    
    # Analizar rutas únicas
    unique_routes = set()
    pol_ports = set()
    pod_ports = set()
    companies = set()
    carriers = set()
    
    for doc in all_documents:
        route_key = doc.metadata.get('route_key', '')
        if route_key:
            unique_routes.add(route_key)
        
        pol = doc.metadata.get('pol', '')
        if pol:
            pol_ports.add(pol)
            
        pod = doc.metadata.get('pod', '')
        if pod:
            pod_ports.add(pod)
            
        company = doc.metadata.get('company', '')
        if company:
            companies.add(company)
            
        carrier = doc.metadata.get('carrier', '')
        if carrier:
            carriers.add(carrier)
    
    print(f"[MARITIME FCL] Rutas únicas: {len(unique_routes)}")
    print(f"[MARITIME FCL] Puertos POL: {sorted(list(pol_ports))}")
    print(f"[MARITIME FCL] Puertos POD: {sorted(list(pod_ports))}")
    print(f"[MARITIME FCL] Compañías: {sorted(list(companies))}")
    print(f"[MARITIME FCL] Carriers: {sorted(list(carriers))}")
    
    return all_documents

####################################################################
#            RETRIEVER Y CHAIN
####################################################################

def create_maritime_fcl_retriever(vector_store, k=20):
    """Crea retriever para documentos de FCL marítimo"""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        }
    )

def create_maritime_fcl_chain(retriever):
    """Crea chain conversacional para FCL marítimo"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para buscar información específica de tarifas FCL marítimas.

ENFOQUE FCL MARÍTIMO:
- Buscar rutas específicas entre puertos (POL → POD)
- Identificar códigos de puertos si están presentes
- Verificar disponibilidad de rutas
- Buscar información de tarifas, carriers y tipos de contenedores

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(get_maritime_fcl_response_template())
    
    doc_prompt = PromptTemplate(
        input_variables=["page_content", "pol", "pod", "company", "carrier", "row_number"],
        template=(
            "{page_content}\n\n"
            "[METADATOS]\n"
            "POL: {pol}\n"
            "POD: {pod}\n"
            "Company: {company}\n"
            "Carrier: {carrier}\n"
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

def validate_maritime_fcl_route(query: str, documents: List[Document]) -> Dict[str, Any]:
    """Valida que la ruta FCL marítima consultada existe"""
    
    validation = {
        'route_exists': False,
        'ports_requested': [],
        'pol_requested': None,
        'pod_requested': None,
        'available_routes': [],
        'suggestions': [],
        'verification_status': 'NOT_FOUND'
    }
    
    # Extraer puertos de la consulta
    port_info = extract_ports_from_query(query)
    validation['ports_requested'] = port_info['ports_found']
    validation['pol_requested'] = port_info['pol_detected']
    validation['pod_requested'] = port_info['pod_detected']
    
    if not port_info['needs_verification']:
        validation['verification_status'] = 'NO_SPECIFIC_ROUTE'
        return validation
    
    # Verificar en documentos
    for doc in documents:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            pol = doc.metadata.get('pol', '')
            pod = doc.metadata.get('pod', '')
            pol_u = (pol or "").upper().strip()
            pod_u = (pod or "").upper().strip()
            route_key = f"{pol_u}_{pod_u}"
            
            validation['available_routes'].append(route_key)
            
            # Verificar coincidencia exacta de ruta
            if port_info['has_route_pattern']:
                if (pol_u == (port_info['pol_detected'] or '').upper().strip() and 
                    pod_u == (port_info['pod_detected'] or '').upper().strip()):
                    validation['route_exists'] = True
                    validation['verification_status'] = 'EXACT_ROUTE_FOUND'
            
            # Verificar puertos individuales
            elif port_info['ports_found']:
                for port in port_info['ports_found']:
                    if port in [pol_u, pod_u]:
                        validation['route_exists'] = True
                        validation['verification_status'] = 'PORT_FOUND'
    
    # Generar sugerencias si no se encuentra la ruta exacta
    if not validation['route_exists'] and validation['available_routes']:
        # Buscar rutas similares
        if port_info['pol_detected']:
            similar_routes = [route for route in validation['available_routes'] 
                            if route.startswith(port_info['pol_detected'])]
            validation['suggestions'].extend(similar_routes[:5])
        
        if port_info['pod_detected']:
            similar_routes = [route for route in validation['available_routes'] 
                            if route.endswith(port_info['pod_detected'])]
            validation['suggestions'].extend(similar_routes[:5])
    
    return validation

def analyze_maritime_fcl_sources(sources: List[Document]) -> Dict[str, Any]:
    """Analiza fuentes de FCL marítimo"""
    
    analysis = {
        "total_routes": 0,
        "pol_ports": set(),
        "pod_ports": set(),
        "companies": set(),
        "carriers": set(),
        "regions_pol": {},
        "regions_pod": {},
        "operation_types": {},
        "container_types": {
            "20GP": 0,
            "40GP": 0,
            "40HQ": 0,
            "40NOR": 0
        }
    }
    
    for doc in sources:
        if doc.metadata.get('verification_status') == 'VERIFIED':
            analysis['total_routes'] += 1
            
            # Puertos
            pol = doc.metadata.get('pol', '')
            pod = doc.metadata.get('pod', '')
            if pol:
                analysis['pol_ports'].add(pol)
                # Contar por región POL
                region = doc.metadata.get('pol_region', '')
                analysis['regions_pol'][region] = analysis['regions_pol'].get(region, 0) + 1
            
            if pod:
                analysis['pod_ports'].add(pod)
                # Contar por región POD
                region = doc.metadata.get('pod_region', '')
                analysis['regions_pod'][region] = analysis['regions_pod'].get(region, 0) + 1
            
            # Compañías
            company = doc.metadata.get('company', '')
            if company:
                analysis['companies'].add(company)
            
            # Carriers
            carrier = doc.metadata.get('carrier', '')
            if carrier:
                analysis['carriers'].add(carrier)
            
            # Tipos de operación
            op_type = doc.metadata.get('operation_type', '')
            if op_type:
                analysis['operation_types'][op_type] = analysis['operation_types'].get(op_type, 0) + 1
            
            # Contenedores disponibles
            if doc.metadata.get('container_20gp'):
                analysis['container_types']['20GP'] += 1
            if doc.metadata.get('container_40gp'):
                analysis['container_types']['40GP'] += 1
            if doc.metadata.get('container_40hq'):
                analysis['container_types']['40HQ'] += 1
            if doc.metadata.get('container_40nor'):
                analysis['container_types']['40NOR'] += 1
    
    return analysis