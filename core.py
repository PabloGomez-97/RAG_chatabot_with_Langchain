import glob
import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Any
import json
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from config import (
    TMP_DIR, LOCAL_VECTOR_STORE_DIR, OPENAI_API_KEY,
    enhanced_seemann_response_template, detect_query_type,
    validate_document_relevance
)

####################################################################
#            PARSER AVANZADO v2.1 - VALIDACION POL/POD
####################################################################

class EnhancedFreightParser:
    """Parser avanzado que maneja CSVs complejos y valida POL/POD"""
    
    def __init__(self):
        self.port_aliases = {
            'shanghai': ['shanghai', 'sha', 'china shanghai', 'shg'],
            'ningbo': ['ningbo', 'ngb', 'china ningbo', 'nnb'],
            'shenzhen': ['shenzhen', 'szn', 'china shenzhen', 'szx'],
            'qingdao': ['qingdao', 'tao', 'china qingdao', 'tad'],
            'tianjin': ['tianjin', 'tja', 'china tianjin', 'tsn'],
            'xiamen': ['xiamen', 'xmn', 'china xiamen', 'xam'],
            'shekou': ['shekou', 'sku', 'china shekou', 'shk'],
            'cbp': ['cbp', 'china basic port', 'china basic', 'basic port'],
            'san antonio': ['san antonio', 'sap', 'chile san antonio', 'val-sap'],
            'valparaiso': ['valparaiso', 'val', 'chile valparaiso', 'vlp'],
            'callao': ['callao', 'cao', 'peru callao', 'lim'],
            'guayaquil': ['guayaquil', 'gye', 'ecuador guayaquil', 'uio'],
        }
        
        self.carrier_aliases = {
            'cosco': ['cosco', 'cosco shipping', 'cscl', 'cosco lines'],
            'msk': ['msk', 'maersk', 'maersk line', 'maersk shipping'],
            'cma': ['cma', 'cma cgm', 'cgm', 'cma-cgm'],
            'msc': ['msc', 'mediterranean shipping', 'msc shipping'],
            'one': ['one', 'ocean network express', 'one line'],
            'pil': ['pil', 'pacific international', 'pil shipping'],
            'ecu': ['ecu', 'ecu worldwide', 'ecu shipping'],
        }
    
    def detect_content_type(self, content: str, filename: str = "") -> str:
        """Detecta automáticamente el tipo de contenido del documento"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Detección de Demurrage/Detention
        if any(term in content_lower for term in [
            'demurrage', 'detention', 'combined', 'free time', 'calendar day',
            'import combined', 'export combined', 'ib dnd', 'ob dnd'
        ]):
            return 'demurrage_detention'
        
        # Detección de Exportación (desde Chile)
        if any(term in content_lower for term in [
            'origin documentation fee', 'gate out charge', 'prepaid tariff',
            'traffic term: cy - cy'
        ]) and any(term in content_lower for term in ['san antonio', 'chile']):
            return 'export_rates'
        
        # Detección por nombre de archivo
        if any(term in filename_lower for term in [
            'demurrage', 'detention', 'dnd'
        ]):
            return 'demurrage_detention'
        
        if any(term in filename_lower for term in [
            'export', 'outbound', 'ob'
        ]):
            return 'export_rates'
        
        # Default: FCL Import
        return 'fcl_rate'
    
    def extract_pol_pod_from_query(self, query: str) -> Dict[str, str]:
        """Extrae POL y POD de la consulta del usuario"""
        query_lower = query.lower()
        
        # Patrones comunes para POL-POD
        patterns = [
            r'desde\s+([^a]+?)\s+a\s+([^?]+)',
            r'from\s+([^to]+?)\s+to\s+([^?]+)',
            r'de\s+([^a]+?)\s+a\s+([^?]+)',
            r'origen\s+([^destino]+?)\s+destino\s+([^?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                pol_raw = match.group(1).strip()
                pod_raw = match.group(2).strip()
                
                # Normalizar puertos
                pol_normalized = self.normalize_port_name(pol_raw)
                pod_normalized = self.normalize_port_name(pod_raw)
                
                return {
                    'pol_raw': pol_raw,
                    'pod_raw': pod_raw,
                    'pol_normalized': pol_normalized,
                    'pod_normalized': pod_normalized
                }
        
        return {}
    
    def validate_pol_pod_match(self, doc_pol: str, doc_pod: str, query_pol: str, query_pod: str) -> bool:
        """Valida si el POL/POD del documento coincide con la consulta"""
        if not query_pol or not query_pod:
            return True  # Si no hay info de query, aceptar
        
        # Normalizar todos los valores
        doc_pol_norm = self.normalize_port_name(doc_pol)
        doc_pod_norm = self.normalize_port_name(doc_pod)
        query_pol_norm = self.normalize_port_name(query_pol)
        query_pod_norm = self.normalize_port_name(query_pod)
        
        # Validar coincidencia exacta
        pol_match = (doc_pol_norm == query_pol_norm) or self._check_port_aliases(doc_pol_norm, query_pol_norm)
        pod_match = (doc_pod_norm == query_pod_norm) or self._check_port_aliases(doc_pod_norm, query_pod_norm)
        
        return pol_match and pod_match
    
    def _check_port_aliases(self, port1: str, port2: str) -> bool:
        """Verifica si dos puertos son aliases del mismo puerto"""
        for canonical, aliases in self.port_aliases.items():
            if port1 in aliases and port2 in aliases:
                return True
        return False
    
    def parse_combined_rates(self, rate_string: str) -> Dict[str, str]:
        """Extrae tarifas combinadas como 'USD2300/2800 per 20/40'"""
        if pd.isna(rate_string) or not rate_string:
            return {"20": "TBD", "40": "TBD"}
        
        rate_string = str(rate_string).strip()
        
        # Patron USD2300/2800 per 20/40
        pattern1 = r'USD(\d+)/(\d+)\s*per\s*20/40'
        match1 = re.search(pattern1, rate_string, re.IGNORECASE)
        if match1:
            return {
                "20": f"USD {match1.group(1)}",
                "40": f"USD {match1.group(2)}"
            }
        
        # Patron $2,605.00/$2,955.00
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
        """Extrae puertos multiples como 'QINGDAO/SHANGHAI/NINGBO/SHENZHEN'"""
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
        
        # Buscar coincidencia directa primero
        for canonical, aliases in self.port_aliases.items():
            if port_lower in aliases:
                return canonical
        
        # Buscar coincidencia parcial
        for canonical, aliases in self.port_aliases.items():
            for alias in aliases:
                if alias in port_lower or port_lower in alias:
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
        """Extrae dias libres de strings como '21days'"""
        if pd.isna(free_time_string) or not free_time_string:
            return "No especificado"
        
        free_time_string = str(free_time_string).strip()
        
        # Buscar patron numero + days
        pattern = r'(\d+)\s*days?'
        match = re.search(pattern, free_time_string, re.IGNORECASE)
        if match:
            return f"{match.group(1)} dias"
        
        # Solo numeros
        pattern2 = r'(\d+)'
        match2 = re.search(pattern2, free_time_string)
        if match2:
            return f"{match2.group(1)} dias"
        
        return "No especificado"

####################################################################
#            PROCESADORES DE CSV AVANZADOS v2.1
####################################################################

def process_vertical_csv(df: pd.DataFrame, csv_file: str, parser: EnhancedFreightParser) -> List:
    """Procesa CSVs formato vertical como NB SEASTAR con validacion POL/POD"""
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
    
    # Extraer informacion especifica
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
        elif 'CARRIER' in key_upper or 'NAVIERA' in key_upper:
            carrier = value
        elif any(term in key_upper for term in ['O/F', 'OF', 'FREIGHT', 'USD']):
            rates_raw = value
        elif 'FREE' in key_upper and 'TIME' in key_upper:
            free_time_raw = value
        elif 'VALIDITY' in key_upper or 'VALID' in key_upper:
            validity_raw = value
    
    # Si no hay POD especifico, usar San Antonio como default para Chile
    if not pod_raw:
        pod_raw = 'San Antonio'
    
    # Procesar puertos multiples
    pol_list = parser.parse_multiple_ports(pol_raw)
    pod_list = parser.parse_multiple_ports(pod_raw) if pod_raw else ['San Antonio']
    
    # Procesar tarifas
    rates = parser.parse_combined_rates(rates_raw)
    free_days = parser.extract_free_time(free_time_raw)
    
    # Crear documentos para cada combinacion POL-POD
    for pol in (pol_list if pol_list else [pol_raw]):
        for pod in pod_list:
            if not pol or not pod:
                continue
            
            # Normalizar para validacion
            pol_normalized = parser.normalize_port_name(pol)
            pod_normalized = parser.normalize_port_name(pod)
                
            content = f"""COTIZACION MARITIMA FCL - FORMATO VERTICAL v2.1
Archivo: {Path(csv_file).name}
Procesamiento: Parser Vertical con Validacion POL/POD

=== INFORMACION DE RUTA VALIDADA ===
NAVIERA: {carrier}
ORIGEN (POL): {pol}
ORIGEN_NORMALIZADO: {pol_normalized}
DESTINO (POD): {pod}
DESTINO_NORMALIZADO: {pod_normalized}
RUTA_COMPLETA: {pol} → {pod}

=== TARIFAS EN USD ===
Contenedor 20': {rates.get('20', 'TBD')}
Contenedor 40'/40'HC: {rates.get('40', 'TBD')}

=== TERMINOS COMERCIALES ===
FREE DAYS: {free_days}
VALIDEZ: {validity_raw}

=== VALIDACION DE RUTA ===
POL_EXACTO: {pol}
POD_EXACTO: {pod}
POL_SEARCH_TERMS: {pol.lower()} {pol_normalized}
POD_SEARCH_TERMS: {pod.lower()} {pod_normalized}

=== DATOS RAW EXTRAIDOS ===
{json.dumps(data_dict, indent=2)}

=== KEYWORDS EXPANDIDAS PARA BUSQUEDA ===
fcl contenedor maritimo {carrier.lower()} {pol.lower()} {pod.lower()}
tarifa oceanic freight {pol_normalized} {pod_normalized} {parser.normalize_carrier_name(carrier)}
precio costo shipping desde {pol.lower()} hacia {pod.lower()}
ruta {pol.lower()} {pod.lower()} naviera {carrier.lower()}
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": csv_file,
                    "source_name": Path(csv_file).name,
                    "carrier": carrier,
                    "carrier_normalized": parser.normalize_carrier_name(carrier),
                    "pol": pol,
                    "pod": pod,
                    "pol_normalized": pol_normalized,
                    "pod_normalized": pod_normalized,
                    "pol_exact": pol.lower().strip(),
                    "pod_exact": pod.lower().strip(),
                    "rate_20": rates.get('20', 'TBD'),
                    "rate_40": rates.get('40', 'TBD'),
                    "free_days": free_days,
                    "validity": validity_raw,
                    "content_type": "fcl_rate",
                    "route_key": f"{pol_normalized}_to_{pod_normalized}",
                    "route_validation": f"{pol.lower()}|{pod.lower()}",
                    "document_type": "csv_tariff_vertical",
                    "processing_method": "vertical_parser_v2_pol_pod_validation"
                }
            )
            documents.append(doc)
    
    return documents

def process_standard_csv(df: pd.DataFrame, csv_file: str, parser: EnhancedFreightParser) -> List:
    """Procesa CSVs formato estandar horizontal con validacion POL/POD"""
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
        
        # Si no hay POD especifico, usar San Antonio como default
        if not pod or pod == 'nan':
            pod = 'San Antonio'
        
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
        
        # Normalizar puertos para validacion
        pol_normalized = parser.normalize_port_name(pol)
        pod_normalized = parser.normalize_port_name(pod)
        
        content = f"""COTIZACION MARITIMA FCL #{idx + 1} - FORMATO ESTANDAR v2.1
Archivo: {Path(csv_file).name}
Procesamiento: Parser Estandar con Validacion POL/POD

=== INFORMACION DE RUTA VALIDADA ===
NAVIERA: {carrier}
ORIGEN (POL): {pol}
ORIGEN_NORMALIZADO: {pol_normalized}
DESTINO (POD): {pod}
DESTINO_NORMALIZADO: {pod_normalized}
TIEMPO DE TRANSITO: {transit_time} dias
RUTA_COMPLETA: {pol} → {pod}

=== TARIFAS EN USD ===
Contenedor 20': {rate_20 if rate_20 and rate_20 != 'nan' else 'TBD'}
Contenedor 40'/40'HC: {rate_40 if rate_40 and rate_40 != 'nan' else 'TBD'}

=== TERMINOS COMERCIALES ===
FREE DAYS: {free_days}
VALIDEZ: {validity}

=== VALIDACION DE RUTA ===
POL_EXACTO: {pol}
POD_EXACTO: {pod}
POL_SEARCH_TERMS: {pol.lower()} {pol_normalized}
POD_SEARCH_TERMS: {pod.lower()} {pod_normalized}

=== DATOS RAW ===
{row.to_string()}

=== KEYWORDS PARA BUSQUEDA VALIDADA ===
fcl contenedor maritimo {carrier.lower()} {pol.lower()} {pod.lower()}
tarifa oceanic freight precio costo shipping
ruta desde {pol.lower()} hacia {pod.lower()} naviera {carrier.lower()}
origen {pol_normalized} destino {pod_normalized}
"""
        
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
                "pol_normalized": pol_normalized,
                "pod_normalized": pod_normalized,
                "pol_exact": pol.lower().strip(),
                "pod_exact": pod.lower().strip(),
                "rate_20": rate_20 if rate_20 and rate_20 != 'nan' else 'TBD',
                "rate_40": rate_40 if rate_40 and rate_40 != 'nan' else 'TBD',
                "transit_time": transit_time,
                "free_days": free_days,
                "validity": validity,
                "content_type": "fcl_rate",
                "route_key": f"{pol_normalized}_to_{pod_normalized}",
                "route_validation": f"{pol.lower()}|{pod.lower()}",
                "document_type": "csv_tariff_standard",
                "processing_method": "standard_parser_v2_pol_pod_validation"
            }
        )
        documents.append(doc)
    
    return documents

def detect_vertical_format(df: pd.DataFrame) -> bool:
    """Detecta si un CSV esta en formato vertical"""
    if len(df.columns) <= 2 and len(df) > 5:
        content_sample = ' '.join([str(cell) for cell in df.iloc[:, 0].dropna()[:10]])
        vertical_patterns = ['POL:', 'POD:', 'CARRIER:', 'O/F:', 'Free time:', 'Validity:']
        return sum(1 for pattern in vertical_patterns if pattern in content_sample) >= 2
    return False

def enhanced_fcl_csv_processor_v2():
    """Procesador principal de CSVs v2.1 con validacion POL/POD"""
    documents = []
    csv_files = glob.glob(TMP_DIR.as_posix() + "/**/*.csv", recursive=True)
    
    parser = EnhancedFreightParser()
    
    for csv_file in csv_files:
        try:
            # Leer CSV con multiples encodings
            df = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                continue
            
            df.columns = df.columns.str.strip()
            file_name = Path(csv_file).name.upper()
            
            # Detectar formato y procesar con validacion POL/POD
            if 'NB SEASTAR' in file_name or 'SEASTAR' in file_name or detect_vertical_format(df):
                docs = process_vertical_csv(df, csv_file, parser)
            else:
                docs = process_standard_csv(df, csv_file, parser)
            
            documents.extend(docs)
                
        except Exception as e:
            continue
    
    return documents

####################################################################
#            CARGADOR DE DOCUMENTOS v2.1
####################################################################

def enhanced_seemann_document_loader_v2():
    """Cargador de documentos v2.1 con parser avanzado y validacion POL/POD"""
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
        pass
    
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
    
    # Usar procesador CSV v2.1 con validacion POL/POD
    csv_documents = enhanced_fcl_csv_processor_v2()
    documents.extend(csv_documents)
    
    return documents

####################################################################
#            SISTEMA DE BUSQUEDA CON VALIDACION POL/POD v2.1
####################################################################

def generate_query_variations_with_pol_pod(original_query: str, parser: EnhancedFreightParser) -> List[str]:
    """Genera variaciones de consulta enfocadas en POL/POD especificos"""
    variations = [original_query]
    query_lower = original_query.lower()
    
    # Extraer POL/POD de la consulta
    route_info = parser.extract_pol_pod_from_query(original_query)
    
    if route_info:
        pol = route_info.get('pol_normalized', '')
        pod = route_info.get('pod_normalized', '')
        
        # Crear variaciones especificas para la ruta
        if pol and pod:
            variations.extend([
                f"POL {pol} POD {pod}",
                f"origen {pol} destino {pod}",
                f"desde {pol} hacia {pod}",
                f"ruta {pol} {pod}",
                f"{pol} to {pod}",
                f"{pol} → {pod}",
                f"ORIGEN_NORMALIZADO: {pol} DESTINO_NORMALIZADO: {pod}",
                f"POL_EXACTO: {route_info.get('pol_raw', pol)} POD_EXACTO: {route_info.get('pod_raw', pod)}"
            ])
    
    # Agregar terminos especificos de FCL
    variations.extend([
        f"{query_lower} fcl contenedor",
        f"{query_lower} tarifa maritima",
        f"{query_lower} cotizacion"
    ])
    
    return list(set(variations))

def create_enhanced_seemann_retriever_v2(vector_store, search_type="mmr", k=20):
    """Retriever avanzado v2.1 con validacion POL/POD"""
    search_kwargs = {
        "k": k,
        "lambda_mult": 0.1,  # Menos diversidad, mas precision
        "fetch_k": k * 3     # Mas candidatos para filtrar
    }
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever

def multi_query_retriever_with_pol_pod_validation(vector_store, original_query: str, parser: EnhancedFreightParser) -> List:
    """Ejecuta busqueda con validacion estricta de POL/POD"""
    
    # Extraer POL/POD de la consulta
    route_info = parser.extract_pol_pod_from_query(original_query)
    query_pol = route_info.get('pol_normalized', '') if route_info else ''
    query_pod = route_info.get('pod_normalized', '') if route_info else ''
    
    query_variations = generate_query_variations_with_pol_pod(original_query, parser)
    all_docs = []
    seen_docs = set()
    
    retriever = create_enhanced_seemann_retriever_v2(vector_store, k=25)
    
    for query in query_variations:
        try:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                doc_hash = hash(doc.page_content[:200])
                if doc_hash not in seen_docs:
                    
                    # Validar POL/POD si hay informacion de ruta en la consulta
                    if query_pol and query_pod:
                        doc_pol = doc.metadata.get('pol', '')
                        doc_pod = doc.metadata.get('pod', '')
                        
                        # Solo incluir si la ruta coincide
                        if parser.validate_pol_pod_match(doc_pol, doc_pod, query_pol, query_pod):
                            all_docs.append(doc)
                            seen_docs.add(doc_hash)
                    else:
                        # Si no hay info de ruta especifica, incluir todos
                        all_docs.append(doc)
                        seen_docs.add(doc_hash)
        except:
            continue
    
    return all_docs[:30]

def create_enhanced_conversational_chain_v2(retriever, chain_type="stuff"):
    """Chain avanzada v2.1 con retrieval exhaustivo y validacion POL/POD"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para maximizar recuperacion de informacion de tarifarios con validacion POL/POD.

IMPORTANTE v2.1 - VALIDACION POL/POD:
- Si mencionan rutas especificas (Shanghai a San Antonio), mantener POL/POD exactos
- Si preguntan por "opciones" o "alternativas", ampliar busqueda pero validar rutas
- Incluye variaciones de puertos asiaticos (Shanghai, Ningbo, Shenzhen, Qingdao)
- Mantener nombres exactos de navieras y puertos
- Incluye terminos: FCL, contenedor, tarifa, precio, cotizacion, POL, POD

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para busqueda exhaustiva con validacion POL/POD v2.1:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(enhanced_seemann_response_template())
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
        chain_type=chain_type,
        verbose=True,
        return_source_documents=True,
        max_tokens_limit=4000
    )

    return chain, memory

def validate_response_completeness(response_text: str, original_query: str, source_docs: List) -> Dict[str, Any]:
    """Valida completitud de respuesta v2.1 con enfoque en POL/POD"""
    
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
            validation['warnings'].append("Informacion de tarifas limitada")
    
    return validation

def enhance_query_for_completeness(original_query: str, first_response: str) -> str:
    """Mejora query para segunda busqueda con enfoque POL/POD"""
    enhanced_terms = []
    
    if 'comparar' in original_query.lower() or 'opcion' in original_query.lower():
        enhanced_terms.extend(['todas las navieras', 'cotizaciones completas', 'alternativas disponibles'])
    
    if any(port in original_query.lower() for port in ['shanghai', 'china']):
        enhanced_terms.extend(['ningbo', 'shenzhen', 'qingdao', 'puertos china'])
    
    enhanced_query = f"{original_query} {' '.join(enhanced_terms)} incluyendo MSK COSCO CMA CGM PIL ONE validacion POL POD"
    return enhanced_query

def create_enhanced_conversational_chain_v2_2(retriever, chain_type="stuff"):
    """Chain avanzada v2.2 con templates dinámicos"""
    
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Reformula la pregunta para maximizar recuperación de información con detección automática de contenido.

IMPORTANTE v2.2 - DETECCIÓN AUTOMÁTICA:
- El sistema detecta automáticamente el tipo de consulta (FCL, Export, D&D, Comparativo)
- Mantén términos específicos de POL/POD para validación de rutas
- Incluye variaciones para diferentes tipos de documentos
- Preserva context clues para detección de contenido

Historial: {chat_history}
Pregunta: {question}

Pregunta reformulada para búsqueda con detección automática v2.2:""",
    )

    # El template se seleccionará dinámicamente en tiempo de ejecución
    answer_prompt = ChatPromptTemplate.from_template(enhanced_seemann_response_template())
    
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
        chain_type=chain_type,
        verbose=True,
        return_source_documents=True,
        max_tokens_limit=4000
    )

    return chain, memory