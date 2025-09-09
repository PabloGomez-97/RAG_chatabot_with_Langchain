import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG SEEMANN GROUP - LCL MARITIME SYSTEM
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado en tarifas LCL marítimas de Seemann Group.

🚢 **SISTEMA ESPECIALIZADO LCL:**
- Consultas de tarifas LCL por puerto de origen y destino
- Información completa de costos por tonelada/m³
- Detalles de tiempos de tránsito y frecuencias
- Información de agentes locales y servicios
- Costos adicionales y observaciones importantes

💡 **Ejemplos de consultas:**
- "¿Cuánto cuesta envío LCL desde Shanghai a San Antonio?"
- "Necesito tarifa desde Buenos Aires a Valparaíso"
- "¿Qué opciones tengo desde Europa a Chile?"
- "Muéstrame todas las rutas desde Asia"

📋 **Información que puedo proporcionar:**
- Costo por tonelada/metro cúbico
- Tarifa mínima aplicable
- Tiempo de tránsito aproximado
- Frecuencia de servicios
- Agentes locales
- Costos adicionales (DDT, VGM, etc.)
- Observaciones especiales por ruta
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
#            MAPEO DE PUERTOS PARA LCL
####################################################################

PORT_ALIASES = {
    # Puertos de América
    'buenos aires': ['buenos aires', 'bue', 'argentina buenos aires', 'ba'],
    'santos': ['santos', 'sao', 'brasil santos', 'santos brasil'],
    'rio de janeiro': ['rio de janeiro', 'rio', 'brasil rio', 'gig'],
    'callao': ['callao', 'cao', 'peru callao', 'lima'],
    'guayaquil': ['guayaquil', 'gye', 'ecuador guayaquil'],
    'san antonio': ['san antonio', 'sap', 'chile san antonio', 'val-sap'],
    'valparaiso': ['valparaiso', 'val', 'chile valparaiso'],
    
    # Puertos de Europa
    'antwerp': ['antwerp', 'anr', 'belgium antwerp', 'amberes'],
    'rotterdam': ['rotterdam', 'rtm', 'netherlands rotterdam', 'holanda'],
    'hamburg': ['hamburg', 'ham', 'germany hamburg', 'alemania'],
    'le havre': ['le havre', 'leh', 'france le havre', 'francia'],
    'valencia': ['valencia', 'vlc', 'spain valencia', 'españa'],
    
    # Puertos de América del Norte
    'miami': ['miami', 'mia', 'usa miami', 'florida'],
    'new york': ['new york', 'nyc', 'usa new york', 'nueva york'],
    'los angeles': ['los angeles', 'lax', 'usa los angeles'],
    'chicago': ['chicago', 'chi', 'usa chicago'],
    'houston': ['houston', 'hou', 'usa houston'],
    
    # Puertos de Asia
    'shanghai': ['shanghai', 'sha', 'china shanghai'],
    'ningbo': ['ningbo', 'ngb', 'china ningbo'],
    'shenzhen': ['shenzhen', 'szn', 'china shenzhen'],
    'qingdao': ['qingdao', 'tao', 'china qingdao'],
    'busan': ['busan', 'pus', 'korea busan', 'corea'],
    'sydney': ['sydney', 'syd', 'australia sydney'],
    'melbourne': ['melbourne', 'mel', 'australia melbourne'],
}

COUNTRY_ALIASES = {
    'argentina': ['argentina', 'arg'],
    'brasil': ['brasil', 'brazil', 'bra'],
    'peru': ['peru', 'per'],
    'ecuador': ['ecuador', 'ecu'],
    'chile': ['chile', 'chi'],
    'belgium': ['belgium', 'bel', 'belgica'],
    'netherlands': ['netherlands', 'net', 'holanda'],
    'germany': ['germany', 'ger', 'alemania'],
    'france': ['france', 'fra', 'francia'],
    'spain': ['spain', 'esp', 'españa'],
    'united states': ['united states', 'usa', 'estados unidos'],
    'china': ['china', 'chn'],
    'australia': ['australia', 'aus'],
    'korea': ['korea', 'kor', 'corea'],
}

####################################################################
#            FUNCIONES DE DETECCIÓN DE CONSULTAS LCL
####################################################################

def detect_lcl_query_type(query: str) -> str:
    """Detecta el tipo específico de consulta LCL"""
    query_lower = query.lower()
    
    # Consulta por ruta específica (origen -> destino)
    if any(pattern in query_lower for pattern in ['desde', 'de', 'from']) and \
       any(pattern in query_lower for pattern in ['hacia', 'hasta', 'a', 'to']):
        return 'route_specific'
    
    # Consulta por región
    if any(region in query_lower for region in ['europa', 'asia', 'america', 'norteamerica']):
        return 'region_query'
    
    # Consulta comparativa
    if any(term in query_lower for term in ['opciones', 'alternativas', 'comparar', 'todas']):
        return 'comparative'
    
    # Consulta por país específico
    for country in COUNTRY_ALIASES.keys():
        if country in query_lower:
            return 'country_specific'
    
    return 'general'

def extract_ports_from_query(query: str) -> dict:
    """Extrae puertos origen y destino de la consulta"""
    query_lower = query.lower()
    
    # Patrones para detectar origen y destino
    patterns = [
        r'desde\s+([^a]+?)\s+(?:hacia|hasta|a)\s+([^?]+)',
        r'de\s+([^a]+?)\s+a\s+([^?]+)',
        r'from\s+([^to]+?)\s+to\s+([^?]+)',
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            origin_raw = match.group(1).strip()
            destination_raw = match.group(2).strip()
            
            # Normalizar puertos
            origin_normalized = normalize_port_name(origin_raw)
            destination_normalized = normalize_port_name(destination_raw)
            
            return {
                'origin_raw': origin_raw,
                'destination_raw': destination_raw,
                'origin_normalized': origin_normalized,
                'destination_normalized': destination_normalized,
                'has_route': True
            }
    
    return {'has_route': False}

def normalize_port_name(port_text: str) -> str:
    """Normaliza nombres de puertos"""
    if not port_text:
        return ""
    
    port_lower = port_text.lower().strip()
    
    # Buscar coincidencia directa
    for canonical, aliases in PORT_ALIASES.items():
        if port_lower in aliases or any(alias in port_lower for alias in aliases):
            return canonical
    
    return port_lower

####################################################################
#            TEMPLATE ESPECIALIZADO PARA LCL
####################################################################

def get_lcl_response_template():
    """Template especializado para respuestas LCL marítimas"""
    return """Eres un especialista en tarifas LCL (Less than Container Load) marítimas de SEEMANN GROUP.

CONTEXTO IMPORTANTE DEL NEGOCIO LCL:
- TODOS los registros del tarifario son para importación HACIA CHILE
- El destino implícito SIEMPRE es Chile (puertos San Antonio o Valparaíso)
- Los usuarios consultan rutas como "desde Shanghai a Chile" o simplemente "desde Shanghai"
- NO existe columna POD porque el destino siempre es Chile

INSTRUCCIONES PARA CONSULTAS LCL:

1. ANÁLISIS DE LA CONSULTA:
   - Identifica puerto origen exacto
   - DESTINO SIEMPRE ES CHILE (San Antonio/Valparaíso)
   - Busca en las regiones correctas (AMERICA, EUROPA, NORTEAMERICA, ASIA)
   - Valida que la ruta origen → Chile existe en nuestro tarifario

2. INFORMACIÓN OBLIGATORIA A MOSTRAR:
   - PUERTO CARGA (origen exacto)
   - PAÍS de origen
   - TON/M3 USD (tarifa por tonelada o metro cúbico)
   - MÍNIMO (tarifa mínima aplicable)
   - T/T APROX (tiempo de tránsito aproximado)
   - FREC (frecuencia del servicio)
   - SERVICIO (tipo: DIRECTO o VÍA otro puerto)
   - AGENTE (agente local)
   - OTROS (costos adicionales como DDT, VGM, etc.)
   - OBSERVACIONES (condiciones especiales)

3. FORMATO DE RESPUESTA OBLIGATORIO:

**🚢 TARIFAS LCL - RUTA: [PUERTO_ORIGEN] → CHILE**

📋 **INFORMACIÓN DETALLADA:**
- **Puerto de Origen:** [PUERTO_EXACTO]
- **País Origen:** [PAÍS]
- **Destino:** Chile (San Antonio/Valparaíso)
- **Tarifa:** [TON/M3] USD por tonelada/m³
- **Mínimo:** [MÍNIMO] USD
- **Tiempo Tránsito:** [T/T] días
- **Frecuencia:** [FREC]
- **Tipo Servicio:** [DIRECTO/VÍA X]
- **Agente Local:** [AGENTE]

💰 **COSTOS ADICIONALES:**
[Detallar OTROS costos como DDT USD 50.00, VGM USD 7.50, etc.]

⚠️ **OBSERVACIONES IMPORTANTES:**
[Incluir todas las observaciones especiales, restricciones, validez de tarifas, etc.]

4. PARA CONSULTAS COMO "desde Shanghai" o "tarifa Shanghai":
   - Entender automáticamente que el destino es Chile
   - Mostrar información completa de la ruta Shanghai → Chile
   - Aclarar que todas las tarifas son para importación hacia Chile

5. PARA CONSULTAS COMPARATIVAS:
   - Mostrar TODAS las opciones desde la región consultada hacia Chile
   - Ordenar por precio o tiempo según consulta
   - Incluir análisis de mejores opciones hacia Chile

6. SI NO EXISTE LA RUTA HACIA CHILE:
   - Informar claramente que no está disponible
   - Sugerir puertos alternativos en la misma región
   - Mostrar rutas más cercanas disponibles hacia Chile

CONTEXTO CONVERSACIÓN: {chat_history}
DOCUMENTOS TARIFARIO: {context}
CONSULTA CLIENTE: {question}

RESPUESTA LCL ESPECIALIZADA (Destino siempre Chile):"""

####################################################################
#            FUNCIÓN DE VALIDACIÓN DE DOCUMENTOS LCL
####################################################################

def validate_lcl_document_relevance(query: str, documents: list) -> list:
    """Filtra documentos relevantes para consultas LCL específicas"""
    
    route_info = extract_ports_from_query(query)
    query_type = detect_lcl_query_type(query)
    
    if not route_info.get('has_route'):
        # Si no hay ruta específica, devolver todos los documentos
        return documents
    
    origin = route_info.get('origin_normalized', '')
    destination = route_info.get('destination_normalized', '')
    
    relevant_docs = []
    
    for doc in documents:
        doc_content = doc.page_content.lower()
        
        # Buscar coincidencia de puertos en el contenido
        origin_match = False
        destination_match = False
        
        if origin:
            # Buscar puerto origen en el contenido
            if origin in doc_content or any(alias in doc_content for alias in PORT_ALIASES.get(origin, [])):
                origin_match = True
        
        if destination:
            # Para destino, buscar en contexto regional apropiado
            if destination in doc_content or any(alias in doc_content for alias in PORT_ALIASES.get(destination, [])):
                destination_match = True
        
        # Si encontramos coincidencias relevantes, incluir documento
        if origin_match or destination_match or query_type == 'comparative':
            relevant_docs.append(doc)
    
    return relevant_docs if relevant_docs else documents  # Fallback


def detect_company_from_excel(file_path: str) -> dict:
    """Detecta la empresa y estructura del archivo Excel
    
    Explicación:
    - Lee solo las primeras 5 filas del Excel para identificar la empresa
    - Busca en celda A1 el identificador (PLUSCARGO vs TRAFICO: X)
    - Retorna configuración específica de cada empresa
    """
    
    try:
        import pandas as pd
        
        # Leer solo primeras filas para performance
        df_first = pd.read_excel(file_path, nrows=5, header=None)
        
        # Verificar celda A1
        cell_a1 = str(df_first.iloc[0, 0]).strip().upper() if not df_first.empty else ""
        
        # Detectar empresa por identificador A1
        if cell_a1 == "PLUSCARGO":
            return {
                "company": "PLUSCARGO",
                "structure_type": "pluscargo_lcl",
                "header_row": 9,  # Fila 10 en Excel
                "has_pod_column": True,
                "identification_cell": "A1",
                "identification_value": "PLUSCARGO"
            }
        elif cell_a1 == "ECU":
            return {
                "company": "ECU",
                "structure_type": "ecu_lcl",
                "header_row": 1,  # Fila 2 en Excel
                "has_pod_column": True,
                "identification_cell": "A1",
                "identification_value": "ECU"
            }
        elif "TRAFICO:" in cell_a1 or "TRÁFICO:" in cell_a1:
            return {
                "company": "MSL",
                "structure_type": "msl_lcl", 
                "header_row": 3,  # Varía entre 3-4 según hoja
                "has_pod_column": False,
                "identification_cell": "A1",
                "identification_value": cell_a1
            }
        else:
            # Si A1 no tiene identificador, buscar en otras celdas
            for row in range(min(5, len(df_first))):
                for col in range(min(3, len(df_first.columns))):
                    cell_value = str(df_first.iloc[row, col]).strip().upper()
                    
                    if "PLUSCARGO" in cell_value:
                        return {
                            "company": "PLUSCARGO",
                            "structure_type": "pluscargo_lcl",
                            "header_row": 9,
                            "has_pod_column": True,
                            "identification_cell": f"{chr(65+col)}{row+1}",
                            "identification_value": cell_value
                        }
                    elif "MSL" in cell_value or "SEEMANN" in cell_value:
                        return {
                            "company": "MSL", 
                            "structure_type": "msl_lcl",
                            "header_row": 3,
                            "has_pod_column": False,
                            "identification_cell": f"{chr(65+col)}{row+1}",
                            "identification_value": cell_value
                        }
        
        # Default si no detecta empresa específica
        return {
            "company": "UNKNOWN",
            "structure_type": "generic",
            "header_row": 0,
            "has_pod_column": False,
            "identification_cell": "A1",
            "identification_value": cell_a1
        }
        
    except Exception as e:
        return {
            "company": "ERROR",
            "structure_type": "error",
            "error": str(e)
        }

def get_company_column_mapping(company: str) -> dict:
    """Retorna mapeo de columnas según la empresa detectada
    
    Explicación:
    - Cada empresa tiene estructura de columnas diferente
    - MSL: PUERTO CARGA, PAIS, TON/M3, etc.
    - PLUSCARGO: País, POL, POD, CBM, etc.
    - Retorna diccionario con mapeo flexible para encontrar columnas
    """
    
    if company == "MSL":
        return {
            "expected_columns": [
                "PUERTO CARGA", "PAIS", "TON / M3 usd", "MINIMO", 
                "T / T APROX.", "FREC.", "OTROS", "SERVICIO", "AGENTE", "OBSERVACIONES"
            ],
            "column_mapping": {
                "puerto_origen": ["PUERTO CARGA", "PUERTO", "ORIGEN"],
                "pais": ["PAIS", "PAÍS", "COUNTRY"],
                "tarifa": ["TON / M3", "TARIFA", "USD"],
                "minimo": ["MINIMO", "MÍNIMO", "MIN"],
                "transito": ["T / T APROX", "TRANSITO", "TT", "APROX"],
                "frecuencia": ["FREC", "FRECUENCIA", "FREQUENCY"],
                "otros": ["OTROS", "OTHER", "ADICIONAL"],
                "servicio": ["SERVICIO", "SERVICE"],
                "agente": ["AGENTE", "AGENT"],
                "observaciones": ["OBSERVACIONES", "OBS", "REMARKS"]
            }
        }
    
    elif company == "PLUSCARGO":
        return {
            "expected_columns": [
                "País", "POL", "POD", "De 0 a 15.00 cbm", "Min", 
                "Frecuencia", "Servicio", "T/T total aprox", "Modo", "Agente"
            ],
            "column_mapping": {
                "pais": ["PAÍS", "PAIS", "COUNTRY"],
                "puerto_origen": ["POL", "PUERTO ORIGEN", "ORIGIN"],
                "puerto_destino": ["POD", "PUERTO DESTINO", "DESTINATION"],
                "tarifa": ["CBM", "W/M", "TARIFA", "USD"],
                "minimo": ["MIN", "MINIMO", "MÍNIMO"],
                "frecuencia": ["FRECUENCIA", "FREQUENCY", "FREC"],
                "servicio": ["SERVICIO", "SERVICE"],
                "transito": ["T/T", "TRANSITO", "APROX"],
                "modo": ["MODO", "MODE"],
                "agente": ["AGENTE", "AGENT"],
                "bl_fee": ["BL FEE", "BLFEE", "FEE"]
            }
        }
    
    elif company == "ECU":
        return {
            "expected_columns": [
                "REGION", "COUNTRY", "FIRST LEG POL", "POL", "RUTA", "POD", 
                "SERVICIO", "CUR", "TON / M3", "BL", "TT estimado", "Validity ETD"
            ],
            "column_mapping": {
                "region": ["REGION"],
                "pais": ["COUNTRY", "PAIS", "PAÍS"],
                "first_leg_pol": ["FIRST LEG POL"],
                "puerto_origen": ["POL", "PUERTO ORIGEN"],
                "ruta": ["RUTA", "ROUTE"],
                "puerto_destino": ["POD", "PUERTO DESTINO"],
                "servicio": ["SERVICIO", "SERVICE"],
                "moneda": ["CUR", "CURRENCY"],
                "tarifa": ["TON / M3", "TON/M3", "CBM"],
                "bl_fee": ["BL", "BL FEE"],
                "transito": ["TT", "FINAL TT", "TT ESTIMADO"],
                "validez": ["VALIDITY", "ETD"]
            }
        }
    
    else:
        # Mapeo genérico para empresas no reconocidas
        return {
            "expected_columns": [],
            "column_mapping": {
                "puerto_origen": ["POL", "PUERTO", "ORIGEN", "ORIGIN"],
                "puerto_destino": ["POD", "DESTINO", "DESTINATION"], 
                "pais": ["PAIS", "PAÍS", "COUNTRY"],
                "tarifa": ["TARIFA", "USD", "PRECIO", "RATE"],
                "frecuencia": ["FRECUENCIA", "FREC", "FREQUENCY"],
                "servicio": ["SERVICIO", "SERVICE"]
            }
        }

def detect_query_company_preference(query: str) -> str:
    """Detecta si el usuario prefiere una empresa específica en su consulta
    
    Explicación:
    - Analiza la consulta del usuario
    - Si menciona "MSL" o "Seemann" -> filtra solo MSL
    - Si menciona "PLUSCARGO" -> filtra solo PLUSCARGO  
    - Si no especifica -> muestra todas las empresas
    """
    query_lower = query.lower()
    
    if any(term in query_lower for term in ['msl', 'seemann']):
        return 'MSL'
    elif any(term in query_lower for term in ['pluscargo', 'plus cargo']):
        return 'PLUSCARGO'
    else:
        return 'ALL'  # Mostrar todas las empresas

def get_multi_company_lcl_template():
    """Template que maneja múltiples empresas LCL
    
    Explicación:
    - Reemplaza el template original get_lcl_response_template()
    - Entiende diferencias entre MSL y PLUSCARGO
    - Genera respuestas adaptadas según empresa(s) detectada(s)
    """
    return """Eres un especialista en tarifas LCL (Less than Container Load) marítimas que maneja múltiples empresas.

EMPRESAS EN EL SISTEMA:
- MSL (Seemann Group): Destino implícito Chile, sin columna POD
- PLUSCARGO: Destino explícito (San Antonio/Valparaíso), con columna POD
- ECU Worldwide: Estructura detallada con FIRST LEG POL y códigos específicos

CONTEXTO IMPORTANTE DEL NEGOCIO LCL:
- MSL: TODOS los registros son para importación HACIA CHILE (destino implícito)
- PLUSCARGO: Tienen columna POD explícita con puerto destino específico
- Adapta respuesta según empresa(s) detectada(s) en los documentos

INSTRUCCIONES MULTI-EMPRESA:

1. IDENTIFICACIÓN AUTOMÁTICA:
   - El sistema detecta automáticamente la empresa
   - Adapta respuesta según empresa detectada en documentos

2. FORMATO DE RESPUESTA SEGÚN EMPRESA DETECTADA:

Si solo hay documentos MSL:
**🚢 TARIFAS LCL - MSL (SEEMANN GROUP)**
**Ruta: [PUERTO_ORIGEN] → Chile (San Antonio/Valparaíso)**
- **Tarifa TON/M³:** [TARIFA] USD
- **Mínimo:** [MINIMO] USD  
- **Tiempo Tránsito:** [TT] días
- **Frecuencia:** [FREC]
- **Servicio:** [DIRECTO/VÍA]
- **Agente:** [AGENTE]
- **Costos Adicionales:** [DDT, VGM, etc.]
- **Observaciones:** [OBS]

Si solo hay documentos PLUSCARGO:
**🚢 TARIFAS LCL - PLUSCARGO**
**Ruta: [POL] → [POD]**
- **Tarifa CBM/W/M:** [TARIFA] USD
- **Mínimo:** [MIN] USD
- **BL Fee:** [FEE] USD (si aplica)
- **Tiempo Tránsito:** [TT] días
- **Frecuencia:** [FREC]
- **Servicio:** [TIPO]
- **Agente:** [AGENTE]

Si hay documentos ECU:
**🚢 TARIFAS LCL - ECU WORLDWIDE**
**Ruta: [FIRST_LEG_POL] → [RUTA] → [POD]**
- **Región:** [REGION]
- **Tarifa TON/M³:** [MONEDA] [TARIFA]
- **BL Fee:** [BL_FEE] (si aplica)
- **Tiempo Tránsito:** [TT] días
- **Servicio:** [ECU CONSOL]
- **Validez:** [VALIDITY_ETD]

Si hay documentos de AMBAS empresas:
**🔄 COMPARACIÓN MULTI-EMPRESA DISPONIBLE**

**🏢 MSL (SEEMANN GROUP):**
[Información MSL como arriba]

**🏢 PLUSCARGO:**
[Información PLUSCARGO como arriba]

**💡 COMPARACIÓN:**
- MSL: Tarifa TON/M³ + costos adicionales
- PLUSCARGO: Tarifa CBM/W/M + BL Fee
- [Análisis de cuál conviene más]

3. MANEJO DE DESTINOS POR EMPRESA:
   - MSL: Siempre aclarar "destino Chile (implícito)"
   - PLUSCARGO: Mostrar POD específico del tarifario

CONTEXTO CONVERSACIÓN: {chat_history}
DOCUMENTOS MULTI-EMPRESA: {context}
CONSULTA CLIENTE: {question}

RESPUESTA ESPECIALIZADA MULTI-EMPRESA:"""