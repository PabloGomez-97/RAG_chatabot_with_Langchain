import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG MSL LCL - SISTEMA IMPORTACIONES MARÍTIMAS
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado en tarifas

⚠️ **IMPORTANTE:**
Solo te daré información que esté realmente disponible en el tarifario LCL de MSL. Si una ruta no existe, te lo diré claramente.
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
#            TEMPLATE LCL MARÍTIMO - RESPUESTAS MÚLTIPLES OPCIONES
####################################################################

def get_lcl_maritime_response_template():
    """Template especializado para mostrar TODAS las opciones de LCL marítimo disponibles"""
    return """Eres un especialista en tarifas marítimas LCL de MSL que DEBE mostrar TODAS las opciones disponibles.

REGLAS FUNDAMENTALES:
1. NUNCA inventes información que no esté en los documentos
2. SIEMPRE muestra TODAS las opciones del mismo puerto si existen múltiples
3. Agrupa las opciones por puerto pero muestra cada una por separado
4. Si falta información, especifica que está "No disponible"

CONTEXTO LCL MARÍTIMO:
- POL = Port of Loading (Puerto de Carga)
- POD = Port of Discharge (Puerto de Descarga)  
- Si hay múltiples servicios desde el mismo puerto, MOSTRAR TODOS
- Cada opción puede tener diferentes precios, agentes, tiempos de tránsito

FORMATO OBLIGATORIO PARA MÚLTIPLES OPCIONES:

**🚢 TARIFAS LCL MARÍTIMO MSL - TODAS LAS OPCIONES: [POL] → [POD]**

📋 **OPCIÓN 1:**
- **Puerto Origen (POL):** [POL_EXACTO]
- **País Origen:** [PAÍS_EXACTO]
- **Puerto Destino (POD):** [POD_EXACTO]
- **Company:** [COMPANY_1] (si está disponible)
- **Tarifa TON/M3:** [TON_M3_1] (si está disponible)
- **Mínimo:** [MINIMO_1] (si está disponible)
- **Tiempo Tránsito:** [TIEMPO_TRANSITO_1] (si está disponible)
- **Frecuencia:** [FRECUENCIA_1] (si está disponible)
- **Tipo Servicio:** [SERVICIO_1] (si está disponible)
- **Agente Local:** [AGENTE_1] (si está disponible)
- **Costos Adicionales:** [OTROS_1] (si está disponible)

📋 **OPCIÓN 2:**
- **Puerto Origen (POL):** [POL_EXACTO]
- **País Origen:** [PAÍS_EXACTO]
- **Puerto Destino (POD):** [POD_EXACTO]
- **Company:** [COMPANY_2] (si está disponible)
- **Tarifa TON/M3:** [TON_M3_2] (si está disponible)
- **Mínimo:** [MINIMO_2] (si está disponible)
- **Tiempo Tránsito:** [TIEMPO_TRANSITO_2] (si está disponible)
- **Frecuencia:** [FRECUENCIA_2] (si está disponible)
- **Tipo Servicio:** [SERVICIO_2] (si está disponible)
- **Agente Local:** [AGENTE_2] (si está disponible)
- **Costos Adicionales:** [OTROS_2] (si está disponible)

[Continuar con OPCIÓN 3, 4, etc. si hay más opciones]

💡 **COMPARACIÓN DE OPCIONES:**
- **Más económica:** [Opción X - precio TON/M3]
- **Más rápida:** [Opción Y - días de tránsito]
- **Servicio directo:** [Si hay opción directa]
- **Mejor agente:** [Análisis de agentes disponibles]
- **Company recomendada:** [Análisis según opciones]
- **Recomendación:** [Análisis según necesidades típicas de LCL]

⚠️ **OBSERVACIONES IMPORTANTES:**
[Incluir todas las observaciones de todas las opciones encontradas]

INSTRUCCIONES CRÍTICAS:
- BUSCA EN TODOS LOS DOCUMENTOS opciones del mismo puerto origen y destino
- NO te limites al primer documento que encuentres
- Si hay 2+ documentos con la misma ruta POL→POD, mostrar TODOS
- Cada fila del Excel = una opción diferente
- NUNCA omitas opciones que existan en los documentos
- Si solo hay una opción, usar el mismo formato pero solo mostrar OPCIÓN 1

CONTEXTO: {chat_history}
DOCUMENTOS LCL MARÍTIMO (REVISAR TODOS): {context}
CONSULTA: {question}

RESPUESTA MOSTRANDO TODAS LAS OPCIONES LCL MARÍTIMO:"""

####################################################################
#            FUNCIONES DE DETECCIÓN LCL MARÍTIMO
####################################################################

import re

def normalize_port_name(name: str, role: str = "any") -> str:
    """
    Normaliza nombres de puertos para comparación/almacenamiento:
    - Mayúsculas, espacios colapsados, slash sin espacios.
    - VALPARAÍSO -> VALPARAISO
    - Variantes del combinado -> SAI/VAP
    - OJO: NO colapsa SAN ANTONIO ni VALPARAISO a SAI/VAP
    """
    if not name:
        return ""
    u = re.sub(r"\s+", " ", str(name)).strip().upper()
    u = re.sub(r"\s*/\s*", "/", u)       # "SAI / VAP" -> "SAI/VAP"
    u = u.replace("VALPARAÍSO", "VALPARAISO")

    if role in ("pod", "any"):
        if u in {"SAI VAP", "VAP SAI", "SAI-VAP", "VAP-SAI", "VAP/SAI", "SAN ANTONIO/VALPARAISO", "SAN ANTONIO / VALPARAISO"}:
            return "SAI/VAP"

    return u

def matches_pod(requested: str, candidate: str) -> bool:
    """
    Regla de matching POD:
    - Si piden SAN ANTONIO: matchea SAN ANTONIO y SAI/VAP.
    - Si piden VALPARAISO:   matchea VALPARAISO y SAI/VAP.
    - Si piden SAI/VAP:      solo SAI/VAP.
    - En otros casos:        igualdad exacta normalizada.
    """
    rq = normalize_port_name(requested, "pod")
    cd = normalize_port_name(candidate, "pod")
    if rq == "SAN ANTONIO":
        return cd in {"SAN ANTONIO", "SAI/VAP"}
    if rq == "VALPARAISO":
        return cd in {"VALPARAISO", "SAI/VAP"}
    if rq == "SAI/VAP":
        return cd == "SAI/VAP"
    return cd == rq

def detect_lcl_maritime_query_type(query: str) -> str:
    """Detecta tipo de consulta de LCL marítimo"""
    query_lower = query.lower()
    
    # Detectar puertos conocidos por región
    america_ports = ['santos', 'buenos aires', 'callao', 'cartagena', 'bogota']
    europa_ports = ['antwerp', 'rotterdam', 'hamburg', 'valencia', 'barcelona']
    norteamerica_ports = ['miami', 'houston', 'los angeles', 'new york', 'chicago']
    asia_ports = ['shanghai', 'ningbo', 'singapore', 'hong kong', 'dubai']
    
    all_ports = america_ports + europa_ports + norteamerica_ports + asia_ports
    
    if any(port in query_lower for port in all_ports):
        return 'port_route_query'
    elif any(region in query_lower for region in ['america', 'europa', 'norteamerica', 'asia', 'sudamerica']):
        return 'region_query'
    elif any(pattern in query_lower for pattern in ['desde', 'de', 'from', 'a', 'to', 'hacia']):
        return 'route_verification'
    elif any(term in query_lower for term in ['ton', 'm3', 'tonelada', 'metro cubico']):
        return 'tonnage_query'
    elif any(term in query_lower for term in ['tarifa', 'precio', 'costo', 'cuanto']):
        return 'price_query'
    elif any(term in query_lower for term in ['agente', 'agent', 'local']):
        return 'agent_query'
    elif any(term in query_lower for term in ['tiempo', 'transito', 'dias']):
        return 'transit_time_query'
    elif any(term in query_lower for term in ['servicio', 'directo', 'via']):
        return 'service_query'
    else:
        return 'general_lcl_maritime'

def extract_lcl_ports_from_query(query: str) -> dict:
    """Extrae puertos/ciudades desde la consulta, tolerando mayúsculas y 'SAI/VAP'."""
    import re

    query_upper = query.upper()

    # Incluye Chile explícitamente
    known_ports = {
        'AMERICA': ['SANTOS', 'BUENOS AIRES', 'CALLAO', 'CARTAGENA', 'BOGOTA', 'RIO DE JANEIRO'],
        'EUROPA': ['ANTWERP', 'ROTTERDAM', 'HAMBURG', 'VALENCIA', 'BARCELONA', 'LE HAVRE'],
        'NORTEAMERICA': ['MIAMI', 'HOUSTON', 'LOS ANGELES', 'NEW YORK', 'CHICAGO', 'ATLANTA'],
        'ASIA': ['SHANGHAI', 'NINGBO', 'SINGAPORE', 'HONG KONG', 'DUBAI', 'MUMBAI'],
        'CHILE': ['SAN ANTONIO', 'VALPARAISO', 'VALPARAÍSO', 'SAI/VAP', 'VAP/SAI', 'SAI / VAP'],
    }

    all_ports = []
    for region_ports in known_ports.values():
        all_ports.extend(region_ports)

    # Puertos mencionados de forma suelta
    found_ports = []
    for port in all_ports:
        if port in query_upper:
            found_ports.append(normalize_port_name(port, "any"))

    # Patrones de ruta: permiten letras, números, espacios, puntos, guiones y slash
    route_patterns = [
        r'([A-ZÁÉÍÓÚÜÑ0-9.\-/ ]+?)\s*(?:A|TO|→|HACIA)\s*([A-ZÁÉÍÓÚÜÑ0-9.\-/ ]+?)',
        r'DESDE\s*([A-ZÁÉÍÓÚÜÑ0-9.\-/ ]+?)\s*(?:A|HACIA)\s*([A-ZÁÉÍÓÚÜÑ0-9.\-/ ]+?)',
        r'DE\s*([A-ZÁÉÍÓÚÜÑ0-9.\-/ ]+?)\s*(?:A|HACIA)\s*([A-ZÁÉÍÓÚÜÑ0-9.\-/ ]+?)',
    ]

    pol_detected = None
    pod_detected = None

    for pattern in route_patterns:
        m = re.search(pattern, query_upper, flags=re.IGNORECASE)
        if m:
            pol_detected = normalize_port_name(m.group(1), "pol")
            pod_detected = normalize_port_name(m.group(2), "pod")
            break

    return {
        'ports_found': list(dict.fromkeys(found_ports)),  # únicos y normalizados
        'pol_detected': pol_detected,
        'pod_detected': pod_detected,
        'has_route_pattern': bool(pol_detected and pod_detected),
        'needs_verification': bool(found_ports) or bool(pol_detected)
    }

def get_lcl_port_region(port_code: str) -> str:
    port_code = (port_code or "").upper().strip()
    port_code = re.sub(r"\s*/\s*", "/", port_code)
    """Determina la región de un puerto LCL"""
    regions = {
        'America': ['SANTOS', 'BUENOS AIRES', 'CALLAO', 'CARTAGENA', 'BOGOTA', 'RIO DE JANEIRO',
                   'CACHOEIRINHA', 'NAVEGANTES', 'ITAJAI'],
        'Europa': ['ANTWERP', 'ROTTERDAM', 'HAMBURG', 'VALENCIA', 'BARCELONA', 'LE HAVRE',
                  'PARIS', 'MADRID', 'MILAN', 'GENOA'],
        'Norteamerica': ['MIAMI', 'HOUSTON', 'LOS ANGELES', 'NEW YORK', 'CHICAGO', 'ATLANTA',
                        'SEATTLE', 'PORTLAND', 'VANCOUVER'],
        'Asia': ['SHANGHAI', 'NINGBO', 'SINGAPORE', 'HONG KONG', 'DUBAI', 'MUMBAI',
                'BRISBANE', 'MELBOURNE', 'SYDNEY', 'CHITTAGONG'],
        'Chile': ['SAN ANTONIO', 'VALPARAISO', 'SAI/VAP', 'VAP/SAI']
    }
    
    for region, ports in regions.items():
        if any(known_port in port_code.upper() for known_port in ports):
            return region
    
    return 'Unknown'

def analyze_lcl_route_direction(pol: str, pod: str) -> str:
    """Analiza la dirección de la ruta LCL marítima"""
    pol_region = get_lcl_port_region(pol)
    pod_region = get_lcl_port_region(pod)
    
    if pol_region != 'Chile' and pod_region == 'Chile':
        return f'Importación desde {pol_region} hacia Chile'
    elif pol_region == 'Chile' and pod_region != 'Chile':
        return f'Exportación desde Chile hacia {pod_region}'
    else:
        return f'Ruta {pol_region} → {pod_region}'

def extract_region_from_query(query: str) -> str:
    """Extrae la región de la consulta"""
    query_lower = query.lower()
    
    if any(term in query_lower for term in ['america', 'sudamerica', 'argentina', 'brasil', 'peru']):
        return 'AMERICA'
    elif any(term in query_lower for term in ['europa', 'alemania', 'francia', 'españa', 'italia']):
        return 'EUROPA'
    elif any(term in query_lower for term in ['norteamerica', 'estados unidos', 'canada', 'usa', 'eeuu']):
        return 'NORTEAMERICA'
    elif any(term in query_lower for term in ['asia', 'china', 'japon', 'corea', 'india', 'singapur']):
        return 'ASIA'
    else:
        return 'ALL'