import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG TRANSCHINACOSCODTASIAECUETC - SISTEMA FCL MARÍTIMO
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado en tarifas marítimas FCL (Full Container Load).

🚢 **SISTEMA FCL MARÍTIMO - CONSULTAS DE CONTENEDORES:**
- Consulta de tarifas entre puertos (POL → POD)
- Información de carriers y servicios navieros
- Tarifas por tipo de contenedor (20GP, 40GP, 40HQ, 40NOR)
- Rutas de importación desde Asia hacia Sudamérica
- Free time y condiciones especiales

💡 **Ejemplos de consultas:**
- "¿Cuál es la tarifa de SHANGHAI a SAI/VAL?"
- "¿Qué opciones hay desde China a Chile?"
- "¿Cuánto cuesta un contenedor 40HQ desde NINGBO a Chile?"
- "¿Qué carriers operan desde BASE PORTS a SAI/VAL?"

⚠️ **IMPORTANTE:**
Solo te daré información que esté realmente disponible en el tarifario FCL marítimo.
Si una ruta no existe, te lo diré claramente.
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
#            TEMPLATE FCL MARÍTIMO - RESPUESTAS MÚLTIPLES OPCIONES
####################################################################

def get_maritime_fcl_response_template():
    """Template especializado para mostrar TODAS las opciones de FCL marítimo disponibles"""
    return """Eres un especialista en tarifas marítimas FCL que DEBE mostrar TODAS las opciones disponibles.

REGLAS FUNDAMENTALES:
1. NUNCA inventes información que no esté en los documentos
2. SIEMPRE muestra TODAS las opciones del mismo puerto si existen múltiples
3. Agrupa las opciones por puerto pero muestra cada una por separado
4. Si falta información, especifica que está "No disponible"

CONTEXTO FCL MARÍTIMO:
- POL = Port of Loading (Puerto de Carga)
- POD = Port of Discharge (Puerto de Descarga)
- Si hay múltiples carriers desde el mismo puerto, MOSTRAR TODOS
- Cada opción puede tener diferentes precios, carriers, free time

FORMATO OBLIGATORIO PARA MÚLTIPLES OPCIONES:

**🚢 TARIFAS FCL MARÍTIMO - TODAS LAS OPCIONES: [POL] → [POD]**

📋 **OPCIÓN 1:**
- **Puerto Origen (POL):** [POL_EXACTO]
- **Puerto Destino (POD):** [POD_EXACTO]
- **Carrier/Servicio:** [CARRIER_1] (si está disponible)
- **Company:** [COMPANY_1] (si está disponible)
- **Contenedor 20GP:** [20GP_USD_1] (si está disponible)
- **Contenedor 40GP:** [40GP_USD_1] (si está disponible)
- **Contenedor 40HQ:** [40HQ_USD_1] (si está disponible)
- **Contenedor 40NOR:** [40NOR_USD_1] (si está disponible)
- **Free time:** [FREE_TIME_1] (si está disponible)
- **Información Adicional:** [OTHER_1] (si está disponible)

📋 **OPCIÓN 2:**
- **Puerto Origen (POL):** [POL_EXACTO]
- **Puerto Destino (POD):** [POD_EXACTO]
- **Carrier/Servicio:** [CARRIER_2] (si está disponible)
- **Company:** [COMPANY_2] (si está disponible)
- **Contenedor 20GP:** [20GP_USD_2] (si está disponible)
- **Contenedor 40GP:** [40GP_USD_2] (si está disponible)
- **Contenedor 40HQ:** [40HQ_USD_2] (si está disponible)
- **Contenedor 40NOR:** [40NOR_USD_2] (si está disponible)
- **Free time:** [FREE_TIME_2] (si está disponible)
- **Información Adicional:** [OTHER_2] (si está disponible)

[Continuar con OPCIÓN 3, 4, etc. si hay más opciones]

💡 **COMPARACIÓN DE OPCIONES:**
- **Más económica 20GP:** [Opción X - precio]
- **Más económica 40GP:** [Opción Y - precio]
- **Más económica 40HQ:** [Opción Z - precio]
- **Mejor free time:** [Opción con más días]
- **Carrier recomendado:** [Análisis de carriers disponibles]
- **Company recomendada:** [Análisis según opciones]
- **Recomendación:** [Análisis según necesidades típicas de FCL]

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
DOCUMENTOS FCL MARÍTIMO (REVISAR TODOS): {context}
CONSULTA: {question}

RESPUESTA MOSTRANDO TODAS LAS OPCIONES FCL MARÍTIMO:"""

####################################################################
#            FUNCIONES DE DETECCIÓN FCL MARÍTIMO
####################################################################

def detect_maritime_fcl_query_type(query: str) -> str:
    """Detecta tipo de consulta de FCL marítimo"""
    query_lower = query.lower()
    
    # Detectar códigos de puertos conocidos
    known_ports = ['shanghai', 'ningbo', 'qingdao', 'shenzhen', 'tianjin', 'xiamen', 
                   'singapore', 'jakarta', 'bangkok', 'haiphong', 'genoa',
                   'sai', 'val', 'callao', 'chancay', 'guayaquil', 'buenaventura', 'san antonio']
    
    if any(port in query_lower for port in known_ports):
        return 'port_route_query'
    elif any(pattern in query_lower for pattern in ['desde', 'de', 'from', 'a', 'to', 'hacia']):
        return 'route_verification'
    elif any(region in query_lower for region in ['asia', 'china', 'sudamerica', 'chile', 'peru', 'ecuador', 'colombia']):
        return 'region_query'
    elif any(container in query_lower for container in ['20gp', '40gp', '40hq', '40nor', 'contenedor']):
        return 'container_query'
    elif any(term in query_lower for term in ['tarifa', 'precio', 'costo', 'cuanto']):
        return 'price_query'
    elif any(term in query_lower for term in ['carrier', 'servicio', 'naviera']):
        return 'carrier_query'
    elif any(term in query_lower for term in ['free time', 'tiempo libre']):
        return 'freetime_query'
    else:
        return 'general_maritime_fcl'

def extract_ports_from_query(query: str) -> dict:
    """Extrae códigos de puertos de la consulta"""
    import re
    
    query_upper = query.upper()
    
    # Puertos conocidos del archivo FCL
    known_ports = {
        'POL': ['SHANGHAI', 'NINGBO', 'QINGDAO', 'SHENZHEN', 'TIANJIN', 'XIAMEN', 
                'SINGAPORE', 'PORTKLANG', 'PENANG', 'JAKARTA', 'SURABAYA', 
                'LAEM CHABANG', 'BANGKOK', 'HAIPHONG', 'HO CHI MINH', 'GENOA', 'BASE PORTS'],
        'POD': ['SAI/VAL', 'SAI', 'CALLAO', 'CHANCAY', 'GUAYAQUIL', 'BUENAVENTURA', 'SAN ANTONIO']
    }
    
    all_ports = set(known_ports['POL'] + known_ports['POD'])
    
    # Buscar códigos de puertos en la consulta
    found_ports = []
    for port in all_ports:
        if port in query_upper:
            found_ports.append(port)
    
    # Detectar patrones de ruta (A → B, desde A a B, etc.)
    route_patterns = [
        r'(\w+(?:\s+\w+)*)\s*(?:a|to|→|hacia)\s*(\w+(?:\s+\w+)*)',
        r'desde\s*(\w+(?:\s+\w+)*)\s*(?:a|hacia)\s*(\w+(?:\s+\w+)*)',
        r'de\s*(\w+(?:\s+\w+)*)\s*(?:a|hacia)\s*(\w+(?:\s+\w+)*)'
    ]
    
    pol_detected = None
    pod_detected = None
    
    for pattern in route_patterns:
        match = re.search(pattern, query_upper)
        if match:
            pol_detected = match.group(1).strip()
            pod_detected = match.group(2).strip()
            break
    
    return {
        'ports_found': found_ports,
        'pol_detected': pol_detected,
        'pod_detected': pod_detected,
        'has_route_pattern': bool(pol_detected and pod_detected),
        'needs_verification': len(found_ports) > 0 or bool(pol_detected)
    }

def get_port_region(port_code: str) -> str:
    """Determina la región de un puerto"""
    regions = {
        'China': ['SHANGHAI', 'NINGBO', 'QINGDAO', 'SHENZHEN', 'TIANJIN', 'XIAMEN', 'BASE PORTS'],
        'Southeast Asia': ['SINGAPORE', 'PORTKLANG', 'PENANG', 'JAKARTA', 'SURABAYA', 
                          'LAEM CHABANG', 'BANGKOK', 'HAIPHONG', 'HO CHI MINH'],
        'Europe': ['GENOA'],
        'Chile': ['SAI/VAL', 'SAI', 'SAN ANTONIO'],
        'Peru': ['CALLAO', 'CHANCAY'],
        'Ecuador': ['GUAYAQUIL'],
        'Colombia': ['BUENAVENTURA']
    }
    
    for region, ports in regions.items():
        if port_code in ports:
            return region
    
    return 'Unknown'

def analyze_maritime_route_direction(pol: str, pod: str) -> str:
    """Analiza la dirección de la ruta marítima"""
    pol_region = get_port_region(pol)
    pod_region = get_port_region(pod)
    
    if pol_region in ['China', 'Southeast Asia', 'Europe'] and pod_region in ['Chile', 'Peru', 'Ecuador', 'Colombia']:
        return f'Importación desde {pol_region} hacia {pod_region}'
    elif pol_region in ['Chile', 'Peru', 'Ecuador', 'Colombia'] and pod_region in ['China', 'Southeast Asia', 'Europe']:
        return f'Exportación desde {pol_region} hacia {pod_region}'
    else:
        return f'Ruta {pol_region} → {pod_region}'

def extract_container_type_from_query(query: str) -> str:
    """Extrae el tipo de contenedor de la consulta"""
    query_lower = query.lower()
    
    if '20gp' in query_lower or '20 gp' in query_lower:
        return '20GP'
    elif '40hq' in query_lower or '40 hq' in query_lower:
        return '40HQ'
    elif '40nor' in query_lower or '40 nor' in query_lower:
        return '40NOR'
    elif '40gp' in query_lower or '40 gp' in query_lower:
        return '40GP'
    else:
        return 'All'