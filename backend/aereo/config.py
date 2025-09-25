import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG CRAFTTRANSWAY - SISTEMA CARGA AEREA
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado en tarifas

⚠️ **IMPORTANTE:**
Solo te daré información que esté realmente disponible en el tarifario de carga aérea.
Si una ruta no existe, te lo diré claramente.
"""

# Available OpenAI models
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4-turbo", 
    "gpt-3.5-turbo-0125",
]

# Rutas base
TMP_DIR = Path(__file__).resolve().parent / "data" / "tmp_aereo"
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent / "data" / "vector_store_aereo"

# Asegurar directorios
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#            TEMPLATE CRAFTTRANSWAY - RESPUESTAS AÉREAS
####################################################################

def get_air_freight_response_template():
    """Template especializado para mostrar TODAS las opciones de carga aérea disponibles"""
    return """Eres un especialista en tarifas de carga aérea de CRAFTTRANSWAY que DEBE mostrar TODAS las opciones disponibles.

REGLAS FUNDAMENTALES:
1. NUNCA inventes información que no esté en los documentos
2. SIEMPRE muestra TODAS las opciones del mismo aeropuerto si existen múltiples
3. Agrupa las opciones por aeropuerto pero muestra cada una por separado
4. Si falta información, especifica que está "No disponible"

CONTEXTO CARGA AÉREA:
- AOL = Airport of Loading (Aeropuerto de Carga)
- AOD = Airport of Discharge (Aeropuerto de Descarga)
- Si hay múltiples servicios desde el mismo aeropuerto, MOSTRAR TODOS
- Cada opción puede tener diferentes precios, airlines, servicios

FORMATO OBLIGATORIO PARA MÚLTIPLES OPCIONES:

**✈️ TARIFAS CARGA AÉREA CRAFTTRANSWAY - TODAS LAS OPCIONES: [AOL] → [AOD]**

📋 **OPCIÓN 1:**
- **Aeropuerto Origen (AOL):** [AOL_EXACTO]
- **País Origen:** [PAÍS_EXACTO]
- **Aeropuerto Destino (AOD):** [AOD_EXACTO]
- **País Destino:** [PAÍS_DESTINO_EXACTO]
- **Company:** [COMPANY_1] (si está disponible)
- **Airline/Servicio:** [AIRLINE_1] (si está disponible)
- **Mínimo por envío:** [MIN_1] (si está disponible)
- **Tarifa por KG:** [FLAT_KG_1] (si está disponible)
- **Frecuencia Salidas:** [SALIDAS_1] (si está disponible)
- **Información Adicional:** [OTROS_1] (si está disponible)

📋 **OPCIÓN 2:**
- **Aeropuerto Origen (AOL):** [AOL_EXACTO]
- **País Origen:** [PAÍS_EXACTO]
- **Aeropuerto Destino (AOD):** [AOD_EXACTO]
- **País Destino:** [PAÍS_DESTINO_EXACTO]
- **Company:** [COMPANY_2] (si está disponible)
- **Airline/Servicio:** [AIRLINE_2] (si está disponible)
- **Mínimo por envío:** [MIN_2] (si está disponible)
- **Tarifa por KG:** [FLAT_KG_2] (si está disponible)
- **Frecuencia Salidas:** [SALIDAS_2] (si está disponible)
- **Información Adicional:** [OTROS_2] (si está disponible)

[Continuar con OPCIÓN 3, 4, etc. si hay más opciones]

💡 **COMPARACIÓN DE OPCIONES:**
- **Más económica:** [Opción X - precio mínimo y tarifa por KG]
- **Mejor servicio:** [Opción Y - airline y frecuencia]
- **Company recomendada:** [CRAFT o TRANSWAY según opciones]
- **Recomendación:** [Análisis según necesidades típicas de carga aérea]

⚠️ **OBSERVACIONES IMPORTANTES:**
[Incluir todas las observaciones de todas las opciones encontradas]

INSTRUCCIONES CRÍTICAS:
- BUSCA EN TODOS LOS DOCUMENTOS opciones del mismo aeropuerto origen y destino
- NO te limites al primer documento que encuentres
- Si hay 2+ documentos con la misma ruta AOL→AOD, mostrar TODOS
- Cada fila del Excel = una opción diferente
- NUNCA omitas opciones que existan en los documentos
- Si solo hay una opción, usar el mismo formato pero solo mostrar OPCIÓN 1

CONTEXTO: {chat_history}
DOCUMENTOS CARGA AÉREA (REVISAR TODOS): {context}
CONSULTA: {question}

RESPUESTA MOSTRANDO TODAS LAS OPCIONES DE CARGA AÉREA:"""

####################################################################
#            FUNCIONES DE DETECCIÓN AÉREA
####################################################################

def detect_air_freight_query_type(query: str) -> str:
    """Detecta tipo de consulta de carga aérea"""
    query_lower = query.lower()
    
    # Detectar códigos de aeropuertos
    if any(code in query_lower for code in ['scl', 'mia', 'fra', 'mad', 'hkg', 'gru', 'lim']):
        return 'airport_route_query'
    elif any(pattern in query_lower for pattern in ['desde', 'de', 'from', 'a', 'to', 'hacia']):
        return 'route_verification'
    elif any(region in query_lower for region in ['europa', 'asia', 'america', 'norteamerica', 'sudamerica']):
        return 'region_query'
    elif any(term in query_lower for term in ['tarifa', 'precio', 'costo', 'cuanto']):
        return 'price_query'
    elif any(term in query_lower for term in ['airline', 'servicio', 'vuelo']):
        return 'service_query'
    else:
        return 'general_air_freight'

def extract_airports_from_query(query: str) -> dict:
    """Extrae códigos de aeropuertos de la consulta"""
    import re
    
    query_upper = query.upper()
    
    # Códigos de aeropuertos conocidos del archivo
    known_airports = {
        'AOL': ['MIA', 'SCL', 'MXP', 'MAD', 'FRA', 'LHR', 'LIS', 'PVG', 'NKG', 'HKG', 'VCP', 'GRU'],
        'AOD': ['SCL', 'LIM', 'GRU', 'BOG', 'MVD', 'MIA', 'UIO']
    }
    
    all_airports = set(known_airports['AOL'] + known_airports['AOD'])
    
    # Buscar códigos de aeropuertos en la consulta
    found_airports = []
    for airport in all_airports:
        if airport in query_upper:
            found_airports.append(airport)
    
    # Detectar patrones de ruta (A → B, desde A a B, etc.)
    route_patterns = [
        r'(\w{3})\s*(?:a|to|→|hacia)\s*(\w{3})',
        r'desde\s*(\w{3})\s*(?:a|hacia)\s*(\w{3})',
        r'de\s*(\w{3})\s*(?:a|hacia)\s*(\w{3})'
    ]
    
    aol_detected = None
    aod_detected = None
    
    for pattern in route_patterns:
        match = re.search(pattern, query_upper)
        if match:
            aol_detected = match.group(1)
            aod_detected = match.group(2)
            break
    
    return {
        'airports_found': found_airports,
        'aol_detected': aol_detected,
        'aod_detected': aod_detected,
        'has_route_pattern': bool(aol_detected and aod_detected),
        'needs_verification': len(found_airports) > 0 or bool(aol_detected)
    }

def get_airport_region(airport_code: str) -> str:
    """Determina la región de un aeropuerto"""
    regions = {
        'North America': ['MIA'],
        'Europe': ['MXP', 'MAD', 'FRA', 'LHR', 'LIS'],
        'Asia': ['PVG', 'NKG', 'HKG'],
        'South America': ['SCL', 'LIM', 'GRU', 'BOG', 'MVD', 'UIO', 'VCP'],
        'Chile': ['SCL']
    }
    
    for region, airports in regions.items():
        if airport_code in airports:
            return region
    
    return 'Unknown'

def analyze_route_direction(aol: str, aod: str) -> str:
    """Analiza la dirección de la ruta"""
    if aol == 'SCL':
        return 'Exportación desde Chile'
    elif aod == 'SCL':
        return 'Importación hacia Chile'
    else:
        return 'Ruta sin conexión directa con Chile'