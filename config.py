import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG MSL - LCL MARITIME SYSTEM
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado en tarifas LCL marítimas de MSL (Seemann Group).

🚢 **SISTEMA MSL LCL:**
- Consultas de tarifas LCL por puerto de origen hacia Chile
- Información completa de costos por tonelada/m³
- Detalles de tiempos de tránsito y frecuencias
- Información de agentes locales y servicios
- Costos adicionales y observaciones importantes

💡 **Ejemplos de consultas:**
- "¿Cuánto cuesta envío LCL desde Shanghai a Chile?"
- "Necesito tarifa desde Buenos Aires"
- "¿Qué opciones tengo desde Europa?"
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
#            MAPEO DE PUERTOS MSL
####################################################################

PORT_ALIASES = {
    # Puertos de América
    'buenos aires': ['buenos aires', 'bue', 'argentina buenos aires', 'ba'],
    'santos': ['santos', 'sao', 'brasil santos', 'santos brasil'],
    'rio de janeiro': ['rio de janeiro', 'rio', 'brasil rio', 'gig'],
    'callao': ['callao', 'cao', 'peru callao', 'lima'],
    'guayaquil': ['guayaquil', 'gye', 'ecuador guayaquil'],
    
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
#            FUNCIONES DE DETECCIÓN DE CONSULTAS MSL
####################################################################

def detect_msl_query_type(query: str) -> str:
    """Detecta el tipo específico de consulta MSL"""
    query_lower = query.lower()
    
    # Consulta por ruta específica (origen -> Chile)
    if any(pattern in query_lower for pattern in ['desde', 'de', 'from']):
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
    """Extrae puertos origen de la consulta (destino siempre es Chile)"""
    query_lower = query.lower()
    
    # Patrones para detectar origen
    patterns = [
        r'desde\s+([^a]+?)(?:\s+(?:hacia|hasta|a)\s+(?:chile|san antonio|valparaiso))?(?:\s|$|[?])',
        r'de\s+([^a]+?)(?:\s+a\s+(?:chile|san antonio|valparaiso))?(?:\s|$|[?])',
        r'from\s+([^to]+?)(?:\s+to\s+(?:chile|san antonio|valparaiso))?(?:\s|$|[?])',
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            origin_raw = match.group(1).strip()
            origin_normalized = normalize_port_name(origin_raw)
            
            return {
                'origin_raw': origin_raw,
                'origin_normalized': origin_normalized,
                'destination': 'Chile',
                'has_route': True
            }
    
    return {'has_route': False, 'destination': 'Chile'}

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
#            TEMPLATE MSL
####################################################################

def get_msl_response_template():
    """Template especializado para respuestas MSL"""
    return """Eres un especialista en tarifas LCL (Less than Container Load) marítimas de MSL (Seemann Group).

CONTEXTO IMPORTANTE MSL:
- TODOS los registros del tarifario MSL son para importación HACIA CHILE
- El destino implícito SIEMPRE es Chile (San Antonio/Valparaíso)
- Los usuarios consultan rutas como "desde Shanghai" o "desde Shanghai a Chile"
- MSL no tiene columna POD porque el destino siempre es Chile

INSTRUCCIONES PARA CONSULTAS MSL:

1. ANÁLISIS DE LA CONSULTA:
   - Identifica puerto origen exacto
   - DESTINO SIEMPRE ES CHILE (San Antonio/Valparaíso)
   - Busca en las regiones correctas (AMERICA, EUROPA, NORTEAMERICA, ASIA)
   - Valida que la ruta origen → Chile existe en el tarifario MSL

2. INFORMACIÓN OBLIGATORIA A MOSTRAR:
   - PUERTO ORIGEN (exacto del tarifario)
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

**🚢 TARIFAS LCL MSL - RUTA: [PUERTO_ORIGEN] → CHILE**

📋 **INFORMACIÓN DETALLADA:**
- **Puerto de Origen:** [PUERTO_EXACTO]
- **País Origen:** [PAÍS]
- **Destino:** Chile (San Antonio/Valparaíso)
- **Tarifa:** [TON/M3] por tonelada/m³
- **Mínimo:** [MÍNIMO]
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
   - Aclarar que todas las tarifas MSL son para importación hacia Chile

5. PARA CONSULTAS COMPARATIVAS:
   - Mostrar TODAS las opciones MSL desde la región consultada hacia Chile
   - Ordenar por precio o tiempo según consulta
   - Incluir análisis de mejores opciones MSL hacia Chile

6. SI NO EXISTE LA RUTA HACIA CHILE:
   - Informar claramente que no está disponible en MSL
   - Sugerir puertos alternativos MSL en la misma región
   - Mostrar rutas MSL más cercanas disponibles hacia Chile

CONTEXTO CONVERSACIÓN: {chat_history}
DOCUMENTOS TARIFARIO MSL: {context}
CONSULTA CLIENTE: {question}

RESPUESTA MSL ESPECIALIZADA (Destino siempre Chile):"""

####################################################################
#            FUNCIONES DE VALIDACIÓN MSL
####################################################################

def validate_msl_document_relevance(query: str, documents: list) -> list:
    """Filtra documentos relevantes para consultas MSL específicas"""
    
    route_info = extract_ports_from_query(query)
    query_type = detect_msl_query_type(query)
    
    if not route_info.get('has_route'):
        # Si no hay ruta específica, devolver todos los documentos MSL
        return documents
    
    origin = route_info.get('origin_normalized', '')
    
    relevant_docs = []
    
    for doc in documents:
        doc_content = doc.page_content.lower()
        
        # Buscar coincidencia de puerto origen en el contenido
        if origin:
            if origin in doc_content or any(alias in doc_content for alias in PORT_ALIASES.get(origin, [])):
                relevant_docs.append(doc)
        elif query_type == 'comparative':
            relevant_docs.append(doc)
    
    return relevant_docs if relevant_docs else documents  # Fallback