import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG MSL - NUEVO SISTEMA VERIFICACIÓN TOTAL
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado en tarifas LCL marítimas de MSL (Seemann Group).

🚢 **SISTEMA MSL LCL - VERIFICACIÓN TOTAL:**
- Verificación completa de datos antes de responder
- Solo información que realmente existe en el tarifario MSL
- Consultas precisas de rutas LCL disponibles
- Destino: Chile (San Antonio/Valparaíso)

💡 **Ejemplos de consultas:**
- "¿Desde qué puertos de Asia puedo enviar a Chile?"
- "¿Cuánto cuesta desde Santos a Chile?"
- "¿Qué rutas directas hay desde Europa?"

⚠️ **IMPORTANTE:**
Solo te daré información que esté realmente disponible en el tarifario MSL.
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
#            TEMPLATE MSL - VERIFICACIÓN ESTRICTA
####################################################################

def get_msl_response_template():
    """Template especializado para mostrar TODAS las opciones disponibles"""
    return """Eres un especialista en tarifas LCL marítimas de MSL que DEBE mostrar TODAS las opciones disponibles.

REGLAS FUNDAMENTALES:
1. NUNCA inventes información que no esté en los documentos
2. SIEMPRE muestra TODAS las opciones del mismo puerto si existen múltiples
3. Agrupa las opciones por puerto pero muestra cada una por separado
4. Si falta información, especifica qué está "No disponible"

CONTEXTO MSL:
- Todos los envíos son hacia Chile (San Antonio/Valparaíso)
- Si hay múltiples servicios desde el mismo puerto, MOSTRAR TODOS
- Cada opción puede tener diferentes precios, tiempos, servicios

FORMATO OBLIGATORIO PARA MÚLTIPLES OPCIONES:

**🚢 TARIFAS LCL MSL - TODAS LAS OPCIONES: [PUERTO] → CHILE**

📋 **OPCIÓN 1:**
- **Puerto de Origen:** [PUERTO_EXACTO]
- **País Origen:** [PAÍS_EXACTO]
- **Company:** [COMPANY_1] (si está disponible)
- **Destino:** Chile (San Antonio/Valparaíso)
- **TON / M3 Usd/Eur:** [TARIFA_1] (si está disponible)
- **Mínimo:** [MÍNIMO_1] (si está disponible)
- **Tiempo Tránsito:** [DÍAS_1] (si está disponible)
- **Frecuencia:** [FRECUENCIA_1] (si está disponible)
- **Tipo Servicio:** [SERVICIO_1] (si está disponible)
- **Agente Local:** [AGENTE_1] (si está disponible)
- **Costos Adicionales:** [OTROS_1] (si está disponible)

📋 **OPCIÓN 2:**
- **Puerto de Origen:** [PUERTO_EXACTO]
- **País Origen:** [PAÍS_EXACTO]
- **Company:** [COMPANY_2] (si está disponible)
- **Destino:** Chile (San Antonio/Valparaíso)
- **TON / M3 Usd/Eur:** [TARIFA_2] (si está disponible)
- **Mínimo:** [MÍNIMO_2] (si está disponible)
- **Tiempo Tránsito:** [DÍAS_2] (si está disponible)
- **Frecuencia:** [FRECUENCIA_2] (si está disponible)
- **Tipo Servicio:** [SERVICIO_2] (si está disponible)
- **Agente Local:** [AGENTE_2] (si está disponible)
- **Costos Adicionales:** [OTROS_2] (si está disponible)

[Continuar con OPCIÓN 3, 4, etc. si hay más opciones]

💡 **COMPARACIÓN DE OPCIONES:**
- **Más económica:** [Opción X - precio]
- **Más rápida:** [Opción Y - días]
- **Servicio directo:** [Si hay opción directa]
- **Por company:** [Agrupar opciones por company si hay múltiples]
- **Recomendación:** [Análisis según necesidades típicas]

⚠️ **OBSERVACIONES IMPORTANTES:**
[Incluir todas las observaciones de todas las opciones]

INSTRUCCIONES CRÍTICAS:
- BUSCA EN TODOS LOS DOCUMENTOS opciones del mismo puerto
- NO te limites al primer documento que encuentres
- Si hay 2+ documentos del mismo puerto, mostrar TODOS
- Cada fila del Excel = una opción diferente
- NUNCA omitas opciones que existan en los documentos

CONTEXTO: {chat_history}
DOCUMENTOS MSL (REVISAR TODOS): {context}
CONSULTA: {question}

RESPUESTA MOSTRANDO TODAS LAS OPCIONES:"""

####################################################################
#            FUNCIONES DE DETECCIÓN Y VERIFICACIÓN
####################################################################

def detect_msl_query_type(query: str) -> str:
    """Detecta tipo de consulta para verificación"""
    query_lower = query.lower()
    
    if any(pattern in query_lower for pattern in ['desde', 'de', 'from']):
        return 'route_verification'
    elif any(region in query_lower for region in ['europa', 'asia', 'america', 'norteamerica']):
        return 'region_verification'
    elif any(term in query_lower for term in ['opciones', 'alternativas', 'disponible']):
        return 'availability_check'
    else:
        return 'general_verification'

def extract_port_for_verification(query: str) -> dict:
    """Extrae puerto para verificar si existe"""
    query_lower = query.lower()
    
    # Patrones más específicos para extracción
    import re
    
    patterns = [
        r'desde\s+([^a-z\s]{2,}(?:\s+[^a-z\s]{2,})*)',  # Desde PUERTO
        r'de\s+([^a-z\s]{2,}(?:\s+[^a-z\s]{2,})*)',     # De PUERTO
        r'tarifa\s+([^a-z\s]{2,}(?:\s+[^a-z\s]{2,})*)', # Tarifa PUERTO
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            port_raw = match.group(1).strip()
            return {
                'port_requested': port_raw,
                'needs_verification': True
            }
    
    return {'needs_verification': False}