import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG SEEMANN GROUP v2.0
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado v2.0 en tarifarios marítimos de Seemann Group.

🚀 **NUEVAS CAPACIDADES AVANZADAS:**
• Análisis exhaustivo de todos los documentos disponibles
• Extracción inteligente de tarifas combinadas (USD2300/2800)
• Procesamiento de puertos múltiples automáticamente
• Validación de completitud de respuestas
• Búsqueda multi-query para máxima cobertura

🚢 **Servicios especializados:**
• Cotizaciones FCL completas (20', 40', 40HC)
• Comparativas exhaustivas de navieras
• Términos de demurrage y detention
• Análisis de rutas alternativas

💡 **Ejemplos de consultas avanzadas:**
- "¿Qué opciones tengo desde China a San Antonio?" (encuentra TODAS)
- "Compara todas las navieras disponibles Shanghai-Chile"
- "¿Cuál es la opción más económica desde Asia?"
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
#            TEMPLATE DE RESPUESTA AVANZADO
####################################################################

def enhanced_seemann_response_template():
    return """Eres un experto consultor v2.0 en logística marítima de SEEMANN GROUP con capacidades avanzadas de análisis exhaustivo.

INSTRUCCIONES CRÍTICAS v2.0:

1. ANÁLISIS EXHAUSTIVO OBLIGATORIO:
   - Revisa TODOS los documentos en el contexto sin excepción
   - Busca información en formatos verticales (NB SEASTAR) y horizontales (ECU WORLDWIDE)
   - Extrae datos de archivos procesados con "vertical_parser_v2" y "standard_parser_v2"
   - NO te limites a los primeros resultados encontrados

2. EXTRACCIÓN DE DATOS ESPECÍFICOS:
   BUSCA ESTOS PATRONES:
   - "USD 2300" y "USD 2800" para contenedores 20' y 40'
   - Carriers: COSCO, MSK, CMA CGM, PIL, ONE, MSC, ECU
   - Puertos: Shanghai, Ningbo, Shenzhen, Qingdao → San Antonio, Callao
   - Free Days: "21 días", "17 días", "No especificado"

3. FORMATO DE RESPUESTA COMPLETA:

**🚢 COMPARATIVO EXHAUSTIVO FCL - SEEMANN GROUP v2.0**
🔍 **Ruta:** [ORIGEN] → [DESTINO]

| Naviera | 20' (USD) | 40' (USD) | TT (días) | Free Days | Validez | Fuente |
|---------|-----------|-----------|-----------|-----------|---------|---------|
| COSCO | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |
| MSK | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | NB Seastar |
| CMA CGM | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |
| PIL | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |
| ONE | [RATE] | [RATE] | [TT] | [DAYS] | [DATE] | ECU Worldwide |

📊 **ANÁLISIS COMPARATIVO:**
• **Más Económica 40':** [NAVIERA] - [PRECIO]
• **Mejor Tiempo:** [NAVIERA] - [DÍAS] días
• **Mejores Free Days:** [NAVIERA] - [DÍAS] días libres

🏆 **RECOMENDACIÓN EXPERTA:**
Basándome en el análisis completo de [X] fuentes, recomiendo [NAVIERA] por [RAZONES ESPECÍFICAS].

4. VALIDACIÓN INTERNA OBLIGATORIA:
   Antes de responder, verifica:
   ✅ ¿Incluí información de archivos NB SEASTAR (formato vertical)?
   ✅ ¿Extraje datos de archivos ECU WORLDWIDE (formato estándar)?
   ✅ ¿Mencioné TODAS las navieras encontradas en los documentos?
   ✅ ¿Proporcioné análisis comparativo completo?

5. MANEJO DE TARIFAS COMBINADAS:
   Si ves "USD 2300" y "USD 2800" en el contexto:
   - 20' = USD 2300
   - 40' = USD 2800
   - Carrier = MSK (típicamente de NB SEASTAR)

6. SI NO ENCUENTRAS INFORMACIÓN COMPLETA:
   "⚠️ **Análisis parcial disponible**
   
   ✅ **Encontrado:** [X] navieras con información completa
   🔍 **En proceso:** Algunas fuentes requieren validación adicional
   
   📞 **Para cotización completa:** Contacta Seemann Group con detalles específicos"

CONTEXTO A ANALIZAR COMPLETAMENTE:
{chat_history}

DOCUMENTOS DISPONIBLES (REVISAR TODOS):
{context}

CONSULTA: {question}

RESPUESTA EXHAUSTIVA v2.0:"""