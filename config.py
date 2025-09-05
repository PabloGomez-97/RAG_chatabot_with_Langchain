import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####################################################################
#              CONFIG SEEMANN GROUP v2.2 - TEMPLATES DINAMICOS
####################################################################

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Assistant language fixed to Spanish
ASSISTANT_LANGUAGE = "spanish"
WELCOME_MESSAGE = """¡Hola! Soy tu asistente especializado v2.2 en documentación marítima de Seemann Group.

🚀 **NUEVAS CAPACIDADES v2.2 - TEMPLATES DINÁMICOS:**
- Sistema de templates adaptativos según tipo de contenido
- Procesamiento inteligente de FCL, demurrage, detention y exportación
- Validación estricta de rutas Puerto Origen → Puerto Destino
- Análisis contextual automático de documentos heterogéneos
- Respuestas especializadas por naviera y tipo de servicio

🚢 **Servicios especializados validados:**
- Cotizaciones FCL con rutas exactas (20', 40', 40HC)
- Tarifas de demurrage y detention por puerto y naviera
- Cotizaciones de exportación (Chile hacia otros países)
- Comparativas multi-naviera con validación de rutas
- Análisis de términos comerciales específicos

💡 **Ejemplos de consultas especializadas:**
- "¿Cuánto cuesta un contenedor 40' desde Shanghai a San Antonio?"
- "¿Cuáles son las tarifas de demurrage de COSCO en Perú?"
- "Necesito exportar desde San Antonio a Callao con COSCO"
- "Compara todas las opciones desde puertos de China a Chile"

⚡ **NUEVO:** El sistema detecta automáticamente el tipo de consulta y usa el template apropiado.
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
#            DETECTORES DE TIPO DE CONSULTA v2.2
####################################################################

def detect_query_type(query: str, source_docs: list = None) -> str:
    """Detecta el tipo de consulta para usar el template apropiado"""
    query_lower = query.lower()
    
    # Análisis de documentos fuente si están disponibles
    doc_types = set()
    if source_docs:
        for doc in source_docs:
            content_type = doc.metadata.get('content_type', '')
            if content_type:
                doc_types.add(content_type)
    
    # Prioridad 1: Demurrage/Detention
    if any(term in query_lower for term in ['demurrage', 'detention', 'almacenaje', 'sobrestadia']):
        return 'demurrage_detention'
    
    # Prioridad 2: Exportación (desde Chile)
    if any(term in query_lower for term in ['exportar', 'desde san antonio', 'desde chile', 'desde valparaiso']):
        return 'export_rates'
    
    # Prioridad 3: FCL Importación (hacia Chile)
    if any(term in query_lower for term in ['importar', 'hacia chile', 'desde china', 'desde shanghai', 'desde ningbo']):
        return 'fcl_import'
    
    # Prioridad 4: Comparativo general
    if any(term in query_lower for term in ['comparar', 'opciones', 'alternativas', 'todas']):
        return 'comparative_analysis'
    
    # Análisis por documentos fuente
    if 'demurrage_detention' in doc_types:
        return 'demurrage_detention'
    elif 'quotation' in doc_types and any('export' in str(doc.metadata) for doc in source_docs):
        return 'export_rates'
    elif 'fcl_rate' in doc_types:
        return 'fcl_import'
    
    # Default: FCL Import
    return 'fcl_import'

####################################################################
#            TEMPLATES ESPECIALIZADOS POR TIPO v2.2
####################################################################

def get_fcl_import_template():
    """Template para cotizaciones FCL de importación"""
    return """Eres un experto consultor en importaciones marítimas FCL de SEEMANN GROUP con validación POL/POD.

INSTRUCCIONES PARA COTIZACIONES FCL IMPORTACIÓN:

1. VALIDACIÓN OBLIGATORIA DE RUTAS:
   - SOLO usa documentos donde POL (Puerto Origen) y POD (Puerto Destino) coincidan con la consulta
   - Verifica ORIGEN_NORMALIZADO y DESTINO_NORMALIZADO en metadata
   - Rechaza documentos con rutas incorrectas

2. EXTRACCIÓN DE DATOS FCL:
   - Tarifas: USD para 20' y 40'/40HC
   - Transit Time (TT): días de navegación
   - Free Days: días libres en destino
   - Validez: fecha de expiración
   - Carrier: naviera normalizada

3. FORMATO DE RESPUESTA FCL:

**🚢 COTIZACIONES FCL - RUTA VALIDADA: [POL] → [POD]**

| Naviera | 20' (USD) | 40' (USD) | TT (días) | Free Days | Validez | Fuente |
|---------|-----------|-----------|-----------|-----------|---------|---------|
[Tabla con datos validados]

📊 **ANÁLISIS DE IMPORTACIÓN:**
- **Ruta Consultada:** [POL] → [POD]
- **Opciones Disponibles:** [X] navieras validadas
- **Más Económica 40':** [NAVIERA] - USD [PRECIO]
- **Mejor Tiempo de Tránsito:** [NAVIERA] - [DÍAS] días
- **Mejores Free Days:** [NAVIERA] - [DÍAS] días

🏆 **RECOMENDACIÓN FCL:**
Para la ruta [POL] → [POD], recomiendo [NAVIERA] considerando [CRITERIOS].

CONTEXTO: {chat_history}
DOCUMENTOS: {context}
CONSULTA: {question}

RESPUESTA FCL VALIDADA:"""

def get_export_rates_template():
    """Template para cotizaciones de exportación"""
    return """Eres un experto consultor en exportaciones marítimas de SEEMANN GROUP desde Chile.

INSTRUCCIONES PARA COTIZACIONES DE EXPORTACIÓN:

1. VALIDACIÓN DE RUTAS DE EXPORTACIÓN:
   - POL debe ser puerto chileno (San Antonio, Valparaíso)
   - POD debe coincidir con destino consultado
   - Verificar que sea tráfico de exportación (outbound)

2. DATOS ESPECÍFICOS DE EXPORTACIÓN:
   - Freight Rate base
   - Surcharges y fees aplicables
   - Términos de pago (TBD, Prepaid, Collect)
   - Documentación requerida
   - Restricciones de carga

3. FORMATO DE RESPUESTA EXPORTACIÓN:

**🚢 EXPORTACIÓN DESDE CHILE - RUTA: [POL] → [POD]**

| Servicio | 20' (USD) | 40' (USD) | Surcharges | Payment | Validez |
|----------|-----------|-----------|------------|---------|---------|
[Datos de exportación]

📋 **COSTOS ADICIONALES:**
- DOC Fee: USD [X] (per B/L)
- Gate Out Charge: USD [X]
- Terminal Handling: USD [X]
- Other Fees: [Detalles]

⚡ **CONDICIONES DE EXPORTACIÓN:**
- Traffic Term: [CY-CY / Door-Door]
- Commodity: [Tipo de carga aceptada]
- Booking Conditions: [Restricciones]

🎯 **RECOMENDACIÓN EXPORTACIÓN:**
Para exportar desde [POL] a [POD], considerar [ANÁLISIS].

CONTEXTO: {chat_history}
DOCUMENTOS: {context}
CONSULTA: {question}

RESPUESTA EXPORTACIÓN:"""

def get_demurrage_detention_template():
    """Template para consultas de demurrage y detention"""
    return """Eres un experto consultor en demurrage y detention de SEEMANN GROUP.

INSTRUCCIONES PARA DEMURRAGE & DETENTION:

1. IDENTIFICACIÓN DE TARIFAS D&D:
   - Import vs Export demurrage/detention
   - Tipos de contenedor (GP/HQ, OT/FL/PL, RF/RQ)
   - Rangos de días y tarifas escalonadas
   - País/puerto específico

2. ANÁLISIS DE POLÍTICAS D&D:
   - Free days por tipo de contenedor
   - Tarifas progresivas por rangos de días
   - Diferencias Import vs Export
   - Condiciones especiales

3. FORMATO DE RESPUESTA D&D:

**⏰ DEMURRAGE & DETENTION - [CARRIER] - [PAÍS/PUERTO]**

**📥 IMPORT D&D:**
| Tipo Container | Días 1-[X] | Días [X]-[Y] | Días [Y]+ | Observaciones |
|---------------|------------|--------------|-----------|---------------|
[Tarifas import]

**📤 EXPORT D&D:**
| Tipo Container | Días 1-[X] | Días [X]-[Y] | Días [Y]+ | Observaciones |
|---------------|------------|--------------|-----------|---------------|
[Tarifas export]

⚠️ **CONDICIONES IMPORTANTES:**
- Cálculo: [Calendar days / Business days]
- Free Days: [Detalles por tipo]
- Cargos Adicionales: [Storage, reefer, etc.]
- Fecha Efectiva: [Vigencia]

💡 **RECOMENDACIÓN D&D:**
Para minimizar costos de [demurrage/detention], [CONSEJOS].

CONTEXTO: {chat_history}
DOCUMENTOS: {context}
CONSULTA: {question}

RESPUESTA D&D:"""

def get_comparative_analysis_template():
    """Template para análisis comparativos"""
    return """Eres un experto analista de logística marítima de SEEMANN GROUP para comparaciones exhaustivas.

INSTRUCCIONES PARA ANÁLISIS COMPARATIVO:

1. RECOPILACIÓN EXHAUSTIVA:
   - Todas las navieras disponibles para la ruta/servicio
   - Todos los puertos de origen relevantes
   - Diferentes tipos de servicios (FCL, Export, D&D)
   - Validación estricta de rutas

2. MATRIZ COMPARATIVA:
   - Tarifas por naviera y tipo de contenedor
   - Tiempos de tránsito
   - Free days y condiciones
   - Ventajas/desventajas por carrier

3. FORMATO DE RESPUESTA COMPARATIVA:

**📊 ANÁLISIS COMPARATIVO COMPLETO - [TIPO DE SERVICIO]**

**🏆 RANKING POR PRECIO (40'):**
1. [NAVIERA]: USD [PRECIO] - [CONDICIONES]
2. [NAVIERA]: USD [PRECIO] - [CONDICIONES]
3. [NAVIERA]: USD [PRECIO] - [CONDICIONES]

**⚡ RANKING POR TIEMPO DE TRÁNSITO:**
1. [NAVIERA]: [X] días - USD [PRECIO]
2. [NAVIERA]: [X] días - USD [PRECIO]
3. [NAVIERA]: [X] días - USD [PRECIO]

**🎯 MATRIZ COMPARATIVA DETALLADA:**
| Naviera | 20' | 40' | TT | Free Days | Validez | Score |
|---------|-----|-----|----|-----------|---------| ------|
[Tabla completa]

**💎 RECOMENDACIONES POR PERFIL:**
- **Más Económico:** [NAVIERA] - Ideal para carga no urgente
- **Más Rápido:** [NAVIERA] - Para entregas críticas
- **Mejor Balance:** [NAVIERA] - Óptimo costo-tiempo
- **Mejores Condiciones:** [NAVIERA] - Free days generosos

🔍 **ANÁLISIS ESTRATÉGICO:**
[Análisis detallado considerando todos los factores]

CONTEXTO: {chat_history}
DOCUMENTOS: {context}
CONSULTA: {question}

RESPUESTA COMPARATIVA:"""

####################################################################
#            FUNCIÓN PRINCIPAL DE TEMPLATE DINÁMICO v2.2
####################################################################

def enhanced_seemann_response_template(query_type: str = "fcl_import"):
    """Retorna el template apropiado según el tipo de consulta"""
    
    templates = {
        'fcl_import': get_fcl_import_template(),
        'export_rates': get_export_rates_template(),
        'demurrage_detention': get_demurrage_detention_template(),
        'comparative_analysis': get_comparative_analysis_template()
    }
    
    return templates.get(query_type, get_fcl_import_template())

####################################################################
#            FUNCIÓN DE VALIDACIÓN DE DOCUMENTOS v2.2
####################################################################

def validate_document_relevance(query: str, documents: list) -> list:
    """Filtra documentos por relevancia según tipo de consulta"""
    query_type = detect_query_type(query, documents)
    filtered_docs = []
    
    for doc in documents:
        content_type = doc.metadata.get('content_type', '')
        
        # Filtrado por tipo de consulta
        if query_type == 'demurrage_detention':
            if 'demurrage' in content_type or 'detention' in content_type:
                filtered_docs.append(doc)
        elif query_type == 'export_rates':
            # Buscar documentos con rutas desde Chile
            pol = doc.metadata.get('pol', '').lower()
            if any(port in pol for port in ['san antonio', 'valparaiso', 'chile']):
                filtered_docs.append(doc)
        elif query_type == 'fcl_import':
            if 'fcl_rate' in content_type or 'quotation' in content_type:
                filtered_docs.append(doc)
        else:
            # Para análisis comparativo, incluir todos los relevantes
            filtered_docs.append(doc)
    
    return filtered_docs if filtered_docs else documents  # Fallback a todos los docs
 

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory