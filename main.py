import streamlit as st
import sys
from pathlib import Path

####################################################################
#            CONFIGURACIÓN PRINCIPAL
####################################################################

st.set_page_config(
    page_title="Sistema de Consulta de Tarifas",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para el navbar y responsive design
st.markdown("""
<style>
    .navbar {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .navbar h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Responsive para navbar */
    @media (max-width: 768px) {
        .navbar h1 {
            font-size: 1.5rem;
        }
        .navbar {
            padding: 0.8rem 1rem;
        }
    }
    
    .system-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        text-align: center;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Responsive para tarjetas */
    @media (max-width: 768px) {
        .system-card {
            margin: 0.5rem 0;
            padding: 1.2rem;
            min-height: 250px;
        }
    }
    
    @media (max-width: 480px) {
        .system-card {
            padding: 1rem;
            min-height: 220px;
        }
    }
    
    .system-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .system-icon {
        font-size: 3.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Responsive para iconos */
    @media (max-width: 768px) {
        .system-icon {
            font-size: 3rem;
            margin-bottom: 0.6rem;
        }
    }
    
    @media (max-width: 480px) {
        .system-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
    }
    
    .system-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: #1f2937;
        line-height: 1.2;
    }
    
    /* Responsive para títulos */
    @media (max-width: 768px) {
        .system-title {
            font-size: 1.4rem;
            margin-bottom: 0.6rem;
        }
    }
    
    @media (max-width: 480px) {
        .system-title {
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }
    }
    
    .system-description {
        color: #6b7280;
        font-size: 0.95rem;
        line-height: 1.4;
        margin-bottom: 1rem;
        flex-grow: 1;
        hyphens: auto;
        word-wrap: break-word;
    }
    
    /* Responsive para descripciones */
    @media (max-width: 768px) {
        .system-description {
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
    }
    
    @media (max-width: 480px) {
        .system-description {
            font-size: 0.85rem;
            margin-bottom: 0.7rem;
        }
    }
    
    .feature-list {
        text-align: left;
        margin: 0.8rem 0;
    }
    
    .feature-item {
        color: #059669;
        font-size: 0.85rem;
        margin: 0.25rem 0;
        line-height: 1.3;
    }
    
    /* Responsive para features */
    @media (max-width: 768px) {
        .feature-item {
            font-size: 0.8rem;
        }
    }
    
    @media (max-width: 480px) {
        .feature-item {
            font-size: 0.75rem;
        }
    }
    
    /* Asegurar que las columnas de Streamlit sean responsive */
    .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }
    
    @media (max-width: 768px) {
        .block-container {
            padding: 0.5rem 1rem;
        }
    }
    
    /* Botones responsive */
    .stButton > button {
        width: 100%;
        height: 50px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    @media (max-width: 480px) {
        .stButton > button {
            height: 45px;
            font-size: 0.9rem;
        }
    }
    
    /* Información adicional responsive */
    .stAlert {
        margin: 0.5rem 0;
    }
    
    @media (max-width: 768px) {
        .stAlert {
            font-size: 0.9rem;
        }
    }
    
    /* Ajustes para el texto de selección */
    h3 {
        margin-bottom: 2rem !important;
        text-align: center;
    }
    
    @media (max-width: 768px) {
        h3 {
            font-size: 1.3rem !important;
            margin-bottom: 1.5rem !important;
        }
    }
    
    @media (max-width: 480px) {
        h3 {
            font-size: 1.1rem !important;
            margin-bottom: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

####################################################################
#            FUNCIONES PRINCIPALES
####################################################################

def show_home():
    """Muestra la página principal con selección de sistemas"""
    
    # Header principal
    st.markdown("""
    <div class="navbar">
        <h1>🌐 Sistema Integral de Consulta de Tarifas</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Selecciona el tipo de transporte para consultar tarifas:")
    
    # Layout de 3 columnas para los sistemas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="system-card">
            <div>
                <div class="system-icon">✈️</div>
                <div class="system-title">CARGA AÉREA</div>
                <div class="system-description">
                    Sistema de consulta de tarifas de carga aérea CRAFTTRANSWAY
                </div>
                <div class="feature-list">
                    <div class="feature-item">✓ Tarifas AOL → AOD</div>
                    <div class="feature-item">✓ Múltiples airlines</div>
                    <div class="feature-item">✓ Precios por KG y mínimos</div>
                    <div class="feature-item">✓ Rutas globales</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 ACCEDER A CARGA AÉREA", key="aereo", use_container_width=True, type="primary"):
            st.session_state.selected_system = "AEREO"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="system-card">
            <div>
                <div class="system-icon">🚢</div>
                <div class="system-title">FCL MARÍTIMO</div>
                <div class="system-description">
                    Sistema de consulta de tarifas de contenedores completos
                </div>
                <div class="feature-list">
                    <div class="feature-item">✓ Contenedores 20GP, 40GP, 40HQ</div>
                    <div class="feature-item">✓ Rutas POL → POD</div>
                    <div class="feature-item">✓ Free time incluido</div>
                    <div class="feature-item">✓ Múltiples carriers</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 ACCEDER A FCL MARÍTIMO", key="fcl", use_container_width=True, type="primary"):
            st.session_state.selected_system = "FCL"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="system-card">
            <div>
                <div class="system-icon">🌊</div>
                <div class="system-title">LCL MARÍTIMO</div>
                <div class="system-description">
                    Sistema MSL de consulta de tarifas de carga consolidada
                </div>
                <div class="feature-list">
                    <div class="feature-item">✓ Tarifas TON/M3</div>
                    <div class="feature-item">✓ Cobertura mundial</div>
                    <div class="feature-item">✓ Agentes locales</div>
                    <div class="feature-item">✓ Importaciones a Chile</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 ACCEDER A LCL MARÍTIMO", key="lcl", use_container_width=True, type="primary"):
            st.session_state.selected_system = "LCL"
            st.rerun()
    
    # Información adicional
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**CARGA AÉREA** - Para envíos urgentes y mercancías de alto valor")
    
    with col2:
        st.info("**FCL MARÍTIMO** - Para grandes volúmenes que llenan contenedores completos")
    
    with col3:
        st.info("**LCL MARÍTIMO** - Para cargas pequeñas que se consolidan con otros envíos")

def run_aereo_system():
    """Ejecuta el sistema de carga aérea"""
    try:
        # Importar y ejecutar el sistema aéreo
        from AEREO.app import main as aereo_main
        
        # Navbar de navegación
        show_navbar("CARGA AÉREA", "✈️")
        
        # Ejecutar sistema aéreo
        aereo_main()
        
    except ImportError as e:
        st.error(f"Error importando sistema aéreo: {e}")
        st.info("Asegúrate de que los archivos del sistema aéreo estén en la carpeta AEREO/")

def run_fcl_system():
    """Ejecuta el sistema FCL marítimo"""
    try:
        # Importar y ejecutar el sistema FCL
        from FCL.app import main as fcl_main
        
        # Navbar de navegación
        show_navbar("FCL MARÍTIMO", "🚢")
        
        # Ejecutar sistema FCL
        fcl_main()
        
    except ImportError as e:
        st.error(f"Error importando sistema FCL: {e}")
        st.info("Asegúrate de que los archivos del sistema FCL estén en la carpeta FCL/")

def run_lcl_system():
    """Ejecuta el sistema LCL marítimo"""
    try:
        # Importar y ejecutar el sistema LCL
        from LCL.app import main as lcl_main
        
        # Navbar de navegación
        show_navbar("LCL MARÍTIMO", "🌊")
        
        # Ejecutar sistema LCL
        lcl_main()
        
    except ImportError as e:
        st.error(f"Error importando sistema LCL: {e}")
        st.info("Asegúrate de que los archivos del sistema LCL estén en la carpeta LCL/")

def show_navbar(system_name, icon):
    """Muestra navbar de navegación con botón de regreso mejorado"""
    
    # CSS adicional para el botón mejorado
    st.markdown("""
    <style>
        .back-button {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white !important;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin: 10px 0;
        }
        
        .back-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4);
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        }
        
        .back-button:active {
            transform: translateY(0px);
        }
        
        .navbar-container {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        .navbar-title {
            color: white;
            text-align: center;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .navbar-grid {
            display: grid;
            grid-template-columns: 1fr 3fr 1fr;
            align-items: center;
            gap: 20px;
        }
        
        @media (max-width: 768px) {
            .navbar-title {
                font-size: 1.6rem;
            }
            .back-button {
                padding: 10px 18px;
                font-size: 14px;
            }
            .navbar-container {
                padding: 15px;
            }
        }
        
        @media (max-width: 480px) {
            .navbar-title {
                font-size: 1.3rem;
            }
            .back-button {
                padding: 8px 15px;
                font-size: 13px;
            }
            .navbar-grid {
                grid-template-columns: 1fr;
                text-align: center;
                gap: 15px;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Contenedor principal del navbar
    st.markdown(f"""
    <div class="navbar-container">
        <div class="navbar-grid">
            <div></div>
            <div>
                <h1 class="navbar-title">{icon} {system_name}</h1>
            </div>
            <div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Botón de regreso mejorado centrado
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("🏠 Volver al Inicio", key="back_home", 
                    use_container_width=True, type="secondary"):
            clear_all_chat_state()  # Limpiar estado al volver
            if 'selected_system' in st.session_state:
                del st.session_state.selected_system
            if 'previous_system' in st.session_state:
                del st.session_state.previous_system
            st.rerun()
    
    st.markdown("---")
####################################################################
#            FUNCIÓN PRINCIPAL
####################################################################

def main():
    """Función principal del dashboard"""
    
    # Inicializar session state
    if 'selected_system' not in st.session_state:
        st.session_state.selected_system = None
    
    # NUEVA FUNCIONALIDAD: Detectar cambio de sistema
    if 'previous_system' not in st.session_state:
        st.session_state.previous_system = None
    
    current_system = st.session_state.selected_system
    
    # Si cambió el sistema, limpiar todo el estado relacionado con chatbots
    if (st.session_state.previous_system is not None and 
        current_system != st.session_state.previous_system):
        
        clear_all_chat_state()  # Llamar a la función de limpieza
        st.session_state.previous_system = current_system
        st.rerun()  # Forzar recarga para aplicar cambios
    
    # Actualizar sistema anterior para la próxima vez
    st.session_state.previous_system = current_system
    
    # El resto del código sigue igual
    if st.session_state.selected_system == "AEREO":
        run_aereo_system()
    elif st.session_state.selected_system == "FCL":
        run_fcl_system()
    elif st.session_state.selected_system == "LCL":
        run_lcl_system()
    else:
        show_home()

def clear_all_chat_state():
    """Limpia todo el estado relacionado con chatbots"""
    
    # Lista de variables que queremos limpiar al cambiar de sistema
    keys_to_clear = [
        'messages',           # Historial del chat
        'memory',            # Memoria de la conversación
        'chain',             # El sistema de preguntas y respuestas
        'retriever',         # El buscador de documentos
        'vector_store',      # La base de datos de vectores
        'uploaded_file_list', # Archivos que subiste
        'vector_store_name', # Nombre de la base de datos
        'selected_vectorstore_name', # Base de datos seleccionada
        'ejemplo_selected',  # Si seleccionaste algún ejemplo
        'initialized'        # Flag de inicialización
    ]
    
    # Eliminar cada una de estas variables del estado
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # También limpiar variables que empiecen con prefijos de los sistemas
    all_keys = list(st.session_state.keys())
    system_prefixes = ['aereo_', 'fcl_', 'lcl_', 'air_', 'maritime_']
    
    for key in all_keys:
        if any(key.startswith(prefix) for prefix in system_prefixes):
            del st.session_state[key]
    
    print("[CLEANUP] Estado del chat limpiado al cambiar sistema")

if __name__ == "__main__":
    main()