import streamlit as st
import sys
from pathlib import Path

####################################################################
#            CONFIGURACIÓN PRINCIPAL
####################################################################

st.set_page_config(
    page_title="Sistema de Consulta de Tarifas",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado más sobrio y completamente responsive
st.markdown("""
<style>
    /* Ocultar elementos de Streamlit por defecto */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Contenedor principal responsive */
    .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    
    @media (min-width: 768px) {
        .block-container {
            padding: 2rem;
        }
    }
    
    /* Navbar simple y sobrio */
    .navbar {
        background: #2c3e50;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .navbar h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 500;
    }
    
    @media (max-width: 768px) {
        .navbar {
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        .navbar h1 {
            font-size: 1.4rem;
        }
    }
    
    @media (max-width: 480px) {
        .navbar {
            padding: 0.8rem;
            margin-bottom: 1rem;
        }
        .navbar h1 {
            font-size: 1.2rem;
        }
    }
    
    /* Tarjetas del sistema - completamente responsive */
    .system-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .system-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    @media (max-width: 1024px) {
        .system-card {
            padding: 1.2rem;
            margin-bottom: 1.5rem;
        }
    }
    
    @media (max-width: 768px) {
        .system-card {
            padding: 1rem;
            margin-bottom: 1rem;
        }
    }
    
    /* Iconos responsive */
    .system-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    @media (max-width: 768px) {
        .system-icon {
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
        }
    }
    
    @media (max-width: 480px) {
        .system-icon {
            font-size: 2rem;
            margin-bottom: 0.6rem;
        }
    }
    
    /* Títulos responsive */
    .system-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #2c3e50;
        line-height: 1.3;
    }
    
    @media (max-width: 768px) {
        .system-title {
            font-size: 1.2rem;
            margin-bottom: 0.6rem;
        }
    }
    
    @media (max-width: 480px) {
        .system-title {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
    }
    
    /* Descripciones responsive */
    .system-description {
        color: #6c757d;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1rem;
        flex-grow: 1;
    }
    
    @media (max-width: 768px) {
        .system-description {
            font-size: 0.85rem;
            margin-bottom: 0.8rem;
        }
    }
    
    @media (max-width: 480px) {
        .system-description {
            font-size: 0.8rem;
            margin-bottom: 0.6rem;
        }
    }
    
    /* Lista de características responsive */
    .feature-list {
        text-align: left;
        margin: 1rem 0;
    }
    
    .feature-item {
        color: #28a745;
        font-size: 0.8rem;
        margin: 0.3rem 0;
        line-height: 1.4;
        display: flex;
        align-items: center;
    }
    
    @media (max-width: 768px) {
        .feature-item {
            font-size: 0.75rem;
            margin: 0.25rem 0;
        }
    }
    
    @media (max-width: 480px) {
        .feature-item {
            font-size: 0.7rem;
            margin: 0.2rem 0;
        }
    }
    
    /* Botones responsive */
    .stButton > button {
        width: 100%;
        height: 48px;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
        border: none;
    }
    
    @media (max-width: 768px) {
        .stButton > button {
            height: 44px;
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .stButton > button {
            height: 40px;
            font-size: 0.85rem;
        }
    }
    
    /* Grid responsive para columnas */
    .responsive-grid {
        display: grid;
        gap: 1rem;
        grid-template-columns: 1fr;
    }
    
    @media (min-width: 768px) {
        .responsive-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }
    }
    
    @media (min-width: 1024px) {
        .responsive-grid {
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
        }
    }
    
    /* Títulos principales responsive */
    h3 {
        text-align: center;
        margin: 1.5rem 0 2rem 0 !important;
        color: #2c3e50;
        font-weight: 500;
    }
    
    @media (max-width: 768px) {
        h3 {
            font-size: 1.2rem !important;
            margin: 1rem 0 1.5rem 0 !important;
        }
    }
    
    @media (max-width: 480px) {
        h3 {
            font-size: 1.1rem !important;
            margin: 0.8rem 0 1rem 0 !important;
        }
    }
    
    /* Información adicional responsive */
    .stAlert {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    @media (max-width: 768px) {
        .stAlert {
            font-size: 0.85rem;
            margin: 0.3rem 0;
        }
    }
    
    @media (max-width: 480px) {
        .stAlert {
            font-size: 0.8rem;
            margin: 0.2rem 0;
        }
    }
    
    /* Navbar de navegación interno más sobrio */
    .navbar-container {
        background: #34495e;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .navbar-title {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 500;
    }
    
    @media (max-width: 768px) {
        .navbar-container {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .navbar-title {
            font-size: 1.3rem;
        }
    }
    
    @media (max-width: 480px) {
        .navbar-container {
            padding: 0.8rem;
        }
        .navbar-title {
            font-size: 1.1rem;
        }
    }
    
    /* Botón de regreso más sobrio */
    .back-button-container {
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Asegurar que las columnas de Streamlit sean responsive */
    @media (max-width: 768px) {
        .stColumns {
            flex-direction: column;
        }
        
        .stColumn {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

####################################################################
#            FUNCIONES PRINCIPALES
####################################################################

def show_home():
    """Muestra la página principal con selección de sistemas"""
    
    # Header principal más sobrio
    st.markdown("""
    <div class="navbar">
        <h1>🌍 Sistema de Consulta de Tarifas</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Selecciona el tipo de transporte para consultar tarifas:")
    
    # Layout responsive - las columnas se adaptan automáticamente
    col1, col2, col3 = st.columns([1, 1, 1])
    
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
    
    # Información adicional responsive
    st.markdown("---")
    
    # Usar flex en lugar de columnas para mejor responsive
    info_col1, info_col2, info_col3 = st.columns([1, 1, 1])
    
    with info_col1:
        st.info("**CARGA AÉREA** - Para envíos urgentes y mercancías de alto valor")
    
    with info_col2:
        st.info("**FCL MARÍTIMO** - Para grandes volúmenes que llenan contenedores completos")
    
    with info_col3:
        st.info("**LCL MARÍTIMO** - Para cargas pequeñas que se consolidan con otros envíos")

def run_aereo_system():
    """Ejecuta el sistema de carga aérea"""
    try:
        # Importar y ejecutar el sistema aéreo
        from AEREO.app import main as aereo_main
        
        # Navbar de navegación más sobrio
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
        
        # Navbar de navegación más sobrio
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
        
        # Navbar de navegación más sobrio
        show_navbar("LCL MARÍTIMO", "🌊")
        
        # Ejecutar sistema LCL
        lcl_main()
        
    except ImportError as e:
        st.error(f"Error importando sistema LCL: {e}")
        st.info("Asegúrate de que los archivos del sistema LCL estén en la carpeta LCL/")

def show_navbar(system_name, icon):
    """Muestra navbar de navegación más sobrio y responsive"""
    
    # Navbar más simple y sobrio
    st.markdown(f"""
    <div class="navbar-container">
        <h1 class="navbar-title">{icon} {system_name}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Botón de regreso centrado y más sobrio
    col1, col2, col3 = st.columns([1, 1, 1])
    
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