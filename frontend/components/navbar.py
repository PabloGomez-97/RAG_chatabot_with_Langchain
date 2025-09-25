import streamlit as st

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