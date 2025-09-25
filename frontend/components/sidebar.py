import streamlit as st

def show_home_selection():
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