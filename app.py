"""
Point d'entrée Streamlit — POC Passages en restauration scolaire.
"""
import streamlit as st

from data import last_sync_time, sync_data

st.set_page_config(
    page_title="POC Passages",
    page_icon="🍽️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Barre latérale — Synchronisation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("POC Passages")
    st.markdown("---")

    last = last_sync_time()
    if last:
        st.caption(f"Dernière synchro : {last}")
    else:
        st.warning("Aucun cache local. Synchronisez les données.")

    if st.button("🔄 Synchroniser les données", use_container_width=True):
        with st.spinner("Connexion à Trino et chargement…"):
            try:
                sync_data()
                st.success("Synchronisation réussie !")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de la synchronisation : {e}")

# ---------------------------------------------------------------------------
# Navigation multipage
# ---------------------------------------------------------------------------
pages = [
    st.Page("pages/1_Tableau_de_bord.py", title="Tableau de bord", icon="📊"),
    st.Page("pages/2_Etablissement.py", title="Par établissement", icon="🏫"),
    st.Page("pages/3_Comparaison_Prod.py", title="Comparaison Prod", icon="🔍"),
    st.Page("pages/4_Monitorage_Modele.py", title="Monitorage modèle", icon="📈"),
]
pg = st.navigation(pages, position="sidebar")
pg.run()
