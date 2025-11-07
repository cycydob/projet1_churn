import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# Configuration de la page
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Option 2: Secrets Streamlit (alternative)
try:
    if hasattr(st, 'secrets') and 'API_URL' in st.secrets:
        API_URL = st.secrets["API_URL"]
except:
    pass

# Styles CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4b4b;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-medium {
        background-color: #ffa500;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background-color: #00cc00;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.75rem;
        border: none;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.title("📊 Prédiction de Churn Client")
st.markdown("### Système intelligent de prédiction du risque de désabonnement")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=100)
    st.markdown("##  Configuration")
    
    # Test de connexion API
    with st.spinner("Vérification de l'API..."):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connectée")
            else:
                st.error("❌ API non disponible")
        except:
            st.error("❌ Impossible de se connecter à l'API")
    
    st.markdown("---")
    st.markdown("### 📖 Guide d'utilisation")
    st.info("""
    1. Remplissez les informations client
    2. Cliquez sur 'Prédire le Churn'
    3. Consultez les résultats et recommandations
    """)

# Onglets principaux
tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📈 Analyse", "ℹ️ À propos"])

with tab1:
    st.markdown("## Informations Client")
    
    # Formulaire de saisie
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Informations Personnelles")
        gender = st.selectbox("Genre", ["Male", "Female"])
        seniorcitizen = st.selectbox("Senior Citizen", [0, 1], 
                                     format_func=lambda x: "Oui" if x == 1 else "Non")
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    
    with col2:
        st.markdown("### 📞 Services Téléphoniques")
        phoneservice = st.selectbox("Service téléphonique", ["Yes", "No"])
        multiplelines = st.selectbox("Lignes multiples", 
                                    ["No", "Yes", "No phone service"])
        
        st.markdown("###  Services Internet")
        internetservice = st.selectbox("Service Internet", 
                                      ["DSL", "Fiber optic", "No"])
        onlinesecurity = st.selectbox("Sécurité en ligne", 
                                     ["No", "Yes", "No internet service"])
        onlinebackup = st.selectbox("Sauvegarde en ligne", 
                                   ["No", "Yes", "No internet service"])
    
    with col3:
        st.markdown("###  Services Additionnels")
        deviceprotection = st.selectbox("Protection appareil", 
                                       ["No", "Yes", "No internet service"])
        techsupport = st.selectbox("Support technique", 
                                  ["No", "Yes", "No internet service"])
        streamingtv = st.selectbox("Streaming TV", 
                                  ["No", "Yes", "No internet service"])
        streamingmovies = st.selectbox("Streaming Films", 
                                      ["No", "Yes", "No internet service"])
    
    # Deuxième ligne de colonnes
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("### 📋 Contrat")
        contract = st.selectbox("Type de contrat", 
                               ["Month-to-month", "One year", "Two year"])
        paperlessbilling = st.selectbox("Facturation électronique", ["Yes", "No"])
    
    with col5:
        st.markdown("### 💳 Paiement")
        paymentmethod = st.selectbox("Méthode de paiement", 
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer (automatic)", 
                                     "Credit card (automatic)"])
    
    with col6:
        st.markdown("### 💰 Finances")
        monthlycharges = st.number_input("Charges mensuelles ($)", 
                                        min_value=0.0, max_value=200.0, 
                                        value=65.0, step=0.5)
        totalcharges = st.number_input("Charges totales ($)", 
                                      min_value=0.0, max_value=10000.0, 
                                      value=780.0, step=10.0)
    
    # Bouton de prédiction
    st.markdown("---")
    if st.button("🔮 Prédire le Churn", type="primary", use_container_width=True):
        # Préparer les données
        customer_data = {
            "gender": gender,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }
        
        # Faire la prédiction
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=customer_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Afficher les résultats
                    st.markdown("---")
                    st.markdown("## 📊 Résultats de la Prédiction")
                    
                    # Métriques principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        churn_emoji = "🔴" if result['churn_prediction'] == "Yes" else "🟢"
                        st.metric(
                            "Prédiction",
                            f"{churn_emoji} {result['churn_prediction']}"
                        )
                    
                    with col2:
                        st.metric(
                            "Probabilité",
                            f"{result['churn_probability']*100:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Confiance",
                            result['confidence']
                        )
                    
                    with col4:
                        risk_color = {
                            "Faible": "🟢",
                            "Moyen": "🟡",
                            "Élevé": "🔴"
                        }
                        st.metric(
                            "Niveau de Risque",
                            f"{risk_color.get(result['risk_level'], '⚪')} {result['risk_level']}"
                        )
                    
                    # Jauge de probabilité
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['churn_probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilité de Churn (%)"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 60], 'color': "yellow"},
                                {'range': [60, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommandations
                    st.markdown("### 💡 Recommandations")
                    for i, rec in enumerate(result['recommendations'], 1):
                        st.info(f"{i}. {rec}")
                    
                else:
                    st.error(f"Erreur: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction: {str(e)}")

with tab2:
    st.markdown("## 📈 Analyse des Facteurs de Risque")
    
    # Graphique des facteurs de risque
    risk_factors = {
        "Contrat court terme": 0.8 if contract == "Month-to-month" else 0.2,
        "Faible ancienneté": 0.7 if tenure < 12 else 0.3,
        "Charges élevées": 0.6 if monthlycharges > 70 else 0.3,
        "Pas de support tech": 0.5 if techsupport == "No" else 0.2,
        "Pas de sécurité": 0.4 if onlinesecurity == "No" else 0.2
    }
    
    df_risk = pd.DataFrame(list(risk_factors.items()), 
                          columns=['Facteur', 'Score de Risque'])
    
    fig = px.bar(df_risk, x='Score de Risque', y='Facteur', 
                 orientation='h',
                 color='Score de Risque',
                 color_continuous_scale='RdYlGn_r',
                 title="Analyse des Facteurs de Risque")
    st.plotly_chart(fig, use_container_width=True)
    
    # Profil client
    st.markdown("### 👤 Profil Client")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Informations démographiques**")
        st.write(f"- Genre: {gender}")
        st.write(f"- Senior: {'Oui' if seniorcitizen == 1 else 'Non'}")
        st.write(f"- Partenaire: {partner}")
        st.write(f"- Personnes à charge: {dependents}")
    
    with col2:
        st.markdown("**Informations contractuelles**")
        st.write(f"- Ancienneté: {tenure} mois")
        st.write(f"- Type de contrat: {contract}")
        st.write(f"- Charges mensuelles: ${monthlycharges:.2f}")
        st.write(f"- Charges totales: ${totalcharges:.2f}")

with tab3:
    st.markdown("## ℹ️ À propos de l'application")
    
    st.markdown("""
    ### 🎯 Objectif
    Cette application utilise le machine learning pour prédire le risque de churn 
    (désabonnement) des clients d'une entreprise de télécommunications.
    
    ### 🤖 Modèle
    - **Algorithme**: Random Forest Classifier
    - **Features**: 19 caractéristiques client
    - **Performance**: ~77% d'accuracy, ROC-AUC ~0.84
    
    ### 📊 Fonctionnalités
    - Prédiction en temps réel du risque de churn
    - Analyse des facteurs de risque
    - Recommandations personnalisées
    - Interface intuitive et interactive
    
    """)
    
    st.markdown("---")
    st.markdown("**Développé par Nancy Dobé**")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📧 Pour toute question ou suggestion, contactez l'équipe de développement</p>
</div>
""", unsafe_allow_html=True)