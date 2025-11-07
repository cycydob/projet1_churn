from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import os

# Initialiser FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API pour prédire le churn des clients télécoms",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle et les preprocesseurs
try:
    model = joblib.load('churn_model.pkl')
    encoders = joblib.load('encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✅ Modèle et preprocesseurs chargés avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    model = None

# Modèle de données pour l'entrée
class CustomerData(BaseModel):
    gender: str = Field(..., description="Male ou Female")
    seniorcitizen: int = Field(..., ge=0, le=1, description="0 ou 1")
    partner: str = Field(..., description="Yes ou No")
    dependents: str = Field(..., description="Yes ou No")
    tenure: int = Field(..., ge=0, description="Nombre de mois")
    phoneservice: str = Field(..., description="Yes ou No")
    multiplelines: str = Field(..., description="Yes, No, ou No phone service")
    internetservice: str = Field(..., description="DSL, Fiber optic, ou No")
    onlinesecurity: str = Field(..., description="Yes, No, ou No internet service")
    onlinebackup: str = Field(..., description="Yes, No, ou No internet service")
    deviceprotection: str = Field(..., description="Yes, No, ou No internet service")
    techsupport: str = Field(..., description="Yes, No, ou No internet service")
    streamingtv: str = Field(..., description="Yes, No, ou No internet service")
    streamingmovies: str = Field(..., description="Yes, No, ou No internet service")
    contract: str = Field(..., description="Month-to-month, One year, ou Two year")
    paperlessbilling: str = Field(..., description="Yes ou No")
    paymentmethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), ou Credit card (automatic)")
    monthlycharges: float = Field(..., gt=0, description="Montant mensuel")
    totalcharges: float = Field(..., ge=0, description="Montant total")

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "seniorcitizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phoneservice": "Yes",
                "multiplelines": "No",
                "internetservice": "Fiber optic",
                "onlinesecurity": "No",
                "onlinebackup": "Yes",
                "deviceprotection": "No",
                "techsupport": "No",
                "streamingtv": "Yes",
                "streamingmovies": "Yes",
                "contract": "Month-to-month",
                "paperlessbilling": "Yes",
                "paymentmethod": "Electronic check",
                "monthlycharges": 85.50,
                "totalcharges": 1026.00
            }
        }

# Modèle de réponse
class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    confidence: str
    risk_level: str
    recommendations: list

def preprocess_input(data: CustomerData):
    """Prétraite les données d'entrée"""
    # Convertir en DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Forcer les colonnes en minuscules
    df.columns = df.columns.str.lower()
    
    # Identifier colonnes numériques et catégorielles
    numeric_features = ['tenure', 'totalcharges', 'monthlycharges']
    categorical_features = [col for col in df.columns if col not in numeric_features]
    
    # Encoder les variables catégorielles
    for col in categorical_features:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except ValueError:
                # Si nouvelle catégorie, utiliser 0
                df[col] = 0
    
    # Normaliser les variables numériques
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # Réorganiser les colonnes dans le bon ordre
    df = df[feature_names]
    
    return df

def get_risk_level(probability: float) -> str:
    """Détermine le niveau de risque"""
    if probability < 0.3:
        return "Faible"
    elif probability < 0.6:
        return "Moyen"
    else:
        return "Élevé"

def get_recommendations(data: CustomerData, probability: float) -> list:
    """Génère des recommandations personnalisées"""
    recommendations = []
    
    if probability > 0.5:
        if data.contract == "Month-to-month":
            recommendations.append("Proposer un contrat longue durée avec réduction")
        
        if data.techsupport == "No":
            recommendations.append("Offrir un support technique gratuit pendant 3 mois")
        
        if data.onlinesecurity == "No":
            recommendations.append("Proposer des services de sécurité en ligne")
        
        if data.monthlycharges > 70:
            recommendations.append("Envisager une offre promotionnelle sur les charges mensuelles")
        
        if data.tenure < 12:
            recommendations.append("Programme de fidélisation pour nouveaux clients")
    
    if not recommendations:
        recommendations.append("Client fidèle - Maintenir la qualité de service actuelle")
    
    return recommendations

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "🚀 Churn Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict - POST pour prédire le churn",
            "health": "/health - GET pour vérifier le statut",
            "docs": "/docs - Documentation interactive"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifie que l'API fonctionne"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Prédit si un client va churner
    
    - **customer**: Données du client
    
    Returns:
    - Prédiction de churn (Yes/No)
    - Probabilité de churn
    - Niveau de confiance
    - Niveau de risque
    - Recommandations personnalisées
    """
    try:
        # Vérifier que le modèle est chargé
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible")
        
        # Prétraiter les données
        X = preprocess_input(customer)
        
        # Faire la prédiction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Déterminer le niveau de confiance
        confidence = "Élevée" if abs(probability - 0.5) > 0.3 else "Moyenne"
        
        # Obtenir le niveau de risque
        risk_level = get_risk_level(probability)
        
        # Générer des recommandations
        recommendations = get_recommendations(customer, probability)
        
        return PredictionResponse(
            churn_prediction="Yes" if prediction == "Yes" else "No",
            churn_probability=round(float(probability), 4),
            confidence=confidence,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Retourne des informations sur le modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "features": feature_names,
        "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None
    }

# Pour exécuter localement: uvicorn backend:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)