from pydantic import BaseModel, Field
from typing import Literal, Annotated
from dataclasses import dataclass

@dataclass

class LoanApplication(BaseModel):
    age: Annotated[int, Field(ge=18, le=100)]
    situation_familiale: Literal["Célibataire", "Marié", "Divorcé", "Veuf"]
    nombre_enfants: Annotated[int, Field(ge=0, le=10)]
    statut_professionnel: Literal["publiqué", "privé"]
    type_contrat: Literal["CDI", "CDD", "Intérim", "Auto-entrepreneur"]
    revenus_mensuels_nets: Annotated[int, Field(ge=0)]
    apport_propre: Annotated[int, Field(ge=0)]
    duree_financement: Annotated[str, Field(pattern=r"^\d+\smois$")]
    historique_credit: Literal["Oui", "Non"]
    montant_credit: Annotated[int, Field(ge=0)]
    anciennete_emploi: Annotated[str, Field(pattern=r"^\d+\sans$")]
    charges_mensuelles: Annotated[int, Field(ge=0)]
    compte_bancaire_actif: Literal["Oui", "Non"]
    incident_bancaire: Literal["Oui", "Non"]
    region: Literal[
        "Tunis",
        "Sfax",
        "Sousse",
        "Nabeul",
        "Ariana",
        "Monastir",
        "Bizerte",
        "Kairouan",
        "Mahdia",
        "Gabès",
    ]
@dataclass

class TopFeature(BaseModel):
    feature: str
    impact: float
@dataclass

class LoanPredictionResult(BaseModel):
    prediction: Annotated[int, Field(description="0 (Not Approved) or 1 (Approved)")]
    probability_not_approved: Annotated[float, Field(ge=0.0, le=1.0)]
    probability_approved: Annotated[float, Field(ge=0.0, le=1.0)]
    top_features: Annotated[list[TopFeature], Field(
        description="Top features contributing to the prediction, each as an object with 'feature' (str) and 'impact' (float)."
    )] = []  # Optional field for SHAP explanation plot
    recommendations: Annotated[str, Field(
        description="List of recommendations based on the prediction result."
    )] = "No recommendations available at this time."
