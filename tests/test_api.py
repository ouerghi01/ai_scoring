from app.main import app


client = app.test_client()


def test_predict():
    data = {
        "age": 35,
        "situation_familiale": "Marié",
        "nombre_enfants": 2,
        "statut_professionnel": "Employé",
        "type_contrat": "CDI",
        "revenus_mensuels_nets": 2500,
        "apport_propre": 5000,
        "duree_financement": "60 mois",
        "historique_credit": "Bon",
        "montant_credit": 30000,
        "anciennete_emploi": "5 ans",
        "charges_mensuelles": 900,
        "compte_bancaire_actif": "Oui",
        "incident_bancaire": "Non",
        "region": "Tunis",
    }

    response = client.post("/predict", json=data)

    print("Status code:", response.status_code)
    print("Response data:", response.data.decode())  # raw response content as string

    assert response.status_code == 200

    json_data = response.get_json()
    print("JSON data:", json_data)  # For debugging purposes

    assert json_data is not None, "Response JSON is None! Check your endpoint."

    assert "prediction" in json_data
    assert "probability_not_approved" in json_data
    assert "probability_approved" in json_data

    # Optional: check value ranges
    assert json_data["prediction"] in [0, 1]
    assert 0.0 <= json_data["probability_not_approved"] <= 1.0
    assert 0.0 <= json_data["probability_approved"] <= 1.0


def test_predict_bad_data():
    # Example of bad data that should lead to rejection (prediction = 0)
    bad_data = {
        "age": 22,  # young age, maybe less stable
        "situation_familiale": "Célibataire",
        "nombre_enfants": 0,
        "statut_professionnel": "Indépendant",
        "type_contrat": "Auto-entrepreneur",
        "revenus_mensuels_nets": 800,  # low income
        "apport_propre": 500,  # very low own capital
        "duree_financement": "12 mois",
        "historique_credit": "Faible",  # bad credit history
        "montant_credit": 20000,
        "anciennete_emploi": "6 mois",  # short job seniority
        "charges_mensuelles": 1100,  # high charges compared to income
        "compte_bancaire_actif": "Oui",
        "incident_bancaire": "Oui",  # previous incidents
        "region": "Kairouan",
    }

    response = client.post("/predict", json=bad_data)

    print("Status code:", response.status_code)
    print("Response data:", response.data.decode())

    assert response.status_code == 200

    json_data = response.get_json()
    assert json_data is not None, "Response JSON is None"

    assert "prediction" in json_data
    assert (
        json_data["prediction"] == 0
    ), "Expected rejection prediction (0) for bad data"

    assert 0.0 <= json_data["probability_not_approved"] <= 1.0
    assert 0.0 <= json_data["probability_approved"] <= 1.0
