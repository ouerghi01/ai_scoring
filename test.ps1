$body = @{
  age = 35
  situation_familiale = "Marié"
  nombre_enfants = 2
  statut_professionnel = "Employé"
  type_contrat = "CDI"
  revenus_mensuels_nets = 2500
  apport_propre = 5000
  duree_financement = "60 mois"
  historique_credit = "Bon"
  montant_credit = 30000
  anciennete_emploi = "5 ans"
  charges_mensuelles = 900
  compte_bancaire_actif = "Oui"
  incident_bancaire = "Non"
  region = "Tunis"
}

# Convert to JSON with UTF8 encoding
$jsonBody = $body | ConvertTo-Json -Depth 3
$utf8Body = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)

$response = Invoke-RestMethod -Uri "http://localhost:5000/predict" `
  -Method POST `
  -Body $utf8Body `
  -ContentType "application/json"

$response
