$body = @{
  age = 58                                # Close to retirement
  situation_familiale = "Divorcé(e)"
  nombre_enfants = 5                      # Very high family burden
  statut_professionnel = "Privé"
  type_contrat = "Sans contrat"          # Invalid or unsupported type
  revenus_mensuels_nets = 1000            # Low income
  apport_propre = 0
  duree_financement = "144 mois"         # Long repayment horizon
  historique_credit = "Non"
  montant_credit = 40000
  anciennete_emploi = "1 an"             # Low job stability
  charges_mensuelles = 1300              # High relative to income
  compte_bancaire_actif = "Non"
  incident_bancaire = "Oui"
  region = "Gafsa"
}


$jsonBody = $body | ConvertTo-Json -Depth 3
$utf8Body = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)

$response = Invoke-RestMethod -Uri "http://localhost:5000/predict" `
  -Method POST `
  -Body $utf8Body `
  -ContentType "application/json"

$response
