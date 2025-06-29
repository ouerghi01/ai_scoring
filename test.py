import requests
import concurrent.futures
import time

url = "http://192.168.1.3:5000/predict"
payload = {
    "age": 58,
    "situation_familiale": "Divorcé(e)",
    "nombre_enfants": 5,
    "statut_professionnel": "Privé",
    "type_contrat": "Sans contrat",
    "revenus_mensuels_nets": 900,
    "apport_propre": 0,
    "duree_financement": "144 mois",
    "historique_credit": "Non",
    "montant_credit": 40000,
    "anciennete_emploi": "1 an",
    "charges_mensuelles": 1300,
    "compte_bancaire_actif": "Non",
    "incident_bancaire": "Oui",
    "region": "Gafsa"
}

def send_request():
    try:
        response = requests.post(url, json=payload, timeout=10)  # timeout after 10 sec
        try:
            data = response.json()
        except ValueError:
            data = {"error": "Non-JSON response"}
        return response.status_code, response.elapsed.total_seconds(), data
    except requests.exceptions.Timeout:
        return "Timeout", 10.0, {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return "Error", 0.0, {"error": str(e)}

# Number of concurrent requests to test
n_requests = 100

start_time = time.time()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
try:
    futures = [executor.submit(send_request) for _ in range(n_requests)]
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        status, elapsed, data = future.result()
        print(f"Request {i+1}: Status={status}, Time={elapsed:.2f}s")
finally:
    executor.shutdown(wait=True)


print(f"\nTotal test time: {time.time() - start_time:.2f} seconds")
