# 🧠 Apprentissage Continu - Guide d'Utilisation

## 📋 Qu'est-ce que l'apprentissage continu?

Le système d'**apprentissage continu** (Online Learning) permet au modèle de **s'améliorer automatiquement** après chaque prédiction en collectant vos retours sur les coûts réels.

### 🔄 Comment ça marche?

```
1. Vous demandez une prédiction
   ↓
2. Le modèle prédit le coût
   ↓
3. Vous fournissez le coût RÉEL du voyage
   ↓
4. Le modèle apprend de son erreur
   ↓
5. Le modèle s'améliore! 📈
```

---

## 🚀 Utilisation

### Mode 1: Interactif (Recommandé pour usage réel)

```powershell
.\.venv\bin\python.exe online_learning.py
```

Choisissez **Mode 1** et suivez les instructions:

1. **Entrez les paramètres du voyage:**
   - Distance (km)
   - Type de route (0=Pavé, 1=Terre, 2=Cassé)
   - Niveau de traffic (0=Faible, 1=Moyen, 2=Élevé)
   - Intensité de la pluie (0.0 à 1.0)
   - Nuit (0=Jour, 1=Nuit)
   - Accident (0=Non, 1=Oui)

2. **Le modèle prédit le coût**

3. **Vous entrez le coût RÉEL** (après avoir fait le voyage)

4. **Le modèle apprend et s'améliore!**

### Mode 2: Démo (Pour tester le système)

```powershell
.\.venv\bin\python.exe online_learning.py
```

Choisissez **Mode 2** pour une simulation automatique de 50 prédictions avec feedbacks.

---

## ⚙️ Configuration

### Fréquence de mise à jour

Par défaut, le modèle se met à jour tous les **10 feedbacks**. Vous pouvez changer cela:

```python
predictor = OnlineLearningPredictor(
    model_path="models/PPO/100000.zip",
    update_frequency=5  # Mise à jour tous les 5 feedbacks
)
```

**Recommandations:**
- `update_frequency=5` : Apprentissage rapide, mais peut être instable
- `update_frequency=10` : **Recommandé** - Bon équilibre
- `update_frequency=20` : Apprentissage lent, mais plus stable

---

## 📊 Données Sauvegardées

Toutes les données sont sauvegardées dans `online_learning_data/`:

```
online_learning_data/
├── feedback_history.json       # Historique de tous les feedbacks
├── model_update_1.zip          # Modèle après 1ère mise à jour
├── model_update_2.zip          # Modèle après 2ème mise à jour
└── ...
```

### Format des feedbacks (JSON)

```json
{
  "timestamp": "2026-01-28T13:20:00",
  "observation": [100, 0, 1, 0.3, 0, 0],
  "predicted_cost": 13500.50,
  "actual_cost": 14200.00,
  "error": 699.50,
  "error_pct": 4.93
}
```

---

## 📈 Statistiques d'Amélioration

Le système affiche automatiquement les statistiques:

```
📊 STATISTIQUES D'APPRENTISSAGE CONTINU
========================================

Nombre total de prédictions: 50
Nombre de mises à jour du modèle: 5

Performance:
  Erreur moyenne: 1,234.56 CFA
  Erreur médiane: 987.23 CFA
  Erreur min: 123.45 CFA
  Erreur max: 3,456.78 CFA
  Erreur % moyenne: 8.5%

📈 Amélioration au fil du temps:
  Erreur moyenne (10 premiers): 1,850.00 CFA
  Erreur moyenne (10 derniers): 890.00 CFA
  Amélioration: +51.9%
```

---

## 🎯 Exemple d'Utilisation Réelle

### Scénario: Vous êtes un chauffeur de taxi

```python
# Démarrer le système
predictor = OnlineLearningPredictor(
    model_path="models/PPO/100000.zip",
    update_frequency=10
)

# Matin - Client 1
predicted, obs = predictor.predict(
    distance=25,      # 25 km
    road_type=0,      # Route pavée
    traffic=2,        # Traffic élevé (heure de pointe)
    rain=0,           # Pas de pluie
    night=0,          # Jour
    accident=0        # Pas d'accident
)
print(f"Prix proposé: {predicted:.0f} CFA")
# Après le voyage, le client a payé 3500 CFA
predictor.add_feedback(obs, predicted, 3500)

# Midi - Client 2
predicted, obs = predictor.predict(
    distance=50,
    road_type=1,      # Route en terre
    traffic=0,        # Traffic faible
    rain=0.6,         # Pluie modérée
    night=0,
    accident=0
)
print(f"Prix proposé: {predicted:.0f} CFA")
# Client a payé 8200 CFA
predictor.add_feedback(obs, predicted, 8200)

# ... Après 10 courses, le modèle se met à jour automatiquement!
# Les prédictions suivantes seront plus précises!
```

---

## 🔧 Intégration dans une Application

### Exemple: API REST

```python
from flask import Flask, request, jsonify
from online_learning import OnlineLearningPredictor

app = Flask(__name__)
predictor = OnlineLearningPredictor(
    model_path="models/PPO/100000.zip",
    update_frequency=10
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    predicted_cost, obs = predictor.predict(
        data['distance'],
        data['road_type'],
        data['traffic'],
        data['rain'],
        data['night'],
        data['accident']
    )
    return jsonify({
        'predicted_cost': predicted_cost,
        'observation_id': len(predictor.feedback_history)
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    predictor.add_feedback(
        np.array(data['observation']),
        data['predicted_cost'],
        data['actual_cost']
    )
    return jsonify({'status': 'success'})

@app.route('/stats', methods=['GET'])
def stats():
    predictor.get_statistics()
    return jsonify({'status': 'printed'})
```

---

## ⚡ Avantages de l'Apprentissage Continu

| Avantage | Description |
|----------|-------------|
| 🎯 **Précision croissante** | Plus vous l'utilisez, plus il devient précis |
| 🌍 **Adaptation locale** | S'adapte aux conditions spécifiques de votre région |
| 📅 **Évolution temporelle** | S'adapte aux changements de prix au fil du temps |
| 🚗 **Personnalisation** | Apprend de VOS données réelles |
| 💾 **Historique complet** | Toutes les données sont sauvegardées |

---

## ⚠️ Limitations et Précautions

### 1. **Qualité des feedbacks**
- ⚠️ Assurez-vous que les coûts réels sont corrects
- ⚠️ Des feedbacks erronés dégradent le modèle

### 2. **Quantité de données**
- ✅ Plus de feedbacks = Meilleur apprentissage
- ⚠️ Minimum 20-30 feedbacks pour voir une amélioration

### 3. **Fréquence de mise à jour**
- ⚠️ Trop fréquent (< 5) = Instable
- ⚠️ Trop rare (> 20) = Apprentissage lent

### 4. **Sauvegarde**
- ✅ Les modèles mis à jour sont sauvegardés automatiquement
- ⚠️ Sauvegardez régulièrement `online_learning_data/`

---

## 🔄 Restaurer un Modèle Précédent

Si le modèle se dégrade après des mauvais feedbacks:

```python
# Charger un modèle précédent
predictor = OnlineLearningPredictor(
    model_path="online_learning_data/model_update_3.zip"
)
```

---

## 📊 Comparer les Versions

Pour voir l'évolution du modèle:

```python
# Évaluer différentes versions
from evaluate_model import evaluate_checkpoint

# Version initiale
errors_v0 = evaluate_checkpoint("models/PPO/100000.zip")

# Après 1ère mise à jour
errors_v1 = evaluate_checkpoint("online_learning_data/model_update_1.zip")

# Après 5ème mise à jour
errors_v5 = evaluate_checkpoint("online_learning_data/model_update_5.zip")

print(f"Erreur initiale: {errors_v0.mean():.2f} CFA")
print(f"Après 1 mise à jour: {errors_v1.mean():.2f} CFA")
print(f"Après 5 mises à jour: {errors_v5.mean():.2f} CFA")
```

---

## 🎓 Cas d'Usage Recommandés

### ✅ Parfait pour:
- Chauffeurs de taxi collectant des données réelles
- Applications de covoiturage
- Entreprises de transport
- Services de livraison
- Études de marché sur les coûts de transport

### ❌ Moins adapté pour:
- Prédictions ponctuelles sans feedback
- Environnements où les coûts réels ne sont pas disponibles
- Cas où les feedbacks sont peu fiables

---

## 🚀 Workflow Complet

```
1. Entraînement initial (une fois)
   → python train_agent.py
   
2. Déploiement avec apprentissage continu
   → python online_learning.py
   
3. Utilisation quotidienne
   → Prédictions + Feedbacks
   
4. Le modèle s'améliore automatiquement!
   → Tous les 10 feedbacks
   
5. Analyse des performances
   → Statistiques automatiques
```

---

## 📞 Support

Pour des questions ou problèmes:
1. Vérifiez que les feedbacks sont corrects
2. Consultez les statistiques régulièrement
3. Sauvegardez vos données fréquemment
4. Testez avec le mode démo d'abord

---

**Le modèle s'améliore avec VOUS! Plus vous l'utilisez, plus il devient précis! 🚀**
