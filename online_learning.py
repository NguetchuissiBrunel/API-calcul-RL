"""
Online Learning System - Le modèle s'améliore après chaque prédiction
Le système collecte les retours réels et met à jour le modèle en continu.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from env import TravelCostEnv
import numpy as np
import os
import json
from datetime import datetime

class OnlineLearningPredictor:
    """
    Système de prédiction avec apprentissage continu.
    Le modèle s'améliore après chaque prédiction en collectant les retours réels.
    """
    
    def __init__(self, model_path=None, update_frequency=10):
        """
        Args:
            model_path: Chemin vers le modèle pré-entraîné
            update_frequency: Nombre de prédictions avant de mettre à jour le modèle
        """
        self.env = TravelCostEnv()
        self.update_frequency = update_frequency
        self.feedback_buffer = []
        self.prediction_count = 0
        self.update_count = 0
        
        # Créer le dossier pour les données
        os.makedirs("online_learning_data", exist_ok=True)
        self.feedback_file = "online_learning_data/feedback_history.json"
        
        # Charger le modèle pré-entraîné ou créer un nouveau
        if model_path and os.path.exists(model_path):
            print(f"✅ Chargement du modèle: {model_path}")
            self.model = PPO.load(model_path, env=self.env)
        else:
            print("⚠️  Aucun modèle trouvé, création d'un nouveau modèle...")
            self.model = PPO("MlpPolicy", self.env, verbose=0)
        
        # Charger l'historique des feedbacks
        self.load_feedback_history()
    
    def load_feedback_history(self):
        """Charge l'historique des feedbacks depuis le fichier JSON."""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                self.feedback_history = json.load(f)
            print(f"📊 {len(self.feedback_history)} feedbacks chargés depuis l'historique")
        else:
            self.feedback_history = []
    
    def save_feedback_history(self):
        """Sauvegarde l'historique des feedbacks."""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def predict(self, distance, road_type, traffic, rain, night, accident):
        """
        Fait une prédiction pour un voyage.
        
        Args:
            distance: Distance en km
            road_type: Type de route (0=Pavé, 1=Terre, 2=Cassé)
            traffic: Niveau de traffic (0=Faible, 1=Moyen, 2=Élevé)
            rain: Intensité de la pluie (0.0 à 1.0)
            night: Nuit (0=Jour, 1=Nuit)
            accident: Accident (0=Non, 1=Oui)
        
        Returns:
            predicted_cost: Coût prédit en CFA
        """
        observation = np.array([distance, road_type, traffic, rain, night, accident], dtype=np.float32)
        action, _ = self.model.predict(observation, deterministic=True)
        predicted_cost = float(action[0])
        
        self.prediction_count += 1
        
        return predicted_cost, observation
    
    def add_feedback(self, observation, predicted_cost, actual_cost):
        """
        Ajoute un feedback avec le coût réel du voyage.
        Le modèle apprendra de cette expérience.
        
        Args:
            observation: Les paramètres du voyage
            predicted_cost: Le coût prédit par le modèle
            actual_cost: Le coût réel du voyage (fourni par l'utilisateur)
        """
        error = abs(predicted_cost - actual_cost)
        error_pct = (error / actual_cost) * 100 if actual_cost > 0 else 0
        
        # Ajouter au buffer
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "observation": observation.tolist(),
            "predicted_cost": predicted_cost,
            "actual_cost": actual_cost,
            "error": error,
            "error_pct": error_pct
        }
        
        self.feedback_buffer.append(feedback)
        self.feedback_history.append(feedback)
        
        print(f"\n📝 Feedback enregistré:")
        print(f"   Prédit: {predicted_cost:,.2f} CFA")
        print(f"   Réel: {actual_cost:,.2f} CFA")
        print(f"   Erreur: {error:,.2f} CFA ({error_pct:.1f}%)")
        
        # Sauvegarder l'historique
        self.save_feedback_history()
        
        # Vérifier si on doit mettre à jour le modèle
        if len(self.feedback_buffer) >= self.update_frequency:
            self.update_model()
    
    def update_model(self):
        """
        Met à jour le modèle avec les feedbacks collectés.
        C'est ici que le modèle apprend et s'améliore!
        """
        if len(self.feedback_buffer) == 0:
            return
        
        print(f"\n🔄 Mise à jour du modèle avec {len(self.feedback_buffer)} nouveaux feedbacks...")
        
        # Créer un environnement spécifique avec les données de feedback uniquement
        update_env = TravelCostEnv(feedback_data=self.feedback_buffer)
        
        # Assigner l'environnement au modèle pour l'entraînement
        self.model.set_env(update_env)
        
        # Nombre d'étapes d'entraînement: on repasse plusieurs fois sur chaque feedback
        # pour s'assurer que le modèle "imprime" bien l'erreur et la correction.
        epochs = 10
        training_steps = len(self.feedback_buffer) * epochs
        
        # Entraîner le modèle sur les feedbacks réels
        self.model.learn(total_timesteps=training_steps, reset_num_timesteps=False)
        
        # Réinitialiser l'environnement de base (si nécessaire)
        self.model.set_env(self.env)
        
        self.update_count += 1
        
        # Sauvegarder le modèle mis à jour
        model_save_path = f"online_learning_data/model_update_{self.update_count}.zip"
        self.model.save(model_save_path)
        
        print(f"✅ Modèle mis à jour et sauvegardé: {model_save_path}")
        print(f"   Total de mises à jour: {self.update_count}")
        print(f"   Total de prédictions: {self.prediction_count}")
        
        # Vider le buffer
        self.feedback_buffer = []
    
    def get_statistics(self):
        """Affiche les statistiques d'apprentissage."""
        if len(self.feedback_history) == 0:
            print("Aucune donnée disponible")
            return
        
        errors = [f["error"] for f in self.feedback_history]
        error_pcts = [f["error_pct"] for f in self.feedback_history]
        
        print("\n" + "="*60)
        print("📊 STATISTIQUES D'APPRENTISSAGE CONTINU")
        print("="*60)
        print(f"\nNombre total de prédictions: {len(self.feedback_history)}")
        print(f"Nombre de mises à jour du modèle: {self.update_count}")
        print(f"\nPerformance:")
        print(f"  Erreur moyenne: {np.mean(errors):,.2f} CFA")
        print(f"  Erreur médiane: {np.median(errors):,.2f} CFA")
        print(f"  Erreur min: {np.min(errors):,.2f} CFA")
        print(f"  Erreur max: {np.max(errors):,.2f} CFA")
        print(f"  Erreur % moyenne: {np.mean(error_pcts):.1f}%")
        
        # Analyser l'amélioration au fil du temps
        if len(self.feedback_history) >= 20:
            first_10_errors = errors[:10]
            last_10_errors = errors[-10:]
            
            improvement = (np.mean(first_10_errors) - np.mean(last_10_errors)) / np.mean(first_10_errors) * 100
            
            print(f"\n📈 Amélioration au fil du temps:")
            print(f"  Erreur moyenne (10 premiers): {np.mean(first_10_errors):,.2f} CFA")
            print(f"  Erreur moyenne (10 derniers): {np.mean(last_10_errors):,.2f} CFA")
            print(f"  Amélioration: {improvement:+.1f}%")
        
        print("="*60)


def interactive_online_learning():
    """Mode interactif avec apprentissage continu."""
    print("\n" + "="*60)
    print("🧠 SYSTÈME D'APPRENTISSAGE CONTINU")
    print("="*60)
    print("\nLe modèle s'améliore après chaque prédiction!")
    print("Vous devrez fournir le coût réel après chaque prédiction.\n")
    
    # Trouver le dernier modèle entraîné
    models_dir = "models/PPO"
    model_path = None
    
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        if models:
            models.sort(key=lambda x: int(x.replace('.zip', '')))
            model_path = os.path.join(models_dir, models[-1])
    
    # Créer le système d'apprentissage
    print("Configuration:")
    update_freq = int(input("Fréquence de mise à jour (nombre de prédictions avant mise à jour, défaut=10): ") or "10")
    
    predictor = OnlineLearningPredictor(model_path=model_path, update_frequency=update_freq)
    
    while True:
        print("\n" + "-"*60)
        print("📍 NOUVELLE PRÉDICTION")
        print("-"*60)
        
        # Collecter les paramètres du voyage
        try:
            distance = float(input("Distance (km): "))
            
            print("\nType de route: 0=Pavé, 1=Terre, 2=Cassé")
            road_type = int(input("Type de route (0-2): "))
            
            print("\nNiveau de traffic: 0=Faible, 1=Moyen, 2=Élevé")
            traffic = int(input("Traffic (0-2): "))
            
            rain = float(input("Intensité de la pluie (0.0-1.0): "))
            night = int(input("Nuit? (0=Jour, 1=Nuit): "))
            accident = int(input("Accident? (0=Non, 1=Oui): "))
            
        except ValueError:
            print("❌ Valeur invalide!")
            continue
        
        # Faire la prédiction
        predicted_cost, observation = predictor.predict(distance, road_type, traffic, rain, night, accident)
        
        print(f"\n💰 COÛT PRÉDIT: {predicted_cost:,.2f} CFA")
        
        # Demander le coût réel
        print("\n" + "-"*60)
        real_cost_input = input("Entrez le coût RÉEL du voyage (ou 'skip' pour passer): ")
        
        if real_cost_input.lower() != 'skip':
            try:
                actual_cost = float(real_cost_input)
                predictor.add_feedback(observation, predicted_cost, actual_cost)
            except ValueError:
                print("❌ Coût invalide, feedback non enregistré")
        else:
            print("⏭️  Feedback ignoré")
        
        # Demander si on continue
        choice = input("\nAutre prédiction? (y/n): ").strip().lower()
        if choice != 'y':
            break
    
    # Afficher les statistiques finales
    predictor.get_statistics()
    
    print("\n✅ Session terminée!")
    print(f"📁 Données sauvegardées dans: online_learning_data/")


def demo_online_learning():
    """Démo automatique de l'apprentissage continu avec simulation."""
    print("\n" + "="*60)
    print("🎬 DÉMO D'APPRENTISSAGE CONTINU")
    print("="*60)
    print("\nSimulation de 50 prédictions avec feedbacks automatiques\n")
    
    from simulation import calculate_true_cost
    
    # Trouver le dernier modèle
    models_dir = "models/PPO"
    model_path = None
    
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        if models:
            models.sort(key=lambda x: int(x.replace('.zip', '')))
            model_path = os.path.join(models_dir, models[-1])
    
    predictor = OnlineLearningPredictor(model_path=model_path, update_frequency=5)
    
    # Faire 50 prédictions avec feedback automatique
    for i in range(50):
        # Générer un voyage aléatoire
        distance = np.random.uniform(10, 300)
        road_type = np.random.randint(0, 3)
        traffic = np.random.randint(0, 3)
        rain = np.random.uniform(0, 1)
        night = 1 if np.random.random() > 0.7 else 0
        accident = 1 if np.random.random() > 0.9 else 0
        
        # Prédiction
        predicted_cost, observation = predictor.predict(distance, road_type, traffic, rain, night, accident)
        
        # Calculer le coût réel
        actual_cost = calculate_true_cost(distance, road_type, traffic, rain, bool(night), bool(accident))
        
        # Ajouter le feedback
        predictor.add_feedback(observation, predicted_cost, actual_cost)
        
        if (i + 1) % 10 == 0:
            print(f"\n✅ {i + 1}/50 prédictions complétées")
    
    # Statistiques finales
    predictor.get_statistics()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 SYSTÈME D'APPRENTISSAGE CONTINU")
    print("="*60)
    print("\nChoisissez le mode:")
    print("  1 = Mode Interactif (Vous entrez les données réelles)")
    print("  2 = Mode Démo (Simulation automatique)")
    
    while True:
        try:
            choice = int(input("\nChoix (1-2): "))
            if choice in [1, 2]:
                break
            print("⚠️  Entrez 1 ou 2!")
        except ValueError:
            print("⚠️  Valeur invalide!")
    
    if choice == 1:
        interactive_online_learning()
    else:
        demo_online_learning()
