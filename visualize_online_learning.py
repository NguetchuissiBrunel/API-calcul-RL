import json
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_progress():
    feedback_file = "online_learning_data/feedback_history.json"
    
    if not os.path.exists(feedback_file):
        print("❌ Aucune donnée d'apprentissage trouvée. Lancez 'online_learning.py' d'abord.")
        return
    
    with open(feedback_file, 'r') as f:
        data = json.load(f)
    
    if len(data) == 0:
        print("❌ L'historique des feedbacks est vide.")
        return
    
    errors = [d['error'] for d in data]
    error_pcts = [d['error_pct'] for d in data]
    indices = list(range(1, len(data) + 1))
    
    # Créer les graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Graphique 1: Erreur Absolue (CFA)
    ax1.plot(indices, errors, marker='o', linestyle='-', color='blue', alpha=0.5, label='Erreur par trajet')
    
    # Ajouter une moyenne mobile pour voir la tendance
    if len(errors) >= 5:
        moving_avg = np.convolve(errors, np.ones(5)/5, mode='valid')
        ax1.plot(indices[4:], moving_avg, color='red', linewidth=2, label='Tendance (Moyenne mobile 5)')
    
    ax1.set_title("Évolution de l'Erreur Absolue (CFA)")
    ax1.set_xlabel("Nombre de prédictions")
    ax1.set_ylabel("Erreur (CFA)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Erreur en Pourcentage (%)
    ax2.plot(indices, error_pcts, marker='s', linestyle='--', color='green', alpha=0.5, label='Erreur %')
    
    if len(error_pcts) >= 5:
        moving_avg_pct = np.convolve(error_pcts, np.ones(5)/5, mode='valid')
        ax2.plot(indices[4:], moving_avg_pct, color='orange', linewidth=2, label='Tendance % (Moyenne mobile 5)')
    
    ax2.set_title("Évolution de l'Erreur en Pourcentage (%)")
    ax2.set_xlabel("Nombre de prédictions")
    ax2.set_ylabel("Erreur (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    os.makedirs("online_learning_data", exist_ok=True)
    save_path = "online_learning_data/learning_progress.png"
    plt.savefig(save_path)
    print(f"✅ Graphique de progression sauvegardé : {save_path}")
    
    # Afficher les statistiques de début vs fin
    if len(errors) >= 10:
        start_avg = np.mean(errors[:5])
        end_avg = np.mean(errors[-5:])
        improvement = (start_avg - end_avg) / start_avg * 100 if start_avg > 0 else 0
        
        print(f"\n📈 Analyse de l'amélioration:")
        print(f"   Moyenne initiale (5 1ers): {start_avg:,.2f} CFA")
        print(f"   Moyenne finale (5 derniers): {end_avg:,.2f} CFA")
        print(f"   Amélioration totale: {improvement:+.1f}%")

if __name__ == "__main__":
    visualize_progress()
