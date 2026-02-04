# Guide de Déploiement : Travel Cost API

Ce guide explique comment mettre en production votre API de prédiction sur **Render**, afin qu'elle soit accessible par votre frontend `farcal`.

## 1. Prérequis
- Un compte sur [Render.com](https://render.com).
- Votre code (le dossier `cameroon_travel_cost_rl`) doit être hébergé sur un dépôt **GitHub** ou **GitLab**.

## 2. Configuration sur Render

1.  **New Web Service** : Connectez votre dépôt GitHub.
2.  **Runtime** : Sélectionnez **Docker**. (Render détectera automatiquement le `Dockerfile` que j'ai créé).
3.  **Plan** : Choisissez "Starter" ou plus élevé (Le modèle RL nécessite environ 512 Mo à 1 Go de RAM pour charger les bibliothèques `torch` et `stable-baselines3`).
4.  **Advanced** : Ajoutez les variables d'environnement si nécessaire (par défaut, aucune n'est requise pour le moment).

## 3. Mise à jour du Frontend (Farcal)

Une fois l'API déployée, Render vous donnera une URL du type `https://votre-api.onrender.com`.

Vous devrez modifier l'URL dans votre frontend Next.js :
1.  Ouvrez `farcal/app/[locale]/LandingPageClient.tsx` (ou votre fichier `.env`).
2.  Remplacez l'ancienne URL `https://farcal-api-coast.onrender.com/predict` par votre nouvelle URL Render.

## 4. Maintenance
Le dossier `models/PPO/` est inclus dans le déploiement. Si vous réentraînez le modèle localement et que vous voulez mettre à jour la production :
1.  Faites un `git commit` des nouveaux fichiers `.zip` dans `models/PPO/`.
2.  `git push` vers GitHub.
3.  Render redéploiera automatiquement la nouvelle version.

---

> [!IMPORTANT]
> **Performance** : Le premier démarrage peut prendre 2-3 minutes car Docker doit installer `torch` et `numpy`. C'est normal.
