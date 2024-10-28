# 📊 Data Science Learning Dashboard

## Description
Le **Data Science Learning Dashboard** est une application interactive créée avec Streamlit, conçue pour fournir une expérience d’apprentissage pratique dans le domaine de la science des données. Ce tableau de bord propose des outils pour l'analyse exploratoire des données (EDA), l'expérimentation de modèles de machine learning, et l'analyse de séries temporelles. Il inclut également une section de documentation avec des ressources utiles.

## Fonctionnalités
### 1. Analyse Exploratoire des Données (EDA)
- **Distribution des Variables** : Visualisez la distribution des différentes variables avec des histogrammes et des box plots.
- **Analyse des Corrélations** : Affichez une matrice de corrélation pour comprendre les relations entre les variables.
- **Objectifs d'Apprentissage** : Comprendre les distributions, détecter les anomalies, analyser les corrélations et visualiser les tendances temporelles.

### 2. Machine Learning
- **Régression et Forêt Aléatoire** : Entraînez et évaluez des modèles de régression linéaire et de forêt aléatoire, avec des métriques de performance (R², MSE, RMSE).
- **Importance des Caractéristiques** : Visualisez l'importance des caractéristiques (features) pour les modèles de forêt aléatoire.
- **Clustering K-means** : Effectuez un clustering sur les données et visualisez les clusters en 3D.

### 3. Séries Temporelles
- **Décomposition des Séries Temporelles** : Analysez la tendance, la saisonnalité et les résidus d’une série temporelle.
- **Analyse de la Saisonnalité** : Examinez les variations des ventes en fonction des jours de la semaine.

### 4. Documentation et Ressources
- Accédez à des ressources d'apprentissage et à des informations sur les bibliothèques utilisées : **Pandas**, **Scikit-learn**, **Plotly**, et **Statsmodels**.

## Prérequis
- **Python 3.8+**
- **Streamlit**
- **Pandas**
- **Plotly**
- **Scikit-learn**
- **Statsmodels**
- **Numpy**

Installez les bibliothèques nécessaires avec la commande suivante :

```bash
pip install streamlit pandas plotly scikit-learn statsmodels numpy
```

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/votre-depot.git
   ```
2. Accédez au dossier du projet :
   ```bash
   cd votre-depot
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Lancement de l'application
Pour lancer l’application, exécutez la commande suivante dans le répertoire du projet :

```bash
streamlit run app.py
```

## Structure du Projet
- `app.py` : Fichier principal contenant le code de l'application Streamlit.
- `README.md` : Ce fichier fournit une documentation détaillée sur le projet.
- `requirements.txt` : Liste des dépendances Python nécessaires pour l'application.

## Utilisation
1. **Choisissez la source des données** :
   - **Données d'exemple** : Données par défaut.
   - **Mes données** : Téléchargez un fichier CSV personnalisé.
   - **Générer des données personnalisées** : Créez des données synthétiques en ajustant les paramètres.

2. **Navigation dans les onglets** :
   - **EDA** : Effectuez une analyse exploratoire des données.
   - **Machine Learning** : Entraînez des modèles de régression ou appliquez le clustering.
   - **Séries Temporelles** : Analysez et décomposez les séries temporelles.
   - **Documentation** : Consultez les ressources d’apprentissage et la documentation des bibliothèques.

3. **Mode Apprentissage** : Activez le mode apprentissage dans la barre latérale pour afficher des explications détaillées et le code correspondant pour chaque étape.

## Exemples de Visualisations
- **Distribution des Variables** : Histogrammes et box plots des variables sélectionnées.
- **Matrice de Corrélation** : Visualisation des corrélations entre les variables.
- **Clusters 3D** : Représentation en 3D des clusters pour le clustering K-means.
- **Décomposition des Séries Temporelles** : Visualisation de la tendance, saisonnalité et résidus.

## Personnalisation
Vous pouvez ajuster les paramètres et le style de l’application :
- **CSS** : Le style est défini dans le fichier `app.py` avec des classes CSS intégrées pour une expérience utilisateur améliorée.
- **Paramètres de génération des données** : Ajustez la saisonnalité, la tendance, et le bruit pour générer des jeux de données personnalisés.

## Contribuer
Les contributions sont les bienvenues ! Si vous souhaitez contribuer, veuillez ouvrir une issue ou soumettre une pull request.

1. Forkez le projet.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AjouterFonctionnalité`).
3. Committez vos modifications (`git commit -m 'Ajouter une nouvelle fonctionnalité'`).
4. Pushez vers la branche (`git push origin feature/AjouterFonctionnalité`).
5. Ouvrez une pull request.

## Auteurs
- **Mathieu Soussigan** - [Mon GitHub](https://github.com/Mathieu-Soussignan)

## Licence
Ce projet est sous licence MIT.

## Remerciements
- **Streamlit** pour fournir un cadre facile à utiliser pour construire des applications web.
- **Plotly** pour ses visualisations interactives.
