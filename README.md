# LifeLongLearning

**➰ Life long learning ➿ in a fast changing world.**

LifeLongLearning est une application interactive permettant de simuler, visualiser et mieux comprendre les dynamiques de l’apprentissage continu et de l’auto-évaluation dans un monde en évolution rapide.

## Fonctionnalités principales

- **Simulation de l’auto-évaluation**
  - Prise en compte de l’auto-évaluation par rapport au niveau réel, avec la possibilité de modéliser des oscillations (sur/sous-évaluation).

La fonction `evalearn` modélise une progression d’auto-évaluation selon le niveau d'apprentissage par rapport à un objectif `R`, avec plusieurs régimes :

- **Croissance initiale (0 ≤ x ≤ R/2)** : composée de deux splines cubiques pour assurer une montée progressive et continue de la valeur, avec un point d’inflexion à `x = R/4`.
- **Transition polynomiale (R/2 < x ≤ R)** : un polynôme de degré 4 prend le relais pour garantir une transition douce jusqu’au seuil `R`.
- **Régime linéaire (x > R)** : au-delà du seuil, la fonction devient strictement linéaire (y = x).

**Cas particuliers** :
- Pour `x < 0`, la valeur retournée est 0.
- Les points de jointure (`x = 0`, `x = R/4`, `x = R/2`, `x = R`) assurent la continuité et la dérivabilité de la fonction.
- Pour `x > R`, la progression suit la première bissectrice du plan (droite d'auto-évaluation réaliste y=x).

Cette construction permet de représenter différentes phases d’apprentissage ou d’évaluation : démarrage progressif, accélération, plafonnement, puis dépassement linéaire de l’objectif.

- **Simulation de l’apprentissage au fil du temps**
  - Modélisation de la progression du niveau d’apprentissage en fonction de paramètres personnalisables tels que :
    - le niveau initial de compétence (`f0`)
    - le taux d’apprentissage (`beta`)
    - le niveau de référence initial, correspondant à l'objectif de compétences à atteindre (`R0`)
    - le taux de croissance exponentielle de l’objectif (`k`), dans la configuration où une quantité importante de nouveaux contenus vient augmenter l'objectif à atteindre.
  - Affichage graphique de l’évolution du niveau réel d’apprentissage (`f(x)`).

- **Paramétrage interactif**
  - Interface utilisateur basée sur Streamlit avec sidebar permettant de régler dynamiquement tous les paramètres des modèles et des courbes.
  - Sélection des plages de temps et des niveaux d’apprentissage affichés.

- **Visualisation graphique**
  - Génération automatique de graphiques interactifs :
    - Courbe d’auto-évaluation en fonction de l’apprentissage réel
    - Évolution comparée de l’apprentissage réel et de l’auto-évaluation dans le temps, avec objectif constant ou croissant exponentiellement


## Utilisation

Lancez l’application avec Streamlit :

```bash
streamlit run app.py
```

Ajustez les paramètres dans la sidebar pour observer l’impact sur les courbes et les dynamiques d’apprentissage/auto-évaluation.

## Dépendances

- Python
- Streamlit
- Matplotlib
- Numpy

## Objectif

LifeLongLearning vise à offrir un outil pédagogique pour explorer expérimentalement les processus d’apprentissage et d’auto-évaluation.
