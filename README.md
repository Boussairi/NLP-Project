# NLP-Project : Sarcasm detection in Arabic and English texts
La détection du sarcasme dans les tweets, notamment dans les contextes informels arabes, est l'un des défis les plus complexes de la classification de texte. Comprendre le sarcasme est crucial pour saisir l'intention réelle des individus. Notre projet vise à améliorer les performances de détection du sarcasme dans les tweets en proposant des approches novatrices. Nous avons effectué une étude exhaustive des travaux connexes et exploré diverses méthodes et architectures existantes. Notre contribution réside dans la sélection et la combinaison des meilleurs modèles et techniques pour améliorer la précision de la détection du sarcasme. Nous avons constaté que l'utilisation de différentes combinaisons de modèles, notamment CNN, SVM, LSTM et Hard voting, permet d'obtenir des résultats prometteurs. Notre travail contribue ainsi à l'amélioration des capacités des modèles d'IA à interpréter le langage humain dans divers domaines d'application.

## Structure du Projet

    ├── Models/
    │   ├── ArabicTweets/
    │   │ ├── CNN.py
    │   │ ├── RNN.py
    │   │ ├── ...
    │   ├── EnglishTweets/
    │   │ ├── LSTM.py
    │   │ ├── SVM.py
    │   │ ├── ...
    ├── prog/
    │   ├── __init__.py
    │   ├── ArabicTextCleaning.py
    │   ├── EnglishTextCleaning.py
    │   ├── RandomSwapDeletion.py
    │   ├── TextDataAugmentation.py    
    ├── test/
    │   ├── task_A_Ar_test.csv
    │   ├── task_A_En_test.csv
    │   ├── task_B_En_test.csv
    ├── train/
    │   ├── train.Ar.csv
    │   ├── train.En.csv
    │   ├── data_for_augmentation.csv  
    ├── third-party-data
        ├── Radmme.md
    ├── ArabicTask.ipynb
    ├── EnglishTask.ipynb
    ├── Task B.ipynb
    ├── requirements.txt
    ├── .gitignore
    └── README.md
    
La structure du projet est organisée comme suit :

- **Models/** :
  - **_init_.py** : Fichier d'initialisation du module Python.
  - **ArabicTweets/**: Répertoire contenant les classes des modèles de langages ainsi que les modèles de classification utilisée pour la tache Arabe.
  - **EnglishTweets/**: Répertoire contenant les classes des modèles de langages ainsi que les modèles de classification utilisée pour la tache Anglaise.

- **prog/** : Répertoire abritant les scripts source du projet :
  - **_init_.py** : Fichier d'initialisation du module Python.
  - **ArabicTextCleaning.py** : Script pour le pretraitement des textes arabes.
  - **EnglishTextCleaning.py** : Script pour le prétraitement des données anglaises.
  - **RandomSwapDeletion.py** : Script pour la methode d'augmentation des données en utilisant le random swap et random deletion.
  - **TextDataAugmentation.py** : Script pour la methode d'augmentation des données en utilisant la traduction.
 
- **test/** : Répertoire contenant les données de test :
  - **task_A_Ar_test.csv** : Données de test pour la tache A (detecter si la phrase est sarcastique ou pas) pour les textes arabes.
  - **task_A_En_test.csv** : Données de test pour la tache A (detecter si la phrase est sarcastique ou pas) pour les textes anglais.
  - **task_b_En_test.csv** : Données de test pour la tache B (detecter le type de sarcasme) pour les textes anglais.
 
- **train/** : Répertoire contenant les données d'entrainement :
  - **train.Ar.csv** : Données d'entrainement contenants les tweets arabes.
  - **train.En.csv** : Données d'entrainement contenants les tweets anglais.
  - **data_for_augmentation.csv** : Données générées en utilisant les 3 methodes d'augmentation de données (double traduction, traduction et random swap and deletion) .
    
- **third-party data/** : Répertoire contenant un Readme où nous avons spécifié le lien du Drive où nous avons mis les données externes utilisées 

- **ArabicTask.ipynb** : Fichier Jupyter Notebook contenant le code principal de la Tache A en Arabe .
- **EnglishTask.ipynb** : Fichier Jupyter Notebook contenant le code principal de la Tache A en Anglais.
- **Task B.ipynb** : Fichier Jupyter Notebook contenant le code principal de la Tache B.


- **requirements.txt** : Fichier spécifiant les dépendances du projet.

- **.gitignore** : Fichier spécifiant les fichiers et répertoires à ignorer lors du suivi avec Git.

- **README.md** : Fichier que vous êtes actuellement en train de lire, contenant des informations générales sur le projet.

## Installation

   ```bash
   # Clonez le dépôt GitHub
   git clone (https://github.com/Boussairi/NLP-Project.git)
   cd nom-du-projet
   
   # Créez un environnement virtuel (optionnel, mais recommandé)
   python -m venv venv

   # Pour activer l'environnement virtuel (sous Windows) 
   venv\Scripts\activate

   # Pour activer l'environnement virtuel (sous macOS/Linux)
   source venv/bin/activate

   #Installez les dépendances
   pip install -r requirements.txt

```

## Utilisation
Une fois que l'installation est terminée, vous pouvez lancer le projet en:
- Pour la tache arabe:  exécutez le notebook  ArabicTask.ipynb qui fait appel aux classes definies dans la structure du projet
- Pour la tache anglaise:  exécutez le notebook EnglishTask.ipynb qui fait appel aux classes definies dans la structure du projet
- Pour la tache B:  exécutez le notebook Task B.ipynb qui fait appel aux classes definies dans la structure du projet

```bash
# Exemple de commande pour exécuter le script d'entraînement
jupyter notebook projet.ipynb
```

Authors
--------

- Abir Jamaly
- Hamza Boussairi
- Jinane Boufaris
