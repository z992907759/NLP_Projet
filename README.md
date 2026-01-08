## **Présentation du projet**

Il s’agit d’un système de questions-réponses basé sur la Retrieval-Augmented Generation (RAG) à partir de documents.

Le système construit un index vectoriel à partir de documents locaux (PDF, TXT).

Lorsqu’une question est posée, il effectue d’abord une recherche sémantique afin d’identifier les passages pertinents, puis combine ces informations avec un modèle de langage pour générer une réponse.

Un **baseline sans mécanisme de recherche** est également fourni afin de comparer les performances avec et sans RAG.

---

## **Fonctionnalités prises en charge**

* Documents non structurés et non formatés en questions-réponses (articles scientifiques, documents techniques)
* Recherche vectorielle combinée à la génération de texte
* Comparaison entre le système RAG et un baseline sans retrieval
* Pipeline d’évaluation automatique simple

---

## **Fonctionnalités principales**

* Extraction et nettoyage de texte à partir de documents PDF et TXT
* Découpage des documents longs en fragments exploitables (chunks)
* Génération de représentations vectorielles à l’aide de **sentence-transformers**
* Construction d’une base vectorielle avec **FAISS**
* Raisonnement RAG basé sur un seuil de similarité

(retour automatique vers le baseline lorsque la similarité est trop faible)
* Prise en charge du multi-query retrieval
* Fourniture de scripts d’évaluation simples pour comparer différentes configurations

---

## **Étapes d’exécution**

### **Préparation des documents**

Placer les fichiers PDF ou TXT dans le répertoire data/raw\_docs.

### **Construction du corpus documentaire**

Exécuter le script docs\_to\_corpus.py afin de générer les fragments de documents nettoyés.

### **Construction de l’index vectoriel**

Exécuter le script build\_index.py pour générer les embeddings et l’index FAISS.

### **Exécution du système de questions-réponses**

Exécuter le script main.py et saisir des questions pour un test interactif.

---

## **Comportement du système RAG et du baseline**

* Lorsque la similarité sémantique entre la question et les documents récupérés dépasse un seuil prédéfini, le système utilise le mécanisme RAG pour générer la réponse.
* Lorsque la similarité est inférieure au seuil, le système indique qu’aucune information pertinente n’a été trouvée dans les documents et bascule automatiquement vers le modèle baseline.
* Le modèle baseline ne repose sur aucun document externe et s’appuie uniquement sur les connaissances internes du modèle de langage.
