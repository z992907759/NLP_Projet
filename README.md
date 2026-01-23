# RAG MÉDICAL
**Par :** Amine AIT MOUSSA, Siham DAANOUNI, Haoxin LIU, Wangjiachen ZHAO

---

## 1. Le Dataset (Corpus Médical)
Pour valider notre approche RAG dans un contexte critique, nous avons constitué une base de connaissances spécialisée en médecine, simulant l'environnement d'un analyste de santé publique.

### Source des Données
Le corpus est composé de rapports techniques officiels provenant majoritairement d'organisations internationales (comme l'OMS/WHO) et d'articles de recherche. Ces documents ont été choisis pour leur densité informationnelle et leur vocabulaire technique.

### Contenu du dossier `data/raw_docs`
Le dataset comprend une vingtaine de documents hétérogènes, incluant :
- **Rapports épidémiologiques :** Statistiques sur le cancer (*Cancer Facts & Figures*) et le VIH.
- **Guides cliniques :** Protocoles de traitement et directives de l'OMS (ex: *Guidelines for the screening of cancer*).
- **Articles scientifiques :** Publications sur la santé publique et les maladies chroniques (Epilepsie, maladies cardiovasculaires).

### Prétraitement
Ces documents bruts (PDF non structurés) ont été transformés en environ **3300 chunks** enrichis de métadonnées. Ce volume permet de tester la capacité du système à retrouver les informations dans le texte.

---

## 2. La Problématique

L'utilisation de Grands Modèles de Langage (LLMs) comme Llama-3 ou GPT-4 dans le domaine médical se heurte à trois obstacles majeurs qui rendent leur utilisation "brute" dangereuse :

### 1. Les Hallucinations
Les LLMs sont entraînés pour générer du texte plausible, pas nécessairement vrai.
* **Problème :** Sur des questions pointues (ex: "Quel est le taux de mortalité dans l'étude S12889 ?"), un LLM classique va souvent inventer un chiffre convaincant mais totalement faux.
* **Risque :** Fournir une information médicale erronée à un praticien ou un patient.

### 2. L'Absence de Connaissances Spécifiques
Les modèles ont une date de coupure (*knowledge cutoff*) et ne connaissent pas les documents internes.
* **Problème :** Si vous interrogez Llama-3 sur un rapport clinique publié hier ou sur un document PDF interne à votre hôpital, il ne peut pas répondre car ce document ne faisait pas partie de son entraînement.

### 3. Le Manque de Traçabilité
Un LLM standard donne une réponse sans dire d'où elle vient.
* **Problème :** Dans un contexte scientifique, une affirmation sans source est inutilisable. Il est impératif de pouvoir vérifier : *"Cette information vient de la page 12 du rapport de l'OMS 2025"*.

### Notre Objectif
Construire un système **RAG (Retrieval-Augmented Generation)** capable de :
1.  **Ancrer** les réponses dans un corpus de documents contrôlés (PDFs médicaux).
2.  **Citer** précisément ses sources (Nom du fichier, extrait du texte).
3.  **Refuser** de répondre si l'information n'est pas présente dans les documents, ou s'il considère qu'il ne peut pas être assez fiable, plutôt que d'inventer.

---

## 3. La Solution Appliquée (Architecture Hybride V4)

Pour répondre aux exigences de précision du domaine médical, nous avons dépassé l'approche RAG naïve (V1) pour développer un pipeline **Hybride avec Reranking (V4)**.

### A. Pipeline Technique (Workflow)

Le traitement d'une question suit 4 étapes rigoureuses pour garantir la fiabilité des réponses :

#### 1. Ingestion et "Smart Chunking"
Contrairement à un découpage arbitraire, nous utilisons un découpage sémantique pour ne pas briser le sens :
* **Segmentation :** Respect des frontières de phrases.
* **Calibrage :** environ 300 mots par chunk, taille idéale pour les modèles d'embedding.
* **Overlap :** 50 mots conservés entre deux blocs consécutifs pour ne jamais perdre le contexte.

#### 2. Recherche Hybride (Hybrid Retrieval)
Pour maximiser le rappel (*Recall*), nous interrogeons le corpus via deux stratégies complémentaires :
* **Dense Retrieval (FAISS) :** Utilise des vecteurs (`all-MiniLM-L6-v2`) pour comprendre le *sens* et les synonymes.
* **Sparse Retrieval (BM25) :** Utilise une approche lexicale pour capturer les *mots-clés exacts*, crucial pour les termes techniques spécifiques (ex: "Étude S12889").

#### 3. Fusion et Réordonnancement (Reranking)
C'est l'étape qui filtre le bruit :
* **Fusion RRF :** Les résultats FAISS et BM25 sont fusionnés via l'algorithme *Reciprocal Rank Fusion*.
* **Cross-Encoder :** Un modèle expert (`ms-marco-MiniLM-L-6-v2`) relit attentivement chaque couple (Question, Document) pour attribuer un score de pertinence précis.
* **Filtrage Strict :** Tout document ayant un score négatif (< 0) est rejeté.

#### 4. Génération Sécurisée
Le LLM (`Llama-3.2-1B`) reçoit uniquement les chunks validés avec une instruction système stricte : *"Answer using ONLY the provided contexts"*.

### B. Stack Technologique
* **Langage :** Python 3.10
* **LLM :** `meta-llama/Llama-3.2-1B-Instruct`
* **Vector Store :** `FAISS`
* **Embeddings :** `sentence-transformers`
* **Reranker :** `cross-encoder`
* **Search Engine :** `rank-bm25`

---

## 4. Les Résultats Obtenus

Pour prouver l'efficacité de notre architecture **Hybride V4**, nous avons mené un benchmark comparatif strict contre une approche **RAG Naïve (V1)** sur un jeu de données de test ("Golden Dataset") composé de 591 questions.

### Protocole de Test
Le benchmark évalue deux axes critiques :
1.  **Précision (Hit Rate) :** Le système retrouve-t-il le document exact contenant la réponse ?
2.  **Sécurité (Rejection Rate) :** Le système est-il capable de dire *"Je ne sais pas"* face à une question piège ?

### Tableau Comparatif (Données issues du Notebook 03)

| Ce que l'on mesure | RAG Naïf (V1) | RAG Hybride (V4) | Évolution | Analyse |
| :--- | :---: | :---: | :---: | :--- |
| **Précision des documents**<br>*(Hit Rate - Potentiel)* | 53.3% | **54.3%** | +1.0% |**Léger gain.** L'ajout de BM25 permet de capturer des termes techniques que la recherche vectorielle seule manquait. |
| **Expérience Utilisateur**<br>*(Réponses affichées)* | 53.1% | 48.5% | -4.6% | **Filtrage Strict.** V4 est plus conservateur. Il préfère ne rien afficher (baisse du taux) plutôt que de montrer une réponse incertaine. |
| **Sécurité / Anti-Hallucination**<br>*(Pièges évités)* | 86.2% | **98.9%** | **+12.8%** | **Gain Massif.** C'est la force majeure de V4. Le système rejette quasi-totalement les questions hors-sujet ou pièges. |

### Exemple Concret (Tiré des logs)
> **Question :** *"For which population groups is assessment of total CVD risk recommended?"*

**1. Réponse du RAG Naïf (V1 - FAISS seul) :**
* **Score de confiance :** 0.64 (Correct mais sans plus).
* **Résultat :** Récupération correcte mais polluée par des documents moyennement pertinents.

**2. Réponse du RAG Hybride (V4 - Architecture Complète) :**
* **Score de confiance :** **3.19** (Score Cross-Encoder très élevé).
* **Comportement :**
    * Le système a identifié **10 candidats uniques** via la fusion.
    * Le Reranker a propulsé le **Chunk #12** en tête.
    * **Résultat :** Liste précise des 7 groupes à risque (Âge >40, Fumeurs, Obésité...), parfaitement ancrée dans le document source.

---

## 5. Installation et Démo

Ce projet utilise le modèle `meta-llama/Llama-3.2-1B-Instruct`. Il nécessite une authentification Hugging Face.

### Pré-requis
* Python 3.10+
* Compte [Hugging Face](https://huggingface.co/) avec Token en lecture (`read`).
* Accès validé sur la page du modèle [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

### Installation
```bash
# 1. Cloner le repository
git clone [VOTRE_LIEN_GITHUB_ICI]
cd [NOM_DU_DOSSIER]

# 2. Installer les dépendances
pip install -r requirements.txt
pip install ipywidgets