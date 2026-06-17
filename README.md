# 📧 CRM AI — Campaign Optimizer

> Application CRM intelligente permettant de **prédire les performances de campagnes email avant envoi** grâce au machine learning, et d'optimiser les décisions marketing grâce à des insights business automatiques.

---

## 🎯 Contexte & Objectif

Les équipes CRM envoient des campagnes email sans savoir à l'avance si elles vont performer. Le taux d'ouverture, le taux de clic et le taux de conversion ne sont connus qu'**après** l'envoi.

**CRM AI Campaign Optimizer** résout ce problème en permettant à un CRM manager de :
- Simuler les performances d'une campagne **avant envoi**
- Comparer deux scénarios de campagne côte à côte
- Recevoir des recommandations business actionnables
- Obtenir un **Campaign Score /100** synthétique et immédiatement lisible

---

## 🖥️ Démonstration

### Prédiction simple
Saisir les paramètres d'une campagne et obtenir en temps réel :
- Open Rate prédit
- Click Rate prédit  
- Conversion Rate prédit
- Campaign Score /100
- Entonnoir estimé (envois → ouvertures → clics → conversions)
- Recommandations contextuelles

### Comparaison de scénarios
Comparer deux versions d'une campagne (ex: segment A vs segment B, mardi vs jeudi) et obtenir un **verdict automatique** avec explication.

---

## 🏗️ Architecture

```
Utilisateur
    ↓
Application Streamlit (app.py)
    ↓
Formulaire paramètres campagne
    ↓
3 modèles RandomForest (scikit-learn)
    ↓
Prédictions : open_rate | click_rate | conversion_rate
    ↓
Campaign Score composite (0-100)
    ↓
Insights business automatiques
```

---

## 🤖 Modèles Machine Learning

| Modèle | Target | MAE | R² |
|--------|--------|-----|----|
| RandomForestRegressor | open_rate | 0.0227 | 0.83 |
| RandomForestRegressor | click_rate | 0.0128 | 0.86 |
| RandomForestRegressor | conversion_rate | 0.0080 | 0.88 |

**Dataset** : 5 000 campagnes email synthétiques avec logiques métier CRM intégrées  
**Features** : 17 variables (segment, type de campagne, jour/heure d'envoi, objet, pression marketing, historique segment...)

### Campaign Score
Score composite pondéré de 0 à 100 :
- Open Rate × 35%
- Click Rate × 35%
- Conversion Rate × 30%

| Score | Interprétation |
|-------|----------------|
| 80-100 | Campagne excellente |
| 65-80 | Bonne campagne |
| 40-65 | Correcte — optimisable |
| 0-40 | Faible — à revoir |

---

## 📊 Insights Business Automatiques

L'application génère automatiquement 8 recommandations actionnables basées sur l'analyse du dataset :

- 🟢 **Segment** : les clients fidèles ont un open rate de 29.2% (+8.4% vs moyenne)
- 📅 **Jour d'envoi** : mardi est le meilleur jour (22.4%), dimanche le pire (18.8%)
- 📝 **Objet** : les objets de 30-50 caractères surperforment (+2% vs objets longs)
- 📨 **Pression marketing** : faible pression (≤2) surperforme forte pression (≥5) de 3.5%
- ✉️ **Personnalisation** : +1.8% d'open rate en activant la personnalisation
- 🛒 **Cart Abandonment** : type de campagne le plus performant (22.7%)

---

## 🛠️ Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Language | Python 3.11+ |
| ML | scikit-learn (RandomForestRegressor) |
| Data | pandas, numpy |
| App | Streamlit |
| Sérialisation | joblib |
| Déploiement | Streamlit Community Cloud / Azure App Service |

---

## 🚀 Lancer le projet en local

### Prérequis
- Python 3.11+
- pip

### Installation

```bash
# Cloner le repository
git clone https://github.com/tokamjohana-stack/crm-ai-campaign-optimizer.git
cd crm-ai-campaign-optimizer

# Installer les dépendances
pip install -r requirements.txt

# Entraîner les modèles
python train_models.py

# Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## 📁 Structure du projet

```
crm-ai-campaign-optimizer/
├── app.py                                    # Application Streamlit
├── train_models.py                           # Script d'entraînement
├── requirements.txt                          # Dépendances
├── crm_email_campaigns_synthetic_5000.csv    # Dataset CRM
└── models/                                   # Modèles entraînés
    ├── model_open_rate.joblib
    ├── model_click_rate.joblib
    ├── model_conversion_rate.joblib
    ├── benchmarks.joblib
    └── insights.joblib
```

---

## 💼 Contexte Portfolio

Ce projet a été développé dans le cadre d'un portfolio professionnel orienté **CRM + IA + Data**, visant des postes de :
- CRM AI Specialist
- AI CRM Consultant
- Data-Driven CRM Analyst
- AI Marketing Specialist

Il démontre la capacité à **transformer un besoin métier CRM concret en solution data/ML opérationnelle**, en combinant :
- Compréhension des KPI CRM (open rate, click rate, conversion)
- Modélisation ML appliquée (RandomForest, pipeline sklearn)
- Logique produit orientée utilisateur métier (CRM manager)
- Déploiement cloud

---

## 👤 Auteur

**Johana Tokam**  
Master Marketing Digital & Data Analytics — EMLV + IIM  
Spécialisation CRM / IA appliquée au marketing  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](www.linkedin.com/in/maelle-johana-simo-tokam-2a8812231)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/tokamjohana-stack)
