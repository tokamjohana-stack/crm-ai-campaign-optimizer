"""
train_models.py
───────────────────────────────────────────────────────
Script d'entraînement CRM AI — version finale

Ce script fait 3 choses :
  1. Entraîne 3 modèles RandomForest (open_rate, click_rate, conversion_rate)
  2. Calcule les benchmarks du dataset (utilisés pour le Campaign Score)
  3. Génère et sauvegarde les insights business automatiques

Outputs produits dans models/ :
  - model_open_rate.joblib
  - model_click_rate.joblib
  - model_conversion_rate.joblib
  - benchmarks.joblib
  - insights.joblib

Usage :
  python train_models.py
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ══════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════

DATA_PATH   = "crm_email_campaigns_synthetic_5000.csv"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TARGETS = ["open_rate", "click_rate", "conversion_rate"]

CATEGORICAL_FEATURES = [
    "campaign_type",
    "offer_type",
    "audience_segment",
    "device_main",
    "send_day",
    "subject_sentiment",
]

NUMERICAL_FEATURES = [
    "send_hour",
    "subject_length",
    "has_personalization",
    "is_urgent",
    "has_image",
    "nb_links",
    "cta_count",
    "audience_size",
    "marketing_pressure",
    "previous_segment_open_rate",
    "previous_segment_ctr",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Pondération du Campaign Score (doit sommer à 1.0)
SCORE_WEIGHTS = {
    "open_rate":       0.35,
    "click_rate":      0.35,
    "conversion_rate": 0.30,
}


# ══════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════

print("═" * 58)
print("  CRM AI — Entraînement des modèles")
print("═" * 58)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

X = df[ALL_FEATURES]
print(f"✅ Features : {len(ALL_FEATURES)} variables")


# ══════════════════════════════════════════════════════
# 2. PREPROCESSING (partagé entre les 3 modèles)
# ══════════════════════════════════════════════════════

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
        # Les features numériques passent sans transformation
        # (RandomForest n'est pas sensible à l'échelle)
        ("num", "passthrough", NUMERICAL_FEATURES),
    ]
)


# ══════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT — 1 modèle par target
# ══════════════════════════════════════════════════════

results = {}

for target in TARGETS:
    print(f"\n{'─' * 58}")
    print(f"  Modèle → {target}")
    print(f"{'─' * 58}")

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,       # plus stable que 100
                    max_depth=12,           # évite l'overfitting
                    min_samples_leaf=5,     # généralisation plus robuste
                    random_state=42,
                    n_jobs=-1,              # utilise tous les cœurs CPU
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

    model_path = os.path.join(MODELS_DIR, f"model_{target}.joblib")
    joblib.dump(pipeline, model_path)
    print(f"  💾 Sauvegardé → {model_path}")

    results[target] = {"mae": mae, "r2": r2, "path": model_path}


# ══════════════════════════════════════════════════════
# 4. BENCHMARKS (référence pour le Campaign Score)
#
#    On utilise min/max du dataset réel.
#    Ces valeurs servent à normaliser les prédictions
#    sur une échelle 0-100 dans l'app Streamlit.
# ══════════════════════════════════════════════════════

print(f"\n{'─' * 58}")
print("  Calcul des benchmarks")
print(f"{'─' * 58}")

benchmarks = {}
for target in TARGETS:
    benchmarks[target] = {
        "min":  round(df[target].min(),  4),
        "mean": round(df[target].mean(), 4),
        "max":  round(df[target].max(),  4),
        "p25":  round(df[target].quantile(0.25), 4),
        "p75":  round(df[target].quantile(0.75), 4),
    }
    print(
        f"  {target:<22} "
        f"min={benchmarks[target]['min']:.3f} | "
        f"mean={benchmarks[target]['mean']:.3f} | "
        f"max={benchmarks[target]['max']:.3f}"
    )

benchmarks["weights"] = SCORE_WEIGHTS

benchmarks_path = os.path.join(MODELS_DIR, "benchmarks.joblib")
joblib.dump(benchmarks, benchmarks_path)
print(f"\n  💾 Benchmarks sauvegardés → {benchmarks_path}")


# ══════════════════════════════════════════════════════
# 5. INSIGHTS BUSINESS AUTOMATIQUES
#
#    Calculés une seule fois a l'entrainement,
#    réutilisés directement dans l'app Streamlit.
# ══════════════════════════════════════════════════════

print(f"\n{'─' * 58}")
print("  Génération des insights business")
print(f"{'─' * 58}")

# Segment
segment_perf  = df.groupby("audience_segment")["open_rate"].mean().sort_values(ascending=False)
best_segment  = segment_perf.idxmax()
worst_segment = segment_perf.idxmin()

# Jour d'envoi
day_perf = df.groupby("send_day")["open_rate"].mean().sort_values(ascending=False)
best_day  = day_perf.idxmax()
worst_day = day_perf.idxmin()

# Type de campagne
campaign_perf      = df.groupby("campaign_type")["open_rate"].mean().sort_values(ascending=False)
best_campaign_type = campaign_perf.idxmax()

# Personnalisation
personalization_perf = df.groupby("has_personalization")["open_rate"].mean()
personalization_diff = personalization_perf.get(1, 0) - personalization_perf.get(0, 0)

# Pression marketing
low_pressure_rate  = df[df["marketing_pressure"] <= 2]["open_rate"].mean()
high_pressure_rate = df[df["marketing_pressure"] >= 5]["open_rate"].mean()
pressure_diff      = low_pressure_rate - high_pressure_rate

# Longueur objet email
def subject_bucket(length):
    if length < 30:    return "court (<30 car.)"
    elif length <= 50: return "moyen (30-50 car.)"
    else:              return "long (>50 car.)"

df["subject_bucket"] = df["subject_length"].apply(subject_bucket)
subject_perf         = df.groupby("subject_bucket")["open_rate"].mean().sort_values(ascending=False)
best_subject_bucket  = subject_perf.idxmax()

# Conversion par segment (cross-target)
segment_conv      = df.groupby("audience_segment")["conversion_rate"].mean().sort_values(ascending=False)
best_segment_conv = segment_conv.idxmax()

insights = {
    "best_segment": {
        "metric_value": round(segment_perf.max(), 4),
        "recommendation": (
            f"Priorisez les campagnes sur les '{best_segment}' : "
            f"open rate moyen de {segment_perf.max():.1%}, "
            f"soit +{(segment_perf.max() - segment_perf.mean()):.1%} vs moyenne."
        ),
    },
    "worst_segment": {
        "metric_value": round(segment_perf.min(), 4),
        "recommendation": (
            f"Les '{worst_segment}' sous-performent ({segment_perf.min():.1%} d'open rate). "
            "Utilisez des campagnes de réactivation dédiées avec une offre forte."
        ),
    },
    "best_day": {
        "metric_value": round(day_perf.max(), 4),
        "recommendation": (
            f"Le meilleur jour d'envoi est le {best_day} ({day_perf.max():.1%} d'open rate). "
            f"Evitez le {worst_day} ({day_perf.min():.1%})."
        ),
    },
    "best_campaign_type": {
        "metric_value": round(campaign_perf.max(), 4),
        "recommendation": (
            f"Les campagnes '{best_campaign_type}' sont les plus performantes "
            f"({campaign_perf.max():.1%} d'open rate) car elles ciblent une intention d'achat réelle."
        ),
    },
    "personalization": {
        "metric_value": round(personalization_diff, 4),
        "recommendation": (
            f"La personnalisation améliore l'open rate de {personalization_diff:.1%} en moyenne. "
            "C'est un levier simple à activer systématiquement."
        ),
    },
    "marketing_pressure": {
        "metric_value": round(pressure_diff, 4),
        "recommendation": (
            f"Une faible pression marketing (<=2) surperforme une forte pression (>=5) "
            f"de {pressure_diff:.1%} en open rate. "
            "Espacez vos envois pour limiter la fatigue email."
        ),
    },
    "subject_length": {
        "metric_value": round(subject_perf.max(), 4),
        "recommendation": (
            f"Les objets '{best_subject_bucket}' obtiennent le meilleur open rate "
            f"({subject_perf.max():.1%}). "
            "Evitez les objets trop longs qui se tronquent sur mobile."
        ),
    },
    "conversion_by_segment": {
        "metric_value": round(segment_conv.max(), 4),
        "recommendation": (
            f"En conversion, le segment '{best_segment_conv}' est aussi le plus performant "
            f"({segment_conv.max():.1%}). "
            "Concentrez vos offres à forte valeur sur ce segment."
        ),
    },
}

for key, insight in insights.items():
    print(f"  + {key}")
    print(f"    {insight['recommendation']}")

insights_path = os.path.join(MODELS_DIR, "insights.joblib")
joblib.dump(insights, insights_path)
print(f"\n  💾 Insights sauvegardés → {insights_path}")


# ══════════════════════════════════════════════════════
# 6. RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════

print(f"\n{'═' * 58}")
print("  RÉSUMÉ DES PERFORMANCES MODÈLES")
print(f"{'═' * 58}")
print(f"  {'Target':<22} {'MAE':>8} {'R²':>8}")
print(f"  {'─' * 40}")
for target, metrics in results.items():
    print(f"  {target:<22} {metrics['mae']:>8.4f} {metrics['r2']:>8.4f}")

print(f"\n{'═' * 58}")
print("  FICHIERS PRODUITS dans models/")
print(f"{'═' * 58}")
for f in sorted(os.listdir(MODELS_DIR)):
    size = os.path.getsize(os.path.join(MODELS_DIR, f))
    print(f"  {f:<38} ({size/1e6:.1f} Mo)")

print(f"\n✅ Entrainement terminé.")
print("   Prochaine étape → app.py (Streamlit)\n")
