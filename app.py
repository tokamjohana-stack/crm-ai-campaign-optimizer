"""
app.py
──────────────────────────────────────────────
CRM AI — Application Streamlit
v2 : Prédiction simple + Comparaison scénarios

Usage :
  python -m streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ══════════════════════════════════════════════
# CONFIGURATION PAGE
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="CRM AI — Campaign Optimizer",
    page_icon="📧",
    layout="wide",
)

# ══════════════════════════════════════════════
# CONFIGURATION FEATURES & MODÈLES
# ══════════════════════════════════════════════

TARGETS = ["open_rate", "click_rate", "conversion_rate"]

CATEGORICAL_FEATURES = [
    "campaign_type", "offer_type", "audience_segment",
    "device_main", "send_day", "subject_sentiment",
]
NUMERICAL_FEATURES = [
    "send_hour", "subject_length", "has_personalization",
    "is_urgent", "has_image", "nb_links", "cta_count",
    "audience_size", "marketing_pressure",
    "previous_segment_open_rate", "previous_segment_ctr",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

SCORE_WEIGHTS = {
    "open_rate": 0.35,
    "click_rate": 0.35,
    "conversion_rate": 0.30,
}

# ══════════════════════════════════════════════
# ENTRAÎNEMENT À LA VOLÉE (si modèles absents)
# ══════════════════════════════════════════════

@st.cache_resource
def load_models():
    """
    Charge les modèles depuis le disque si disponibles,
    sinon les entraîne à la volée depuis le CSV.
    """
    models_dir = "models"
    model_path = os.path.join(models_dir, "model_open_rate.joblib")

    # Cas 1 : modèles déjà présents sur disque
    if os.path.exists(model_path):
        models = {
            t: joblib.load(os.path.join(models_dir, f"model_{t}.joblib"))
            for t in TARGETS
        }
        benchmarks = joblib.load(os.path.join(models_dir, "benchmarks.joblib"))
        insights   = joblib.load(os.path.join(models_dir, "insights.joblib"))
        return models, benchmarks, insights

    # Cas 2 : entraînement à la volée (Streamlit Cloud)
    with st.spinner("⚙️ Initialisation des modèles IA... (environ 60 secondes)"):
        df = pd.read_csv("crm_email_campaigns_synthetic_5000.csv")
        X  = df[ALL_FEATURES]

        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ])

        models = {}
        for target in TARGETS:
            y = df[target]
            X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=200, max_depth=12,
                    min_samples_leaf=5, random_state=42, n_jobs=-1,
                )),
            ])
            pipeline.fit(X_train, y_train)
            models[target] = pipeline

        # Benchmarks
        benchmarks = {}
        for target in TARGETS:
            benchmarks[target] = {
                "min":  round(df[target].min(),  4),
                "mean": round(df[target].mean(), 4),
                "max":  round(df[target].max(),  4),
            }
        benchmarks["weights"] = SCORE_WEIGHTS

        # Insights
        segment_perf  = df.groupby("audience_segment")["open_rate"].mean().sort_values(ascending=False)
        day_perf      = df.groupby("send_day")["open_rate"].mean().sort_values(ascending=False)
        campaign_perf = df.groupby("campaign_type")["open_rate"].mean().sort_values(ascending=False)
        perso_perf    = df.groupby("has_personalization")["open_rate"].mean()
        perso_diff    = perso_perf.get(1, 0) - perso_perf.get(0, 0)
        low_p  = df[df["marketing_pressure"] <= 2]["open_rate"].mean()
        high_p = df[df["marketing_pressure"] >= 5]["open_rate"].mean()

        def subject_bucket(l):
            if l < 30:    return "court (<30 car.)"
            elif l <= 50: return "moyen (30-50 car.)"
            else:         return "long (>50 car.)"

        df["subject_bucket"] = df["subject_length"].apply(subject_bucket)
        subj_perf = df.groupby("subject_bucket")["open_rate"].mean().sort_values(ascending=False)
        seg_conv  = df.groupby("audience_segment")["conversion_rate"].mean().sort_values(ascending=False)

        insights = {
            "best_segment": {"metric_value": round(segment_perf.max(), 4), "recommendation": f"Priorisez les campagnes sur les '{segment_perf.idxmax()}' : open rate moyen de {segment_perf.max():.1%}, soit +{(segment_perf.max() - segment_perf.mean()):.1%} vs moyenne."},
            "worst_segment": {"metric_value": round(segment_perf.min(), 4), "recommendation": f"Les '{segment_perf.idxmin()}' sous-performent ({segment_perf.min():.1%} d'open rate). Utilisez des campagnes de réactivation dédiées avec une offre forte."},
            "best_day": {"metric_value": round(day_perf.max(), 4), "recommendation": f"Le meilleur jour d'envoi est le {day_perf.idxmax()} ({day_perf.max():.1%} d'open rate). Evitez le {day_perf.idxmin()} ({day_perf.min():.1%})."},
            "best_campaign_type": {"metric_value": round(campaign_perf.max(), 4), "recommendation": f"Les campagnes '{campaign_perf.idxmax()}' sont les plus performantes ({campaign_perf.max():.1%} d'open rate)."},
            "personalization": {"metric_value": round(perso_diff, 4), "recommendation": f"La personnalisation améliore l'open rate de {perso_diff:.1%} en moyenne. C'est un levier simple à activer systématiquement."},
            "marketing_pressure": {"metric_value": round(low_p - high_p, 4), "recommendation": f"Une faible pression marketing (<=2) surperforme une forte pression (>=5) de {(low_p - high_p):.1%} en open rate. Espacez vos envois."},
            "subject_length": {"metric_value": round(subj_perf.max(), 4), "recommendation": f"Les objets '{subj_perf.idxmax()}' obtiennent le meilleur open rate ({subj_perf.max():.1%}). Evitez les objets trop longs qui se tronquent sur mobile."},
            "conversion_by_segment": {"metric_value": round(seg_conv.max(), 4), "recommendation": f"En conversion, le segment '{seg_conv.idxmax()}' est aussi le plus performant ({seg_conv.max():.1%}). Concentrez vos offres à forte valeur sur ce segment."},
        }

    return models, benchmarks, insights

models, benchmarks, insights = load_models()


# ══════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════

def normalize(value, min_val, max_val):
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def compute_campaign_score(predictions, benchmarks):
    weights = benchmarks["weights"]
    score = 0
    for target, weight in weights.items():
        norm = normalize(
            predictions[target],
            benchmarks[target]["min"],
            benchmarks[target]["max"],
        )
        score += norm * weight
    return round(score * 100, 1)

def score_label(score):
    if score >= 80:   return "Excellente", "#2ecc71"
    elif score >= 65: return "Bonne", "#27ae60"
    elif score >= 40: return "Correcte — optimisable", "#f39c12"
    else:             return "Faible — à revoir", "#e74c3c"

def predict(input_dict):
    """Lance les 3 modèles sur un dict de paramètres."""
    df = pd.DataFrame([input_dict])
    return {
        target: round(model.predict(df)[0], 4)
        for target, model in models.items()
    }

def build_recommendations(params, predictions):
    """Génère des recommandations contextuelles."""
    reco_list = []
    if params["audience_segment"] == "inactive_customer":
        reco_list.append("🔴 Segment inactif : pensez à une offre de réactivation forte.")
    elif params["audience_segment"] == "loyal_customer":
        reco_list.append("🟢 Excellent choix de segment : les clients fidèles sont votre audience la plus réceptive.")
    if params["send_day"] in ["saturday", "sunday"]:
        reco_list.append(f"⚠️ Le {params['send_day']} est l'un des jours les moins performants. Privilégiez mardi ou mercredi.")
    elif params["send_day"] in ["tuesday", "wednesday"]:
        reco_list.append(f"✅ Bon choix : le {params['send_day']} est parmi les meilleurs jours d'envoi.")
    if params["marketing_pressure"] >= 5:
        reco_list.append("🔴 Pression marketing élevée : vos contacts ont été très sollicités récemment.")
    elif params["marketing_pressure"] <= 2:
        reco_list.append("✅ Bonne pression marketing : l'audience n'est pas saturée.")
    if params["subject_length"] > 60:
        reco_list.append(f"⚠️ Objet long ({params['subject_length']} car.) : risque de troncature sur mobile.")
    elif 30 <= params["subject_length"] <= 50:
        reco_list.append(f"✅ Longueur d'objet optimale ({params['subject_length']} car.).")
    if not params["has_personalization"]:
        reco_list.append("💡 Activez la personnalisation : +1.8% d'open rate en moyenne.")
    if not reco_list:
        reco_list.append("✅ Les paramètres semblent bien configurés.")
    return reco_list

def campaign_form(key_prefix, default_segment="loyal_customer", default_day="tuesday"):
    """
    Formulaire de campagne réutilisable.
    Retourne un dict de paramètres.
    """
    segment_labels = {
        "loyal_customer":    "🟢 Clients fidèles",
        "new_customer":      "🔵 Nouveaux clients",
        "inactive_customer": "🔴 Clients inactifs",
        "prospect":          "🟡 Prospects",
    }
    campaign_labels = {
        "promo":            "🏷️ Promo",
        "newsletter":       "📰 Newsletter",
        "cart_abandonment": "🛒 Cart Abandonment",
        "reactivation":     "🔄 Réactivation",
        "onboarding":       "👋 Onboarding",
    }
    offer_labels = {
        "discount":      "💰 Réduction",
        "bundle":        "📦 Bundle",
        "limited_offer": "⏳ Offre limitée",
        "free_shipping": "🚚 Livraison gratuite",
        "no_offer":      "— Pas d'offre",
    }
    day_labels = {
        "monday": "Lundi", "tuesday": "Mardi", "wednesday": "Mercredi",
        "thursday": "Jeudi", "friday": "Vendredi",
        "saturday": "Samedi", "sunday": "Dimanche",
    }
    sentiment_labels = {
        "positive": "😊 Positif",
        "neutral":  "😐 Neutre",
        "urgent":   "🚨 Urgent",
    }

    st.markdown("**Audience**")
    audience_segment = st.selectbox(
        "Segment cible", list(segment_labels.keys()),
        format_func=lambda x: segment_labels[x],
        index=list(segment_labels.keys()).index(default_segment),
        key=f"{key_prefix}_segment",
    )
    previous_segment_open_rate = st.slider(
        "Open rate historique du segment (%)",
        5, 50, 20, key=f"{key_prefix}_open_hist",
    ) / 100
    previous_segment_ctr = st.slider(
        "CTR historique du segment (%)",
        1, 25, 8, key=f"{key_prefix}_ctr_hist",
    ) / 100
    audience_size = st.number_input(
        "Taille audience (contacts)",
        min_value=1000, max_value=500000, value=50000, step=1000,
        key=f"{key_prefix}_audience_size",
    )

    st.markdown("---")
    st.markdown("**Campagne**")
    campaign_type = st.selectbox(
        "Type de campagne", list(campaign_labels.keys()),
        format_func=lambda x: campaign_labels[x],
        key=f"{key_prefix}_campaign_type",
    )
    offer_type = st.selectbox(
        "Type d'offre", list(offer_labels.keys()),
        format_func=lambda x: offer_labels[x],
        key=f"{key_prefix}_offer_type",
    )
    marketing_pressure = st.slider(
        "Pression marketing", 0, 6, 2,
        key=f"{key_prefix}_pressure",
    )

    st.markdown("---")
    st.markdown("**Objet & Contenu**")
    subject_length = st.slider(
        "Longueur objet (car.)", 10, 100, 45,
        key=f"{key_prefix}_subject_length",
    )
    subject_sentiment = st.selectbox(
        "Ton de l'objet", list(sentiment_labels.keys()),
        format_func=lambda x: sentiment_labels[x],
        key=f"{key_prefix}_sentiment",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        has_personalization = st.toggle("Personnalisation", value=True, key=f"{key_prefix}_perso")
        is_urgent           = st.toggle("Mention urgence",  value=False, key=f"{key_prefix}_urgent")
    with col_b:
        has_image = st.toggle("Image dans l'email", value=True, key=f"{key_prefix}_image")

    nb_links  = st.slider("Nombre de liens", 1, 10, 3, key=f"{key_prefix}_links")
    cta_count = st.slider("Nombre de CTA",   1,  5, 2, key=f"{key_prefix}_cta")

    st.markdown("---")
    st.markdown("**Envoi**")
    days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    col_c, col_d = st.columns(2)
    with col_c:
        send_day = st.selectbox(
            "Jour d'envoi", days,
            format_func=lambda x: day_labels[x],
            index=days.index(default_day),
            key=f"{key_prefix}_day",
        )
    with col_d:
        send_hour = st.selectbox(
            "Heure d'envoi", list(range(6, 22)),
            index=4,
            format_func=lambda h: f"{h}h00",
            key=f"{key_prefix}_hour",
        )
    device_main = st.radio(
        "Device principal", ["mobile", "desktop"],
        horizontal=True,
        format_func=lambda x: "📱 Mobile" if x == "mobile" else "💻 Desktop",
        key=f"{key_prefix}_device",
    )

    return {
        "campaign_type":              campaign_type,
        "offer_type":                 offer_type,
        "audience_segment":           audience_segment,
        "device_main":                device_main,
        "send_day":                   send_day,
        "subject_sentiment":          subject_sentiment,
        "send_hour":                  send_hour,
        "subject_length":             subject_length,
        "has_personalization":        int(has_personalization),
        "is_urgent":                  int(is_urgent),
        "has_image":                  int(has_image),
        "nb_links":                   nb_links,
        "cta_count":                  cta_count,
        "audience_size":              audience_size,
        "marketing_pressure":         marketing_pressure,
        "previous_segment_open_rate": previous_segment_open_rate,
        "previous_segment_ctr":       previous_segment_ctr,
    }

def display_results(predictions, params):
    """Affiche score, KPI, entonnoir et recommandations."""
    score = compute_campaign_score(predictions, benchmarks)
    label, color = score_label(score)

    st.markdown(
        f"""
        <div style="
            background:{color}18;border:2px solid {color};
            border-radius:12px;padding:20px;text-align:center;margin-bottom:20px;
        ">
            <div style="font-size:13px;color:#666;margin-bottom:4px;">CAMPAIGN SCORE</div>
            <div style="font-size:52px;font-weight:800;color:{color};line-height:1;">
                {score}<span style="font-size:22px">/100</span>
            </div>
            <div style="font-size:17px;color:{color};margin-top:6px;font-weight:600;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mean_open  = benchmarks["open_rate"]["mean"]
    mean_click = benchmarks["click_rate"]["mean"]
    mean_conv  = benchmarks["conversion_rate"]["mean"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Open Rate",        f"{predictions['open_rate']:.1%}",       f"{(predictions['open_rate']  - mean_open):+.1%} vs moy.")
    k2.metric("Click Rate",       f"{predictions['click_rate']:.1%}",      f"{(predictions['click_rate'] - mean_click):+.1%} vs moy.")
    k3.metric("Conversion Rate",  f"{predictions['conversion_rate']:.1%}", f"{(predictions['conversion_rate'] - mean_conv):+.1%} vs moy.")

    st.markdown("**Entonnoir estimé**")
    audience_size = params["audience_size"]
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("📤 Envois",      f"{audience_size:,}")
    f2.metric("👁️ Ouvertures",  f"{int(audience_size * predictions['open_rate']):,}")
    f3.metric("🖱️ Clics",       f"{int(audience_size * predictions['click_rate']):,}")
    f4.metric("💰 Conversions", f"{int(audience_size * predictions['conversion_rate']):,}")

    st.markdown("**💡 Recommandations**")
    for reco in build_recommendations(params, predictions):
        st.markdown(f"- {reco}")

    return score


# ══════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════

st.title("📧 CRM AI — Campaign Optimizer")
st.markdown(
    "Prédisez les performances de votre campagne email **avant envoi** "
    "grâce au machine learning."
)
st.divider()


# ══════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════

tab1, tab2 = st.tabs(["🔮 Prédiction", "⚖️ Comparaison de scénarios"])


# ══════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION SIMPLE
# ══════════════════════════════════════════════

with tab1:
    col_form, col_results = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("⚙️ Paramètres de la campagne")
        with st.form("form_single"):
            params = campaign_form("single")
            submitted = st.form_submit_button(
                "🔮 Prédire les performances",
                use_container_width=True,
                type="primary",
            )

    with col_results:
        st.subheader("📊 Résultats")
        if not submitted:
            st.info("Remplissez le formulaire et cliquez sur **Prédire les performances**.")
        else:
            predictions = predict(params)
            display_results(predictions, params)

    # Insights globaux
    st.divider()
    st.subheader("📌 Insights business — Données historiques")
    st.markdown("Ces insights sont calculés automatiquement sur l'ensemble du dataset CRM.")
    cols = st.columns(2)
    for i, (key, insight) in enumerate(insights.items()):
        with cols[i % 2]:
            st.info(insight["recommendation"])


# ══════════════════════════════════════════════
# ONGLET 2 — COMPARAISON DE SCÉNARIOS
# ══════════════════════════════════════════════

with tab2:
    st.markdown(
        "Configurez deux versions de votre campagne et comparez leurs performances "
        "pour choisir la meilleure avant envoi."
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.subheader("🅰️ Scénario A")
        with st.form("form_scenario_a"):
            params_a = campaign_form("scen_a", default_segment="loyal_customer", default_day="tuesday")
            submit_a = st.form_submit_button("Calculer Scénario A", use_container_width=True, type="primary")

    with col_b:
        st.subheader("🅱️ Scénario B")
        with st.form("form_scenario_b"):
            params_b = campaign_form("scen_b", default_segment="inactive_customer", default_day="thursday")
            submit_b = st.form_submit_button("Calculer Scénario B", use_container_width=True, type="primary")

    # Résultats comparatifs
    if submit_a or submit_b:
        st.divider()
        st.subheader("📊 Comparaison des résultats")

        pred_a = predict(params_a)
        pred_b = predict(params_b)
        score_a = compute_campaign_score(pred_a, benchmarks)
        score_b = compute_campaign_score(pred_b, benchmarks)
        label_a, color_a = score_label(score_a)
        label_b, color_b = score_label(score_b)

        # ── TABLEAU COMPARATIF ──
        st.markdown("**Vue synthétique**")

        winner = "A" if score_a >= score_b else "B"
        diff   = abs(score_a - score_b)

        comparison_data = {
            "Métrique":            ["Campaign Score", "Open Rate", "Click Rate", "Conversion Rate"],
            "🅰️ Scénario A":       [
                f"{score_a}/100",
                f"{pred_a['open_rate']:.1%}",
                f"{pred_a['click_rate']:.1%}",
                f"{pred_a['conversion_rate']:.1%}",
            ],
            "🅱️ Scénario B":       [
                f"{score_b}/100",
                f"{pred_b['open_rate']:.1%}",
                f"{pred_b['click_rate']:.1%}",
                f"{pred_b['conversion_rate']:.1%}",
            ],
            "Écart (A - B)": [
                f"{score_a - score_b:+.1f} pts",
                f"{(pred_a['open_rate']  - pred_b['open_rate']):+.1%}",
                f"{(pred_a['click_rate'] - pred_b['click_rate']):+.1%}",
                f"{(pred_a['conversion_rate'] - pred_b['conversion_rate']):+.1%}",
            ],
        }
        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True,
            hide_index=True,
        )

        # ── VERDICT ──
        winner_color = color_a if winner == "A" else color_b
        winner_score = score_a if winner == "A" else score_b
        winner_label = label_a if winner == "A" else label_b
        winner_segment = params_a["audience_segment"] if winner == "A" else params_b["audience_segment"]
        winner_day     = params_a["send_day"]         if winner == "A" else params_b["send_day"]

        st.markdown(
            f"""
            <div style="
                background:{winner_color}18;border:2px solid {winner_color};
                border-radius:12px;padding:24px;margin-top:16px;
            ">
                <div style="font-size:20px;font-weight:700;color:{winner_color};margin-bottom:8px;">
                    ✅ Scénario {winner} recommandé — {winner_score}/100 ({winner_label})
                </div>
                <div style="font-size:15px;color:#444;line-height:1.6;">
                    Le scénario {winner} surperforme de <strong>{diff:.1f} points</strong>
                    grâce au segment <strong>{winner_segment}</strong>
                    envoyé le <strong>{winner_day}</strong>.
                    {"Activez la personnalisation pour maximiser les résultats."
                      if (params_a if winner=="A" else params_b)["has_personalization"] == 0
                      else "La personnalisation est activée — bon paramétrage."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── DÉTAIL CÔTE À CÔTE ──
        st.markdown("---")
        st.markdown("**Détail par scénario**")
        res_a, res_b = st.columns(2, gap="large")

        with res_a:
            st.markdown("**🅰️ Scénario A**")
            display_results(pred_a, params_a)

        with res_b:
            st.markdown("**🅱️ Scénario B**")
            display_results(pred_b, params_b)

