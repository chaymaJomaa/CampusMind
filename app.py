import html
import re
from datetime import datetime, timedelta, time
from core.ml_engine import predict_priority
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from core.smart_planner import generate_smart_plan
from core.time_coach import explain_schedule

from core.quiz_engine import (
    generate_quiz,
    evaluate_quiz,
    save_quiz_result,
    get_weak_topics
)

from core.llm_engine import (
    check_ollama,
    generate_answer,
    summarize_document
)


import hashlib

from core.semantic_study import (
    extract_document,
    build_chunks,
    build_index,
    semantic_search
)

from core.feedback_manager import (
    save_feedback,
    get_feedback_stats
)


from core.notification_manager import (
    decide_and_save, release_pending, history, pending_items
)

from data.database import (
    init_database,
    add_course,
    add_exam,
    add_task,
    get_records,
)

st.set_page_config(
    page_title="CampusMind",
    page_icon=":material/school:",
    layout="wide",
)

init_database()

JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

:root {
    --ink: #1c2430;
    --muted: #6b7280;
    --line: #e4e0d6;
    --paper: #f7f5f0;
    --card: #ffffff;
    --accent: #1f5f5b;
    --accent-soft: #e2eeec;
    --amber: #a85d00;
    --amber-soft: #fbefdc;
    --red: #b42318;
    --red-soft: #fbe9e7;
}

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: var(--paper); }
#MainMenu, footer, [data-testid="stDeployButton"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }

/* Barre latérale */
[data-testid="stSidebar"] { background: #efece4; border-right: 1px solid var(--line); }
.brand { font-family: 'Fraunces', Georgia, serif; font-size: 1.55rem; font-weight: 600;
         color: var(--ink); letter-spacing: -0.01em; line-height: 1.1; }
.brand-sub { font-size: .78rem; color: var(--muted); margin: .15rem 0 1.4rem; }
.side-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .09em;
              color: var(--muted); margin: 1.6rem 0 .4rem; font-weight: 600; }

/* En-tête de page */
.eyebrow { font-size: .78rem; text-transform: uppercase; letter-spacing: .09em;
           color: var(--accent); font-weight: 600; }
.page-title { font-family: 'Fraunces', Georgia, serif; font-size: 2.3rem; font-weight: 600;
              color: var(--ink); margin: .15rem 0 .2rem; letter-spacing: -0.02em; }
.page-sub { color: var(--muted); font-size: .98rem; margin-bottom: 1.6rem; }

/* Cartes */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        padding: 1.2rem 1.35rem; margin-bottom: 1rem; }
.card-title { font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
              color: var(--muted); font-weight: 600; margin-bottom: .8rem; }

.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: .9rem; margin-bottom: 1.4rem; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.15rem; }
.kpi-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.kpi-value { font-family: 'Fraunces', Georgia, serif; font-size: 1.95rem; color: var(--ink);
             font-weight: 600; margin: .25rem 0 .1rem; line-height: 1.1; }
.kpi-note { font-size: .82rem; color: var(--muted); }

/* Recommandation */
.reco { border-left: 4px solid var(--accent); }
.reco.muted { border-left-color: var(--line); }
.reco-title { font-family: 'Fraunces', Georgia, serif; font-size: 1.45rem; color: var(--ink);
              font-weight: 600; margin-bottom: .35rem; }
.reco-text { color: #3b4452; font-size: .95rem; line-height: 1.5; }
.reco-meta { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .9rem; align-items: center; }
.reco-foot { font-size: .8rem; color: var(--muted); margin-top: .9rem; }

/* Pastilles */
.chip { display: inline-block; padding: .18rem .6rem; border-radius: 999px; font-size: .76rem; font-weight: 500; }
.chip-high { background: var(--red-soft); color: var(--red); }
.chip-mid { background: var(--amber-soft); color: var(--amber); }
.chip-low { background: var(--accent-soft); color: var(--accent); }
.chip-neutral { background: #efece4; color: #4b5563; }

/* Listes */
.row { display: flex; align-items: center; justify-content: space-between; gap: 1rem;
       padding: .7rem 0; border-bottom: 1px solid var(--line); }
.row:last-child { border-bottom: none; padding-bottom: 0; }
.row:first-of-type { padding-top: 0; }
.row-main { min-width: 0; }
.row-title { color: var(--ink); font-weight: 500; font-size: .95rem; }
.row-sub { color: var(--muted); font-size: .82rem; margin-top: .1rem; }
.row-time { font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 500; font-size: .9rem; white-space: nowrap; }
.now-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
           background: var(--accent); margin-right: .4rem; vertical-align: middle; }
.empty { color: var(--muted); font-size: .92rem; padding: .3rem 0; }

/* Planning hebdomadaire */
.week { display: grid; grid-template-columns: repeat(7, 1fr); gap: .6rem; }
.day { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
       padding: .75rem .65rem; min-height: 230px; }
.day.today { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
.day-head { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.day-num { font-family: 'Fraunces', Georgia, serif; font-size: 1.35rem; color: var(--ink); font-weight: 600; margin-bottom: .6rem; }
.slot { background: var(--accent-soft); border-radius: 8px; padding: .45rem .55rem; margin-bottom: .4rem; }
.slot b { display: block; font-size: .84rem; color: var(--ink); font-weight: 600; }
.slot span { font-size: .74rem; color: var(--accent); font-variant-numeric: tabular-nums; }
.slot.exam { background: var(--red-soft); }
.slot.exam span { color: var(--red); }
.slot-none { color: #b5b0a3; font-size: .8rem; }

/* Formulaires et onglets */
[data-testid="stForm"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.2rem 1.35rem; }
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 9px; font-weight: 500; border: 1px solid var(--accent);
    background: var(--accent); color: #fff; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #174a47; border-color: #174a47; color: #fff; }
button[role="tab"] { font-weight: 500; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }

@media (max-width: 960px) {
    .kpis { grid-template-columns: repeat(2, 1fr); }
    .week { grid-template-columns: repeat(2, 1fr); }
    .page-title { font-size: 1.9rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------

def render(markup):
    """Affiche du HTML sans que Markdown ne l'interprète comme du code."""
    st.markdown("".join(l.strip() for l in markup.splitlines()), unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value))


def fr_date(d):
    return f"{JOURS[d.weekday()]} {d.day} {MOIS[d.month - 1]}"


def to_date(value):
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def field(rec, *names, default=None):
    for name in names:
        if rec.get(name) not in (None, ""):
            return rec[name]
    return default


def countdown(days):
    if days < 0:
        return f"En retard de {-days} j"
    if days == 0:
        return "Aujourd'hui"
    if days == 1:
        return "Demain"
    return f"Dans {days} jours"


def urgency(days):
    if days <= 1:
        return "high", "Élevée"
    if days <= 7:
        return "mid", "Moyenne"
    return "low", "Normale"


def chip(text, tone="neutral"):
    return f'<span class="chip chip-{tone}">{esc(text)}</span>'


# --------------------------------------------------------------------------
# Accès aux données
# --------------------------------------------------------------------------

TIME_RE = re.compile(r"^\d{1,2}:\d{2}")


def load_courses():
    courses = []
    for raw in get_records("courses"):
        rec = dict(raw)
        times = [str(v)[:5] for v in rec.values() if isinstance(v, str) and TIME_RE.match(v)]
        start = field(rec, "start_time", "start", default=times[0] if times else "")
        end = field(rec, "end_time", "end", default=times[1] if len(times) > 1 else "")
        courses.append({
            "subject": field(rec, "subject", default="—"),
            "day": int(field(rec, "day_of_week", default=0)),
            "start": str(start)[:5],
            "end": str(end)[:5],
        })
    return sorted(courses, key=lambda c: (c["day"], c["start"]))


def load_exams(today):
    exams = []
    for raw in get_records("exams"):
        rec = dict(raw)
        d = to_date(field(rec, "exam_date", default=""))
        if d:
            exams.append({
                "subject": field(rec, "subject", default="—"),
                "date": d,
                "days": (d - today).days,
            })
    return sorted(exams, key=lambda e: e["date"])


def load_tasks(today):
    tasks = []
    for raw in get_records("tasks"):
        rec = dict(raw)
        d = to_date(field(rec, "deadline", "due_date", "date_limite", default=""))
        if not d:
            continue
        if int(field(rec, "completed", default=0)) == 1:
            continue
        tasks.append({
            "title": field(rec, "title", default="—"),
            "subject": field(rec, "subject", default=""),
            "date": d,
            "days": (d - today).days,
            "minutes": int(field(
                rec, "duration", "estimated_duration", "duration_minutes",
                "estimated_minutes", default=30,
            )),
        })
    return sorted(tasks, key=lambda t: t["date"])


# --------------------------------------------------------------------------
# Moteurs Context-Aware et recommandations — intégrés à VOTRE app.py
# --------------------------------------------------------------------------

def detect_context(courses, now, declared_status="Automatique", limit_minutes=180):
    """Déduit le contexte depuis l'emploi du temps hebdomadaire.

    Les cours sont supposés se répéter chaque semaine. L'override 'Libre'
    signifie que l'étudiant déclare être libre malgré un cours planifié.
    """
    now_hm = now.strftime("%H:%M")
    day_courses = sorted(
        [c for c in courses if c["day"] == now.weekday()],
        key=lambda c: c["start"],
    )
    current = next(
        (c for c in day_courses if c["start"] <= now_hm < c["end"]),
        None,
    )
    following = next((c for c in day_courses if c["start"] > now_hm), None)
    if following:
        start = datetime.combine(now.date(), time.fromisoformat(following["start"]))
        free = max(0, int((start - now).total_seconds() // 60))
    else:
        # Pas de cours restant : limite configurable plutôt que prétendre
        # que toutes les heures avant minuit sont des heures de travail.
        free = limit_minutes

    status = "Occupé(e)" if current else "Libre"
    if declared_status != "Automatique":
        status = declared_status
    if status != "Libre":
        free = 0
    return {
        "activity": status,
        "free_minutes": min(limit_minutes, free),
        "current_course": current,
        "next_course": following,
        "available": status == "Libre" and free >= 15,
        "today_courses": day_courses,
    }


def recommend_all(exams, tasks, activity, free_time):
    """Classement explicable, fondé sur des règles (pas encore une IA ML)."""
    if activity != "Libre":
        return [], "busy"
    if free_time < 15:
        return [], "short"

    candidates = []
    for e in exams:
        days = e["days"]
        if days < 0:
            continue
        duration = min(45, free_time)
        score = .8 * min(100, 100 / (1 + days / 2)) + .2 * (100 * duration / free_time)
        candidates.append({
            "type": "Examen", "title": f"Réviser {e['subject']}",
            "why": f"Examen {countdown(days).lower()} ({e['date']:%d/%m}).",
            "days": days, "minutes": duration, "score": round(score, 2),
        })
    for t in tasks:
        duration = t["minutes"]
        if duration <= 0 or duration > free_time:
            continue
        days = t["days"]
        urgency_score = 100 if days <= 0 else min(100, 100 / (1 + days / 2))
        score = .8 * urgency_score + .2 * (100 * duration / free_time)
        candidates.append({
            "type": "Tâche", "title": t["title"],
            "why": f"{countdown(days)} · {t['subject']} · {duration} min.",
            "days": days, "minutes": duration, "score": round(score, 2),
        })
    candidates.sort(key=lambda item: (-item["score"], item["days"], item["title"]))
    return candidates, ("ok" if candidates else "empty")


def recommend(exams, tasks, activity, free_time):
    """Adaptateur : préserve l'appel déjà présent dans le tableau de bord."""
    ranked, state = recommend_all(exams, tasks, activity, free_time)
    return {"state": "ok", **ranked[0]} if ranked else {"state": state}


# --------------------------------------------------------------------------
# Données du jour et simulation
# --------------------------------------------------------------------------

real_now = datetime.now()
courses = load_courses()

with st.sidebar:
    render("""
        <div class="brand">CampusMind</div>
        <div class="brand-sub">Assistant académique</div>
    """)
    page = st.radio(
        "Navigation",
        ["Tableau de bord", "Emploi du temps", "Gestion académique","Évaluation IA","Assistant IA","Planning intelligent"],
        label_visibility="collapsed",
    )
    render('<div class="side-label">Situation actuelle</div>')
    simulation = st.toggle("Mode simulation", value=False)
    if simulation:
        sim_day = st.date_input("Date simulée", real_now.date())
        sim_hour = st.time_input("Heure simulée", real_now.time().replace(second=0, microsecond=0))
        now = datetime.combine(sim_day, sim_hour)
    else:
        now = real_now
    declared_status = st.selectbox(
        "Disponibilité déclarée", ["Automatique", "Libre", "Occupé(e)", "En repos"]
    )
    horizon = st.slider("Créneau maximal à considérer (min)", 15, 180, 120, step=5)
    manual_time = st.toggle("Corriger le temps libre manuellement", value=False)
    detected = detect_context(courses, now, declared_status, horizon)
    activity = detected["activity"]
    free_time = detected["free_minutes"]
    if manual_time and activity == "Libre":
        free_time = st.slider("Temps libre réel (min)", 0, 180, free_time, step=5)
    st.caption(f"Contexte : {activity} · {free_time} min libres")
    if detected["current_course"]:
        st.caption(f"Cours planifié : {detected['current_course']['subject']}")
    notifications = st.toggle("Rappels contextuels", value=True)

today = now.date()
exams = load_exams(today)
tasks = load_tasks(today)
todays_courses = detected["today_courses"]
current_course = detected["current_course"]


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def page_header(title, subtitle):
    render(f"""
        <div class="eyebrow">{esc(fr_date(today))}</div>
        <div class="page-title">{esc(title)}</div>
        <div class="page-sub">{esc(subtitle)}</div>
    """)


def greeting():
    if now.hour < 12:
        return "Bonjour"
    if now.hour < 18:
        return "Bon après-midi"
    return "Bonsoir"


@st.fragment(run_every="30s")
def notification_panel():
    """Recalcule le contexte à chaque rafraîchissement sans toucher au design."""
    actual_time = now if simulation else datetime.now()
    live_courses = load_courses()
    live_context = detect_context(
        live_courses, actual_time, declared_status, horizon
    )
    live_activity = live_context["activity"]
    live_free = live_context["free_minutes"]
    if manual_time and live_activity == "Libre":
        live_free = free_time
    available = live_activity == "Libre" and live_free >= 15

    live_exams = load_exams(actual_time.date())
    live_tasks = load_tasks(actual_time.date())
    ranked, _ = recommend_all(
        live_exams, live_tasks, "Libre",
        live_free if available else max(15, horizon)
    )
    best = ranked[0] if ranked else None

    if not notifications:
        st.caption("🔕 Mode silencieux : aucun nouveau rappel affiché.")
    elif not best:
        st.caption("🔔 Aucun rappel à créer : aucune activité compatible.")
    elif not available:
        status = decide_and_save(
            best, available=False, enabled=True, at=actual_time
        )
        st.caption("🔕 Rappel différé pendant une période occupée." if status == "deferred"
                   else "Aucun nouveau rappel pendant cette période.")
    else:
        # On délivre d'abord un rappel en attente ; un seul par rafraîchissement.
        pending = release_pending(available=True, enabled=True, at=actual_time)
        if pending:
            st.toast(f"🔔 Rappel différé : {pending}")
            st.success(f"Rappel différé délivré : {pending}")
        else:
            outcome = decide_and_save(
                best, available=True, enabled=True, at=actual_time
            )
            if outcome == "displayed":
                st.toast(f"🔔 CampusMind : {best['title']}")
                st.success(f"Nouveau rappel : {best['title']}")
            else:
                st.caption("✓ Aucun nouveau rappel : déjà affiché aujourd'hui.")

    pending_count = len(pending_items())
    if pending_count:
        st.caption(f"{pending_count} rappel(s) en attente.")

    with st.expander("Historique des rappels"):
        records = history()
        if records:
            st.dataframe(
                pd.DataFrame([
                    {"Activité": rec["title"],
                     "Statut": "En attente" if rec["status"] == "deferred" else "Affiché",
                     "Date": rec["updated_at"].replace("T", " ")}
                    for rec in records
                ]),
                hide_index=True, width="stretch"
            )
        else:
            st.caption("Aucun rappel enregistré.")


def view_dashboard():
    page_header(greeting(), "Voici où vous en êtes aujourd'hui.")

    upcoming_exams = [e for e in exams if e["days"] >= 0]
    soon_tasks = [t for t in tasks if 0 <= t["days"] <= 7]
    workload = sum(t["minutes"] for t in soon_tasks)

    if upcoming_exams:
        nxt = upcoming_exams[0]
        exam_value, exam_note = f"J-{nxt['days']}", f"{nxt['subject']} · {nxt['date']:%d/%m}"
    else:
        exam_value, exam_note = "—", "Aucun examen planifié"

    next_course = detected["next_course"]
    course_note = (
        f"Prochain à {next_course['start']}" if next_course
        else ("Terminé pour aujourd'hui" if todays_courses else "Journée libre")
    )

    render(f"""
        <div class="kpis">
            <div class="kpi"><div class="kpi-label">Prochain examen</div>
                <div class="kpi-value">{esc(exam_value)}</div><div class="kpi-note">{esc(exam_note)}</div></div>
            <div class="kpi"><div class="kpi-label">Cours aujourd'hui</div>
                <div class="kpi-value">{len(todays_courses)}</div><div class="kpi-note">{esc(course_note)}</div></div>
            <div class="kpi"><div class="kpi-label">À rendre sous 7 jours</div>
                <div class="kpi-value">{len(soon_tasks)}</div><div class="kpi-note">{workload} min de travail estimé</div></div>
            <div class="kpi"><div class="kpi-label">Temps disponible</div>
                <div class="kpi-value">{free_time} min</div><div class="kpi-note">{esc(activity)}</div></div>
        </div>
    """)

    left, right = st.columns([3, 2], gap="large")

    with left:
        reco = recommend(exams, tasks, activity, free_time)

        if reco["state"] == "ok":
            tone, label = urgency(reco["days"])
            foot = (
                "Rappels contextuels activés." if notifications
                else "Mode silencieux : aucun rappel ne sera envoyé."
            )
            render(f"""
                <div class="card reco">
                    <div class="card-title">À faire maintenant</div>
                    <div class="reco-title">{esc(reco['title'])}</div>
                    <div class="reco-text">{esc(reco['why'])}</div>
                    <div class="reco-meta">{chip('Priorité ' + label.lower(), tone)}{chip(str(reco['minutes']) + ' min')}{chip(str(reco['score']) + '/100')}</div>
                    <div class="reco-foot">{foot}</div>
                </div>
            """)
        
            # ==========================================
            # COMPARAISON : REGLES VS RANDOM FOREST
            # ==========================================

            st.markdown("### Comparaison des moteurs")

            # Déterminer le type de l'activité
            is_exam = any(
                reco["title"] == f"Réviser {e['subject']}"
                for e in exams
            )

            activity_type = 1 if is_exam else 0

            # Valeur du moteur à règles
            rule_tone, rule_priority = urgency(
                reco["days"]
            )

            # Vérifier la validité des données
            if reco["days"] < 0:
                st.info(
                    "Activité échue : comparaison IA "
                    "non disponible."
                )

            elif reco["minutes"] > free_time:
                st.info(
                    "Activité incompatible avec "
                    "le temps disponible."
                )

            else:
                try:
                    # Prédiction Random Forest
                    ml_result = predict_priority(
                        days_remaining=reco["days"],
                        duration=reco["minutes"],
                        free_time=free_time,
                        activity_type=activity_type
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            "**Moteur à règles**"
                        )

                        st.metric(
                            "Priorité calculée",
                            rule_priority
                        )

                        st.caption(
                            "Décision basée sur "
                            "l'échéance académique."
                        )

                    with col2:
                        st.markdown(
                            "**Random Forest**"
                        )

                        st.metric(
                            "Priorité prédite",
                            ml_result["priority"]
                        )

                        confidence = ml_result[
                            "confidence"
                        ]

                        if confidence is not None:
                            st.caption(
                                f"Confiance du modèle : "
                                f"{confidence:.2f} %"
                            )

                    # Comparaison des décisions
                    if (
                        rule_priority
                        == ml_result["priority"]
                    ):
                        st.success(
                            "Les deux moteurs donnent "
                            "la même priorité."
                        )

                    else:
                        st.warning(
                            "Les deux moteurs donnent "
                            "des priorités différentes."
                        )

                except (
                    FileNotFoundError,
                    ValueError,
                    OSError
                ) as error:

                    st.error(
                        f"Erreur du modèle IA : {error}"
                    )
        
            # ======================================
            # APPRENTISSAGE DES PRÉFÉRENCES
            # ======================================

            st.markdown("### Votre avis sur la recommandation")

            st.caption(
                "CampusMind utilise vos réponses "
                "pour préparer des recommandations "
                "plus personnalisées."
            )

            # Identifier le type d'activité
            is_exam_feedback = any(
                reco["title"] == f"Réviser {e['subject']}"
                for e in exams
            )

            feedback_reco = {
                "title": reco["title"],
                "days": reco["days"],
                "minutes": reco["minutes"],
                "activity_type": (
                    1 if is_exam_feedback else 0
                )
            }

            c1, c2 = st.columns(2)

            with c1:
                if st.button(
                    "👍 Recommandation utile",
                    key="feedback_yes"
                ):
                    save_feedback(
                        feedback_reco,
                        free_time,
                        accepted=1,
                        at=now
                    )
                    st.success("Avis positif enregistré.")

            with c2:
                if st.button(
                    "👎 Pas utile",
                    key="feedback_no"
                ):
                    save_feedback(
                        feedback_reco,
                        free_time,
                        accepted=0,
                        at=now
                    )
                    st.info("Avis négatif enregistré.")

            stats = get_feedback_stats()

            st.caption(
                f"Interactions enregistrées : "
                f"{stats['total']} | "
                f"Acceptées : {stats['accepted']} | "
                f"Refusées : {stats['rejected']}"
            )
 
        else:
            messages = {
                "busy": ("Rien pour l'instant", "Vous êtes pris(e) : la recommandation est mise de côté jusqu'à votre prochain créneau libre."),
                "short": ("Créneau trop court", "Il faut au moins 15 minutes pour une séance utile. Augmentez le temps disponible ou attendez le prochain créneau."),
                "empty": ("Tout est à jour", "Aucun examen ni aucune tâche ne correspond à ce créneau. Ajoutez-en dans Gestion académique."),
            }
            title, text = messages[reco["state"]]
            render(f"""
                <div class="card reco muted">
                    <div class="card-title">À faire maintenant</div>
                    <div class="reco-title">{title}</div>
                    <div class="reco-text">{text}</div>
                </div>
            """)

        # Les recommandations restent visibles sans être envoyées de force.
        ranked, decision = recommend_all(exams, tasks, activity, free_time)
        if ranked:
            with st.expander("Voir les autres recommandations"):
                for i, candidate in enumerate(ranked, 1):
                    st.write(
                        f"{i}. **{candidate['title']}** — "
                        f"{candidate['minutes']} min · score {candidate['score']}/100"
                    )
                    st.caption(candidate["why"])

        # Zone autonome, actualisée toutes les 30 secondes (onglet ouvert).
        notification_panel()

        if todays_courses:
            rows = "".join(
                f"""<div class="row">
                    <div class="row-main"><div class="row-title">{'<span class="now-dot"></span>' if c is current_course else ''}{esc(c['subject'])}</div></div>
                    <div class="row-time">{esc(c['start'])} – {esc(c['end'])}</div>
                </div>"""
                for c in todays_courses
            )
        else:
            rows = '<div class="empty">Aucun cours aujourd\'hui.</div>'
        render(f'<div class="card"><div class="card-title">Cours du jour</div>{rows}</div>')

    with right:
        items = [
            {"kind": "Examen", "title": e["subject"], "sub": "", "date": e["date"], "days": e["days"]}
            for e in upcoming_exams
        ] + [
            {"kind": "Tâche", "title": t["title"], "sub": t["subject"], "date": t["date"], "days": t["days"]}
            for t in tasks
        ]
        items.sort(key=lambda i: i["date"])

        if items:
            rows = ""
            for it in items[:7]:
                tone = "high" if it["days"] <= 1 else ("mid" if it["days"] <= 7 else "neutral")
                sub = " · ".join(x for x in (it["kind"], it["sub"], f"{it['date']:%d/%m}") if x)
                rows += f"""<div class="row">
                    <div class="row-main"><div class="row-title">{esc(it['title'])}</div>
                    <div class="row-sub">{esc(sub)}</div></div>
                    {chip(countdown(it['days']), tone)}
                </div>"""
        else:
            rows = '<div class="empty">Aucune échéance à venir.</div>'
        render(f'<div class="card"><div class="card-title">Échéances</div>{rows}</div>')


def view_schedule():
    page_header("Emploi du temps", "Votre semaine, cours et examens.")

    monday = today - timedelta(days=today.weekday())
    columns = ""

    for i, name in enumerate(JOURS):
        day = monday + timedelta(days=i)
        slots = "".join(
            f'<div class="slot"><b>{esc(c["subject"])}</b><span>{esc(c["start"])} – {esc(c["end"])}</span></div>'
            for c in courses if c["day"] == i
        )
        slots += "".join(
            f'<div class="slot exam"><b>{esc(e["subject"])}</b><span>Examen</span></div>'
            for e in exams if e["date"] == day
        )
        if not slots:
            slots = '<div class="slot-none">Libre</div>'

        cls = "day today" if day == today else "day"
        columns += f"""<div class="{cls}">
            <div class="day-head">{name[:3]}.</div>
            <div class="day-num">{day.day}</div>{slots}</div>"""

    render(f'<div class="week">{columns}</div>')


def view_management():
    page_header("Gestion académique", "Cours, examens et tâches.")

    tab_courses, tab_exams, tab_tasks = st.tabs(["Cours", "Examens", "Tâches"])

    with tab_courses:
        form_col, list_col = st.columns([2, 3], gap="large")

        with form_col:
            with st.form("course_form", clear_on_submit=True):
                subject = st.text_input("Matière", key="course_subject")
                day_name = st.selectbox("Jour", JOURS)
                c1, c2 = st.columns(2)
                start = c1.time_input("Début", value=datetime.strptime("08:00", "%H:%M").time())
                end = c2.time_input("Fin", value=datetime.strptime("10:00", "%H:%M").time())
                submitted = st.form_submit_button("Enregistrer le cours")

                if submitted:
                    if not subject.strip():
                        st.error("Indiquez la matière.")
                    elif end <= start:
                        st.error("L'heure de fin doit suivre l'heure de début.")
                    else:
                        try:
                            add_course(
                                subject.strip(),
                                JOURS.index(day_name),
                                start.strftime("%H:%M"),
                                end.strftime("%H:%M"),
                            )
                            st.toast("Cours enregistré")
                        except ValueError as error:
                            st.error(str(error))

        with list_col:
            current = load_courses()
            if current:
                st.dataframe(
                    pd.DataFrame([
                        {"Matière": c["subject"], "Jour": JOURS[c["day"]], "Début": c["start"], "Fin": c["end"]}
                        for c in current
                    ]),
                    hide_index=True,
                    width="stretch",
                )
            else:
                render('<div class="empty">Aucun cours enregistré.</div>')

    with tab_exams:
        form_col, list_col = st.columns([2, 3], gap="large")

        with form_col:
            with st.form("exam_form", clear_on_submit=True):
                subject = st.text_input("Matière", key="exam_subject")
                exam_date = st.date_input("Date", value=today, format="DD/MM/YYYY", key="exam_date")
                submitted = st.form_submit_button("Enregistrer l'examen")

                if submitted:
                    if subject.strip():
                        add_exam(subject.strip(), exam_date.isoformat())
                        st.toast("Examen enregistré")
                    else:
                        st.error("Indiquez la matière.")

        with list_col:
            current = load_exams(today)
            if current:
                st.dataframe(
                    pd.DataFrame([
                        {"Matière": e["subject"], "Date": f"{e['date']:%d/%m/%Y}",
                         "Échéance": countdown(e["days"]) if e["days"] >= 0 else "Passé"}
                        for e in current
                    ]),
                    hide_index=True,
                    width="stretch",
                )
            else:
                render('<div class="empty">Aucun examen enregistré.</div>')

    with tab_tasks:
        form_col, list_col = st.columns([2, 3], gap="large")

        with form_col:
            with st.form("task_form", clear_on_submit=True):
                title = st.text_input("Titre", key="task_title")
                subject = st.text_input("Matière", key="task_subject")
                c1, c2 = st.columns(2)
                deadline = c1.date_input("Date limite", value=today, format="DD/MM/YYYY", key="task_deadline")
                duration = c2.number_input("Durée (min)", min_value=5, value=30, step=5)
                submitted = st.form_submit_button("Enregistrer la tâche")

                if submitted:
                    if title.strip() and subject.strip():
                        add_task(title.strip(), subject.strip(), deadline.isoformat(), duration)
                        st.toast("Tâche enregistrée")
                    else:
                        st.error("Renseignez le titre et la matière.")

        with list_col:
            current = load_tasks(today)
            if current:
                st.dataframe(
                    pd.DataFrame([
                        {"Tâche": t["title"], "Matière": t["subject"],
                         "Date limite": f"{t['date']:%d/%m/%Y}", "Durée": f"{t['minutes']} min"}
                        for t in current
                    ]),
                    hide_index=True,
                    width="stretch",
                )
            else:
                render('<div class="empty">Aucune tâche enregistrée.</div>')


# ============================================================
# PAGE : ÉVALUATION DE L'INTELLIGENCE ARTIFICIELLE
# ============================================================

def view_evaluation():

    page_header(
        "Évaluation IA",
        "Analyse comparative du moteur à règles et de Random Forest."
    )

    results_dir = (
        Path(__file__).resolve().parent / "results"
    )

    metrics_path = results_dir / "metrics.json"
    comparison_path = results_dir / "comparison.csv"

    if not metrics_path.exists() or not comparison_path.exists():
        st.warning(
            "Aucun résultat d'évaluation disponible. "
            "Exécutez : python -m core.evaluate_model"
        )
        return

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    df = pd.read_csv(comparison_path)

    # ======================================
    # 1. INDICATEURS
    # ======================================

    st.subheader("Indicateurs expérimentaux")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Scénarios évalués",
        metrics["evaluated_scenarios"]
    )

    col2.metric(
        "Taux d'accord",
        f'{metrics["agreement_rate"]:.2f} %'
    )

    col3.metric(
        "Désaccords",
        metrics["disagreements"]
    )

    col4.metric(
        "Scénarios exclus",
        metrics["excluded_scenarios"]
    )

    st.divider()

    # ======================================
    # 2. CONCORDANCE
    # ======================================

    st.subheader("Accord entre les deux moteurs")

    agreement_data = pd.DataFrame({
        "Décision": ["Accord", "Désaccord"],
        "Nombre": [
            metrics["agreements"],
            metrics["disagreements"]
        ]
    })

    st.bar_chart(
        agreement_data.set_index("Décision"),
        y="Nombre"
    )

    st.caption(
        "Le taux d'accord mesure la concordance "
        "des décisions et non la précision réelle "
        "des recommandations."
    )

    # ======================================
    # 3. COMPARAISON DES TEMPS
    # ======================================

    st.subheader("Temps moyen de prédiction")

    time_data = pd.DataFrame({
        "Moteur": [
            "Règles",
            "Random Forest"
        ],
        "Temps (ms)": [
            metrics["avg_rule_time_ms"],
            metrics["avg_ml_time_ms"]
        ]
    })

    st.dataframe(
        time_data,
        hide_index=True,
        width="stretch"
    )

    st.bar_chart(
        time_data.set_index("Moteur"),
        y="Temps (ms)"
    )

    # ======================================
    # 4. ANALYSE DES DÉSACCORDS
    # ======================================

    st.subheader("Analyse des désaccords")

    # Accepte les booléens et les chaînes CSV
    agreement_mask = (
        df["agreement"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    disagreements = df.loc[
        ~agreement_mask
    ].copy()

    if disagreements.empty:
        st.success(
            "Aucun désaccord dans les scénarios évalués."
        )

    else:
        st.warning(
            f"{len(disagreements)} scénarios présentent "
            "des priorités différentes."
        )

        st.dataframe(
            disagreements[
                [
                    "days_remaining",
                    "duration",
                    "free_time",
                    "activity_type",
                    "rule_priority",
                    "ml_priority",
                    "ml_confidence"
                ]
            ],
            hide_index=True,
            width="stretch"
        )

        st.subheader("Désaccords selon l'échéance")

        counts = (
            disagreements.groupby("days_remaining")
            .size()
            .reset_index(name="Nombre")
        )

        st.bar_chart(
            counts.set_index("days_remaining"),
            y="Nombre"
        )

    # ======================================
    # 5. DONNÉES COMPLÈTES
    # ======================================

    with st.expander("Consulter tous les scénarios"):

        st.dataframe(
            df,
            hide_index=True,
            width="stretch"
        )

    st.download_button(
        "Télécharger les résultats CSV",
        data=comparison_path.read_bytes(),
        file_name="campusmind_evaluation.csv",
        mime="text/csv"
    )

    # ======================================
    # 6. LIMITES
    # ======================================

    st.info(
        "Cette expérience utilise des scénarios "
        "synthétiques. Elle mesure la concordance "
        "avec le moteur à règles, pas la pertinence "
        "des recommandations pour de vrais étudiants."
    )


# ==========================================
# INTERFACE QCM INTELLIGENT
# ==========================================

def render_quiz_section(index):

    st.divider()

    st.markdown("### 4. Quiz intelligent")

    st.caption(
        "Générez des questions à partir "
        "de vos documents de cours."
    )

    number_questions = st.slider(
        "Nombre de questions",
        min_value=3,
        max_value=10,
        value=5,
        key="quiz_question_count"
    )

    if "quiz_version" not in st.session_state:
        st.session_state["quiz_version"] = 0

    # ======================================
    # 1. GÉNÉRER UN QUIZ
    # ======================================

    if st.button(
        "Générer un quiz IA",
        key="generate_quiz_button"
    ):

        with st.spinner(
            "Qwen prépare les questions..."
        ):

            try:
                questions = generate_quiz(
                    index["chunks"],
                    n_questions=number_questions
                )

                st.session_state[
                    "current_quiz"
                ] = questions

                st.session_state[
                    "quiz_result"
                ] = None

                st.session_state[
                    "quiz_version"
                ] += 1

                st.success(
                    f"{len(questions)} questions "
                    "générées avec succès."
                )

            except Exception as error:
                st.error(
                    f"Erreur de génération : {error}"
                )

    questions = st.session_state.get(
        "current_quiz"
    )

    if not questions:
        return

    st.markdown("### Répondez aux questions")

    # ======================================
    # 2. FORMULAIRE DU QUIZ
    # ======================================

    version = st.session_state["quiz_version"]

    with st.form(
        f"quiz_form_{version}"
    ):

        answers = []

        for i, question in enumerate(
            questions
        ):

            st.markdown(
                f"**Question {i + 1}**"
            )

            st.write(question["question"])

            answer = st.radio(
                "Sélectionnez une réponse",
                options=list(range(4)),
                format_func=lambda x, opts=question["options"]:
                    opts[x],
                index=None,
                key=f"quiz_answer_{version}_{i}"
            )

            answers.append(answer)

            st.divider()

        submitted = st.form_submit_button(
            "Corriger mon quiz"
        )

    # ======================================
    # 3. CORRECTION
    # ======================================

    if submitted:

        if any(
            answer is None
            for answer in answers
        ):
            st.warning(
                "Répondez à toutes les questions "
                "avant de valider."
            )

        elif st.session_state.get(
            "quiz_result"
        ) is not None:

            st.info(
                "Ce quiz a déjà été corrigé."
            )

        else:

            try:
                result = evaluate_quiz(
                    questions,
                    answers
                )

                save_quiz_result(
                    questions,
                    answers
                )

                st.session_state[
                    "quiz_result"
                ] = result

            except Exception as error:
                st.error(
                    f"Erreur de correction : {error}"
                )

    result = st.session_state.get(
        "quiz_result"
    )

    if result is not None:

        st.markdown("### Résultat du quiz")

        col1, col2 = st.columns(2)

        col1.metric(
            "Score",
            f"{result['score']} %"
        )

        col2.metric(
            "Bonnes réponses",
            f"{result['correct']}/{result['total']}"
        )

        st.progress(
            result["score"] / 100
        )

        # Explication de chaque question
        for i, detail in enumerate(
            result["details"],
            start=1
        ):

            with st.expander(
                f"Question {i} — "
                + (
                    "Correcte"
                    if detail["is_correct"]
                    else "Incorrecte"
                ),
                expanded=not detail["is_correct"]
            ):

                st.write(
                    detail["question"]
                )

                st.markdown(
                    "**Bonne réponse :** "
                    + detail["correct_answer"]
                )

                st.markdown(
                    "**Explication :** "
                    + detail["explanation"]
                )

                st.caption(
                    f"Notion : {detail['topic']} | "
                    f"Source : {detail['source']} "
                    f"— page {detail['page']}"
                )

    # ======================================
    # 4. ANALYSE DES DIFFICULTÉS
    # ======================================

    st.markdown(
        "### Notions à retravailler"
    )

    weak_topics = get_weak_topics()

    weak_topics = [
        item
        for item in weak_topics
        if item["success_rate"] < 70
    ]

    if weak_topics:

        for item in weak_topics:

            st.write(
                f"**{item['topic']}** — "
                f"Réussite : {item['success_rate']} % "
                f"({item['attempts']} réponse(s))"
            )

            st.progress(
                item["success_rate"] / 100
            )

        st.info(
            "CampusMind recommande de revoir "
            "ces notions avant le prochain quiz."
        )

    else:

        st.caption(
            "Aucune difficulté identifiée "
            "dans les réponses enregistrées."
        )

# ============================================================
# ASSISTANT IA - RECHERCHE SÉMANTIQUE
# ============================================================

def view_ai_assistant():

    page_header(
        "Assistant pédagogique IA",
        "Interrogez vos documents avec Qwen 2.5."
    )

    # =====================================
    # 1. CONNEXION OLLAMA
    # =====================================

    status = check_ollama()

    if status["connected"] and status["model_available"]:
        st.success("Qwen 2.5 3B connecté")

    else:
        st.error(
            "Ollama ou le modèle Qwen n'est pas disponible."
        )

        st.code("ollama pull qwen2.5:3b")
        return

    # =====================================
    # 2. IMPORTATION DES DOCUMENTS
    # =====================================

    st.markdown("### 1. Documents de cours")

    uploaded_files = st.file_uploader(
        "Importer des documents PDF ou TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="study_documents"
    )

    if not uploaded_files:
        st.info("Importez un document pour commencer.")
        return

    file_data = [
        (file.name, file.getvalue())
        for file in uploaded_files
    ]

    digest = hashlib.sha256()

    for filename, content in file_data:
        digest.update(filename.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    document_key = digest.hexdigest()

    if (
        st.session_state.get("study_document_key")
        != document_key
    ):
        st.session_state["study_document_key"] = None
        st.session_state["study_index"] = None
        st.session_state["study_answer"] = None
        st.session_state["study_sources"] = []

    if st.button(
        "Analyser les documents",
        key="build_study_index"
    ):

        with st.spinner("Indexation des documents..."):

            try:
                pages = []

                for filename, content in file_data:
                    pages.extend(
                        extract_document(filename, content)
                    )

                chunks = build_chunks(pages)

                if not chunks:
                    st.error(
                        "Aucun texte exploitable dans le document."
                    )
                    return

                embeddings = build_index(chunks)

                st.session_state["study_index"] = {
                    "chunks": chunks,
                    "embeddings": embeddings
                }

                st.session_state[
                    "study_document_key"
                ] = document_key

                st.success(
                    f"{len(chunks)} passages indexés."
                )

            except Exception as error:
                st.error(f"Erreur : {error}")
                return

    index = st.session_state.get("study_index")

    if (
        index is None
        or st.session_state.get("study_document_key")
        != document_key
    ):
        st.info(
            "Cliquez sur Analyser les documents."
        )
        return

    st.caption(
        f"{len(index['chunks'])} passages disponibles."
    )

    # =====================================
    # 3. RÉSUMÉ DU DOCUMENT
    # =====================================

    st.markdown("### 2. Résumé du document")

    if st.button(
        "Générer une synthèse IA",
        key="summary_button"
    ):
        with st.spinner("Qwen prépare la synthèse..."):

            try:
                summary = summarize_document(
                    index["chunks"],
                    max_chunks=6
                )

                st.session_state["study_summary"] = summary

            except Exception as error:
                st.error(str(error))

    if st.session_state.get("study_summary"):

        st.markdown(
            st.session_state["study_summary"]
        )

        st.caption(
            "Synthèse partielle des premiers passages, "
            "pas nécessairement du document entier."
        )

    # =====================================
    # 4. QUESTIONS À L'IA
    # =====================================

    st.markdown("### 3. Poser une question")

    question = st.text_area(
        "Votre question",
        placeholder=(
            "Quel est l'objectif principal du document ?"
        ),
        key="study_question"
    )

    top_k = st.slider(
        "Nombre de passages utilisés",
        1, 5, 3,
        key="study_top_k"
    )

    if st.button(
        "Obtenir une réponse IA",
        key="generate_answer_button"
    ):

        if not question.strip():
            st.warning("Saisissez une question.")

        else:
            with st.spinner(
                "CampusMind prépare votre réponse..."
            ):
                try:
                    sources = semantic_search(
                        question,
                        index["chunks"],
                        index["embeddings"],
                        top_k=top_k
                    )

                    if not sources:
                        st.warning(
                            "Aucun passage pertinent disponible."
                        )
                    else:
                        answer = generate_answer(
                            question,
                            sources
                        )

                        st.session_state[
                            "study_answer"
                        ] = answer

                        st.session_state[
                            "study_sources"
                        ] = sources

                except Exception as error:
                    st.error(
                        f"Erreur Ollama : {error}"
                    )

    # =====================================
    # 5. RÉPONSE GÉNÉRÉE
    # =====================================

    answer = st.session_state.get("study_answer")

    if answer:

        st.markdown("### Réponse de CampusMind IA")

        with st.chat_message("assistant"):
            st.markdown(answer)

        with st.expander("Sources utilisées"):

            for i, source in enumerate(
                st.session_state.get(
                    "study_sources", []
                ),
                start=1
            ):

                st.markdown(
                    f"**[Source {i}] — "
                    f"{source['source']} "
                    f"(page {source['page']})**"
                )

                st.write(source["text"])

                st.caption(
                    f"Similarité : {source['score']:.3f}"
                )

                st.divider()

        st.caption(
            "Réponse générée automatiquement. "
            "Vérifiez les références dans le PDF."
        )
    render_quiz_section(index)

def view_smart_planning():

    page_header(
        "Planning intelligent",
        "Organisation automatique de votre temps académique."
    )

    selected_date = st.date_input(
        "Date à organiser",
        value=datetime.now().date(),
        key="planning_date"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        start_hour = st.slider(
            "Début",
            6, 12, 8,
            key="plan_start"
        )

    with c2:
        end_hour = st.slider(
            "Fin",
            16, 23, 20,
            key="plan_end"
        )

    with c3:
        break_time = st.slider(
            "Pause (minutes)",
            5, 20, 10, 5,
            key="plan_break"
        )

    # Données académiques existantes
    courses_data = load_courses()
    tasks_data = load_tasks(selected_date)
    exams_data = load_exams(selected_date)

    if st.button(
        "Générer mon planning intelligent",
        key="generate_plan"
    ):

        current = datetime.now()

        earliest = (
            current.time()
            if selected_date == current.date()
            else None
        )

        try:
            planning = generate_smart_plan(
                target_date=selected_date,
                courses=courses_data,
                tasks=tasks_data,
                exams=exams_data,
                start_hour=start_hour,
                end_hour=end_hour,
                break_minutes=break_time,
                earliest_time=earliest
            )

            st.session_state["smart_plan"] = planning
            st.session_state["smart_plan_date"] = selected_date
            st.session_state["planning_ai_advice"] = None

        except Exception as error:
            st.error(f"Erreur de planification : {error}")

    planning = st.session_state.get("smart_plan", [])

    if st.session_state.get("smart_plan_date") != selected_date:
        st.info("Générez le planning pour cette date.")
        return

    if not planning:
        st.info("Aucune activité planifiable.")
        return

    study_minutes = sum(
        item["duration"]
        for item in planning
        if item["type"] in ("Tâche", "Révision")
    )

    course_minutes = sum(
        item["duration"]
        for item in planning
        if item["type"] == "Cours"
    )

    sessions = sum(
        item["type"] in ("Tâche", "Révision")
        for item in planning
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Temps d'étude", f"{study_minutes} min")
    c2.metric("Cours", f"{course_minutes} min")
    c3.metric("Séances", sessions)

    st.divider()

    for item in planning:
        with st.container(border=True):

            c1, c2 = st.columns([1, 3])

            with c1:
                st.markdown(f"**{item['start_time']}**")
                st.caption(item["end_time"])

            with c2:
                st.markdown(f"**{item['title']}**")
                st.caption(
                    f"{item['type']} - "
                    f"{item['duration']} minutes"
                )

    with st.expander("Planning détaillé"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Début": item["start_time"],
                    "Fin": item["end_time"],
                    "Activité": item["title"],
                    "Type": item["type"],
                    "Durée": item["duration"]
                }
                for item in planning
            ]),
            hide_index=True,
            width="stretch"
        )

    # Analyse avec Qwen
    st.divider()

    st.markdown("### Coach IA — Gestion du temps")

    if st.button(
        "Analyser mon planning avec Qwen",
        key="coach_plan"
    ):

        with st.spinner("Qwen analyse le planning..."):
            try:
                advice = explain_schedule(
                    planning,
                    selected_date
                )

                st.session_state["planning_ai_advice"] = advice

            except Exception as error:
                st.error(str(error))

    if st.session_state.get("planning_ai_advice"):
        with st.container(border=True):
            st.markdown(
                st.session_state["planning_ai_advice"]
            )


if page == "Tableau de bord":
    view_dashboard()

elif page == "Emploi du temps":
    view_schedule()

elif page == "Gestion académique":
    view_management()

elif page == "Évaluation IA":
    view_evaluation()

elif page == "Assistant IA":
    view_ai_assistant()
elif page == "Planning intelligent":
    view_smart_planning()