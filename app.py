
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="JobShield AI",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_model():
    model_dir = Path(__file__).resolve().parent / "models"
    model = joblib.load(model_dir / "fake_job_model.pkl")
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")

    return model, vectorizer


@st.cache_resource
def load_naive_bayes_model():
    model_dir = Path(__file__).resolve().parent / "models"
    model_path = model_dir / "fake_job_nb_model_v2.pkl"
    vectorizer_path = model_dir / "fake_job_nb_tfidf_vectorizer_v2.pkl"
    missing_files = [
        path.name for path in (model_path, vectorizer_path) if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing Naive Bayes artifact(s): " + ", ".join(missing_files)
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def predict_with_confidence(model, vectorizer, job_description):
    text_vector = vectorizer.transform([job_description])
    prediction = model.predict(text_vector)[0]
    predict_proba = getattr(model, "predict_proba", None)

    if not callable(predict_proba):
        raise AttributeError(
            f"{type(model).__name__} does not support predict_proba()."
        )

    probabilities = predict_proba(text_vector)[0]
    confidence = max(probabilities) * 100

    return prediction, confidence


def get_final_decision(logistic_prediction, naive_bayes_prediction):
    if logistic_prediction == 0 and naive_bayes_prediction == 0:
        return (
            "Likely Legitimate",
            "Both models found legitimate signals in this posting. "
            "Continue with normal verification before applying.",
            "success",
        )

    if logistic_prediction == 1 and naive_bayes_prediction == 1:
        return (
            "High Fraud Risk",
            "Both models identified this posting as potentially fraudulent. "
            "Do not send money or sensitive information.",
            "error",
        )

    return (
        "Mixed Signals – Verification Required",
        "The models disagree and cannot determine this posting's authenticity "
        "consistently. Treat it as unverified until you complete the checks below.",
        "warning",
    )


def render_analysis_results(
    logistic_result,
    naive_bayes_result,
    job_description,
):
    results = [result for result in (logistic_result, naive_bayes_result) if result]
    if not results:
        return

    if logistic_result is not None and naive_bayes_result is not None:
        decision, explanation, alert_type = get_final_decision(
            logistic_result[0],
            naive_bayes_result[0],
        )
        final_prediction = (
            1 if decision == "High Fraud Risk"
            else 0 if decision == "Likely Legitimate"
            else None
        )
    else:
        final_prediction = results[0][0]
        decision = (
            "Potentially Fraudulent"
            if final_prediction == 1
            else "Legitimate"
        )
        explanation = (
            "The available analysis signal indicates that this posting "
            "should be independently verified before you apply."
        )
        alert_type = "warning"

    authenticity = (
        "Potentially Fraudulent"
        if final_prediction != 0
        else "Legitimate"
    )
    result_class = "danger" if final_prediction != 0 else "safe"
    status_text = "REVIEW REQUIRED" if result_class == "danger" else "LOWER RISK SIGNAL"
    score = max(result[1] for result in results)
    score = max(0, min(score, 100))

    job_text = job_description.lower()
    indicator_rules = (
        (
            ("registration fee", "application fee", "processing fee", "pay a fee"),
            "Upfront registration or application fees",
        ),
        (
            ("guaranteed income", "guaranteed salary", "make $", "per day", "per week"),
            "Unusually strong or guaranteed earnings claims",
        ),
        (
            ("social security", "ssn", "passport", "bank account", "credit card"),
            "Requests for sensitive personal information",
        ),
        (
            ("whatsapp", "telegram", "wire transfer", "send money"),
            "Unusual contact or payment instructions",
        ),
    )
    indicators = [
        label
        for terms, label in indicator_rules
        if any(term in job_text for term in terms)
    ]
    company_terms = ("company:", "employer:", "organization:", "inc.", "llc", "ltd")
    if not any(term in job_text for term in company_terms):
        indicators.append("Company details are not clearly identified")

    indicator_markup = "".join(
        f'<li><span class="js-indicator-dot"></span>{indicator}</li>'
        for indicator in indicators
    )
    if not indicator_markup:
        indicator_markup = (
            '<li class="js-no-indicators">'
            '<span class="js-indicator-dot"></span>'
            "No specific warning signs were detected in the submitted text."
            "</li>"
        )

    recommendation_title = (
        "Recommended next step"
        if alert_type == "success"
        else "Safety recommendation"
    )
    st.html(f"""
    <section class="js-analysis">
        <div class="js-analysis-header">
            <div>
                <p class="js-section-kicker">Analysis results</p>
                <h2 class="js-analysis-title">Job authenticity</h2>
            </div>
            <span class="js-status {result_class}">
                <span class="js-status-dot"></span>{status_text}
            </span>
        </div>

        <div class="js-authenticity {result_class}">
            <div>
                <p class="js-card-label">Final assessment</p>
                <h3>{authenticity}</h3>
                <p>{explanation}</p>
            </div>
            <span class="js-authenticity-mark">{"!" if result_class == "danger" else "✓"}</span>
        </div>

        <div class="js-risk-card">
            <div class="js-confidence-line">
                <span>Fraud risk score <small>(model confidence, not probability)</small></span>
                <strong>{score:.2f}%</strong>
            </div>
            <div class="js-progress {result_class}" style="--js-confidence: {score:.2f}%">
                <span></span>
            </div>
        </div>

        <div class="js-detail-grid">
            <div class="js-detail-card">
                <p class="js-card-label">Suspicious indicators</p>
                <ul class="js-indicators">{indicator_markup}</ul>
            </div>
            <div class="js-detail-card js-recommendation">
                <p class="js-card-label">{recommendation_title}</p>
                <p>{explanation}</p>
                <p class="js-recommendation-note">
                    Verify the employer through its official website, never pay to apply,
                    and avoid sharing sensitive information before verification.
                </p>
            </div>
        </div>
    </section>
    """)


model, vectorizer = load_model()
try:
    nb_model, nb_vectorizer = load_naive_bayes_model()
    nb_load_error = None
except (FileNotFoundError, OSError, ValueError, ImportError, ModuleNotFoundError) as exc:
    nb_model = None
    nb_vectorizer = None
    nb_load_error = str(exc)


st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --js-ink: #111111;
    --js-muted: #777777;
    --js-border: #e7e7e7;
    --js-blue: #477bfa;
    --js-blue-soft: #eef4ff;
    --js-green: #287c58;
    --js-green-soft: #edf8f2;
    --js-red: #b55252;
    --js-red-soft: #fff1f1;
}

html, body, [class*="css"], button, textarea {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    color: var(--js-ink);
    background: #ffffff;
}

[data-testid="stHeader"] {
    background: #ffffff;
}

.block-container {
    max-width: 1080px;
    padding: 0 2rem 3.5rem;
}

.js-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.35rem 0;
    border-bottom: 1px solid #eeeeee;
}

.js-brand {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    color: var(--js-ink);
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.js-logo {
    display: grid;
    width: 2rem;
    height: 2rem;
    place-items: center;
    border-radius: 0.65rem;
    background: var(--js-ink);
    color: #ffffff;
    font-size: 0.95rem;
}

.js-nav-note {
    color: var(--js-muted);
    font-size: 0.78rem;
}

.js-hero {
    padding: 6rem 1rem 5.5rem;
    text-align: center;
}

.js-eyebrow {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    border: 1px solid #dce7ff;
    border-radius: 999px;
    background: var(--js-blue-soft);
    color: var(--js-blue);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
}

.js-hero-title {
    margin: 1.35rem 0 1rem;
    color: var(--js-ink);
    font-size: clamp(2.8rem, 7vw, 5rem);
    font-weight: 700;
    letter-spacing: -0.075em;
    line-height: 1.02;
}

.js-gradient {
    color: var(--js-blue);
}

.js-hero-copy {
    max-width: 37rem;
    margin: 0 auto;
    color: var(--js-muted);
    font-size: 1.05rem;
    line-height: 1.75;
}

.js-panel {
    padding: clamp(1.35rem, 4vw, 2.25rem);
    border: 1px solid var(--js-border);
    border-radius: 1rem;
    background: #ffffff;
}

.js-panel-heading {
    margin: 0;
    color: var(--js-ink);
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

.js-panel-copy {
    margin: 0.45rem 0 0;
    color: var(--js-muted);
    font-size: 0.9rem;
}

.st-key-job_description {
    margin-top: 0.85rem;
}

.st-key-job_description [data-testid="stTextAreaRootElement"],
.st-key-job_description [data-testid="stTextAreaRootElement"]:hover,
.st-key-job_description [data-testid="stTextAreaRootElement"]:focus-within {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 0.8rem !important;
    box-shadow: none !important;
    color: #222222 !important;
    color-scheme: light !important;
}

.st-key-job_description textarea,
.st-key-job_description textarea:hover,
.st-key-job_description textarea:focus {
    min-height: 13rem;
    background: #FFFFFF !important;
    color: #222222 !important;
    color-scheme: light !important;
    caret-color: #222222 !important;
    font-size: 0.98rem !important;
    line-height: 1.65 !important;
    outline: none !important;
    transition: border-color 180ms ease, box-shadow 180ms ease;
}

.st-key-job_description textarea::placeholder {
    color: #999999 !important;
    opacity: 1;
}

.st-key-job_description textarea::selection {
    background: #BFDBFE !important;
    color: #222222 !important;
}

.st-key-job_description [data-testid="stTextAreaRootElement"]:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
}

.st-key-job_description label {
    display: none;
}

.stButton {
    margin-top: 0.75rem;
}

.stButton > button {
    min-height: 3.15rem;
    border: 1px solid #111111;
    border-radius: 0.7rem;
    background: #111111;
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 700;
    transition: background 180ms ease, border-color 180ms ease, transform 180ms ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--js-blue);
    background: var(--js-blue);
}

.stButton > button:focus-visible {
    outline: 3px solid rgba(71, 123, 250, 0.24);
    outline-offset: 3px;
}

.js-analysis {
    margin-top: 1.5rem;
    padding: clamp(1.25rem, 3vw, 2rem);
    border: 1px solid var(--js-border);
    border-radius: 1rem;
    background: #ffffff;
}

.js-analysis-header,
.js-confidence-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.js-section-kicker,
.js-card-label {
    margin: 0;
    color: var(--js-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.js-analysis-title {
    margin: 0.3rem 0 0;
    color: var(--js-ink);
    font-size: 1.45rem;
    letter-spacing: -0.035em;
}

.js-status {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.7rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    white-space: nowrap;
}

.js-status.safe {
    background: var(--js-green-soft);
    color: var(--js-green);
}

.js-status.danger {
    background: var(--js-red-soft);
    color: var(--js-red);
}

.js-status-dot,
.js-indicator-dot {
    width: 0.45rem;
    height: 0.45rem;
    flex: 0 0 auto;
    border-radius: 50%;
    background: currentColor;
}

.js-authenticity {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 1.25rem;
    padding: 1.25rem;
    border: 1px solid var(--js-border);
    border-left: 4px solid;
    border-radius: 0.8rem;
}

.js-authenticity.safe {
    border-left-color: var(--js-green);
    background: var(--js-green-soft);
}

.js-authenticity.danger {
    border-left-color: var(--js-red);
    background: var(--js-red-soft);
}

.js-authenticity h3 {
    margin: 0.35rem 0;
    color: var(--js-ink);
    font-size: 1.45rem;
    letter-spacing: -0.04em;
}

.js-authenticity p:not(.js-card-label),
.js-recommendation p {
    margin: 0;
    color: var(--js-muted);
    font-size: 0.9rem;
    line-height: 1.6;
}

.js-authenticity-mark {
    display: grid;
    width: 2.5rem;
    height: 2.5rem;
    flex: 0 0 auto;
    place-items: center;
    border: 1px solid currentColor;
    border-radius: 50%;
    font-size: 1.1rem;
    font-weight: 700;
}

.js-authenticity.safe .js-authenticity-mark {
    color: var(--js-green);
}

.js-authenticity.danger .js-authenticity-mark {
    color: var(--js-red);
}

.js-risk-card {
    margin-top: 1rem;
    padding: 1rem 1.25rem;
    border: 1px solid var(--js-border);
    border-radius: 0.8rem;
}

.js-confidence-line {
    margin-bottom: 0.45rem;
    color: var(--js-muted);
    font-size: 0.78rem;
}

.js-confidence-line strong {
    color: var(--js-ink);
    font-size: 0.95rem;
}

.js-confidence-line small {
    color: #999999;
    font-size: 0.7rem;
}

.js-progress {
    height: 0.45rem;
    overflow: hidden;
    border-radius: 999px;
    background: #edf0f4;
}

.js-progress.safe > span {
    background: var(--js-green);
}

.js-progress.danger > span {
    background: var(--js-red);
}

.js-progress > span {
    display: block;
    width: var(--js-confidence);
    height: 100%;
    border-radius: inherit;
}

.js-detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.js-detail-card {
    padding: 1.1rem 1.25rem;
    border: 1px solid var(--js-border);
    border-radius: 0.8rem;
}

.js-indicators {
    display: grid;
    gap: 0.65rem;
    margin: 0.9rem 0 0;
    padding: 0;
    color: var(--js-ink);
    font-size: 0.86rem;
    line-height: 1.4;
    list-style: none;
}

.js-indicators li {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}

.js-indicator-dot {
    margin-top: 0.38rem;
    background: var(--js-red);
}

.js-indicators .js-no-indicators {
    color: var(--js-muted);
}

.js-no-indicators .js-indicator-dot {
    background: var(--js-green);
}

.js-recommendation-note {
    margin-top: 0.8rem !important;
    color: var(--js-muted);
    font-size: 0.78rem !important;
}

.js-footer {
    padding: 4rem 0 0.5rem;
    color: #999999;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-align: center;
}

[data-testid="stMetric"] {
    border: 1px solid var(--js-border);
    border-radius: 0.8rem;
    background: #ffffff;
    box-shadow: none;
}

[data-testid="stMetricLabel"] {
    color: var(--js-muted);
}

[data-testid="stMetricValue"] {
    color: var(--js-ink);
}

.stAlert {
    border-radius: 0.7rem;
}

@media (max-width: 640px) {
    .block-container { padding: 0 1rem 2.5rem; }
    .js-nav-note { display: none; }
    .js-hero { padding: 4rem 0.25rem 3.5rem; }
    .js-hero-copy { font-size: 0.95rem; }
    .js-analysis-header { align-items: flex-start; flex-direction: column; }
    .js-detail-grid { grid-template-columns: 1fr; }
}
</style>
""")

st.html("""
<nav class="js-nav">
    <div class="js-brand"><span class="js-logo">✓</span> JobShield AI</div>
    <div class="js-nav-note">Trust starts with verification</div>
</nav>
""")

st.html("""
<section class="js-hero">
    <div class="js-eyebrow">✦ AI-POWERED SECURITY</div>
    <h1 class="js-hero-title">JobShield <span class="js-gradient">AI</span></h1>
    <p class="js-hero-copy">
        Intelligent job screening powered by machine learning.
     Identify potential recruitment fraud and make more informed career decisions.
    </p>
</section>
""")

# ---------------- INPUT SECTION ----------------

st.html("""
<section class="js-panel">
    <h2 class="js-panel-heading">Analyze a job posting</h2>
    <p class="js-panel-copy">
    </p>
</section>
""")

job_description = st.text_area(
    "Job Description",
    height=230,
    placeholder="Paste the complete job description here...\n\nExample:\nJob Title: Python Developer\nCompany: ABC Technologies\nRequirements: Python, SQL, Machine Learning\nExperience: 0-2 years\nResponsibilities: Develop and maintain applications.",
    label_visibility="collapsed",
    key="job_description"
)

st.write("")

analyze = st.button("✦  Analyze Job Posting", width="content")

# ---------------- PREDICTION ----------------

if analyze:

    if len(job_description.strip()) < 30:
        st.warning("Please enter a more detailed job description.")

    else:

        with st.spinner("AI is analyzing the job posting..."):
            logistic_result = None
            naive_bayes_result = None
            prediction_errors = []

            try:
                logistic_result = predict_with_confidence(
                    model, vectorizer, job_description
                )
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                prediction_errors.append(f"Logistic Regression: {exc}")

            if nb_model is None or nb_vectorizer is None:
                prediction_errors.append(
                    f"Naive Bayes unavailable: {nb_load_error}"
                )
            else:
                try:
                    naive_bayes_result = predict_with_confidence(
                        nb_model, nb_vectorizer, job_description
                    )
                except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                    prediction_errors.append(f"Naive Bayes: {exc}")

        st.divider()

        if prediction_errors:
            st.warning(
                "Some analysis checks were unavailable. Review the result with "
                "extra caution."
            )

        render_analysis_results(
            logistic_result,
            naive_bayes_result,
            job_description,
        )

        st.write("")

        st.metric("Characters analyzed", f"{len(job_description):,}")

        


st.html("""
<footer class="js-footer">
    JOBSHIELD AI · <br>
    Built with Python · Scikit-learn · Streamlit
</footer>
""")
