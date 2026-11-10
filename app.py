
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
    model_dir = Path(__file__).resolve().parent
    model = joblib.load(model_dir / "fake_job_model.pkl")
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")

    return model, vectorizer


@st.cache_resource
def load_naive_bayes_model():
    model_dir = Path(__file__).resolve().parent
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


def render_prediction_result(model_name, prediction, confidence):
    if prediction == 1:
        status = "FAKE RISK DETECTED"
        title = "⚠️ Potentially fraudulent"
        result_class = "danger"
        copy = (
            "This model classified the job posting as potentially fake. "
            "Verify the company and recruiter before sharing personal "
            "information or making payments."
        )
    else:
        status = "LEGITIMATE SIGNALS"
        title = "✓ Likely legitimate"
        result_class = "safe"
        copy = (
            "This model classified the job posting as likely legitimate. "
            "Always independently verify the employer before applying."
        )

    st.html(f"""
    <section class="js-result {result_class}">
        <span class="js-status {result_class}">{status}</span>
        <h3 class="js-result-title">{model_name}: {title}</h3>
        <p class="js-result-copy">{copy}</p>
        <div class="js-confidence">
            <div class="js-confidence-line">
                <span>Model confidence</span><strong>{confidence:.2f}%</strong>
            </div>
            <div class="js-progress" style="--js-confidence: {confidence:.2f}%">
                <span></span>
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

.js-result {
    margin-top: 1.5rem;
    padding: 1.5rem;
    border: 1px solid var(--js-border);
    border-radius: 1rem;
    background: #ffffff;
}

.js-result.safe {
    border-left: 4px solid var(--js-green);
}

.js-result.danger {
    border-left: 4px solid var(--js-red);
}

.js-status {
    display: inline-block;
    margin-bottom: 0.8rem;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.js-status.safe {
    background: var(--js-green-soft);
    color: var(--js-green);
}

.js-status.danger {
    background: var(--js-red-soft);
    color: var(--js-red);
}

.js-result-title {
    margin: 0 0 0.45rem;
    color: var(--js-ink);
    font-size: 1.35rem;
    font-weight: 700;
}

.js-result-copy {
    margin: 0;
    color: var(--js-muted);
    font-size: 0.9rem;
    line-height: 1.65;
}

.js-confidence {
    margin-top: 1.25rem;
}

.js-confidence-line {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.45rem;
    color: var(--js-muted);
    font-size: 0.78rem;
}

.js-confidence-line strong {
    color: var(--js-ink);
}

.js-progress {
    height: 0.42rem;
    overflow: hidden;
    border-radius: 999px;
    background: #edf0f4;
}

.js-progress > span {
    display: block;
    width: var(--js-confidence);
    height: 100%;
    border-radius: inherit;
    background: var(--js-blue);
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
        Your intelligent shield against fraudulent job postings.<br>
        Analyze job descriptions with confidence using machine learning.
    </p>
</section>
""")

# ---------------- INPUT SECTION ----------------

st.html("""
<section class="js-panel">
    <h2 class="js-panel-heading">Analyze a job posting</h2>
    <p class="js-panel-copy">
        Paste the complete job description below to check its authenticity.
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
        st.subheader("Analysis results")

        if prediction_errors:
            for prediction_error in prediction_errors:
                st.error(prediction_error)

        result_columns = st.columns(2)
        if logistic_result is not None:
            with result_columns[0]:
                render_prediction_result(
                    "Logistic Regression",
                    logistic_result[0],
                    logistic_result[1],
                )

        if naive_bayes_result is not None:
            with result_columns[1]:
                render_prediction_result(
                    "Naive Bayes",
                    naive_bayes_result[0],
                    naive_bayes_result[1],
                )

        if logistic_result is not None and naive_bayes_result is not None:
            decision, explanation, alert_type = get_final_decision(
                logistic_result[0],
                naive_bayes_result[0],
            )
            st.subheader("Final recommendation")
            getattr(st, alert_type)(
                explanation,
                title=decision,
            )
            st.markdown(
                """
                **Before taking action:**
                - Verify the employer through its official website.
                - Verify the recruiter's identity independently.
                - Never pay registration or interview fees.
                - Avoid sharing sensitive information before verification.
                """
            )
            st.caption(
                "The two model confidence scores are shown separately above. "
                "They are not averaged or combined."
            )

        st.write("")

        metric_columns = st.columns(3)

        with metric_columns[0]:
            if logistic_result is not None:
                st.metric(
                    "Logistic confidence",
                    f"{logistic_result[1]:.2f}%",
                )
            else:
                st.metric("Logistic confidence", "Unavailable")

        with metric_columns[1]:
            if naive_bayes_result is not None:
                st.metric(
                    "Naive Bayes confidence",
                    f"{naive_bayes_result[1]:.2f}%",
                )
            else:
                st.metric("Naive Bayes confidence", "Unavailable")

        with metric_columns[2]:
            st.metric("Characters analyzed", f"{len(job_description):,}")

        st.caption(
            "Each confidence score reflects that            http://localhost:8501 model's prediction for this "
            "text. It is not a real-world probability of job authenticity, "
            "and the scores are not combined."
        )


st.html("""
<footer class="js-footer">
    JOBSHIELD AI · <br>
    Built with Python · Scikit-learn · Streamlit
</footer>
""")
