# JobShield AI

JobShield AI is a Streamlit application for screening job postings for potential
fraud signals. It uses Logistic Regression as the primary model and Naive Bayes V2
as a secondary comparison model.

## Run locally

```text
python -m pip install -r requirements.txt
streamlit run app.py
```

The application requires these model artifacts in the project root:

- `fake_job_model.pkl`
- `tfidf_vectorizer.pkl`
- `fake_job_nb_model_v2.pkl`
- `fake_job_nb_tfidf_vectorizer_v2.pkl`

The app preserves independent model predictions and confidence scores. If the
models disagree, it displays a verification warning rather than classifying the
posting as legitimate.

## Streamlit Community Cloud

Create a Streamlit Community Cloud app from this repository and select `app.py`
as the main file. Keep the four model artifacts and `requirements.txt` in the
repository root. Do not upload the local virtual environment, training dataset,
or notebook.
