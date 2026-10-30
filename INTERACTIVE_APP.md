# Interactive console predictor

Run the model directly in a terminal; this mode does not start a web server or open a browser. From the project root:

```powershell
python -m pip install -r requirements-console.txt
python console_app.py
```

The predictor loads the included trained models and course/presentation lookup. It asks for a hypothetical learner profile using activity from the first 28 course days, then prints the four-category probabilities and the binary risk probability. Press Enter to use the displayed defaults; enter `y` to score another profile. Press Ctrl+C to exit. Profiles entered at the prompt are not saved.

The models were trained on OULAD data from one UK university. Treat results as a research demonstration for hypothetical profiles, not decisions about real students.

## Optional browser dashboard

`app.py` remains available if a browser dashboard is wanted. Install the full dependencies from `requirements.txt` and run `python -m streamlit run app.py`. The console predictor does not need Streamlit.
