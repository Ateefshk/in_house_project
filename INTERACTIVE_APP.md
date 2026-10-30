# Interactive console predictor

From the project root, install the console dependencies and start the predictor:

```powershell
python -m pip install -r requirements-console.txt
python main.py
```

`python main.py predict` does the same thing. The program loads the included models and course/presentation lookup, then asks for a hypothetical learner profile using activity from the first 28 course days. Press Enter to accept the displayed defaults, enter `y` for another prediction, or press Ctrl+C to exit. It does not open a browser or save entered profiles.

To run the separate full model-training pipeline, first download the OULAD CSV files into `data/original/`, install full dependencies with `python -m pip install -r requirements.txt`, and run `python main.py train`.

The models use OULAD data from one UK university. Treat predictions as a research demonstration for hypothetical profiles.

`app.py` remains an optional Streamlit dashboard; console mode does not need Streamlit.
