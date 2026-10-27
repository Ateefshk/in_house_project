# Interactive dashboard

The project includes a local Streamlit dashboard. From the project root, install dependencies and launch it in VS Code's terminal:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit opens the dashboard in your browser. If the browser does not open automatically, visit the `Local URL` printed in the terminal. The main modeling pipeline must have been run first so the dashboard can load the saved models, metrics, and figures:

```powershell
python main.py
```

The dashboard has four sections: a hypothetical-profile prediction form, test-set model results and threshold trade-offs, cohort/figure exploration, and project notes. It uses the saved expanded-feature models trained on the first 28 course days. It does not save profiles entered in the app. Predictions are for research demonstration and should not be used as decisions about real students.
