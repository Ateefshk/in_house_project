# Data provenance and placement

The repository excludes raw student-level OULAD tables and row-level derivatives. Download the official archive from the [UCI OULAD page](https://archive.ics.uci.edu/dataset/349/open%2Buniversity%2B), extract its CSV files into `data/original/`, then run `python main.py` to reproduce the analysis. The dashboard uses included fitted models, aggregate results, and `course_presentations.csv`; it does not require raw tables.

OULAD is distributed under CC BY 4.0. Cite Kuzilek, Hlosta, and Zdrahal, *Open University Learning Analytics dataset*, Scientific Data 4, 170171 (2017), https://doi.org/10.1038/sdata.2017.171, and UCI dataset DOI 10.24432/C5KK69. Keep attribution with redistributed data or derived files.

Preserve official file names and columns. `src/inspect_data.py` validates required schemas. Do not commit student-level data, split assignments, or row-level predictions.
