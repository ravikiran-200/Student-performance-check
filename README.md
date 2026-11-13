# Student Support Dashboard

A simple local website for exploring student performance by subject. It uses Flask, HTML, CSS, and JavaScript. It does not use Streamlit.

## Run it in VS Code

1. Open the `student-performance-web` folder in VS Code.
2. Open **Terminal → New Terminal**. In PowerShell, create the project's Python environment and install its packages:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Start the website:

```powershell
.\.venv\Scripts\python.exe server.py
```

4. Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Keep the terminal open while using the site. Press **Ctrl+C** to stop it. If Windows cannot find the Python path in the first command, replace it with the location of your installed Python 3.14 `python.exe`.

## Deploy on Render

The repository includes `render.yaml`, `.python-version`, and Gunicorn configuration for Render. Create a private GitHub repository for this project and push the project files (do not include `.venv`). In Render, choose **New → Blueprint**, connect that repository, and deploy the `render.yaml` Blueprint. Render will install `requirements.txt` and start `gunicorn server:app`. The app uses Render's assigned port and binds to `0.0.0.0` in production. The free service may spin down when idle.

## What it does

- Choose Applied Data Science, Machine Learning, Data Warehouse and Data Mining, Coding Skills, or Principles of Big Data Management.
- Set the mark cutoff. By default, a mark below 50 out of 100 is labelled at risk.
- Compare Logistic Regression, Decision Tree, and Random Forest.
- Compare accuracy, precision, recall, F1, ROC AUC, and fairness gaps by gender.
- Search the names of students the Random Forest flags in the held-out test group.
- View the actual mark for the selected subject and download the flagged list as a CSV.
- Select a test student and see a SHAP explanation of one model prediction.
- Show a MapReduce-style semester aggregation with mapper batches, a shuffle/group step, and reducer statistics for risk rate, mean marks, and attendance.
- Connect the implementation to course topics in Big Data, Hadoop, HiveQL, and analytics with R.

## Big data course concepts applied

The `/api/big-data` endpoint demonstrates a batch processing workflow over the project CSV. The mapper emits `(semester, values)` pairs in batches of 100 records; the shuffle groups values by semester; and the reducer calculates student counts, counts and rates below a 50 mark average, mean overall average, and mean attendance. The dashboard displays the results.

This is an educational MapReduce-style simulation implemented in local Python. It does not start Hadoop, use HDFS, or execute work in parallel. The dataset has only 600 rows, so it demonstrates the processing pattern rather than a big data scale workload. The Hive panel shows an equivalent HiveQL aggregation. R is a course connection for exploratory analysis and graphs; this application runs its analysis in Python.

| Course area | Project connection |
| --- | --- |
| Big Data concepts, types, and storage | Structured CSV data; local storage and a note about limits of this small dataset |
| Parallel and distributed processing | Map, shuffle, and reduce stages are shown, with the local sequential simulation clearly identified |
| Hadoop / HDFS | Described as a possible deployment path; not installed or claimed as active |
| NoSQL and CAP theorem | Dashboard explains the consistency/availability tradeoff during a network partition and identifies this app as a single-machine CSV design |
| HDFS architecture and replication | Dashboard explains NameNode metadata, DataNode blocks, and replicated copies as a scale-out design concept |
| Hive / HiveQL | Equivalent `GROUP BY semester` query shown in the dashboard |
| Data analytics using R | Existing statistical summaries, model metrics, fairness comparison, and explanations; R is noted as an alternative tool |

The supplied `risk_by_semester.txt` is not used as a source of results; the displayed semester table is computed from the CSV at request time.

## Project dataset

`data/indian_student_performance_demo.csv` is a newly generated, fictional dataset with 600 records. It has Indian-style fictional names, semester, attendance, study hours, assignment and lab completion, participation, marks for all five subjects, and an overall average. It contains no real school records and should not be described as real data from India.

`generate_dataset.py` recreates the synthetic CSV using Python's standard library.

For each analysis, the selected subject's mark is the prediction target. Names, gender, subject marks, and overall average are excluded from model inputs to avoid identity leakage and to avoid using final marks to predict themselves. Gender is kept separately for the fairness check. The at-risk names shown are from the 25% held-out test group whose Random Forest risk estimate meets the chosen alert threshold; the list is not a prediction for every student in the dataset.

## Notes

The dataset uses made-up values and a fixed split to demonstrate the workflow. Its model scores do not show how the system would perform at a real Indian college. A prediction is only a prompt to review a student's situation; it should not decide academic support on its own.

The server listens only on `127.0.0.1` and is for local use. Put it online only after preparing a production setup, reviewing privacy and consent, and choosing a host. Do not upload real student records or secrets to GitHub.
