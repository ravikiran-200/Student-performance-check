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

## Student demo login

Sign in with the student ID as the username and the matching name as the password. The dataset includes named classroom demo profiles, so this login is only an educational access demo; it is not secure authentication. The listed marks for these profiles are illustrative values and are not verified assessment results.

| Username | Password |
| --- | --- |
| `STU0601` | `RAJU` |
| `STU0602` | `RAVI KIRAN` |
| `STU0603` | `TANAMAI` |
| `STU0604` | `BALA BHASKAR` |
| `STU0605` | `ADITYA` |

Render generates and stores a secret key for signed login sessions through the Blueprint configuration.

## Admin dashboard

The administrator signs in with the configured admin username and password, sees the class overview, searches the demo roster, and can open an individual student's tracking view. Use **Admin overview** to return to the class list. Student accounts remain restricted to their own record.

For security, the admin password is not stored in this repository. Set `ADMIN_PASSWORD` as a secret environment variable in the Render service settings before using admin sign-in. The Blueprint sets `ADMIN_USERNAME` to `Imandi Raju` and marks `ADMIN_PASSWORD` as a value to supply privately. For local use, set `ADMIN_PASSWORD` in the shell before starting Flask. Do not put the password in this file, source code, or GitHub.

## Student dashboard

After signing in, a student sees only the record linked to their own student ID: their subject marks, average, semester, attendance, study hours, assignment and lab completion, and participation score. The app does not show classmates, class rankings, cohort summaries, or other students' at-risk lists in a student session. The server blocks the class-wide analysis and download endpoints for student accounts as well.

The CSV has one current row per student, not dated snapshots, so the dashboard displays a current profile rather than a historical trend chart. The five named classroom profiles use illustrative marks and are not verified assessment results.

## Big data course concepts applied

The `/api/big-data` endpoint demonstrates a batch processing workflow over the project CSV. The mapper emits `(semester, values)` pairs in batches of 100 records; the shuffle groups values by semester; and the reducer calculates student counts, counts and rates below a 50 mark average, mean overall average, and mean attendance. The dashboard displays the results.

This is an educational MapReduce-style simulation implemented in local Python. It does not start Hadoop, use HDFS, or execute work in parallel. The dataset has only 605 rows, so it demonstrates the processing pattern rather than a big data scale workload. A HiveQL `GROUP BY` query could express a similar aggregation, but the app does not execute HiveQL. R is a course connection for exploratory analysis and graphs; this application runs its analysis in Python.

An equivalent HiveQL-style semester summary would be:

```sql
SELECT semester, COUNT(*) AS students,
       AVG(overall_average) AS mean_mark
FROM student_performance
GROUP BY semester;
```

| Course area | Project connection |
| --- | --- |
| Big Data concepts, types, and storage | Structured CSV data; local storage and a note about limits of this small dataset |
| Parallel and distributed processing | Map, shuffle, and reduce stages are shown, with the local sequential simulation clearly identified |
| Hadoop / HDFS | Described as a possible deployment path; not installed or claimed as active |
| NoSQL and CAP theorem | README explains the consistency/availability tradeoff during a network partition and identifies this app as a single-machine CSV design |
| HDFS architecture and replication | README explains NameNode metadata, DataNode blocks, and replicated copies as a scale-out design concept |
| Hive / HiveQL | README includes an equivalent `GROUP BY semester` query; the app does not run Hive |
| Data analytics using R | Existing statistical summaries, model metrics, fairness comparison, and explanations; R is noted as an alternative tool |

The supplied `risk_by_semester.txt` is not used as a source of results; the displayed semester table is computed from the CSV at request time.

## Project dataset

`data/indian_student_performance_demo.csv` contains 600 generated demo profiles and five named classroom demo profiles. The other named profiles have illustrative scores because no assessed marks were supplied. Do not treat these values as official results or use them for academic decisions.

`generate_dataset.py` recreates the synthetic CSV using Python's standard library.

For each analysis, the selected subject's mark is the prediction target. Names, gender, subject marks, and overall average are excluded from model inputs to avoid identity leakage and to avoid using final marks to predict themselves. Gender is kept separately for the fairness check. The at-risk names shown are from the 25% held-out test group whose Random Forest risk estimate meets the chosen alert threshold; the list is not a prediction for every student in the dataset.

## Notes

The generated dataset uses illustrative values and a fixed split to demonstrate the workflow. Its model scores do not show how the system would perform at a real college. A prediction is only a prompt to review a student's situation; it should not decide academic support on its own.

The server listens only on `127.0.0.1` and is for local use. Put it online only after preparing a production setup, reviewing privacy and consent, and choosing a host. Do not upload real student records or secrets to GitHub.
