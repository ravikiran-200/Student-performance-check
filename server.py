from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from ml_engine import MODEL_NAMES, local_shap_values, prepare_data, run_benchmark


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "indian_student_performance_demo.csv"
SUBJECTS = {
    "applied_data_science_marks": "Applied Data Science",
    "machine_learning_marks": "Machine Learning",
    "data_warehouse_and_data_mining_marks": "Data Warehouse and Data Mining",
    "coding_skills_marks": "Coding Skills",
    "principles_of_big_data_management_marks": "Principles of Big Data Management",
}
AUDIT_COLUMN = "gender"
DATA_NOTE = (
    "Project-generated demonstration data with fictional Indian names. "
    "It contains no real student records."
)
RUNS: OrderedDict[str, dict] = OrderedDict()
MAX_CACHED_RUNS = 6
MODEL_LABELS = list(MODEL_NAMES)

app = Flask(__name__)


def _json_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (float, int, str, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict]:
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _load_frame() -> pd.DataFrame:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {DATA_FILE.name}")
    frame = pd.read_csv(DATA_FILE)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _semester_mapreduce(frame: pd.DataFrame) -> dict:
    """Run a small, local MapReduce-style semester aggregation demonstration."""
    required = {"semester", "overall_average", "attendance_percent"}
    if not required.issubset(frame.columns):
        return {"error": "Semester, average marks, and attendance columns are required."}

    # Mapper: transform each record into a key/value pair. Split into batches to
    # make the map/shuffle/reduce stages explicit; this is not a Hadoop cluster.
    mapped_batches = []
    batch_size = 100
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        mapped_batches.append(
            [
                (
                    int(row.semester),
                    (float(row.overall_average), float(row.attendance_percent),
                     int(float(row.overall_average) < 50)),
                )
                for row in batch.itertuples(index=False)
            ]
        )

    # Shuffle and sort: group mapped values by their semester key.
    shuffled = defaultdict(list)
    for batch in mapped_batches:
        for key, value in batch:
            shuffled[key].append(value)

    # Reducer: combine each semester's values into summary statistics.
    results = []
    for semester in sorted(shuffled):
        values = shuffled[semester]
        count = len(values)
        results.append({
            "semester": semester,
            "students": count,
            "at_risk": sum(value[2] for value in values),
            "at_risk_rate": sum(value[2] for value in values) / count,
            "mean_average": sum(value[0] for value in values) / count,
            "mean_attendance": sum(value[1] for value in values) / count,
        })
    return {
        "records": len(frame),
        "map_batches": len(mapped_batches),
        "batch_size": batch_size,
        "risk_cutoff": 50,
        "semesters": results,
        "execution": "Local sequential MapReduce-style simulation; no Hadoop cluster is running.",
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/subjects")
def list_subjects():
    try:
        frame = _load_frame()
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500
    return jsonify(
        {
            "subjects": [
                {"column": column, "label": label}
                for column, label in SUBJECTS.items()
            ],
            "students": int(len(frame)),
            "note": DATA_NOTE,
        }
    )


@app.get("/api/big-data")
def big_data_summary():
    try:
        frame = _load_frame()
        summary = _semester_mapreduce(frame)
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 500
    return jsonify(summary)


@app.post("/api/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    subject_column = str(body.get("subject_column", ""))
    try:
        cutoff = float(body.get("grade_cutoff", 50))
        threshold = float(body.get("alert_threshold", 0.5))
        if subject_column not in SUBJECTS:
            raise ValueError("Choose one of the five subjects in the list.")
        if not 1 <= cutoff <= 99:
            raise ValueError("The at-risk mark cutoff must be between 1 and 99.")
        if not 0.1 <= threshold <= 0.9:
            raise ValueError("The alert threshold must be between 0.10 and 0.90.")
        frame = _load_frame()
        prepared = prepare_data(frame, subject_column, risk_cutoff=cutoff)
        if AUDIT_COLUMN not in prepared.audit_columns:
            raise ValueError("The dataset needs both gender groups for the fairness check.")
        benchmark = run_benchmark(prepared, AUDIT_COLUMN, threshold=threshold)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.exception("Could not analyze the selected subject")
        return jsonify({"error": f"Analysis failed: {error}"}), 500

    run_id = uuid4().hex
    subject_label = SUBJECTS[subject_column]
    run = {
        "subject_column": subject_column,
        "subject_label": subject_label,
        "prepared": prepared,
        "benchmark": benchmark,
        "cutoff": cutoff,
        "threshold": threshold,
        "risk_students": [],
    }
    RUNS[run_id] = run
    while len(RUNS) > MAX_CACHED_RUNS:
        RUNS.popitem(last=False)

    model_rows = []
    for name in MODEL_LABELS:
        result = benchmark["runs"][name]
        model_rows.append(
            {
                "model": name,
                **{key: _json_value(value) for key, value in result["metrics"].items()},
                **{key: _json_value(value) for key, value in result["fairness"].items()},
                "train_seconds": _json_value(result["fit_seconds"]),
                "prediction_ms_per_student": _json_value(result["latency_ms"]),
            }
        )

    test_indices = benchmark["test_indices"]
    probabilities = benchmark["selected_run"]["probabilities"]
    examples = []
    risk_students = []
    for position, source_index in enumerate(test_indices):
        source_index = int(source_index)
        source_row = prepared.source_rows.iloc[source_index]
        name = str(source_row.get("student_name", f"Test sample {position + 1}"))
        probability = float(probabilities[position])
        examples.append(
            {
                "position": position,
                "label": name,
                "actual_at_risk": int(prepared.target.iloc[source_index]),
                "group": _json_value(benchmark["sensitive_test"].iloc[position]),
                "baseline_probability": _json_value(probability),
            }
        )
        if probability >= threshold:
            risk_students.append(
                {
                    "student_id": _json_value(source_row.get("student_id")),
                    "student_name": name,
                    "gender": _json_value(source_row.get(AUDIT_COLUMN)),
                    "semester": _json_value(source_row.get("semester")),
                    "subject_mark": _json_value(source_row.get(subject_column)),
                    "overall_average": _json_value(source_row.get("overall_average")),
                    "risk_probability": probability,
                }
            )
    risk_students.sort(key=lambda row: row["risk_probability"], reverse=True)
    run["risk_students"] = risk_students

    fairness_aware = benchmark["fair_result"]
    if fairness_aware and "error" in fairness_aware:
        fairness_aware = {"error": fairness_aware["error"]}
    elif fairness_aware:
        fairness_aware = {
            "label": fairness_aware["label"],
            "metrics": {
                key: _json_value(value)
                for key, value in fairness_aware["metrics"].items()
            },
            "fairness": {
                key: _json_value(value)
                for key, value in fairness_aware["fairness"].items()
            },
            "group_metrics": _records(fairness_aware["group_metrics"]),
        }
        model_rows.append(
            {
                "model": "Random Forest · equal-opportunity adjustment",
                **fairness_aware["metrics"],
                **fairness_aware["fairness"],
                "train_seconds": None,
                "prediction_ms_per_student": None,
            }
        )

    return jsonify(
        {
            "run_id": run_id,
            "subject": {"column": subject_column, "label": subject_label},
            "data_note": DATA_NOTE,
            "summary": {
                "students": int(len(prepared.target)),
                "at_risk_rate": _json_value(prepared.positive_rate),
                "observed_at_risk_count": int(prepared.target.sum()),
                "predictors": int(prepared.features.shape[1]),
                "target": prepared.target_label,
                "audit_column": AUDIT_COLUMN,
                "grade_cutoff": cutoff,
                "alert_threshold": threshold,
            },
            "models": model_rows,
            "fairness_aware": fairness_aware,
            "baseline_group_metrics": _records(
                benchmark["selected_run"]["group_metrics"]
            ),
            "risk_students": risk_students,
            "examples": examples,
        }
    )


@app.post("/api/explain")
def explain():
    body = request.get_json(silent=True) or {}
    run_id = str(body.get("run_id", ""))
    model_name = str(body.get("model", "Random Forest"))
    try:
        position = int(body.get("sample_position", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Choose a valid student example."}), 400
    run = RUNS.get(run_id)
    if run is None:
        return jsonify({"error": "This analysis has expired. Run the comparison again."}), 410
    if model_name not in MODEL_LABELS:
        return jsonify({"error": "Choose one of the available models."}), 400
    benchmark = run["benchmark"]
    test_indices = benchmark["test_indices"]
    if position < 0 or position >= len(test_indices):
        return jsonify({"error": "Choose a valid student example."}), 400

    source_index = int(test_indices[position])
    prepared = run["prepared"]
    model_result = benchmark["runs"][model_name]
    student = prepared.features.iloc[[source_index]]
    source_row = prepared.source_rows.iloc[source_index]
    try:
        contributions = local_shap_values(
            model_result["model"], benchmark["x_train"], student
        ).head(12)
        probability = float(model_result["model"].predict_proba(student)[0, 1])
    except Exception as error:
        app.logger.exception("Could not create the SHAP explanation")
        return jsonify({"error": f"Could not explain this prediction: {error}"}), 500

    return jsonify(
        {
            "student_name": _json_value(source_row.get("student_name")),
            "gender": _json_value(source_row.get(AUDIT_COLUMN)),
            "actual_score": _json_value(source_row.get(run["subject_column"])),
            "overall_average": _json_value(source_row.get("overall_average")),
            "risk_probability": probability,
            "alert": bool(probability >= run["threshold"]),
            "threshold": run["threshold"],
            "profile": {
                str(key): _json_value(value)
                for key, value in student.iloc[0].items()
            },
            "contributions": _records(contributions),
        }
    )


@app.get("/api/results/<run_id>.csv")
def download_results(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        return jsonify({"error": "This analysis has expired. Run the comparison again."}), 410
    benchmark = run["benchmark"]
    output = StringIO()
    fields = [
        "Subject", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC",
        "Balanced accuracy", "Demographic parity difference",
        "Equal opportunity difference", "Training seconds",
        "Prediction milliseconds per student",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for name in MODEL_LABELS:
        result = benchmark["runs"][name]
        writer.writerow(
            {
                "Subject": run["subject_label"],
                "Model": name,
                "Accuracy": result["metrics"]["Accuracy"],
                "Precision": result["metrics"]["Precision"],
                "Recall": result["metrics"]["Recall"],
                "F1": result["metrics"]["F1"],
                "ROC AUC": result["metrics"]["ROC AUC"],
                "Balanced accuracy": result["metrics"]["Balanced accuracy"],
                "Demographic parity difference": result["fairness"]["demographic_parity_difference"],
                "Equal opportunity difference": result["fairness"]["equal_opportunity_difference"],
                "Training seconds": result["fit_seconds"],
                "Prediction milliseconds per student": result["latency_ms"],
            }
        )
    fair_result = benchmark["fair_result"]
    if fair_result and "error" not in fair_result:
        writer.writerow(
            {
                "Subject": run["subject_label"],
                "Model": "Random Forest · equal-opportunity adjustment",
                "Accuracy": fair_result["metrics"]["Accuracy"],
                "Precision": fair_result["metrics"]["Precision"],
                "Recall": fair_result["metrics"]["Recall"],
                "F1": fair_result["metrics"]["F1"],
                "ROC AUC": fair_result["metrics"]["ROC AUC"],
                "Balanced accuracy": fair_result["metrics"]["Balanced accuracy"],
                "Demographic parity difference": fair_result["fairness"]["demographic_parity_difference"],
                "Equal opportunity difference": fair_result["fairness"]["equal_opportunity_difference"],
                "Training seconds": "",
                "Prediction milliseconds per student": "",
            }
        )
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=subject-model-results.csv"},
    )


@app.get("/api/at-risk/<run_id>.csv")
def download_risk_students(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        return jsonify({"error": "This analysis has expired. Run the comparison again."}), 410
    output = StringIO()
    fields = [
        "Student ID", "Student name", "Gender", "Semester", "Subject",
        "Subject mark out of 100", "Overall average", "Predicted risk probability",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for student in run["risk_students"]:
        writer.writerow(
            {
                "Student ID": student["student_id"],
                "Student name": student["student_name"],
                "Gender": student["gender"],
                "Semester": student["semester"],
                "Subject": run["subject_label"],
                "Subject mark out of 100": student["subject_mark"],
                "Overall average": student["overall_average"],
                "Predicted risk probability": student["risk_probability"],
            }
        )
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=students-for-review.csv"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    print(f"Student Support Dashboard is running on port {port}")
    app.run(host=host, port=port, debug=False)
