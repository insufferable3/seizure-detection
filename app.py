from pathlib import Path
import csv
import io
import json
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from inference.predictor import predict_edf, validate_edf

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_EDF = BASE_DIR / "sample" / "chb03_36_sample.edf"
CACHE_DIR = BASE_DIR / ".result_cache"
CACHE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


def _save_result(result):
    job_id = result["job_id"]
    (CACHE_DIR / f"{job_id}.json").write_text(json.dumps(result))
    return job_id


def _load_result(job_id):
    path = CACHE_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@app.route("/api/validate", methods=["POST"])
def api_validate():
    uploaded = request.files.get("edf_file")
    if not uploaded or not uploaded.filename:
        return jsonify(ok=False, error="Please select an EDF file."), 400
    if not uploaded.filename.lower().endswith(".edf"):
        return jsonify(ok=False, error="Only .edf files are supported."), 400
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(uploaded.filename).name}"
    try:
        uploaded.save(path)
        info = validate_edf(path)
        return jsonify(ok=True, filename=Path(uploaded.filename).name, size_bytes=path.stat().st_size, **info)
    except Exception as exc:
        return jsonify(ok=False, error=f"{type(exc).__name__}: {exc}"), 400
    finally:
        path.unlink(missing_ok=True)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    uploaded = request.files.get("edf_file")
    use_sample = request.form.get("use_sample") == "1"
    threshold = request.form.get("threshold", "0.50")
    merge_gap = request.form.get("merge_gap", "30")
    min_duration = request.form.get("min_duration", "10")

    if use_sample:
        if not SAMPLE_EDF.exists():
            return jsonify(ok=False, error="Sample EDF is not bundled. Add sample/chb03_36_sample.edf."), 400
        path = SAMPLE_EDF
        cleanup = False
    else:
        if not uploaded or not uploaded.filename:
            return jsonify(ok=False, error="Please select an EDF file."), 400
        if not uploaded.filename.lower().endswith(".edf"):
            return jsonify(ok=False, error="Only .edf files are supported."), 400
        path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(uploaded.filename).name}"
        uploaded.save(path)
        cleanup = True

    try:
        result = predict_edf(path, float(threshold), float(merge_gap), float(min_duration))
        result["source"] = "sample" if use_sample else "upload"
        _save_result(result)
        return jsonify(ok=True, result=result)
    except Exception as exc:
        return jsonify(ok=False, error=f"{type(exc).__name__}: {exc}"), 400
    finally:
        if cleanup:
            path.unlink(missing_ok=True)


@app.route("/result/<job_id>")
def result(job_id):
    data = _load_result(job_id)
    if not data:
        return render_template("predict.html", error="This result has expired or could not be found."), 404
    return render_template("result.html", result=data)


@app.route("/download/<job_id>.csv")
def download_csv(job_id):
    result = _load_result(job_id)
    if not result:
        return "Result not found", 404
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["window_start", "window_end", "probability", "prediction"])
    for row in result["windows"]:
        writer.writerow([row["start"], row["end"], row["probability"], row["prediction"]])
    return send_file(io.BytesIO(out.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name=f"{Path(result['filename']).stem}_neuroguard_predictions.csv")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
