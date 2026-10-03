NEUROGUARD — RESEARCH + PREDICTION DASHBOARD

This build adds the upgraded prediction workflow:
- Editable detection threshold, merge gap and minimum event duration sliders.
- Real drag-and-drop EDF upload with filename/size display.
- Client + server EDF validation, including extension and channel compatibility.
- Bundled sample EDF button for a zero-file demo.
- Loading UI with pipeline stages: Windowing -> Features -> Prediction -> Events.
- Probability-over-time Chart.js result view with threshold line and shaded event intervals.
- Event table with start, end, duration and confidence.
- Window-level CSV download.
- Clinical/research disclaimer placed directly on the results page.

IMPORTANT MODEL FILES
The web app expects the trained artifacts at:
  models/neuroguard_rf.joblib
  models/neuroguard_scaler.joblib
  models/selected_features.txt

Copy your real NeuroGuard model artifacts into the models/ folder before running inference.
Do NOT replace your trained artifacts with demo/random models.

RUN
1. cd into this folder (or merge these files into your existing seizure detection project).
2. Activate your conda environment.
3. pip install -r requirements.txt
4. python app.py
5. Open http://127.0.0.1:5000

SAMPLE EDF
The included sample/chb03_36_sample.edf is a small 23-channel, 256 Hz, 60-second EDF generated for interface/demo testing. It is NOT a clinical CHB-MIT recording and its synthetic signal should not be presented as real patient data.

If you want the demo button to run your real trained model on the real CHB03_36 recording, replace/use the sample path with your actual EDF locally. The app itself does not claim the bundled synthetic EDF is a real clinical recording.

CSV
The download contains one row per 5-second window: window_start, window_end, probability, prediction.

NOTE
The loading stage animation is a UI progress indicator while the single Flask inference request runs; it does not expose server-side progress percentages.
