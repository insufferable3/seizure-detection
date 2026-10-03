from pathlib import Path
import joblib,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data/processed"; MODELS=ROOT/"models"; MODELS.mkdir(exist_ok=True)
df=pd.read_csv(DATA/"features.csv"); selected=pd.read_csv(DATA/"selected_features.csv")["feature"].tolist()
X=df[selected].replace([float("inf"),-float("inf")],pd.NA).fillna(0); y=df.label.astype(int)
scaler=StandardScaler(); Xs=scaler.fit_transform(X)
model=RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1); model.fit(Xs,y)
joblib.dump(model,MODELS/"neuroguard_rf.joblib"); joblib.dump(scaler,MODELS/"neuroguard_scaler.joblib"); (MODELS/"selected_features.txt").write_text("\n".join(selected))
print("Live inference model artifacts saved.")
