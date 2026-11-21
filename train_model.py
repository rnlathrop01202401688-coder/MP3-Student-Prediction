from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'augmented_balanced_dataset_m.csv')
features = [c for c in df.columns if c != 'rating']
X, y = df[features], df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {'accuracy':float(accuracy_score(y_test,pred)), 'training_rows':len(X_train), 'test_rows':len(X_test), 'classification_report':classification_report(y_test,pred,output_dict=True), 'confusion_matrix':confusion_matrix(y_test,pred,labels=['High','Low']).tolist(), 'feature_importances':dict(zip(features,map(float,model.feature_importances_)))}
joblib.dump({'model':model,'features':features},ROOT/'model.joblib')
(ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(f"Test accuracy: {metrics['accuracy']:.2%}")
