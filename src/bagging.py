import numpy as np
import joblib
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocess import (load_all_subjects_cached,
                        balance_smote,
                        extract_features,
                        create_windows)

from sklearn.ensemble import (BaggingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder,
                                   label_binarize)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold,
                                     cross_val_score,
                                     train_test_split)
from sklearn.metrics import (classification_report,
                             roc_auc_score,
                             roc_curve,
                             auc)


def train_bagging(X, y):
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=15,
                class_weight='balanced'),
            n_estimators=100,
            random_state=42,
            n_jobs=-1))
    ])

    print("Training Bagging classifier...")
    model.fit(X, y_encoded)

    train_acc = model.score(X, y_encoded)
    print(f"Train accuracy: {train_acc*100:.2f}%")

    return model, le


def evaluate_cv(model, X, y, le):
    y_encoded = le.transform(y)

    cv = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y_encoded,
        cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"\nCross-Validation Results (10-fold):")
    print(f"  Per fold: {[f'{s*100:.1f}%' for s in scores]}")
    print(f"  Mean:     {scores.mean()*100:.2f}%")
    print(f"  Std:      ±{scores.std()*100:.2f}%")

    return scores.mean(), scores.std()


def evaluate_test(model, X, y, le):
    y_encoded = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2,
        stratify=y_encoded, random_state=42)

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)

    print(f"\nTrain/Test Evaluation:")
    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test  accuracy: {test_acc*100:.2f}%")

    y_pred = model.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=3))

    return X_test, y_test


def compute_auc_roc(model, le, X_test, y_test):
    n_classes = len(le.classes_)
    y_binary  = label_binarize(
        y_test, classes=range(n_classes))

    y_prob = model.predict_proba(X_test)

    auc_scores = {}

    print(f"\n{'═'*50}")
    print(f"  AUC-ROC Analysis — Bagging + SMOTE")
    print(f"{'═'*50}")
    print(f"\n  Per-class AUC-ROC:")
    print(f"  {'─'*38}")
    print(f"  {'Emotion':<12} {'AUC':>8}  {'Bar'}")
    print(f"  {'─'*38}")

    for i, emotion in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(
            y_binary[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        auc_scores[emotion] = class_auc

        bar   = '█' * int(class_auc * 20)
        empty = '░' * (20 - int(class_auc * 20))

        if class_auc >= 0.90:
            quality = "🟢"
        elif class_auc >= 0.80:
            quality = "🟡"
        elif class_auc >= 0.70:
            quality = "🟠"
        else:
            quality = "🔴"

        print(f"  {emotion:<12} {class_auc:>8.4f}  "
              f"{bar}{empty} {quality}")

    macro_auc = roc_auc_score(
        y_binary, y_prob,
        multi_class='ovr', average='macro')
    weighted_auc = roc_auc_score(
        y_binary, y_prob,
        multi_class='ovr', average='weighted')

    auc_scores['macro_avg']    = macro_auc
    auc_scores['weighted_avg'] = weighted_auc

    print(f"  {'─'*38}")
    print(f"  {'Macro Avg':<12} {macro_auc:>8.4f}")
    print(f"  {'Weighted':<12} {weighted_auc:>8.4f}")

    if macro_auc >= 0.90:
        grade = "🟢 Excellent"
    elif macro_auc >= 0.80:
        grade = "🟡 Good"
    elif macro_auc >= 0.70:
        grade = "🟠 Fair"
    else:
        grade = "🔴 Poor"

    print(f"\n  Grade: {grade}")
    print(f"{'═'*50}")

    return auc_scores


def save_model(model, le,
               model_path='models/emotion_model.pkl',
               le_path='models/label_encoder.pkl'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    print(f"\n  Model saved   → {model_path}")
    print(f"  Encoder saved → {le_path}")


def load_model(
        model_path='models/emotion_model.pkl',
        le_path='models/label_encoder.pkl'):
    model = joblib.load(model_path)
    le    = joblib.load(le_path)
    return model, le


def predict_emotion(model, le, features):
    features   = np.array(features).reshape(1, -1)
    pred       = model.predict(features)
    proba      = model.predict_proba(features)[0]
    emotion    = le.inverse_transform(pred)[0]
    confidence = round(float(proba.max()) * 100, 1)
    return emotion, confidence


if __name__ == "__main__":
    print("=" * 55)
    print("  ThetaPlay — Bagging + SMOTE")
    print("=" * 55)

    print("\nStep 1: Loading data...")
    X_raw, y_raw = load_all_subjects_cached()

    print("\nStep 2: Applying SMOTE...")
    X_bal, y_bal = balance_smote(X_raw, y_raw)

    print("\nStep 3: Training Bagging...")
    model, le = train_bagging(X_bal, y_bal)

    print("\nStep 4: Cross validation...")
    mean_acc, std_acc = evaluate_cv(model, X_bal, y_bal, le)

    print("\nStep 5: Test set evaluation...")
    X_test, y_test = evaluate_test(model, X_bal, y_bal, le)

    print("\nStep 6: AUC-ROC...")
    auc_scores = compute_auc_roc(model, le, X_test, y_test)

    print("\nStep 7: Saving model...")
    save_model(model, le)

    print("\nStep 8: Testing predictions...")
    print(f"  {'─'*45}")
    for i, emotion_name in enumerate(le.classes_):
        idx = np.where(le.transform(y_bal) == i)[0][0]
        pred, conf = predict_emotion(model, le, X_bal[idx])
        status = "✅" if pred == emotion_name else "❌"
        print(f"  {status} True: {emotion_name:10s} → "
              f"Predicted: {pred:10s} ({conf}%)")

    print(f"\n{'='*55}")
    print(f"  ✅ Bagging + SMOTE Complete!")
    print(f"{'─'*55}")
    print(f"  Balancing:      SMOTE")
    print(f"  Classifier:     Bagging")
    print(f"  CV Accuracy:    {mean_acc*100:.2f}% "
          f"(±{std_acc*100:.2f}%)")
    print(f"  Macro AUC-ROC:  "
          f"{auc_scores['macro_avg']:.4f}")
    print(f"  Weighted AUC:   "
          f"{auc_scores['weighted_avg']:.4f}")
    print(f"{'─'*55}")
    print(f"  Per-emotion AUC:")
    for emotion in le.classes_:
        print(f"    {emotion:<12}: "
              f"{auc_scores[emotion]:.4f}")
    print(f"{'='*55}")