import os
import warnings, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HIST = True
except ImportError:
    HAS_HIST = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings('ignore')
np.random.seed(42)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

CSV_PATH = "df.csv"
SEP  = "═" * 68
SEP2 = "─" * 68


def load_data(path=CSV_PATH):
 
    print("  STEP 1/4 — Loading Dataset")


    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df['Count'] = pd.to_numeric(df['Count'], errors='coerce').fillna(0)

    print(f"  ✓ Rows loaded        : {len(df):,}")
    print(f"  ✓ Cities             : {df['Million Plus Cities'].nunique()}")
    print(f"  ✓ Cause categories   : {df['Cause category'].nunique()} → {list(df['Cause category'].unique())}")
    print(f"  ✓ Subcategories      : {df['Cause Subcategory'].nunique()}")
    print(f"  ✓ Outcomes tracked   : {list(df['Outcome of Incident'].unique())}")
    print(f"  ✓ Total count range  : {df['Count'].min():.0f} – {df['Count'].max():.0f}")

    return df

def explore(df):

    print("  STEP 2/4 — Exploratory Analysis")


    # ── Top 10 cities by total accidents ──────────────────────
    accidents = df[df['Outcome of Incident'] == 'Total number of Accidents']
    city_totals = accidents.groupby('Million Plus Cities')['Count'].sum().sort_values(ascending=False)

    print(f"\n  ── Top 10 Cities by Total Accidents {'─'*29}")
    max_count = city_totals.max()
    for i, (city, cnt) in enumerate(city_totals.head(10).items(), 1):
        bar = "█" * int(cnt / max_count * 35)
        print(f"  {i:>2}. {city:<25} {bar:<35} {cnt:,.0f}")

    # ── Deaths by cause category ───────────────────────────────
    killed = df[df['Outcome of Incident'] == 'Persons Killed']
    cause_deaths = killed.groupby('Cause category')['Count'].sum().sort_values(ascending=False)

    print(f"\n  ── Deaths by Cause Category {'─'*38}")
    max_d = cause_deaths.max()
    for cause, cnt in cause_deaths.items():
        bar = "█" * int(cnt / max_d * 35)
        print(f"  {cause:<30} {bar:<35} {cnt:,.0f}")

    # ── Top 10 deadliest subcategories ────────────────────────
    sub_deaths = killed.groupby('Cause Subcategory')['Count'].sum().sort_values(ascending=False)
    print(f"\n  ── Top 10 Deadliest Subcategories {'─'*32}")
    max_s = sub_deaths.max()
    for i, (sub, cnt) in enumerate(sub_deaths.head(10).items(), 1):
        bar = "█" * int(cnt / max_s * 30)
        print(f"  {i:>2}. {sub:<45} {bar:<30} {cnt:,.0f}")

    # ── Injury severity ratio ──────────────────────────────────
    print(f"\n  ── Injury Severity Overview {'─'*38}")
    for outcome in ['Total number of Accidents', 'Persons Killed',
                    'Total Injured', 'Greviously Injured', 'Minor Injury']:
        total = df[df['Outcome of Incident'] == outcome]['Count'].sum()
        print(f"  {outcome:<35} : {total:>10,.0f}")

    # ── Fatality rate by cause ─────────────────────────────────
    print(f"\n  ── Fatality Rate per Accident by Cause {'─'*26}")
    for cause in df['Cause category'].unique():
        acc = df[(df['Cause category'] == cause) &
                 (df['Outcome of Incident'] == 'Total number of Accidents')]['Count'].sum()
        dth = df[(df['Cause category'] == cause) &
                 (df['Outcome of Incident'] == 'Persons Killed')]['Count'].sum()
        rate = (dth / acc * 100) if acc > 0 else 0
        bar = "█" * int(rate / 5)
        print(f"  {cause:<30} {bar:<20} {rate:.1f}% fatality rate")

    return city_totals, cause_deaths


def build_features(df):
    """
    Target: HIGH RISK = city–cause combo where fatality rate > median
    Features: encoded cause, subcategory, city + derived count features
    """
    # Pivot to get per-combo statistics
    accidents = df[df['Outcome of Incident'] == 'Total number of Accidents'][
        ['Million Plus Cities', 'Cause category', 'Cause Subcategory', 'Count']
    ].rename(columns={'Count': 'accidents'})

    killed = df[df['Outcome of Incident'] == 'Persons Killed'][
        ['Million Plus Cities', 'Cause category', 'Cause Subcategory', 'Count']
    ].rename(columns={'Count': 'killed'})

    injured = df[df['Outcome of Incident'] == 'Total Injured'][
        ['Million Plus Cities', 'Cause category', 'Cause Subcategory', 'Count']
    ].rename(columns={'Count': 'injured'})

    grievous = df[df['Outcome of Incident'] == 'Greviously Injured'][
        ['Million Plus Cities', 'Cause category', 'Cause Subcategory', 'Count']
    ].rename(columns={'Count': 'grievous'})

    keys = ['Million Plus Cities', 'Cause category', 'Cause Subcategory']
    merged = accidents \
        .merge(killed,   on=keys, how='left') \
        .merge(injured,  on=keys, how='left') \
        .merge(grievous, on=keys, how='left')

    merged = merged.fillna(0)
    merged['fatality_rate']  = merged.apply(
        lambda r: r['killed'] / r['accidents'] if r['accidents'] > 0 else 0, axis=1)
    merged['injury_rate']    = merged.apply(
        lambda r: r['injured'] / r['accidents'] if r['accidents'] > 0 else 0, axis=1)
    merged['grievous_rate']  = merged.apply(
        lambda r: r['grievous'] / r['accidents'] if r['accidents'] > 0 else 0, axis=1)
    merged['severity_score'] = (
        merged['fatality_rate'] * 3 +
        merged['grievous_rate'] * 2 +
        merged['injury_rate']
    )

    # ── Target: HIGH RISK if fatality_rate > median ────────────
    threshold = merged[merged['accidents'] > 0]['fatality_rate'].median()
    merged['high_risk'] = ((merged['fatality_rate'] > threshold) &
                           (merged['accidents'] > 0)).astype(int)

    print(f"\n  ✓ Feature matrix     : {len(merged):,} city–cause combinations")
    print(f"  ✓ Fatality threshold : {threshold:.3f} ({threshold*100:.1f}%)")
    print(f"  ✓ High-risk combos   : {merged['high_risk'].sum():,} "
          f"({merged['high_risk'].mean()*100:.1f}% of total)")

    return merged, threshold


def preprocess(merged):
    proc = merged.copy()
    label_encoders = {}
    for col in ['Million Plus Cities', 'Cause category', 'Cause Subcategory']:
        le = LabelEncoder()
        proc[col + '_enc'] = le.fit_transform(proc[col].astype(str))
        label_encoders[col] = le

    feature_cols = [
        'Million Plus Cities_enc',
        'Cause category_enc',
        'Cause Subcategory_enc',
        'accidents',
        'killed',
        'injured',
        'grievous',
        'injury_rate',
        'grievous_rate',
        'severity_score',
    ]

    num_scale = ['accidents', 'killed', 'injured', 'grievous',
                 'injury_rate', 'grievous_rate', 'severity_score']
    scaler = StandardScaler()
    proc[num_scale] = scaler.fit_transform(proc[num_scale])

    X = proc[feature_cols]
    y = proc['high_risk']
    return X, y, label_encoders, scaler, feature_cols, num_scale


def get_subcategories_by_category(df):
    mapping = {}
    grouped = df.groupby('Cause category')['Cause Subcategory']
    for category, subcategories in grouped:
        mapping[category] = sorted(subcategories.dropna().astype(str).unique().tolist())
    return mapping


def train_prediction_bundle(path=CSV_PATH):
    df = load_data(path)
    merged, threshold = build_features(df)
    X, y, label_encoders, scaler, feature_cols, num_scale = preprocess(merged)
    model, model_name = train_models(X, y)

    return {
        'dataframe': df,
        'merged_df': merged,
        'threshold': threshold,
        'model': model,
        'model_name': model_name,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'num_scale': num_scale,
        'category_to_subcategories': get_subcategories_by_category(df),
    }


def predict_risk_score(bundle, city, cause_cat, cause_sub):
    label_encoders = bundle['label_encoders']
    merged_df = bundle['merged_df']
    scaler = bundle['scaler']
    feature_cols = bundle['feature_cols']
    num_scale = bundle['num_scale']
    model = bundle['model']

    row = {}
    for col, val in [('Million Plus Cities', city),
                     ('Cause category', cause_cat),
                     ('Cause Subcategory', cause_sub)]:
        le = label_encoders[col]
        enc_val = le.transform([val])[0] if val in le.classes_ \
                  else le.transform([le.classes_[0]])[0]
        row[col + '_enc'] = enc_val

    mask = ((merged_df['Million Plus Cities'] == city) &
            (merged_df['Cause category'] == cause_cat) &
            (merged_df['Cause Subcategory'] == cause_sub))
    if mask.any():
        ref = merged_df[mask].iloc[0]
    else:
        ref = merged_df.mean(numeric_only=True)

    row['accidents'] = ref.get('accidents', 0)
    row['killed'] = ref.get('killed', 0)
    row['injured'] = ref.get('injured', 0)
    row['grievous'] = ref.get('grievous', 0)
    row['injury_rate'] = ref.get('injury_rate', 0)
    row['grievous_rate'] = ref.get('grievous_rate', 0)
    row['severity_score'] = ref.get('severity_score', 0)

    row_df = pd.DataFrame([row])
    row_df[num_scale] = scaler.transform(row_df[num_scale])
    probability = model.predict_proba(row_df[feature_cols])[0, 1] * 100

    return {
        'risk_probability': probability,
        'reference_stats': {
            'accidents': float(ref.get('accidents', 0)),
            'killed': float(ref.get('killed', 0)),
            'injured': float(ref.get('injured', 0)),
            'grievous': float(ref.get('grievous', 0)),
            'fatality_rate': float(ref.get('fatality_rate', 0)) * 100,
            'injury_rate': float(ref.get('injury_rate', 0)) * 100,
            'grievous_rate': float(ref.get('grievous_rate', 0)) * 100,
            'severity_score': float(ref.get('severity_score', 0)),
        },
    }


def train_models(X, y):

    print("  STEP 3/4 — Model Training & Comparison")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    models = {}
    if HAS_HIST:
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=8, random_state=42)
    # Use a single worker so training also works in restricted Windows
    # environments where joblib multiprocessing resources are blocked.
    models["RandomForest"]     = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=1, random_state=42)
    models["ExtraTrees"]       = ExtraTreesClassifier(
        n_estimators=100, max_depth=10, n_jobs=1, random_state=42)
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

    results = {}
    best_name = None
    best_f1   = -1

    header = f"  {'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Time':>6}"
    print(header)
    print("  " + SEP2)

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)

        results[name] = dict(model=clf, acc=acc, prec=prec,
                             rec=rec, f1=f1, auc=auc,
                             y_pred=y_pred, y_prob=y_prob)

        marker = "  ◄ BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1   = f1
            best_name = name

        print(f"  {name:<25} {acc*100:>5.1f}% {prec*100:>5.1f}% "
              f"{rec*100:>5.1f}% {f1*100:>5.1f}% {auc*100:>5.1f}% "
              f"{elapsed:>5.1f}s{marker}")

    # ── Detailed report for best ───────────────────────────────
    r = results[best_name]
 
    print(f"  BEST MODEL: {best_name}")
    print(SEP)
    print(f"  Accuracy  : {r['acc']*100:.2f}%")
    print(f"  Precision : {r['prec']*100:.2f}%")
    print(f"  Recall    : {r['rec']*100:.2f}%")
    print(f"  F1 Score  : {r['f1']*100:.2f}%")
    print(f"  ROC-AUC   : {r['auc']*100:.2f}%")

    cm = confusion_matrix(y_test, r['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  ── Confusion Matrix {'─'*46}")
    print(f"  {'':22}  Predicted LOW   Predicted HIGH")
    print(f"  {'Actual LOW (safe)':22}  {tn:>13,}   {fp:>13,}")
    print(f"  {'Actual HIGH (risk)':22}  {fn:>13,}   {tp:>13,}")
    print(f"\n  Correctly caught high-risk: {tp:,}  |  Missed: {fn:,}  |  False alarms: {fp:,}")

    # ── Feature importance ─────────────────────────────────────
    clf = r['model']
    if hasattr(clf, 'feature_importances_'):
        print(f"\n  ── Feature Importances {'─'*42}")
        imps = pd.Series(clf.feature_importances_, index=X.columns)
        top = imps.sort_values(ascending=False)
        for feat, imp in top.items():
            bar = "█" * int(imp / top.max() * 35)
            print(f"  {feat:<35} {bar:<35} {imp*100:.2f}%")

    return results[best_name]['model'], best_name


def predict_scenarios(model, label_encoders, scaler, feature_cols,
                      num_scale, merged_df):

    print("  STEP 4/4 — Scenario Predictions")
    print(SEP)

    def predict_one(city, cause_cat, cause_sub):
        row = {}
        for col, val in [('Million Plus Cities', city),
                         ('Cause category', cause_cat),
                         ('Cause Subcategory', cause_sub)]:
            le = label_encoders[col]
            enc_val = le.transform([val])[0] if val in le.classes_ \
                      else le.transform([le.classes_[0]])[0]
            row[col + '_enc'] = enc_val

        # Use mean counts for that city–cause combo if known
        mask = ((merged_df['Million Plus Cities'] == city) &
                (merged_df['Cause category']      == cause_cat) &
                (merged_df['Cause Subcategory']   == cause_sub))
        if mask.any():
            ref = merged_df[mask].iloc[0]
        else:
            ref = merged_df.mean(numeric_only=True)

        row['accidents']     = ref.get('accidents', 0)
        row['killed']        = ref.get('killed', 0)
        row['injured']       = ref.get('injured', 0)
        row['grievous']      = ref.get('grievous', 0)
        row['injury_rate']   = ref.get('injury_rate', 0)
        row['grievous_rate'] = ref.get('grievous_rate', 0)
        row['severity_score']= ref.get('severity_score', 0)

        row_df = pd.DataFrame([row])
        row_df[num_scale] = scaler.transform(row_df[num_scale])
        prob = model.predict_proba(row_df[feature_cols])[0]
        return prob[1] * 100

    def risk_label(p):
        if p < 25:  return "🟢 LOW RISK"
        if p < 50:  return "🟡 MEDIUM RISK"
        if p < 75:  return "🟠 HIGH RISK"
        return "🔴 CRITICAL RISK"

    def bar(p, w=35):
        f = int(p / 100 * w)
        c = ("░" if p < 25 else "▒" if p < 50 else "▓" if p < 75 else "█")
        return c * f + "─" * (w - f)

    scenarios = [
        ("Delhi",     "Traffic Violation", "Drunken Driving/ Consumption of alcohol and drug"),
        ("Mumbai",    "Traffic Violation", "Jumping Red Light"),
        ("Bengaluru", "Road Features",     "Pot Holes"),
        ("Chennai",   "Weather",           "Rainy"),
        ("Hyderabad", "Junction",          "Four arm Junction"),
        ("Kolkata",   "Traffic Control",   "Traffic Light Signal"),
        ("Jaipur",    "Road Features",     "Curved Road"),
        ("Pune",      "Traffic Violation", "Use of Mobile Phone"),
    ]

    for city, cat, sub in scenarios:
        risk = predict_one(city, cat, sub)
        print(f"\n  ┌─ {city} | {cat} → {sub}")
        print(f"  │  [{bar(risk)}] {risk:.1f}%  {risk_label(risk)}")
        print(f"  └{'─'*65}")

    # ── Interactive ────────────────────────────────────────────

    print("  CUSTOM PREDICTION — Enter Your Own Conditions")
    print(SEP)

    all_cities  = sorted(label_encoders['Million Plus Cities'].classes_)
    all_causes  = sorted(label_encoders['Cause category'].classes_)
    all_subs    = sorted(label_encoders['Cause Subcategory'].classes_)

    print(f"\n  Cities available : {', '.join(all_cities)}")
    print(f"  Cause categories : {', '.join(all_causes)}")

    try:
        print("\n  (Press Ctrl+C to skip)\n")
        city = input("  City              : ").strip()
        if city not in all_cities:
            print(f"  ⚠ '{city}' not recognised. Choose from: {', '.join(all_cities)}")
            city = input("  City              : ").strip()

        cat = input("  Cause category    : ").strip()
        if cat not in all_causes:
            print(f"  ⚠ '{cat}' not recognised. Choose from: {', '.join(all_causes)}")
            cat = input("  Cause category    : ").strip()

        print(f"\n  Subcategories     : {', '.join(all_subs)}")
        sub = input("  Cause subcategory : ").strip()
        if sub not in all_subs:
            print(f"  ⚠ '{sub}' not recognised. Choose from: {', '.join(all_subs)}")
            sub = input("  Cause subcategory : ").strip()

            risk  = predict_one(city, cat, sub)
            label = risk_label(risk)
            b     = bar(risk)

        print(f"\n  ╔{'═'*65}╗")
        print(f"  ║  City     : {city:<53}║")
        print(f"  ║  Cause    : {cat} → {sub[:40]:<40}  ║")
        print(f"  ║  [{b}] {risk:.1f}%  ║")
        print(f"  ║  Result   : {label:<53}║")
        print(f"  ╚{'═'*65}╝")

    except (KeyboardInterrupt, EOFError):
        print("\n  Skipped.")

if __name__ == "__main__":

    print("  INDIA ACCIDENT RISK ANALYSER  v1.0")
    print(f"  Dataset : {CSV_PATH}")
    print(SEP)

    df = load_data()
    explore(df)

    print("  STEP 3/4 — Building ML Features")
    print(SEP)
    merged, threshold = build_features(df)
    X, y, label_encoders, scaler, feature_cols, num_scale = preprocess(merged)

    best_model, best_name = train_models(X, y)

    predict_scenarios(best_model, label_encoders, scaler,
                      feature_cols, num_scale, merged)

    print(f"\n{SEP}")
    print(f"  ✓ Done!  Best model: {best_name}")
    print(SEP + "\n")
