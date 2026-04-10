import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')
np.random.seed(42)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSV_PATH   = "df.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BG='#0d1117'; CARD='#161b22'; CARD2='#1c2230'
ACC1='#00d4ff'; ACC2='#ff6b35'; ACC3='#7c3aed'
ACC4='#10b981'; WARN='#f59e0b'; TXT='#e6edf3'
TXTSUB='#8b949e'; GRID='#21262d'

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':CARD,'axes.edgecolor':GRID,
    'axes.labelcolor':TXT,'xtick.color':TXTSUB,'ytick.color':TXTSUB,
    'text.color':TXT,'grid.color':GRID,'grid.linewidth':0.5
})

def sax(ax, title='', grid=True):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(GRID)
    if title: ax.set_title(title, color=TXT, fontsize=11, fontweight='bold', pad=10)
    if grid:  ax.grid(True, alpha=0.25, color=GRID)

def save(fig, name):
    p = os.path.abspath(os.path.join(OUTPUT_DIR, name))
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"   ✓ {p}")

def load_dataset():
    print(f"\n{'═'*60}")
    print("  STEP 1 — LOADING DATASET")
    print(f"{'═'*60}")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = ['City','CauseCategory','CauseSubcategory','Outcome','Count']
    df['Count'] = pd.to_numeric(df['Count'], errors='coerce').fillna(0)

    print(f"  Raw shape            : {df.shape}")
    print(f"  Cities               : {df['City'].nunique()}")
    print(f"  Cause categories     : {df['CauseCategory'].nunique()}")
    print(f"  Cause subcategories  : {df['CauseSubcategory'].nunique()}")
    print(f"  Outcomes             : {list(df['Outcome'].unique())}")
    print(f"  Total count (sum)    : {df['Count'].sum():,.0f}")
    print(f"  Missing Count rows   : {df['Count'].isna().sum()}")

    accidents = df[df['Outcome']=='Total number of Accidents'].groupby('City')['Count'].sum()
    killed    = df[df['Outcome']=='Persons Killed'].groupby('City')['Count'].sum()
    injured   = df[df['Outcome']=='Total Injured'].groupby('City')['Count'].sum()

    print(f"\n  Top 5 cities by accidents:")
    for city, v in accidents.sort_values(ascending=False).head(5).items():
        print(f"    {city:<25}: {v:,.0f}")

    return df


def visualise_raw(df):
    print(f"\n{'═'*60}")
    print("  STEP 2 — RAW VISUALISATIONS")
    print(f"{'═'*60}")

    acc  = df[df['Outcome']=='Total number of Accidents']
    kill = df[df['Outcome']=='Persons Killed']
    inj  = df[df['Outcome']=='Total Injured']
    sev  = df[df['Outcome']=='Greviously Injured']

    total_acc  = acc['Count'].sum()
    total_kill = kill['Count'].sum()
    total_inj  = inj['Count'].sum()
    fatality_r = (total_kill / total_acc * 100) if total_acc > 0 else 0

    # — 01  KPI summary ———————————————————————————————————————
    print("\n  RAW_01_kpi_summary.png")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=BG)
    fig.suptitle('RAW DATASET — KPI Summary | India Road Accidents (Million+ Cities)',
                 fontsize=13, fontweight='bold', color=TXT, y=1.02)
    kpis = [('Total Accidents', f'{total_acc:,.0f}', ACC1),
            ('Persons Killed',  f'{total_kill:,.0f}', ACC2),
            ('Total Injured',   f'{total_inj:,.0f}', WARN),
            ('Fatality Rate',   f'{fatality_r:.1f}%', ACC3)]
    for ax,(label,val,col) in zip(axes,kpis):
        ax.set_facecolor(CARD2); ax.set_xlim(0,1); ax.set_ylim(0,1)
        for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5,0.62,val,   ha='center',va='center',fontsize=22,fontweight='bold',color=col)
        ax.text(0.5,0.18,label, ha='center',va='center',fontsize=10,color=TXTSUB)
    fig.tight_layout()
    save(fig,'RAW_01_kpi_summary.png')

    # — 02  Accidents by city (top 20) ————————————————————————
    print("  RAW_02_accidents_by_city.png")
    city_acc = acc.groupby('City')['Count'].sum().sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(10,10), facecolor=BG)
    sax(ax, 'Total Accidents by City (Top 20)')
    colors = [WARN if v == city_acc.max() else ACC1 for v in city_acc.values]
    bars = ax.barh(city_acc.index, city_acc.values, color=colors, alpha=0.88)
    for b in bars:
        ax.text(b.get_width()+50, b.get_y()+b.get_height()/2,
                f'{int(b.get_width()):,}', va='center', fontsize=8, color=TXT)
    ax.set_xlabel('Number of Accidents', color=TXT)
    fig.tight_layout()
    save(fig,'RAW_02_accidents_by_city.png')

    # — 03  Fatality by city (top 15) ——————————————————————————
    print("  RAW_03_fatality_by_city.png")
    city_kill = kill.groupby('City')['Count'].sum().sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10,8), facecolor=BG)
    sax(ax, 'Persons Killed by City (Top 15)')
    bars = ax.barh(city_kill.index, city_kill.values, color=ACC2, alpha=0.88)
    for b in bars:
        ax.text(b.get_width()+10, b.get_y()+b.get_height()/2,
                f'{int(b.get_width()):,}', va='center', fontsize=9, color=TXT)
    ax.set_xlabel('Persons Killed', color=TXT)
    fig.tight_layout()
    save(fig,'RAW_03_fatality_by_city.png')

    # — 04  Accidents by cause category ———————————————————————
    print("  RAW_04_cause_category.png")
    cat_acc = acc.groupby('CauseCategory')['Count'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11,5), facecolor=BG)
    sax(ax, 'Total Accidents by Cause Category')
    pal = [ACC1,ACC2,ACC3,ACC4,WARN,'#ff2d55']
    bars = ax.bar(cat_acc.index, cat_acc.values, color=pal[:len(cat_acc)], alpha=0.88, width=0.55)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+cat_acc.max()*0.01,
                f'{int(b.get_height()):,}', ha='center', fontsize=9, fontweight='bold', color=TXT)
    ax.set_ylabel('Number of Accidents', color=TXT)
    ax.set_xticklabels(cat_acc.index, rotation=20, ha='right', fontsize=9)
    fig.tight_layout()
    save(fig,'RAW_04_cause_category.png')

    # — 05  Outcome distribution (pie) ————————————————————————
    print("  RAW_05_outcome_distribution.png")
    out_tot = df.groupby('Outcome')['Count'].sum()
    out_plot = out_tot.drop('Total number of Accidents', errors='ignore')
    fig, ax = plt.subplots(figsize=(8,7), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, texts, auts = ax.pie(
        out_plot.values, labels=out_plot.index, autopct='%1.1f%%',
        colors=[ACC2,WARN,ACC3,ACC1][:len(out_plot)],
        startangle=130, pctdistance=0.78,
        textprops={'color':TXT,'fontsize':10},
        wedgeprops={'edgecolor':BG,'linewidth':2})
    for at in auts: at.set_color(BG); at.set_fontweight('bold')
    ax.set_title('Outcome Distribution (Injuries & Deaths)', color=TXT,
                 fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    save(fig,'RAW_05_outcome_distribution.png')

    # — 06  Subcategory accidents (top 15) ————————————————————
    print("  RAW_06_subcategory_accidents.png")
    sub_acc = acc.groupby('CauseSubcategory')['Count'].sum().sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10,8), facecolor=BG)
    sax(ax, 'Top 15 Cause Subcategories by Accident Count')
    colors = [ACC3 if v == sub_acc.max() else ACC1 for v in sub_acc.values]
    bars = ax.barh(sub_acc.index, sub_acc.values, color=colors, alpha=0.88)
    for b in bars:
        ax.text(b.get_width()+50, b.get_y()+b.get_height()/2,
                f'{int(b.get_width()):,}', va='center', fontsize=8, color=TXT)
    ax.set_xlabel('Number of Accidents', color=TXT)
    fig.tight_layout()
    save(fig,'RAW_06_subcategory_accidents.png')

    # — 07  Weather-related accidents ————————————————————————
    print("  RAW_07_weather_accidents.png")
    weather_df = acc[acc['CauseCategory']=='Weather']
    if not weather_df.empty:
        w_acc = weather_df.groupby('CauseSubcategory')['Count'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9,5), facecolor=BG)
        sax(ax, 'Accidents by Weather Condition')
        bars = ax.bar(w_acc.index, w_acc.values,
                      color=[ACC1,ACC4,WARN,ACC2][:len(w_acc)], alpha=0.88, width=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+w_acc.max()*0.02,
                    f'{int(b.get_height()):,}', ha='center', fontsize=10, fontweight='bold', color=TXT)
        ax.set_ylabel('Number of Accidents', color=TXT)
        fig.tight_layout()
        save(fig,'RAW_07_weather_accidents.png')

    # — 08  Traffic violation breakdown ———————————————————————
    print("  RAW_08_traffic_violations.png")
    viol_df = acc[acc['CauseCategory']=='Traffic Violation']
    if not viol_df.empty:
        v_acc = viol_df.groupby('CauseSubcategory')['Count'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10,5), facecolor=BG)
        sax(ax, 'Accidents by Traffic Violation Type')
        bars = ax.bar(v_acc.index, v_acc.values,
                      color=[ACC2,ACC3,WARN,ACC1,ACC4][:len(v_acc)], alpha=0.88, width=0.55)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+v_acc.max()*0.015,
                    f'{int(b.get_height()):,}', ha='center', fontsize=9, fontweight='bold', color=TXT)
        ax.set_ylabel('Number of Accidents', color=TXT)
        ax.set_xticklabels(v_acc.index, rotation=20, ha='right', fontsize=9)
        fig.tight_layout()
        save(fig,'RAW_08_traffic_violations.png')

    # — 09  Fatality rate by cause category ——————————————————
    print("  RAW_09_fatality_rate_by_cause.png")
    acc_cat  = acc.groupby('CauseCategory')['Count'].sum()
    kill_cat = kill.groupby('CauseCategory')['Count'].sum()
    fat_rate = (kill_cat / acc_cat * 100).dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG)
    sax(ax, 'Fatality Rate (%) by Cause Category')
    colors = [WARN if v == fat_rate.max() else ACC2 for v in fat_rate.values]
    bars = ax.bar(fat_rate.index, fat_rate.values, color=colors, alpha=0.88, width=0.5)
    for b,v in zip(bars, fat_rate.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.2, f'{v:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color=TXT)
    ax.set_ylabel('Fatality Rate (%)', color=TXT)
    ax.set_xticklabels(fat_rate.index, rotation=15, ha='right', fontsize=9)
    fig.tight_layout()
    save(fig,'RAW_09_fatality_rate_by_cause.png')

    # — 10  City heatmap: cause vs outcome ————————————————————
    print("  RAW_10_cause_outcome_heatmap.png")
    pivot = df.groupby(['CauseCategory','Outcome'])['Count'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(13,7), facecolor=BG)
    sax(ax, 'Cause Category vs Outcome — Count Heatmap', grid=False)
    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
                annot_kws={'size':9}, linewidths=0.5, linecolor=GRID,
                cbar_kws={'shrink':0.8})
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    fig.tight_layout()
    save(fig,'RAW_10_cause_outcome_heatmap.png')

    # — 11  Road features accidents ——————————————————————————
    print("  RAW_11_road_features.png")
    road_df = acc[acc['CauseCategory']=='Road Features']
    if not road_df.empty:
        r_acc = road_df.groupby('CauseSubcategory')['Count'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10,5), facecolor=BG)
        sax(ax, 'Accidents by Road Feature Type')
        bars = ax.bar(r_acc.index, r_acc.values,
                      color=[ACC4,ACC1,WARN,ACC3,ACC2,TXTSUB][:len(r_acc)], alpha=0.88, width=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+r_acc.max()*0.015,
                    f'{int(b.get_height()):,}', ha='center', fontsize=9, fontweight='bold', color=TXT)
        ax.set_ylabel('Number of Accidents', color=TXT)
        ax.set_xticklabels(r_acc.index, rotation=15, ha='right', fontsize=9)
        fig.tight_layout()
        save(fig,'RAW_11_road_features.png')

    print(f"\n  Total Accidents  : {total_acc:,.0f}")
    print(f"  Persons Killed   : {total_kill:,.0f}")
    print(f"  Total Injured    : {total_inj:,.0f}")
    print(f"  Fatality Rate    : {fatality_r:.1f}%")


def preprocess(df):
    print(f"\n{'═'*60}")
    print("  STEP 3 — PREPROCESSING")
    print(f"{'═'*60}")

    proc = df.copy()

    le_city = LabelEncoder()
    le_cat  = LabelEncoder()
    le_sub  = LabelEncoder()
    le_out  = LabelEncoder()

    proc['City_enc']     = le_city.fit_transform(proc['City'])
    proc['CatEnc']       = le_cat.fit_transform(proc['CauseCategory'])
    proc['SubEnc']       = le_sub.fit_transform(proc['CauseSubcategory'])
    proc['OutcomeEnc']   = le_out.fit_transform(proc['Outcome'])

    proc['Count'] = proc['Count'].fillna(0)

    acc_per_city = proc[proc['Outcome']=='Total number of Accidents'].groupby('City')['Count'].sum()
    kill_per_city = proc[proc['Outcome']=='Persons Killed'].groupby('City')['Count'].sum()

    proc['city_total_acc']  = proc['City'].map(acc_per_city).fillna(0)
    proc['city_total_kill'] = proc['City'].map(kill_per_city).fillna(0)
    proc['fatality_rate']   = (proc['city_total_kill'] / proc['city_total_acc'].replace(0, np.nan)).fillna(0)

    proc['is_fatal_outcome']   = (proc['Outcome']=='Persons Killed').astype(int)
    proc['is_weather_cause']   = (proc['CauseCategory']=='Weather').astype(int)
    proc['is_violation_cause'] = (proc['CauseCategory']=='Traffic Violation').astype(int)
    proc['is_road_feature']    = (proc['CauseCategory']=='Road Features').astype(int)
    proc['is_high_count']      = (proc['Count'] > proc['Count'].quantile(0.75)).astype(int)

    proc['log_count'] = np.log1p(proc['Count'])

    scaler = StandardScaler()
    proc[['Count_scaled','log_count_scaled','city_total_acc_scaled','fatality_rate_scaled']] = \
        scaler.fit_transform(proc[['Count','log_count','city_total_acc','fatality_rate']])

    proc['is_accident'] = (proc['Outcome']=='Total number of Accidents').astype(int)

    before = len(proc)
    proc = proc.dropna().reset_index(drop=True)
    print(f"  Rows dropped (NaN)   : {before-len(proc)}")
    print(f"  Final shape          : {proc.shape}")
    print(f"  NaN remaining        : {proc.isnull().sum().sum()}")
    print(f"  Class 0 (non-acc)    : {(proc['is_accident']==0).sum():,}")
    print(f"  Class 1 (accident)   : {(proc['is_accident']==1).sum():,}")
    print(f"  Engineered features  : city_total_acc, city_total_kill, fatality_rate,")
    print(f"                         is_fatal_outcome, is_weather_cause, is_violation_cause,")
    print(f"                         is_road_feature, is_high_count, log_count")
    return proc


# ════════════════════════════════════════════════════════════════
# 4  PREPROCESSED VISUALISATIONS
# ════════════════════════════════════════════════════════════════
def visualise_preprocessed(proc, raw):
    print(f"\n{'═'*60}")
    print("  STEP 4 — PREPROCESSED VISUALISATIONS")
    print(f"{'═'*60}")

    n = len(proc)

    # — P01  KPI summary ——————————————————————————————————————
    print("\n  PROC_01_kpi_summary.png")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=BG)
    fig.suptitle('PREPROCESSED DATASET — KPI Summary',
                 fontsize=13, fontweight='bold', color=TXT, y=1.02)
    eng_feats = sum(1 for c in ['city_total_acc','city_total_kill','fatality_rate',
                                'is_fatal_outcome','is_weather_cause','is_violation_cause',
                                'is_road_feature','is_high_count','log_count'] if c in proc.columns)
    kpis = [('Clean Records', f'{n:,}', ACC4),
            ('Missing Values','0', ACC4),
            ('Total Features', str(proc.shape[1]), ACC1),
            ('Engineered Feats', f'+{eng_feats}', WARN)]
    for ax,(label,val,col) in zip(axes,kpis):
        ax.set_facecolor(CARD2); ax.set_xlim(0,1); ax.set_ylim(0,1)
        for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5,0.62,val,   ha='center',va='center',fontsize=26,fontweight='bold',color=col)
        ax.text(0.5,0.18,label, ha='center',va='center',fontsize=10,color=TXTSUB)
    fig.tight_layout()
    save(fig,'PROC_01_kpi_summary.png')

    # — P02  Correlation heatmap ——————————————————————————————
    print("  PROC_02_correlation_heatmap.png")
    num_cols = ['City_enc','CatEnc','SubEnc','Count','log_count',
                'city_total_acc','city_total_kill','fatality_rate',
                'is_fatal_outcome','is_weather_cause','is_violation_cause',
                'is_road_feature','is_high_count','is_accident']
    num_cols = [c for c in num_cols if c in proc.columns]
    corr = proc[num_cols].corr()
    fig, ax = plt.subplots(figsize=(13,11), facecolor=BG)
    sax(ax, 'Feature Correlation Matrix', grid=False)
    sns.heatmap(corr, ax=ax, cmap=sns.diverging_palette(240,10,as_cmap=True),
                center=0, vmin=-1, vmax=1, annot=True, fmt='.2f',
                annot_kws={'size':7.5}, linewidths=0.5, linecolor=GRID,
                cbar_kws={'shrink':0.8})
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    fig.tight_layout()
    save(fig,'PROC_02_correlation_heatmap.png')

    print(f"  Top correlations with is_accident:")
    corr_acc = corr['is_accident'].drop('is_accident').sort_values(key=abs, ascending=False)
    for feat, val in corr_acc.head(6).items():
        print(f"    {feat:<30}: {val:+.3f}")

    # — P03  Log-transformed count distribution ———————————————
    print("  PROC_03_log_count_distribution.png")
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5), facecolor=BG)
    sax(ax1,'Raw Count Distribution')
    ax1.hist(proc['Count'].clip(upper=proc['Count'].quantile(0.99)),
             bins=50, color=ACC2, alpha=0.78, edgecolor='none')
    ax1.set_xlabel('Count', color=TXT); ax1.set_ylabel('Frequency', color=TXT)
    sax(ax2,'Log-Transformed Count Distribution')
    ax2.hist(proc['log_count'], bins=50, color=ACC4, alpha=0.78, edgecolor='none')
    ax2.axvline(proc['log_count'].mean(), color=ACC2, lw=2, ls='--',
                label=f'Mean = {proc["log_count"].mean():.2f}')
    ax2.legend(fontsize=9, facecolor=CARD2, labelcolor=TXT)
    ax2.set_xlabel('log(1 + Count)', color=TXT); ax2.set_ylabel('Frequency', color=TXT)
    fig.suptitle('Count Before vs After Log Transformation',
                 color=TXT, fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig,'PROC_03_log_count_distribution.png')

    # — P04  Fatality rate by city ————————————————————————————
    print("  PROC_04_fatality_rate_by_city.png")
    fat_city = proc[proc['Outcome']=='Persons Killed'].groupby('City').first()['fatality_rate']
    fat_city = fat_city.sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(10,9), facecolor=BG)
    sax(ax, 'Engineered Fatality Rate by City (Top 20)')
    colors = [WARN if v==fat_city.max() else ACC2 for v in fat_city.values]
    ax.barh(fat_city.index, fat_city.values*100, color=colors, alpha=0.88)
    ax.set_xlabel('Fatality Rate (%)', color=TXT)
    fig.tight_layout()
    save(fig,'PROC_04_fatality_rate_by_city.png')

    # — P05  Feature importance ———————————————————————————————
    print("  PROC_05_feature_importance.png")
    feat_cols = ['City_enc','CatEnc','SubEnc','Count_scaled','log_count_scaled',
                 'city_total_acc_scaled','fatality_rate_scaled',
                 'is_fatal_outcome','is_weather_cause','is_violation_cause',
                 'is_road_feature','is_high_count']
    feat_cols = [c for c in feat_cols if c in proc.columns]
    Xq = proc[feat_cols]
    yq = proc['is_accident']
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(Xq, yq)
    fi = pd.Series(rf.feature_importances_, index=Xq.columns).sort_values()
    fig, ax = plt.subplots(figsize=(10,7), facecolor=BG)
    sax(ax, 'Feature Importance — Random Forest (50 trees)')
    colors_fi = [ACC2 if v>fi.median() else ACC1 for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors_fi, alpha=0.88)
    ax.axvline(fi.median(), color=WARN, ls='--', lw=1.5, alpha=0.8, label='Median')
    ax.legend(fontsize=9, facecolor=CARD2, labelcolor=TXT)
    ax.set_xlabel('Importance Score', color=TXT)
    print(f"  Top 5 features (RF)  :")
    for feat, val in fi.sort_values(ascending=False).head(5).items():
        print(f"    {feat:<30}: {val:.4f}")
    fig.tight_layout()
    save(fig,'PROC_05_feature_importance.png')

    # — P06  Class balance ————————————————————————————————————
    print("  PROC_06_class_balance.png")
    cv = proc['is_accident'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7,5), facecolor=BG)
    sax(ax, 'Class Balance — is_accident Label')
    bars = ax.bar(['Non-Accident\nOutcome (0)','Accident\nOutcome (1)'],
                  cv.values, color=[ACC4,ACC2], alpha=0.88, width=0.45)
    for i,v in enumerate(cv.values):
        ax.text(i, v+max(cv)*0.015, f'{v:,}\n({v/n*100:.1f}%)',
                ha='center', fontsize=11, fontweight='bold', color=TXT)
    ax.set_ylabel('Count', color=TXT)
    fig.tight_layout()
    save(fig,'PROC_06_class_balance.png')

    # — P07  Engineered feature impact ———————————————————————
    print("  PROC_07_engineered_features.png")
    feats_eng = ['is_weather_cause','is_violation_cause','is_road_feature','is_high_count']
    feats_eng = [f for f in feats_eng if f in proc.columns]
    feat_rates  = [proc[proc[f]==1]['is_accident'].mean()*100 for f in feats_eng]
    base_rate   = proc['is_accident'].mean()*100
    labels_eng  = ['Weather\nCause','Traffic\nViolation','Road\nFeature','High Count\nRecord']
    fig, ax = plt.subplots(figsize=(9,5), facecolor=BG)
    sax(ax, 'Engineered Feature Flag — Accident Rate Impact')
    bars = ax.bar(labels_eng, feat_rates, color=[ACC3,WARN,ACC2,ACC1], alpha=0.88, width=0.5)
    ax.axhline(base_rate, color=ACC4, ls='--', lw=2, label=f'Overall avg {base_rate:.1f}%')
    for b,v in zip(bars,feat_rates):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color=TXT)
    ax.legend(fontsize=9, facecolor=CARD2, labelcolor=TXT)
    ax.set_ylabel('Accident Rate (%)', color=TXT)
    print(f"  Engineered feat rates:")
    for f,v in zip(feats_eng, feat_rates):
        print(f"    {f:<30}: {v:.1f}%  (vs baseline {base_rate:.1f}%)")
    fig.tight_layout()
    save(fig,'PROC_07_engineered_features.png')


# ════════════════════════════════════════════════════════════════
# 5  COMPARISON
# ════════════════════════════════════════════════════════════════
def visualise_comparison(raw, proc):
    print(f"\n{'═'*60}")
    print("  STEP 5 — RAW vs PREPROCESSED COMPARISON")
    print(f"{'═'*60}")

    # — C01  Count before vs after log transform ——————————————
    print("\n  COMP_01_count_before_after.png")
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6), facecolor=BG)
    sax(ax1, 'Count — RAW (Skewed)')
    raw_counts = raw['Count'].dropna()
    ax1.hist(raw_counts.clip(upper=raw_counts.quantile(0.99)),
             bins=50, color=ACC2, alpha=0.78)
    ax1.axvline(raw_counts.mean(), color=WARN, lw=2, ls='--',
                label=f'Mean={raw_counts.mean():.1f}')
    ax1.text(0.97,0.95, f'Missing: {raw["Count"].isna().sum()} rows',
             transform=ax1.transAxes, ha='right', va='top', fontsize=9, color=WARN,
             bbox=dict(boxstyle='round', facecolor=CARD2, edgecolor=WARN))
    ax1.legend(fontsize=9, facecolor=CARD2, labelcolor=TXT)
    ax1.set_xlabel('Count (raw)', color=TXT)
    ax1.set_ylabel('Frequency', color=TXT)

    sax(ax2, 'Count — PROCESSED (Log-scaled + Normalised)')
    ax2.hist(proc['log_count_scaled'], bins=50, color=ACC4, alpha=0.78)
    ax2.axvline(0, color=WARN, lw=2, ls='--', label='μ=0')
    ax2.text(0.97,0.95, '✓  0 Missing rows',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9, color=ACC4,
             bbox=dict(boxstyle='round', facecolor=CARD2, edgecolor=ACC4))
    ax2.legend(fontsize=9, facecolor=CARD2, labelcolor=TXT)
    ax2.set_xlabel('log(Count) — z-score', color=TXT)
    fig.suptitle('Count Distribution: Before vs After Preprocessing',
                 color=TXT, fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig,'COMP_01_count_before_after.png')

    # — C02  Summary metrics comparison ——————————————————————
    print("  COMP_02_key_metrics.png")
    raw_miss = raw['Count'].isna().sum()
    fig, axes = plt.subplots(1,3, figsize=(15,5), facecolor=BG)
    metrics = [('Missing Values', [raw_miss, 0],           [ACC2,ACC4]),
               ('Feature Count',  [len(raw.columns), proc.shape[1]], [ACC1,ACC3]),
               ('Total Records',  [len(raw), len(proc)],   [WARN,ACC4])]
    for ax,(label,vals,colors) in zip(axes, metrics):
        sax(ax, label)
        ax.bar(['Raw','Processed'], vals, color=colors, alpha=0.88, width=0.4)
        for j,v in enumerate(vals):
            ax.text(j, v*1.025+0.5, f'{v:,}', ha='center', fontsize=12,
                    fontweight='bold', color=TXT)
    fig.suptitle('Raw vs Preprocessed — Key Metrics',
                 color=TXT, fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig,'COMP_02_key_metrics.png')
    print(f"  Missing values: {raw_miss} → 0")
    print(f"  Feature count : {len(raw.columns)} → {proc.shape[1]}")
    print(f"  Record count  : {len(raw):,} → {len(proc):,}")

    # — C03  Pipeline diagram ————————————————————————————————
    print("  COMP_03_pipeline_diagram.png")
    fig, ax = plt.subplots(figsize=(18,4), facecolor=BG)
    ax.set_facecolor(BG); ax.set_xlim(0,10); ax.set_ylim(0,1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Preprocessing Pipeline', color=TXT, fontsize=13, fontweight='bold', pad=12)
    steps = [('1. Load\nCSV',0.6,ACC1),('2. Rename\nColumns',1.85,ACC3),
             ('3. Fill\nMissing',3.1,WARN),('4. Encode\nCategoricals',4.4,ACC3),
             ('5. Engineer\nFeatures',5.75,ACC4),('6. Log\nTransform',7.0,ACC1),
             ('7. Scale\nNumerics',8.25,WARN),('8. Train/Test\nSplit',9.5,ACC4)]
    for label,x,col in steps:
        rect = mpatches.FancyBboxPatch((x-0.52,0.2),1.05,0.56,
            boxstyle='round,pad=0.06', facecolor=CARD2, edgecolor=col, lw=2.5)
        ax.add_patch(rect)
        ax.text(x,0.48,label, ha='center',va='center',fontsize=8.5,color=TXT)
        if x < 9.5:
            ax.annotate('', xy=(x+0.58,0.48), xytext=(x+0.53,0.48),
                        arrowprops=dict(arrowstyle='->',color=TXTSUB,lw=2))
    fig.tight_layout()
    save(fig,'COMP_03_pipeline_diagram.png')


# ════════════════════════════════════════════════════════════════
# 6  TRAIN MODELS
# ════════════════════════════════════════════════════════════════
def train_models(proc):
    print(f"\n{'═'*60}")
    print("  STEP 6 — TRAINING ML MODELS")
    print(f"{'═'*60}")

    cols_without = ['City_enc','CatEnc','SubEnc','Count_scaled','log_count_scaled']
    cols_without = [c for c in cols_without if c in proc.columns]

    cols_with = cols_without + ['city_total_acc_scaled','fatality_rate_scaled',
                                'is_fatal_outcome','is_weather_cause',
                                'is_violation_cause','is_road_feature','is_high_count']
    cols_with = [c for c in cols_with if c in proc.columns]

    y = proc['is_accident']
    X_with    = proc[cols_with]
    X_without = proc[cols_without]

    Xwi_tr,Xwi_te,y_tr,y_te = train_test_split(X_with,    y, test_size=0.2, random_state=42, stratify=y)
    Xwo_tr,Xwo_te,_,_        = train_test_split(X_without, y, test_size=0.2, random_state=42, stratify=y)

    print(f"  Without FE features  : {cols_without}")
    print(f"  With FE features     : {cols_with}")
    print(f"  Train size           : {len(y_tr):,}  |  Test size: {len(y_te):,}")

    MODELS = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes':         GaussianNB(),
        'LinearSVC':           CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3),
    }

    def eval_m(model, Xtr, Xte, ytr, yte):
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:,1]
        return {'Accuracy': accuracy_score(yte,pred)*100,
                'Precision':precision_score(yte,pred,zero_division=0)*100,
                'Recall':   recall_score(yte,pred,zero_division=0)*100,
                'F1':       f1_score(yte,pred,zero_division=0)*100,
                'AUC-ROC':  roc_auc_score(yte,prob)*100}

    res_with = {}; res_without = {}
    print(f"\n  {'Model':<25} {'Time':>6}  {'F1(w/FE)':>9}  {'F1(w/o FE)':>11}  {'AUC(w/FE)':>10}")
    print(f"  {'─'*25} {'─'*6}  {'─'*9}  {'─'*11}  {'─'*10}")

    for name,m in MODELS.items():
        t0 = time.time()
        try:
            res_with[name]    = eval_m(clone(m), Xwi_tr, Xwi_te, y_tr, y_te)
            res_without[name] = eval_m(clone(m), Xwo_tr, Xwo_te, y_tr, y_te)
            elapsed = time.time()-t0
            print(f"  {name:<25} {elapsed:5.1f}s  "
                  f"{res_with[name]['F1']:>8.1f}%  "
                  f"{res_without[name]['F1']:>10.1f}%  "
                  f"{res_with[name]['AUC-ROC']:>9.1f}%")
        except Exception as e:
            elapsed = time.time()-t0
            print(f"  {name:<25} ✗ FAILED {elapsed:.1f}s  →  {e}")
            z = {'Accuracy':0,'Precision':0,'Recall':0,'F1':0,'AUC-ROC':0}
            res_with[name]=z.copy(); res_without[name]=z.copy()

    df_with    = pd.DataFrame(res_with).T
    df_without = pd.DataFrame(res_without).T
    print(f"\n  Best model (F1 w/FE)   : {df_with['F1'].idxmax()}  ({df_with['F1'].max():.1f}%)")
    print(f"  Best model (F1 w/o FE) : {df_without['F1'].idxmax()}  ({df_without['F1'].max():.1f}%)")
    print(f"  Avg F1 gain from FE    : +{(df_with['F1'].mean()-df_without['F1'].mean()):.1f}%")
    print(f"  Avg AUC gain from FE   : +{(df_with['AUC-ROC'].mean()-df_without['AUC-ROC'].mean()):.1f}%")
    return df_with, df_without


# ════════════════════════════════════════════════════════════════
# 7  MODEL COMPARISON VISUALISATIONS
# ════════════════════════════════════════════════════════════════
def visualise_models(df_with, df_without):
    print(f"\n{'═'*60}")
    print("  STEP 7 — MODEL COMPARISON VISUALISATIONS")
    print(f"{'═'*60}")

    names   = list(df_with.index)
    metrics = ['Accuracy','Precision','Recall','F1','AUC-ROC']
    x = np.arange(len(names)); w = 0.38
    short = [n.replace(' ','\n') for n in names]

    # — M01  Accuracy comparison ——————————————————————————————
    print("\n  MODEL_01_accuracy_comparison.png")
    fig, ax = plt.subplots(figsize=(13,6), facecolor=BG)
    sax(ax, 'Accuracy — Without FE vs With FE')
    ax.bar(x-w/2, df_without['Accuracy'], w, color=ACC2, alpha=0.78, label='Without FE')
    bars = ax.bar(x+w/2, df_with['Accuracy'], w, color=ACC4, alpha=0.88, label='With FE')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(50,105); ax.legend(fontsize=10, facecolor=CARD2, labelcolor=TXT)
    ax.set_ylabel('Accuracy (%)', color=TXT)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
                f'{b.get_height():.1f}', ha='center', fontsize=8, color=TXT)
    fig.tight_layout()
    save(fig,'MODEL_01_accuracy_comparison.png')

    # — M02  F1 comparison ————————————————————————————————————
    print("  MODEL_02_f1_comparison.png")
    fig, ax = plt.subplots(figsize=(13,6), facecolor=BG)
    sax(ax, 'F1 Score — Without FE vs With FE')
    ax.bar(x-w/2, df_without['F1'], w, color=ACC3, alpha=0.78, label='Without FE')
    bars2 = ax.bar(x+w/2, df_with['F1'], w, color=ACC1, alpha=0.88, label='With FE')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.legend(fontsize=10, facecolor=CARD2, labelcolor=TXT)
    ax.set_ylabel('F1 Score (%)', color=TXT)
    for b in bars2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
                f'{b.get_height():.1f}', ha='center', fontsize=8, color=TXT)
    fig.tight_layout()
    save(fig,'MODEL_02_f1_comparison.png')

    # — M03  AUC-ROC comparison ———————————————————————————————
    print("  MODEL_03_auc_comparison.png")
    fig, ax = plt.subplots(figsize=(13,6), facecolor=BG)
    sax(ax, 'AUC-ROC — Without FE vs With FE')
    ax.bar(x-w/2, df_without['AUC-ROC'], w, color=ACC2, alpha=0.78, label='Without FE')
    ax.bar(x+w/2, df_with['AUC-ROC'],    w, color=WARN,  alpha=0.88, label='With FE')
    ax.axhline(50, color=TXTSUB, ls='--', lw=1.5, alpha=0.6, label='Random baseline (50%)')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(40,105); ax.legend(fontsize=10, facecolor=CARD2, labelcolor=TXT)
    ax.set_ylabel('AUC-ROC (%)', color=TXT)
    fig.tight_layout()
    save(fig,'MODEL_03_auc_comparison.png')
    

    # — M04  Precision & Recall ———————————————————————————————
    print("  MODEL_04_precision_recall.png")
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6), facecolor=BG)
    sax(ax1, 'Precision — With vs Without FE')
    ax1.bar(x-w/2, df_without['Precision'], w, color=ACC3, alpha=0.78, label='Without FE')
    ax1.bar(x+w/2, df_with['Precision'],    w, color=ACC1, alpha=0.88, label='With FE')
    ax1.set_xticks(x); ax1.set_xticklabels(short, fontsize=8)
    ax1.legend(fontsize=9, facecolor=CARD2, labelcolor=TXT)
    ax1.set_ylabel('Precision (%)', color=TXT)
    sax(ax2, 'Recall — With vs Without FE')
    ax2.bar(x-w/2, df_without['Recall'], w, color=ACC3, alpha=0.78, label='Without FE')
    ax2.bar(x+w/2, df_with['Recall'],    w, color=ACC1, alpha=0.88, label='With FE')
    ax2.set_xticks(x); ax2.set_xticklabels(short, fontsize=8)
    ax2.legend(fontsize=9, facecolor=CARD2, labelcolor=TXT)
    ax2.set_ylabel('Recall (%)', color=TXT)
    fig.tight_layout()
    save(fig,'MODEL_04_precision_recall.png')

    # — M05  Radar best model —————————————————————————————————
    print("  MODEL_05_radar_best_model.png")
    best = df_with['F1'].idxmax()
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':'polar'}, facecolor=BG)
    ax.set_facecolor(CARD)
    ax.set_title(f'Metric Radar — {best}\n(Best Model by F1)',
                 color=TXT, fontsize=12, fontweight='bold', pad=20)
    ang = np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist(); ang+=ang[:1]
    vw  = [df_with.loc[best,m]/100    for m in metrics]+[df_with.loc[best,metrics[0]]/100]
    vwo = [df_without.loc[best,m]/100 for m in metrics]+[df_without.loc[best,metrics[0]]/100]
    ax.plot(ang,vw, color=ACC4,lw=2.5,label='With FE');    ax.fill(ang,vw, color=ACC4,alpha=0.2)
    ax.plot(ang,vwo,color=ACC2,lw=2.5,ls='--',label='Without FE'); ax.fill(ang,vwo,color=ACC2,alpha=0.1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(metrics, fontsize=11, color=TXT)
    ax.set_ylim(0,1); ax.spines['polar'].set_color(GRID); ax.grid(color=GRID, alpha=0.4)
    ax.legend(fontsize=10, loc='lower right', facecolor=CARD2, labelcolor=TXT)
    fig.tight_layout()
    save(fig,'MODEL_05_radar_best_model.png')

    # — M06  Delta F1 gain ————————————————————————————————————
    print("  MODEL_06_f1_gain_from_fe.png")
    delta = df_with['F1'] - df_without['F1']
    fig, ax = plt.subplots(figsize=(11,5), facecolor=BG)
    sax(ax, 'F1 Score Gain from Feature Engineering  (With FE − Without FE)')
    colors_d = [ACC4 if v>=0 else ACC2 for v in delta.values]
    brs = ax.bar(names, delta.values, color=colors_d, alpha=0.88, width=0.55)
    ax.axhline(0, color=TXTSUB, lw=1.5)
    ax.axhline(delta.mean(), color=WARN, ls='--', lw=1.5, label=f'Avg gain {delta.mean():+.1f}%')
    for b,v in zip(brs,delta.values):
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height()+0.15 if v>=0 else b.get_height()-0.8,
                f'{v:+.1f}%', ha='center', fontsize=9, fontweight='bold', color=TXT)
    ax.legend(fontsize=10, facecolor=CARD2, labelcolor=TXT)
    ax.set_ylabel('ΔF1 (%)', color=TXT)
    ax.set_xticklabels(names, rotation=15, fontsize=9)
    fig.tight_layout()
    save(fig,'MODEL_06_f1_gain_from_fe.png')

    # — M07  Full results table ———————————————————————————————
    print("  MODEL_07_full_results_table.png")
    fig, ax = plt.subplots(figsize=(18,5), facecolor=BG)
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Full Model Results — With vs Without Feature Engineering',
                 color=TXT, fontsize=13, fontweight='bold', pad=12)
    for sp in ax.spines.values(): sp.set_visible(False)
    hdr  = ['Model','Acc\n(w/o FE)','Acc\n(w/ FE)','F1\n(w/o FE)','F1\n(w/ FE)',
            'AUC\n(w/o FE)','AUC\n(w/ FE)','ΔF1','Best?']
    rows = []
    best_f1 = df_with['F1'].max()
    for mn in names:
        b=df_without.loc[mn]; f=df_with.loc[mn]; d=f['F1']-b['F1']
        rows.append([mn,
                     f'{b["Accuracy"]:.1f}%', f'{f["Accuracy"]:.1f}%',
                     f'{b["F1"]:.1f}%',       f'{f["F1"]:.1f}%',
                     f'{b["AUC-ROC"]:.1f}%',  f'{f["AUC-ROC"]:.1f}%',
                     f'{d:+.1f}%',
                     '★ BEST' if f['F1']==best_f1 else ''])
    tbl = ax.table(cellText=rows, colLabels=hdr, loc='center',
                   cellLoc='center', bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_facecolor(CARD if r%2==0 else CARD2)
        cell.set_edgecolor(GRID); cell.set_text_props(color=TXT)
        if r==0:
            cell.set_facecolor(ACC3); cell.set_text_props(color='white',fontweight='bold')
        if c==7 and r>0:
            try:
                vf=float(rows[r-1][7].replace('%',''))
                cell.set_facecolor('#1a3a1a' if vf>=0 else '#3a1a1a')
                cell.set_text_props(color=ACC4 if vf>=0 else ACC2, fontweight='bold')
            except: pass
        if c==8 and r>0 and rows[r-1][8]=='★ BEST':
            cell.set_facecolor('#2a2a00')
            cell.set_text_props(color=WARN, fontweight='bold')
    fig.tight_layout()
    save(fig,'MODEL_07_full_results_table.png')


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {os.path.abspath(OUTPUT_DIR)}\n")

    raw_df              = load_dataset()
    visualise_raw(raw_df)
    proc_df             = preprocess(raw_df)
    visualise_preprocessed(proc_df, raw_df)
    visualise_comparison(raw_df, proc_df)
    df_with, df_without = train_models(proc_df)
    visualise_models(df_with, df_without)

    print(f"\n{'═'*60}")
    print(f"  ALL OUTPUTS SAVED TO: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'═'*60}")
    print("  RAW CHARTS        : RAW_01  to RAW_11   (11 files)")
    print("  PROCESSED CHARTS  : PROC_01 to PROC_07  ( 7 files)")
    print("  COMPARISON CHARTS : COMP_01 to COMP_03  ( 3 files)")
    print("  MODEL CHARTS      : MODEL_01 to MODEL_07 ( 7 files)")
    print(f"{'═'*60}")
