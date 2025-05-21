import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Réglages visuels
sns.set(style="whitegrid")

# Répertoire de sortie
REPORT_DIR = "data_quality_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_dataset(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif ext == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError("❌ Format non supporté : " + ext)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        sys.exit(1)

def analyze_missing(df):
    missing = df.isnull().sum()
    percent = 100 * df.isnull().mean()
    return pd.DataFrame({"Valeurs manquantes": missing, "%": percent}).sort_values(by="%", ascending=False)

def analyze_duplicates(df):
    return df.duplicated().sum()

def analyze_types(df):
    return df.dtypes.reset_index().rename(columns={"index": "Colonne", 0: "Type"})

def analyze_cardinality(df):
    return df.nunique().sort_values(ascending=False)

def analyze_statistics(df):
    return df.describe(include="all").T

def detect_outliers(df):
    numeric_df = df.select_dtypes(include=[np.number])
    outlier_info = {}
    for col in numeric_df.columns:
        z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std(ddof=0))
        iqr = numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25)
        iqr_bounds = ((numeric_df[col] < (numeric_df[col].quantile(0.25) - 1.5 * iqr)) |
                      (numeric_df[col] > (numeric_df[col].quantile(0.75) + 1.5 * iqr)))
        outlier_info[col] = {
            "Z-score > 3": (z_scores > 3).sum(),
            "IQR": iqr_bounds.sum()
        }
    return pd.DataFrame(outlier_info).T.sort_values(by="Z-score > 3", ascending=False)

def plot_distributions(df, output_path):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution - {col}")
        plt.savefig(os.path.join(output_path, f"hist_{col}.png"))
        plt.close()

def generate_html_report(file_path, df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"rapport_qualite_{timestamp}.html")

    html = f"<h1>🧠 Rapport de Qualité - {os.path.basename(file_path)}</h1>"
    html += f"<p><b>Taille :</b> {df.shape[0]} lignes, {df.shape[1]} colonnes</p>"

    # 1. Valeurs manquantes
    html += "<h2>1. Valeurs manquantes</h2>"
    html += analyze_missing(df).to_html(classes="table table-striped", border=0)

    # 2. Doublons
    html += "<h2>2. Doublons</h2>"
    html += f"<p>Nombre de doublons : <b>{analyze_duplicates(df)}</b></p>"

    # 3. Types de données
    html += "<h2>3. Types de données</h2>"
    html += analyze_types(df).to_html(classes="table table-bordered", index=False)

    # 4. Cardinalité
    html += "<h2>4. Cardinalité</h2>"
    html += analyze_cardinality(df).to_frame("Niveaux uniques").to_html(classes="table", border=0)

    # 5. Statistiques descriptives
    html += "<h2>5. Statistiques descriptives</h2>"
    html += analyze_statistics(df).to_html(classes="table table-sm", border=0)

    # 6. Outliers
    html += "<h2>6. Détection de valeurs aberrantes</h2>"
    html += detect_outliers(df).to_html(classes="table table-hover", border=0)

    # 7. Distributions
    dist_path = os.path.join(REPORT_DIR, f"dist_{timestamp}")
    os.makedirs(dist_path, exist_ok=True)
    plot_distributions(df, dist_path)

    html += "<h2>7. Distributions</h2><ul>"
    for img in sorted(os.listdir(dist_path)):
        html += f'<li><img src="{os.path.join("dist_" + timestamp, img)}" width="500"/></li>'
    html += "</ul>"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'><style>body{{font-family:sans-serif}}img{{margin:10px}}.table{{border-collapse:collapse}}</style></head><body>{html}</body></html>")

    print(f"\n✅ Rapport généré : {report_path}")

def main():
    if len(sys.argv) < 2:
        print("❗ Usage : python data_quality_checker.py chemin/vers/fichier.csv")
        return
    file_path = sys.argv[1]
    df = load_dataset(file_path)
    generate_html_report(file_path, df)

if __name__ == "__main__":
    main()
