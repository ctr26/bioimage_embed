import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import json

pd.set_option('display.max_colwidth', None)

df = pd.read_csv("clustered_data.csv")

df.insert(0, 'label', df['fname'].str.extract(r'^(?:[^/]*/){7}([^/]*)').squeeze())
df.insert(0, 'n_label', df['label'].apply(lambda x: 0 if x == 'alive' else 1))

new_df = df.iloc[:, :-4]

y = new_df.iloc[:, 0] 
X = new_df.iloc[:, 2:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_and_evaluate_model(clf, X_train, y_train, X_test, y_test):
    model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, whiten=True, random_state=42)),
                ("clf", clf),
            ]
        )

    pipeline = model.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"Classification score: {score}")

    y_pred = pipeline.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Cross-validation
    cv_results = cross_validate(pipeline, X, y, cv=5)
    print("Cross-validation results:")
    print(cv_results)

    # Plot and save the confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix for {clf.__class__.__name__}')
    plt.savefig(f'confusion_matrix_{clf.__class__.__name__}.png')
    plt.clf()  # Clear the current figure

    return score, cm, cv_results

classifiers = [RandomForestClassifier(), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), svm.SVC()]

results = []

for clf in classifiers:
    score, cm, cv_results = build_and_evaluate_model(clf, X_train, y_train, X_test, y_test)
    results.append((clf.__class__.__name__, score, cm, cv_results))

known_labels = list(y[:50])
unknown_labels = [-1]*len(y[50:])
partial_labels = known_labels + unknown_labels

reducer = umap.UMAP()
embedding = reducer.fit_transform(X, y=partial_labels)

plt.scatter(embedding[:, 0], embedding[:, 1], c=partial_labels, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the dataset', fontsize=24)

plt.savefig('umap_visualization.png')
plt.clf()  # Clear the current figure

# Generate LaTeX report
with open('final_report.tex', 'w') as f:
    f.write("\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{longtable}\n\\usepackage{listings}\n\\begin{document}\n")
    for name, score, cm, cv_results in results:
        f.write(f"\\section*{{Results for {name}}}\n")
        f.write("\\begin{longtable}{|l|l|}\n")
        f.write("\\hline\n")
        f.write(f"Classification Score & {score} \\\\\n")
        f.write("\\hline\n")
        f.write("Confusion Matrix & \\\\\n")
        f.write("\\begin{lstlisting}\n")
        f.write(np.array2string(cm).replace('\n', ' \\\\\n'))
        f.write("\\end{lstlisting}\n")
        f.write("\\hline\n")
        f.write("Cross-validation Results & \\\\\n")
        f.write("\\begin{lstlisting}\n")
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df = cv_results_df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        f.write(cv_results_df.to_string().replace('\n', ' \\\\\n'))
        f.write("\\end{lstlisting}\n")
        f.write("\\hline\n")
        f.write("\\end{longtable}\n")
    f.write("\\section*{UMAP visualization}\n")
    f.write("\\includegraphics[width=\\textwidth]{umap_visualization.png}\n")
    f.write("\\end{document}\n")

os.system('pdflatex final_report.tex')

# Generate CSV report
report_df = pd.DataFrame(results, columns=['Classifier', 'Score', 'Confusion Matrix', 'Cross-validation Results'])
report_df['Cross-validation Results'] = report_df['Cross-validation Results'].apply(lambda x: pd.DataFrame(x).applymap(lambda y: y.tolist() if isinstance(y, np.ndarray) else y).to_dict())
report_df.to_csv('final_report.csv', index=False)
