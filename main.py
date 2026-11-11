import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, silhouette_score,
                             adjusted_rand_score)

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}

results = {}
for name, model in models.items():
    acc = cross_val_score(model, X_train, y_train, cv=5)
    results[name] = acc
    print(name, "-> mean accuracy:", round(acc.mean(), 3))

plt.boxplot(list(results.values()), labels=list(results.keys()))
plt.title("Model comparison (5-fold CV)")
plt.ylabel("accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# find a good k for knn
params = {"kneighborsclassifier__n_neighbors": range(1, 21),
          "kneighborsclassifier__weights": ["uniform", "distance"]}
grid = GridSearchCV(models["KNN"], params, cv=5)
grid.fit(X_train, y_train)
print("\nbest knn params:", grid.best_params_)

best_knn = grid.best_estimator_
preds = best_knn.predict(X_test)
print("test accuracy:", round(accuracy_score(y_test, preds), 3))
print(classification_report(y_test, preds, target_names=class_names))

ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=class_names, cmap="Blues")
plt.title("KNN confusion matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# kmeans part, no labels used here
X_scaled = StandardScaler().fit_transform(X)

k_values = range(2, 9)
inertias = []
silhouettes = []
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(k_values, inertias, marker="o")
ax[0].set_title("Elbow plot")
ax[0].set_xlabel("k")
ax[1].plot(k_values, silhouettes, marker="o", color="green")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
plt.tight_layout()
plt.savefig("kmeans_elbow_silhouette.png")
plt.close()

km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
print("\nkmeans silhouette (k=3):", round(silhouette_score(X_scaled, km3.labels_), 3))
print("adjusted rand index:", round(adjusted_rand_score(y, km3.labels_), 3))

# 2D view using PCA
points = PCA(n_components=2).fit_transform(X_scaled)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(points[:, 0], points[:, 1], c=y, cmap="viridis")
ax[0].set_title("Actual species")
ax[1].scatter(points[:, 0], points[:, 1], c=km3.labels_, cmap="viridis")
ax[1].set_title("K-means clusters")
plt.tight_layout()
plt.savefig("pca_clusters_vs_species.png")
plt.close()

print("\ndone, plots saved")
