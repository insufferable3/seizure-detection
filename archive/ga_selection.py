import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

# ==============================
# 1. LOAD DATA
# ==============================

df = pd.read_csv("features.csv")

feature_cols = [
    col for col in df.columns
    if col not in ["file", "start", "end", "label"]
]

X = df[feature_cols].values
y = df["label"].values

print("Total features:", len(feature_cols))
print("Dataset:", X.shape)

# ==============================
# 2. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 3. FITNESS FUNCTION
# ==============================

def fitness(chromosome):

    selected = np.where(chromosome == 1)[0]

    # Don't allow empty feature sets
    if len(selected) == 0:
        return 0

    X_selected = X_train[:, selected]

    model = RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_selected, y_train)

    pred = model.predict(X_selected)

    f1 = f1_score(y_train, pred, zero_division=0)
    recall = recall_score(y_train, pred, zero_division=0)

    # Reward both F1 and seizure recall
    score = 0.5 * f1 + 0.5 * recall

    return score


# ==============================
# 4. CREATE INITIAL POPULATION
# ==============================

population_size = 20
generations = 10

population = np.random.randint(
    0,
    2,
    size=(population_size, len(feature_cols))
)

# ==============================
# 5. GENETIC ALGORITHM
# ==============================

for generation in range(generations):

    scores = []

    for chromosome in population:
        score = fitness(chromosome)
        scores.append(score)

    scores = np.array(scores)

    # Best chromosome
    best_index = np.argmax(scores)
    best_score = scores[best_index]
    best_chromosome = population[best_index].copy()

    selected_count = np.sum(best_chromosome)

    print(
        f"Generation {generation + 1}/{generations} | "
        f"Fitness: {best_score:.4f} | "
        f"Features: {selected_count}"
    )

    # ==========================
    # SELECTION
    # ==========================

    top_indices = np.argsort(scores)[-5:]

    parents = population[top_indices]

    # ==========================
    # NEW POPULATION
    # ==========================

    new_population = []

    # Keep best solution
    new_population.append(best_chromosome)

    while len(new_population) < population_size:

        parent1 = parents[np.random.randint(len(parents))]
        parent2 = parents[np.random.randint(len(parents))]

        # Crossover
        point = np.random.randint(1, len(feature_cols) - 1)

        child = np.concatenate([
            parent1[:point],
            parent2[point:]
        ])

        # Mutation
        mutation_rate = 0.02

        mutation = np.random.rand(len(feature_cols)) < mutation_rate

        child[mutation] = 1 - child[mutation]

        new_population.append(child)

    population = np.array(new_population)


# ==============================
# 6. FINAL BEST SOLUTION
# ==============================

final_scores = np.array([
    fitness(chromosome)
    for chromosome in population
])

best_index = np.argmax(final_scores)

best_chromosome = population[best_index]

selected_indices = np.where(best_chromosome == 1)[0]

selected_features = [
    feature_cols[i]
    for i in selected_indices
]

print("\n==============================")
print("GA COMPLETE")
print("==============================")

print("Best fitness:", final_scores[best_index])
print("Selected features:", len(selected_features))

print("\nSelected features:")

for feature in selected_features:
    print(feature)

# Save selected feature names
pd.DataFrame({
    "feature": selected_features
}).to_csv(
    "selected_features.csv",
    index=False
)

print("\nSaved as selected_features.csv")