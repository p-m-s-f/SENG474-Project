import csv
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent
cache = torch.load(
    project_root / "outputs" / "development_features.pt",
    map_location="cpu",
    weights_only=True,
)

row_by_filename = {
    filename: index for index, filename in enumerate(cache["filenames"])
}

if len(row_by_filename) != len(cache["filenames"]):
    raise ValueError("Duplicate filenames in the feature cache")


def load_fold_split(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    indices = [row_by_filename[record["filename"]] for record in records]
    labels = [record["label"] for record in records]

    if labels != [cache["labels"][index] for index in indices]:
        raise ValueError(f"Feature labels do not match {csv_path}")

    return cache["features"][indices], labels


class_names = ["no_fire", "fire_low", "fire_medium", "fire_high"]
label_to_id = {label: index for index, label in enumerate(class_names)}


def macro_f1_score(actual_ids, predictions):
    scores = []

    for class_id in range(len(class_names)):
        actual = actual_ids == class_id
        predicted = predictions == class_id
        true_positives = (actual & predicted).sum().item()
        denominator = actual.sum().item() + predicted.sum().item()
        scores.append(2 * true_positives / denominator if denominator else 0.0)

    return sum(scores) / len(scores)

def logistic_predict(train_scaled, train_ids, val_scaled, l2_penalty):
    torch.manual_seed(0)
    model = torch.nn.Linear(train_scaled.shape[1], len(class_names))
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(train_scaled), train_ids)
        loss = loss + (l2_penalty / 2) * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        return model(val_scaled).argmax(dim=1)

scores_by_k = {3: [], 5: [], 9: []}

scores_by_l2 = {0.01: [], 0.1: [], 1.0: []}



for fold in (1, 2, 3):
    fold_dir = project_root / "splits" / "cv" / f"fold_{fold}"
    train_features, train_labels = load_fold_split(fold_dir / "train.csv")
    val_features, val_labels = load_fold_split(fold_dir / "val.csv")

    train_ids = torch.tensor([label_to_id[label] for label in train_labels])
    val_ids = torch.tensor([label_to_id[label] for label in val_labels])

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False)
    std[std < 1e-6] = 1.0

    train_scaled = (train_features - mean) / std
    val_scaled = (val_features - mean) / std

    for l2, scores in scores_by_l2.items():
        predictions = logistic_predict(train_scaled, train_ids, val_scaled, l2)
        score = macro_f1_score(val_ids, predictions)
        scores.append(score)
        print(f"Fold {fold}, logistic regression (L2={l2}): {score:.2%} macro-F1")

    distances = torch.cdist(val_scaled, train_scaled)
    nearest = distances.argsort(dim=1, stable=True)


    for k, scores in scores_by_k.items():
        neighbors = nearest[:, :k]
        votes = torch.nn.functional.one_hot(
            train_ids[neighbors], num_classes=len(class_names)
        )
        predictions = votes.sum(dim=1).argmax(dim=1)
        score = macro_f1_score(val_ids, predictions)
        scores.append(score)
        print(f"Fold {fold}, k={k}: {score:.2%} macro-F1")

mean_by_k = {k: sum(scores) / len(scores) for k, scores in scores_by_k.items()}

for k, mean in mean_by_k.items():
    print(f"k={k}: {mean:.2%} mean macro-F1")

best_k = max(mean_by_k, key=mean_by_k.get)
print(f"Selected k: {best_k}")

mean_by_l2 = {
    l2: sum(scores) / len(scores)
    for l2, scores in scores_by_l2.items()
}

for l2, mean in mean_by_l2.items():
    print(f"L2={l2}: {mean:.2%} mean macro-F1")

best_l2 = max(mean_by_l2, key=mean_by_l2.get)
print(f"Selected L2: {best_l2}")

test_cache = torch.load(
    project_root / "outputs" / "test_features.pt",
    map_location="cpu",
    weights_only=True,
)

if set(cache["filenames"]) & set(test_cache["filenames"]):
    raise ValueError("Development and test feature caches overlap")

development_features = cache["features"]
development_ids = torch.tensor([label_to_id[label] for label in cache["labels"]])
test_ids = torch.tensor([label_to_id[label] for label in test_cache["labels"]])

mean = development_features.mean(dim=0)
std = development_features.std(dim=0, unbiased=False)
std[std < 1e-6] = 1.0

development_scaled = (development_features - mean) / std
test_scaled = (test_cache["features"] - mean) / std

logistic_predictions = logistic_predict(
    development_scaled, development_ids, test_scaled, best_l2
)

distances = torch.cdist(test_scaled, development_scaled)
neighbors = distances.argsort(dim=1, stable=True)[:, :best_k]
votes = torch.nn.functional.one_hot(
    development_ids[neighbors], num_classes=len(class_names)
)
knn_predictions = votes.sum(dim=1).argmax(dim=1)

for name, predictions in [
    (f"Logistic regression (L2={best_l2})", logistic_predictions),
    (f"k-NN (k={best_k})", knn_predictions),
]:
    accuracy = (predictions == test_ids).float().mean().item()
    macro_f1 = macro_f1_score(test_ids, predictions)
    print(f"{name}: test accuracy {accuracy:.2%}, test macro-F1 {macro_f1:.2%}")