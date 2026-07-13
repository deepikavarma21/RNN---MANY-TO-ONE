"""
RNN (LSTM) classifier: predicts student performance category (High / Medium / Low)
from academic + demographic data.

Dataset: StudentsPerformance.csv (gender, race/ethnicity, parental level of education,
lunch, test preparation course, math score, reading score, writing score)

Design choice:
- The three subject scores (math, reading, writing) are treated as a 3-step sequence
  fed into a recurrent layer (SimpleRNN/LSTM), which is what justifies using an RNN
  at all here rather than a plain feed-forward net.
- Demographic features are one-hot encoded and concatenated with the RNN's output
  before the final classification head.
- Target label (High/Medium/Low) is derived from the average of the three scores.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("StudentsPerformance.csv")
df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]
print("Columns:", list(df.columns))
print(df.head())

# ---------------------------------------------------------------------------
# 2. Feature engineering: build the target label from average score
# ---------------------------------------------------------------------------
score_cols = ["math_score", "reading_score", "writing_score"]
df["average_score"] = df[score_cols].mean(axis=1)

def categorize(avg):
    if avg >= 80:
        return "High"
    elif avg >= 60:
        return "Medium"
    else:
        return "Low"

df["final_grade"] = df["average_score"].apply(categorize)
print("\nClass distribution:\n", df["final_grade"].value_counts())

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
categorical_cols = ["gender", "race_ethnicity", "parental_level_of_education",
                     "lunch", "test_preparation_course"]

# Sequence features (the "time steps" fed to the RNN)
seq_features = df[score_cols].values.astype("float32")
seq_scaler = StandardScaler()
seq_features_scaled = seq_scaler.fit_transform(seq_features)
# reshape to (samples, timesteps=3, features_per_step=1)
seq_features_scaled = seq_features_scaled.reshape(-1, 3, 1)

# Static/context features (demographics), one-hot encoded
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
static_features = ohe.fit_transform(df[categorical_cols])

# Target
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(df["final_grade"])   # High/Low/Medium -> 0,1,2
class_names = label_encoder.classes_
y_cat = to_categorical(y_int)

print("\nLabel mapping:", dict(zip(class_names, range(len(class_names)))))

# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------
X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test, y_int_train, y_int_test = \
    train_test_split(
        seq_features_scaled, static_features, y_cat, y_int,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_int
    )

print(f"\nTrain samples: {X_seq_train.shape[0]}  Test samples: {X_seq_test.shape[0]}")

# ---------------------------------------------------------------------------
# 5. Build the RNN model (functional API: sequence branch + static branch)
# ---------------------------------------------------------------------------
n_timesteps = X_seq_train.shape[1]
n_seq_feat = X_seq_train.shape[2]
n_static_feat = X_static_train.shape[1]
n_classes = y_cat.shape[1]

seq_input = Input(shape=(n_timesteps, n_seq_feat), name="score_sequence")
x = LSTM(32, return_sequences=True)(seq_input)
x = LSTM(16)(x)
x = Dropout(0.3)(x)

static_input = Input(shape=(n_static_feat,), name="demographics")
s = Dense(16, activation="relu")(static_input)
s = Dropout(0.3)(s)

merged = Concatenate()([x, s])
h = Dense(32, activation="relu")(merged)
h = Dropout(0.2)(h)
output = Dense(n_classes, activation="softmax")(h)

model = Model(inputs=[seq_input, static_input], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    [X_seq_train, X_static_train], y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=2
)

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate([X_seq_test, X_static_test], y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict([X_seq_test, X_static_test])
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
report = classification_report(y_int_test, y_pred, target_names=class_names, digits=3)
print(report)

precision, recall, f1, support = precision_recall_fscore_support(
    y_int_test, y_pred, average="weighted"
)
print(f"Weighted Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

cm = confusion_matrix(y_int_test, y_pred)

# ---------------------------------------------------------------------------
# 8. Save plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# training curves
axes[0].plot(history.history["accuracy"], label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[2])
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("\nSaved plot: training_results.png")

# Save the trained model and the classification report
model.save("student_performance_rnn.keras")
with open("classification_report.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

print("\nSaved model: student_performance_rnn.keras")
print("Saved report: classification_report.txt")
