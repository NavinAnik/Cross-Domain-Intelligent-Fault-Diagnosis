import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Load the trained model
model = tf.keras.models.load_model("base_encoder.h5")
classifier = tf.keras.models.load_model("classifier_model.h5")

# Predict and evaluate
y_pred_probs = classifier.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ROC AUC (one-vs-rest)
roc_auc = roc_auc_score(y_test_cat, y_pred_probs, multi_class='ovr')
print(f"ROC AUC Score: {roc_auc:.4f}")
