import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn as sns
import mplcyberpunk as cyberpunk
plt.style.use("cyberpunk")

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

# Plot ROC curves for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure()
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()  

#plot presision-recall-f1  bar chat for ech class of each epoch
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
plt.figure(figsize=(10, 6))
x = np.arange(10)
width = 0.25
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1 Score')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1 Score for Each Class')
plt.xticks(x, [str(i) for i in range(10)])
plt.legend()
plt.show()





# ROC AUC (one-vs-rest)
roc_auc = roc_auc_score(y_test_cat, y_pred_probs, multi_class='ovr')
print(f"ROC AUC Score: {roc_auc:.4f}")
