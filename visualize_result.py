from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import preprocessing as pdata
import model as m

print(classification_report(pdata.y_test, m.y_pred))

# Create report as a DataFrame
report = classification_report(pdata.y_test, m.y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Drop accuracy/avg rows for per-class plot
df_per_class = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])

# Plot precision, recall, F1
plt.figure(figsize=(10,6))
sns.barplot(data=df_per_class[['precision','recall','f1-score']])
plt.title("Classification Metrics per Fast Food Company")
plt.ylabel("Score")
plt.xlabel("Company")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()

# - Confusion matrix
cm = confusion_matrix(pdata.y_test, m.y_pred, labels=m.model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=m.model.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix: Fast Food Brand Prediction")
plt.show()

# - Feature Importance
importances = pd.Series(m.model.feature_importances_, index=pdata.X_train.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance in Predicting Fast Food Company")
plt.xlabel("Importance")
plt.show()