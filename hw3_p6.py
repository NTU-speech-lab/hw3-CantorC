from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, validation, normalize="true")
df_cm = pd.DataFrame(cm, range(11), range(11))
plt.figure(figsize = (10,10))
ax = sn.heatmap(df_cm, annot=True)
figure = ax.get_figure()
plt.title("Confusion matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
figure.savefig("confusion_matrix_{}.png".format(exp_id),dpi=400)