import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", show=True, save_path=None):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[OK] Saved confusion matrix to {save_path}")
    if show:
        plt.show()
