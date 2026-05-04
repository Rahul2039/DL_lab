# -------------------------------
# Fix for OpenMP Error (IMPORTANT)
# -------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------
# Libraries
# -------------------------------
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Input Sentence
# -------------------------------
sentence = ["I", "love", "deep", "learning"]

# -------------------------------
# 2. Attention Scores
# -------------------------------
attention_scores = torch.tensor([0.1, 0.3, 0.4, 0.2])

# -------------------------------
# 3. Softmax Normalization
# -------------------------------
attention_weights = F.softmax(attention_scores, dim=0).detach().numpy()

print("Attention Weights:", attention_weights)

# -------------------------------
# 4. Heatmap Visualization
# -------------------------------
plt.figure(figsize=(8, 2))

sns.heatmap(
    [attention_weights],
    annot=True,
    cmap="Blues",
    xticklabels=sentence,
    yticklabels=["Attention"]
)

plt.title("Attention Heatmap Visualization")
plt.xlabel("Words")
plt.tight_layout()
plt.show()
