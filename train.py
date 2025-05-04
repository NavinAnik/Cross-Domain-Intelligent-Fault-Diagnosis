import subprocess
import sys
from utils.utils import cyber_prank
import os
from dotenv import load_dotenv
load_dotenv("config.env")

#create a directory for the models
if not os.path.exists("model"):
    os.makedirs("model")
if not os.path.exists("results"):
    os.makedirs("results")

# Step 1: Train with contrastive learning
print("Training with contrastive learning...")
subprocess.run([sys.executable, "train_contrastive.py"], check=True)

# Step 2: Fine-tune classifier on labeled data
print("Fine-tuning classifier...")
subprocess.run([sys.executable, "train_classifier.py"], check=True)
cyber_prank()
print("Training and fine-tuning complete.")
