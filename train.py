import subprocess
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv("config.env")

# Step 1: Train with contrastive learning
print("Training with contrastive learning...")
subprocess.run([sys.executable, "train_contrastive.py"], check=True)

# Step 2: Fine-tune classifier on labeled data
print("Fine-tuning classifier...")
subprocess.run([sys.executable, "train_classifier.py"], check=True)

print("Training and fine-tuning complete.")
