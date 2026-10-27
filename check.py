import os
import torch
from datetime import datetime

for filename in ["gpt_best.pth", "gpt_checkpoint.pth"]:
    if os.path.exists(filename):
        modified = datetime.fromtimestamp(
            os.path.getmtime(filename)
        )
        size_mb = os.path.getsize(filename) / (1024 * 1024)

        print(f"\nFile: {filename}")
        print(f"Last modified: {modified}")
        print(f"Size: {size_mb:.2f} MB")

        if filename == "gpt_checkpoint.pth":
            checkpoint = torch.load(
                filename, map_location="cpu"
            )
            print("Saved training step:", checkpoint["step"])
            print("Best validation loss:",
                  checkpoint["best_val_loss"])
    else:
        print(f"\n{filename} not found!")