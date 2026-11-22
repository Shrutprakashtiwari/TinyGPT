from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized"
)

print(dataset)
print(dataset["train_prefs"][0])
dataset.save_to_disk("./ultrafeedback_data")
