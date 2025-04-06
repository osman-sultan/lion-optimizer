import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset
from lion_pytorch import Lion


# Create a callback to track losses during training
class LossTrackingCallback(TrainerCallback):
    def __init__(self):
        self.training_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.training_losses.append(logs["loss"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

task = "mrpc"
model_name = "albert-base-v1"

dataset = load_dataset("glue", task)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(
    device
)


def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Define a compute_metrics function for accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


training_args = TrainingArguments(
    output_dir="./results",
    run_name="run1",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

# Initialize loss tracking callback for Lion
lion_callback = LossTrackingCallback()

# First training run using the Lion optimizer
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    callbacks=[lion_callback],
)

print("Training with Lion optimizer...")
trainer.train()

eval_results = trainer.evaluate()
print("Evaluation results with Lion optimizer:", eval_results)
print(f"Lion Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")

# Clear GPU memory before running the next training run
torch.cuda.empty_cache()

# Initialize loss tracking callback for Adam
adam_callback = LossTrackingCallback()

# Now training using the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    callbacks=[adam_callback],
)

print("Training with Adam optimizer...")
trainer.train()

eval_results = trainer.evaluate()
print("Evaluation results with Adam optimizer:", eval_results)
print(f"Adam Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")

# Plot the training losses
plt.figure(figsize=(10, 6))
plt.plot(lion_callback.training_losses, label="Lion")
plt.plot(adam_callback.training_losses, label="Adam")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Comparison between Lion and Adam Optimizers")
plt.legend()
plt.grid(True)
plt.savefig("optimizer_loss_comparison.png")
plt.show()
