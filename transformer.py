from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch

bert_model_path = "google-bert/bert-base-uncased"
fine_tuned_model_path = "./fine-tuned/checkpoint-1788" # replace with created model after training
model_output_dir = "output_model"
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

def main():
  # modify parameters to either train a new model or evaluate an existing one
  create_trainer(fine_tuned_model_path, model_output_dir, eval_only=True, enable_CUDA=True)

# Check for GPU CUDA device, print message and set device if available
def check_CUDA():
  print(torch.version.cuda)
  if torch.cuda.is_available():
      print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
      torch.cuda.set_device(0)
      return True
  else:
      print("No GPU available. Training will run on CPU.")
      return False

def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

# Create a trainer object to be either trained or only evaluated based on job data
def create_trainer(model_path, output_dir, eval_only=False, enable_CUDA=False):
  job_listing_data = load_dataset("csv", data_files="combined_text.csv")
  job_listing_data = job_listing_data["train"].train_test_split(test_size=0.2)

  tokenized_job_listing_data = job_listing_data.map(preprocess_function, batched=True)

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # define classifiers
  id2label = {0: "Real", 1: "Fraudulent"}
  label2id = {"Real": 0, "Fraudulent": 1}

  model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=2, id2label=id2label, label2id=label2id
  )
  # if CUDA available, switch training device
  if (enable_CUDA and check_CUDA()):
    model.to('cuda')

  training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    push_to_hub=False,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_job_listing_data["train"],
    eval_dataset=tokenized_job_listing_data["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
  )

  # skip training if in eval only mode
  if (eval_only):
    trainer.evaluate()
  else:
    trainer.train()
  
# Display confusion matrix and classification report based on test split
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    confusion = confusion_matrix(labels, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Real", "Fraudulent"])
    display.plot()
    plt.show()
    print(classification_report(labels, predictions, target_names = ['Real','Fraudulent']))
    return accuracy.compute(predictions=predictions, references=labels)

main()