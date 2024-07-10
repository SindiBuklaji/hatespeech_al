from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset

# Load the labeled data
labeled_data = pd.read_csv("sample_for_labeling.csv", encoding='utf-8')

# Initialize the multilingual BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# Tokenize the data
inputs = tokenizer(labeled_data['Comment'].tolist(), max_length=128, padding=True, truncation=True, return_tensors="pt")

# Create attention masks
attention_masks = inputs['attention_mask']

# Labels should be integers representing classes
labels = torch.tensor(labeled_data['Label'].values, dtype=torch.long)

# Verify the lengths of inputs and labels
print(len(inputs['input_ids']), len(labels))  # Ensure these lengths match

# Split the data into training and test sets
train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
    inputs['input_ids'], labels, attention_masks, test_size=0.2, random_state=42
)

# Define the Dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = inputs
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

# Create Dataset instances
train_dataset = HateSpeechDataset(train_inputs, train_labels, train_masks)
test_dataset = HateSpeechDataset(test_inputs, test_labels, test_masks)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

print(f"Evaluation results: {eval_result}")