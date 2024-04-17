from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
import torch
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def preprocess_data(sentences, labels, tokenizer, max_len=256):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

with open("S_P_Wiki/test1/positive_train.txt", "r") as file:
    positive_train_sentences = file.readlines()
    positive_train_labels = [1] * len(positive_train_sentences)
with open("S_P_Wiki/test1/negative_train.txt", "r") as file:
    negative_train_sentences = file.readlines()
    negative_train_labels = [0] * len(negative_train_sentences)
with open("S_P_Wiki/test1/positive_test.txt", "r") as file:
    positive_test_sentences = file.readlines()
    positive_test_labels = [1] * len(positive_test_sentences)
with open("S_P_Wiki/test1/negative_test.txt", "r") as file:
    negative_test_sentences = file.readlines()
    negative_test_labels = [1] * len(negative_test_sentences)


train_sentences = positive_train_sentences + negative_train_sentences
train_labels = positive_train_labels + negative_train_labels

test_sentences = positive_test_sentences + negative_test_sentences
test_labels = positive_test_labels + negative_test_labels

train_dataset = preprocess_data(train_sentences, train_labels, tokenizer)
test_dataset = preprocess_data(test_sentences, test_labels, tokenizer)

# Create DataLoaders
batch_size = 16  # You can adjust the batch size
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
epochs = 3  # Number of training epochs. Recommended range: 2-4
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.train()

# Training loop
for epoch in range(epochs):
    print(f'======== Epoch {epoch + 1} / {epochs} ========')
    total_loss = 0

    for batch in tqdm(train_dataloader, desc="Iteration"):
        batch = tuple(t.to(device) for t in batch)  # Move batch to GPU
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()        

        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

        logits = outputs.logits
        loss = torch.nn.CrossEntropyLoss()(logits, b_labels)
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients
        loss.backward()

        # Clip the norm of the gradients to 1.0 to help prevent the "exploding gradients" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update the learning rate
        scheduler.step()

    # Calculate the average loss over the training data
    avg_train_loss = total_loss / len(train_dataloader)            
    
    print(f"Average train loss: {avg_train_loss}")

print("Training complete.")

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Testing
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
  
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

# Use the predictions and true labels to calculate the accuracy or other metrics
from sklearn.metrics import classification_report

# Flatten the predictions and true values for entire dataset
flat_predictions = np.concatenate(predictions, axis=0)
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_predictions = torch.argmax(torch.tensor(flat_predictions), dim=1).flatten().numpy()
flat_true_labels = np.concatenate(true_labels, axis=0)

print(classification_report(flat_true_labels, flat_predictions, target_names=['Negative', 'Positive']))

config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2
config.save_pretrained("Replication-Model")

model.save_pretrained("Replication-Model")
tokenizer.save_pretrained("Replication-Model")