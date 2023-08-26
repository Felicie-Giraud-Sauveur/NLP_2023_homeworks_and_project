import os
from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines

import torch
import torch.utils.data
from torch import nn, optim
import numpy as np 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup


# 1.1
class NLIDataset(Dataset):
    """
    Implement NLIDataset in Pytorch
    """
    def __init__(self, data_repo, tokenizer, sent_max_length=128):
        
        self.label_to_id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self.id_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        self.tokenizer = tokenizer
        
        # Get the special token and token id for PAD from defined tokenizer (self.tokenizer)
        self.pad_token = self.tokenizer.pad_token
        self.pad_id = self.tokenizer.pad_token_id

        self.text_samples = []
        self.samples = []
        
        print("Building NLI Dataset...")
        
        with jsonlines.open(data_repo, "r") as reader:

            for sample in tqdm(reader.iter()):
                
                self.text_samples.append(sample)

                # Build input token indices(input_ids):
                    # Get split tokens (subtokens), truncate each list of tokens if it exceeds the sent_max_length and map each text token to id
                p_ids = tokenizer(sample["premise"], max_length=sent_max_length, truncation=True)['input_ids'][1:-1]
                h_ids = tokenizer(sample["hypothesis"], max_length=sent_max_length, truncation=True)['input_ids'][1:-1]  
                    # Combine hypothesis and premise sentences with (self.tokenizer.build_inputs_with_special_tokens)
                input_ids = self.tokenizer.build_inputs_with_special_tokens(p_ids, h_ids)

                label = self.label_to_id.get(sample["label"], None)
                self.samples.append({"ids": input_ids, "label": label})

                
    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, index):
        return deepcopy(self.samples[index])
    

    def padding(self, inputs, max_length=-1):
        """
        Pad inputs to the max_length.
        
        INPUT: 
          - inputs: input token ids
          - max_length: the maximum length you should add padding to.
          
        OUTPUT: 
          - pad_inputs: token ids padded to `max_length` """

        if max_length < 0:
            max_length = max(list(map(len, inputs)))
        
        pad_inputs = []
        # Padding
        for input_ in inputs:
            pad_inputs.append(input_ + [self.pad_id]*(max_length-len(input_)))
        return pad_inputs
    
        
    def collate_fn(self, batch):
        """
        Convert batch inputs to tensor of batch_ids and labels.
        
        INPUT: 
          - batch: batch input, with format List[Dict1{"ids":..., "label":...}, Dict2{...}, ..., DictN{...}]
          
        OUTPUT: 
          - tensor_batch_ids: torch tensor of token ids of a batch, with format Tensor(List[ids1, ids2, ..., idsN])
          - tensor_labels: torch tensor for corresponding labels, with format Tensor(List[label1, label2, ..., labelN])
        """
        # collabte_fn for batchify input into preferable format
        
        batch_ids = []
        batch_labels = []
        
        for x in batch:
            batch_ids.append(x["ids"])
            batch_labels.append(x["label"])
        
        tensor_batch_ids = torch.tensor(self.padding(batch_ids))
        tensor_labels = torch.tensor(batch_labels).long()

        return tensor_batch_ids, tensor_labels
    
    
    def get_text_sample(self, index):
        return deepcopy(self.text_samples[index])
    
    
    def decode_class(self, class_ids):
        """
        Decode to output the predicted class name.
        
        INPUT: 
          - class_ids: index of each class.
          
        OUTPUT: 
          - labels_from_ids: a list of label names. """
        
        # Class decoding function
        label_name_list = []
        for id_ in class_ids:
            label_name_list.append(self.id_to_label[id_])
        
        return label_name_list

    
    
# 1.2
def compute_metrics(predictions, gold_labels):
    """
    Compute evaluation metrics (accuracy and F1 score) for NLI task.
    
    INPUT: 
      - gold_labels: real labels;
      - predictions: model predictions.
    OUTPUT: 4 float scores
      - accuracy score (float);
      - f1 score for each class (3 classes in total).
    """
    # Metrics computation
    
    # Calculate accuracy
    acc = len([i for i, j in zip(gold_labels, predictions) if i==j]) / float(len(gold_labels))
    
    # F1 score for class0
    tp0 = len([i for i, j in zip([c==0 for c in gold_labels], [c==0 for c in predictions]) if (i==j & i==True)])
    fp0 = len([i for i, j in zip([((c==1) | (c==2)) for c in gold_labels], [c==0 for c in predictions]) if (i==j & i==True)])
    fn0 = len([i for i, j in zip([c==0 for c in gold_labels], [((c==1) | (c==2)) for c in predictions]) if (i==j & i==True)])
    if (tp0+fp0!=0) & (tp0+fn0!=0):
        precision0 = tp0/(tp0+fp0)
        recall0 = tp0/(tp0+fn0)
        f10 = (2*precision0*recall0)/(precision0+recall0)
    else:
        print("Division by 0 for class0")
        f10 = 0
    
    # F1 score for class1
    tp1 = len([i for i, j in zip([c==1 for c in gold_labels], [c==1 for c in predictions]) if (i==j & i==True)])
    fp1 = len([i for i, j in zip([((c==0) | (c==2)) for c in gold_labels], [c==1 for c in predictions]) if (i==j & i==True)])
    fn1 = len([i for i, j in zip([c==1 for c in gold_labels], [((c==0) | (c==2)) for c in predictions]) if (i==j & i==True)])
    if (tp1+fp1!=0) & (tp1+fn1!=0):
        precision1 = tp1/(tp1+fp1)
        recall1 = tp1/(tp1+fn1)
        f11 = (2*precision1*recall1)/(precision1+recall1)
    else:
        print("Division by 0 for class1")
        f11 = 0

    # F1 score for class2
    tp2 = len([i for i, j in zip([c==2 for c in gold_labels], [c==2 for c in predictions]) if (i==j & i==True)])
    fp2 = len([i for i, j in zip([((c==0) | (c==1)) for c in gold_labels], [c==2 for c in predictions]) if (i==j & i==True)])
    fn2 = len([i for i, j in zip([c==2 for c in gold_labels], [((c==0) | (c==1)) for c in predictions]) if (i==j & i==True)])
    if (tp2+fp2!=0) & (tp2+fn2!=0):
        precision2 = tp2/(tp2+fp2)
        recall2 = tp2/(tp2+fn2)
        f12 = (2*precision2*recall2)/(precision2+recall2)
    else:
        print("Division by 0 for class2")
        f12 = 0
    
    f1 = [f10, f11, f12]
    
    return acc, f1[0], f1[1], f1[2]


def train(train_dataset, dev_dataset, model, device, batch_size, epochs,
          learning_rate, warmup_percent, max_grad_norm, model_save_root):
    '''
    Train models with predefined datasets.

    INPUT:
      - train_dataset: dataset for training
      - dev_dataset: dataset for evaluation
      - model: model to train
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - epochs: total epochs to train the model
      - learning_rate: learning rate of optimizer
      - warmup_percent: percentage of warmup steps
      - max_grad_norm: maximum gradient for clipping
      - model_save_root: path to save model checkpoints
    '''
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn
    )
    
    # Define optimizer and learning rate scheduler with learning rate and warmup steps
    
        # calculate total training steps (epochs * number of data batches per epoch)
    total_steps = epochs*batch_size
    warmup_steps = int(warmup_percent*total_steps)
    
        # set up AdamW optimizer and constant learning rate scheduleer with warmup (get_constant_schedule_with_warmup)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps)
    
    model.zero_grad()
    model.train()
    best_dev_macro_f1 = 0
    save_repo = model_save_root + 'lr{}-warmup{}'.format(learning_rate, warmup_percent)
    
    for epoch in range(epochs):
        
        train_loss_accum = 0
        epoch_train_step = 0
        
        # Training process: calculate the loss then update the model with optimizer
        # Also keep track on the training step and update the learning rate scheduler

        for batch in tqdm(train_dataloader, desc="Training"):
            
            # Set the gradients of all optimized parameters to zero
            optimizer.zero_grad()

            epoch_train_step += 1

            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            # get model's single-batch outputs and loss
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            # conduct back-propagation
            loss.backward()

            # truncate gradient to max_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            train_loss_accum += loss.mean().item()

            # step forward optimizer and scheduler
            optimizer.step()
            scheduler.step()
        
        epoch_train_loss = train_loss_accum / epoch_train_step

        # epoch evaluation
        dev_loss, acc, f1_ent, f1_neu, f1_con = evaluate(dev_dataset, model, device, batch_size)
        macro_f1 = (f1_ent + f1_neu + f1_con)/3
        
        print(f'Epoch: {epoch} | Training Loss: {epoch_train_loss:.3f} | Validation Loss: {dev_loss:.3f}')
        print(f'Epoch {epoch} NLI Validation:')
        print(f'Accuracy: {acc*100:.2f}% | F1: ({f1_ent*100:.2f}%, {f1_neu*100:.2f}%, {f1_con*100:.2f}%) | Macro-F1: {macro_f1*100:.2f}%')
        
        # Update the highest macro_f1. Save best model and tokenizer to <save_repo>
        if macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = macro_f1
            model.save_pretrained(save_repo)
            train_dataset.tokenizer.save_pretrained(save_repo)
            print("Model Saved!")


def evaluate(eval_dataset, model, device, batch_size, no_labels=False, result_save_file=None):
    '''
    Evaluate the trained model.

    INPUT: 
      - eval_dataset: dataset for evaluation
      - model: trained model
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - no_labels: whether the labels should be used as one input to the model
      - result_save_file: path to save the prediction results
    '''
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    
    eval_loss_accum = 0
    eval_step = 0
    batch_preds = []
    batch_labels = []
    
    model.eval()
    
    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        
        eval_step += 1
        
        with torch.no_grad():
            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            # Get model outputs, loss and logits
            if not no_labels:
                outputs = model(input_ids, labels=labels)
            else:
                outputs = model(input_ids)
            
            loss = outputs.loss
            logits = outputs.logits
            
            batch_preds.append(logits.detach().cpu().numpy())
            
            if not no_labels:
                batch_labels.append(labels.detach().cpu().numpy())
                eval_loss_accum += loss.mean().item()

    # Get model predicted labels
    pred_labels = []
    for batch in batch_preds:
        for logits in batch:
            pred_labels.append(logits.argmax().item())
    
    if result_save_file:
        if not os.path.exists(os.path.dirname(result_save_file)):
            os.makedirs(os.path.dirname(result_save_file))
        pred_results = eval_dataset.decode_class(pred_labels)
        with jsonlines.open(result_save_file, mode="w") as writer:
            for sid, pred in enumerate(pred_results):
                sample = eval_dataset.get_text_sample(sid)
                sample["prediction"] = pred
                writer.write(sample)
    
    if not no_labels:
        eval_loss = eval_loss_accum / eval_step
        gold_labels = list(np.concatenate(batch_labels))
        acc, f1_ent, f1_neu, f1_con = compute_metrics(pred_labels, gold_labels)
        return eval_loss, acc, f1_ent, f1_neu, f1_con
    else:
        return None
