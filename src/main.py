from pathlib import Path
import argparse
import time
import csv
import random
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import AdamW


# parse arguments
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--mode",
                    required=True,
                    help="Mode can be one of (train, continue_train, test, test_without_train)")

# Optional parameters
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--t5_model",
                    default="t5-base",
                    type=str,
                    help="T5 Transformer pre-trained model")
parser.add_argument("--cache_dir",
                    default="./cache/",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--output_dir",
                    default="./out/",
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--train_batch_size",
                    default=16,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--num_train_epochs",
                    default=10,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--start_epoch",
                    default=1,
                    type=int,
                    help="When do continue training, start from start_epoch.")
# optimization hyperparameters
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                    help="Epsilon for Adam optimizer.")
# parser.add_argument("--warmup_steps", default=0, type=int,
#                     help="Linear warmup over warmup_steps.")
# set random seed for all packages
parser.add_argument('--seed',
                    type=int,
                    default=2020,
                    help="random seed for initialization")

args = parser.parse_args()

# make output directory and cache directory
Path(args.output_dir).mkdir(exist_ok=True)
Path(args.cache_dir).mkdir(exist_ok=True)

# Save file names for both training and testing
output_model_file = Path.cwd() / args.output_dir / "model.bin"
output_config_file = Path.cwd() / args.output_dir / 'config.bin'

# device information
device = torch.device("cuda")

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_tokenizer():
  tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
  # tokenizer.add_tokens(['<choice1>', '<choice2>', '<choice3>', '<choice4>', '<choice5>'])
  return tokenizer


def get_model(tokenizer_len=None):
  if args.mode == 'train' or args.mode == 'test_without_train':
    model = T5ForConditionalGeneration.from_pretrained(
        args.t5_model, cache_dir=args.cache_dir)
    if tokenizer_len is not None:
      model.resize_token_embeddings(tokenizer_len)
  elif args.mode == 'test' or args.mode == 'continue_train':
    model = T5ForConditionalGeneration(
        T5Config.from_json_file(output_config_file))
    model.load_state_dict(torch.load(output_model_file))
  else:
    raise NotImplementedError(
        f'No such mode called {args.mode}, error raised from get_model.')

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  return model.to(device)


def get_optimizer(model):
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(
          nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
      {'params': [p for n, p in model.named_parameters() if any(
          nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters,
                    lr=args.learning_rate, eps=args.adam_epsilon)
  return optimizer


def convert_string_to_ids(tokenizer, strings):
  token_info = tokenizer.batch_encode_plus(
      strings,
      max_length=args.max_seq_length,
      pad_to_max_length=True,
      return_attention_masks=True,
      return_tensors='pt')
  return token_info['input_ids'], token_info['attention_mask']


def load_dataset(train_test_dev, tokenizer):
  input_texts, target_texts = [], []
  path = Path('data4T5/') / (train_test_dev + '.jsonl')
  with open(path, 'r') as f:
    for line in f:
      d = json.loads(line)
      input_texts.append(d['input_text'])
      target_texts.append(d['target_text'])

  input_ids, input_attention_mask = convert_string_to_ids(
      tokenizer, input_texts)
  target_ids, target_attention_mask = convert_string_to_ids(
      tokenizer, target_texts)

  # mask target_ids to lm_labels (-100 are ignored)
  target_attention_mask = torch.ones_like(
      target_attention_mask, dtype=torch.long) - target_attention_mask
  target_ids -= 100 * target_attention_mask
  dataset = TensorDataset(input_ids, input_attention_mask, target_ids)
  if train_test_dev == 'train':
    return DataLoader(
        dataset, shuffle=True, batch_size=args.train_batch_size, pin_memory=True, num_workers=4)
  else:
    return DataLoader(
        dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=4)


# def save_results_to_tsv(train_or_test, all_doc_id, all_inputs, all_labels, all_preds):
#   label_map_invert = {value: key for key, value in label_map.items()}
#   with open(Path.cwd() / args.output_dir / (train_or_test + '_result.tsv'), 'w') as file:
#     writer = csv.writer(file, delimiter='\t')
#     writer.writerow(['DocID', 'Example', 'Ground_truth', 'Prediction'])

#     for doc_id, inputs, labels, preds in zip(all_doc_id, all_inputs, all_labels, all_preds):
#       writer.writerow(
#           [doc_id, inputs, label_map_invert[labels], label_map_invert[preds]])

def compare_output(target, output):
  n_correct = torch.eq(target[:, 1], output[:, 2]).sum().item()
  return len(target), n_correct


def main():
  # Get tokenizer
  tokenizer = get_tokenizer()

  # Create model
  model = get_model()

  # Training phase
  if args.mode == 'train' or args.mode == 'continue_train':
    # Prepare train set
    train_dataloader = load_dataset('train', tokenizer)
    dev_dataloader = load_dataset('dev', tokenizer)

    # Prepare optimizer
    optimizer = get_optimizer(model)

    # Start training
    print('Start training...')

    for epoch in range(args.num_train_epochs):
      model.train()
      # Train for one epoch, and evaluate later
      train_loss = 0

      for input_ids, attention_mask, output_ids in train_dataloader:
        outputs = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        lm_labels=output_ids.to(device))

        loss = outputs[0].mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

      # eval on dev set
      model.eval()
      eval_loss = 0
      with torch.no_grad():
        # eval on dev set
        n_totals, n_corrects = 0, 0
        for input_ids, attention_mask, output_ids in dev_dataloader:
          outputs = model(input_ids=input_ids.to(device),
                          attention_mask=attention_mask.to(device),
                          lm_labels=output_ids.to(device))

          loss = outputs[0].mean()
          eval_loss += loss.item()

          outputs = model.module.generate(input_ids.to(
              device), attention_mask=attention_mask.to(device)).cpu()
          n_total, n_correct = compare_output(target_ids, outputs)
          n_totals += n_total
          n_corrects += n_correct

        # do the same for training set
        n_totals_train, n_corrects_train = 0, 0
        for input_ids, attention_mask, target_ids in train_dataloader:
          outputs = model.module.generate(input_ids.to(
              device), attention_mask=attention_mask.to(device)).cpu()
          n_total, n_correct = compare_output(target_ids, outputs)
          n_totals_train += n_total
          n_corrects_train += n_correct

      train_loss = train_loss / len(train_dataloader)
      train_acc = n_corrects_train / n_totals_train
      eval_loss = eval_loss / len(dev_dataloader)
      eval_acc = n_corrects / n_totals
      print(
          f'Train epoch {args.start_epoch + epoch} loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val loss: {eval_loss:.3f}, val accuracy: {eval_acc:.3f}')

    # save final model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), output_model_file)
    # save config file
    model_to_save.config.to_json_file(output_config_file)

  # Test phase
  # if args.mode == ''

  print(f"program run time: {(time.time() - start) / 60 :.0f} mins")
