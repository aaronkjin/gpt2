'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from evaluation import test_sonnet

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.lm_head = nn.Linear(args.d, self.tokenizer.vocab_size)

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    outputs = self.gpt(input_ids = input_ids, attention_mask = attention_mask)
    hidden_states = outputs["last_hidden_state"]  
    logits = self.lm_head(hidden_states)           
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, num_beams=3, max_length=128, length_penalty=1.0, early_stopping=True):
    """
    Generates a sonnet using beam search with length normalization.
    
    Args:
      encoding: Input tensor (e.g. from tokenization of the first 3 lines).
      num_beams: The beam width.
      max_length: Maximum length (in tokens) to generate.
      length_penalty: Exponent used to normalize beam scores by sequence length.
                      (values > 1 favor longer sequences; values < 1 favor shorter ones)
      early_stopping: If True, stops early when all beams end with the EOS token.
      
    Returns:
      A tuple (None, [decoded_output_string]).
    """
    return self._beam_search_generate(
        encoding, 
        num_beams=num_beams, 
        max_length=max_length, 
        length_penalty=length_penalty, 
        early_stopping=early_stopping
    )

  def _beam_search_generate(self, encoding, num_beams=3, max_length=128, length_penalty=1.0, early_stopping=True):
    """
    Performs beam search with length normalization.
    Each candidate is represented as a tuple: (token_ids, cumulative_log_prob).
    """
    device = self.get_device()
    # Initialize beam with the given encoding and zero log probability.
    beam = [(encoding.to(device), 0.0)]
    completed = []

    for step in range(max_length):
      new_beam = []
      for tokens, cum_log_prob in beam:
        # If candidate already ends with EOS, mark it as complete.
        if tokens[0, -1].item() == self.tokenizer.eos_token_id:
          completed.append((tokens, cum_log_prob))
          continue

          attention_mask = torch.ones(tokens.shape, dtype=torch.int64).to(device)
          logits = self.forward(tokens, attention_mask)
          logits_last = logits[:, -1, :]
          log_probs = torch.log_softmax(logits_last, dim=-1)

          # Expand candidate: get top `num_beams` next tokens.
          topk_log_probs, topk_indices = torch.topk(log_probs, k=num_beams)
          for i in range(topk_indices.shape[-1]):
            next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
            new_tokens = torch.cat([tokens, next_token], dim=1)
            new_score = cum_log_prob + topk_log_probs[0, i].item()
            new_beam.append((new_tokens, new_score))

      if not new_beam:
          break

        # Apply length normalization: normalized_score = score / (sequence_length^length_penalty)
      new_beam = sorted(
          new_beam,
          key=lambda x: x[1] / (x[0].shape[1] ** length_penalty),
          reverse=True
      )
      beam = new_beam[:num_beams]

      # If early stopping is enabled and all beams have ended with EOS, break.
      if early_stopping and all(tokens[0, -1].item() == self.tokenizer.eos_token_id for tokens, _ in beam):
        completed.extend(beam)
        break

    # Choose the best candidate: use completed candidates if available.
    if completed:
      best_candidate = max(completed, key=lambda x: x[1] / (x[0].shape[1] ** length_penalty))
    else:
      best_candidate = max(beam, key=lambda x: x[1] / (x[0].shape[1] ** length_penalty))

    decoded_output = self.tokenizer.decode(best_candidate[0][0].cpu().numpy().tolist())[3:]
    return None, [decoded_output]

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  train_dataset = SonnetsDataset(args.sonnet_path)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                            collate_fn=train_dataset.collate_fn)
  
  args = add_arguments(args)
  model = SonnetGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr)

  for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=TQDM_DISABLE):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      batch_count += 1

    avg_train_loss = total_loss / batch_count
    print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.3f}")

    # Optionally, generate a few output sonnets for qualitative check.
    model.eval()
    held_out_dataset = SonnetsDataset(args.held_out_sonnet_path)
    for batch in held_out_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      # Call generate() without temperature/top_p since beam search is used.
      _, output = model.generate(encoding['input_ids'], num_beams=3, max_length=128)
      print(f'{batch[1]}\nGenerated: {output[0]}\n')
        
        # Save checkpoint after each epoch.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    _, generated = model.generate(encoding['input_ids'], num_beams=3, max_length=128)
    full_sonnet = f'{generated[0]}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'Generated for {sonnet_id}:\n{generated[0]}\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)
  score = test_sonnet()
  print("chrF score:", score)