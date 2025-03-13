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
    self.lm_head = nn.Linear(args.d, self.gpt.config.vocab_size, bias=False)


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
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs["last_hidden_state"]
    logits = self.lm_head(hidden_states)
    return logits
    


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, input_ids, temperature=0.7, top_k=50, beam_width=1, max_length=128, do_sample=True):
    """
    Generates an original sonnet using advanced sampling techniques:
    
    Beam Search: If beam_width > 1, uses beam search with length normalization (exponent=0.7).
    Top-K Sampling: If do_sample is True and beam_width==1, at each generation step only the top_k tokens are considered.
    Greedy: If do_sample is False and beam_width==1, picks the highest probability token.
    
    Temperature scaling is applied to the logits before sampling.
    
    NEW MODIFICATIONS:
      - Enforces a 14-line limit (Shakespearean sonnet format).
      - Enforces the rhyme scheme (ABABCDCDEFEFGG) by biasing the generation of line endings.
    """
    if beam_width > 1:
      best_seq = self._beam_search_generate(input_ids, temperature, top_k, beam_width, max_length)
      generated_output = self.tokenizer.decode(best_seq[0].cpu().numpy().tolist())[3:]
      return best_seq, generated_output
    else:
      # --- NEW: Structural Constraints Setup ---
      max_lines = 14
      # Identify the newline token id (used to count line breaks)
      newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
      # Decode the prompt and count how many lines are already present.
      decoded_prompt = self.tokenizer.decode(input_ids[0].cpu().numpy().tolist())
      current_line_count = decoded_prompt.count("\n")
      # Define the Shakespearean rhyme scheme: ABAB CDCDEFEFGG
      rhyme_pattern = ['A','B','A','B','C','D','C','D','E','F','E','F','G','G']
      # For lines already in the prompt, extract their ending word to serve as the target rhyme.
      rhyme_map = {}  # Mapping: rhyme letter -> target ending word.
      lines = decoded_prompt.split("\n")
      for i, line in enumerate(lines):
        if i < max_lines and line.strip():
          last_word = line.strip().split()[-1]
          letter = rhyme_pattern[i]
          if letter not in rhyme_map:
            rhyme_map[letter] = last_word
      # ------------------------------------------------

      token_ids = input_ids.to(self.get_device())
      attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

      while token_ids.shape[1] < max_length:
        # Forward pass to get logits
        logits_sequence = self.forward(token_ids, attention_mask)
        logits_last_token = logits_sequence[:, -1, :] / temperature  # Temperature scaling

        # --- NEW: Rhyme Enforcement ---
        # Determine which line is being generated (0-indexed)
        current_line_index = current_line_count  
        # If the current line has a rhyme requirement (i.e. a previous line with the same rhyme letter exists),
        # then bias the logits for candidate tokens that rhyme with the target.
        if current_line_index < max_lines and rhyme_pattern[current_line_index] in rhyme_map:
          target_rhyme = rhyme_map[rhyme_pattern[current_line_index]]
          candidate_ids = self.get_rhyme_token_ids(target_rhyme)
          for cid in candidate_ids:
            logits_last_token[0, cid] += 2.0  # Boost factor (tunable)
        # ---------------------------------

        if do_sample:
          # Top-K sampling: select from top_k tokens
          probs = torch.softmax(logits_last_token, dim=-1)
          topk_probs, topk_indices = torch.topk(probs, k=top_k)
          topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # Re-normalize
          sampled_index = torch.multinomial(topk_probs, 1)
          sampled_token = topk_indices.gather(dim=-1, index=sampled_index)
        else:
          # Greedy: choose the highest probability token
          _, sampled_token = torch.max(logits_last_token, dim=-1, keepdim=True)

        # Stop if EOS token is reached
        if sampled_token.item() == self.tokenizer.eos_token_id:
          break

        # Append the sampled token
        token_ids = torch.cat([token_ids, sampled_token], dim=1)
        attention_mask = torch.cat(
          [attention_mask, torch.ones((token_ids.shape[0], 1), dtype=torch.int64).to(self.get_device())],
          dim=1
        )

        # --- NEW: Line Count and Rhyme Map Update ---
        # Check if the generated token is a newline (i.e. end of a line).
        if sampled_token.item() == newline_token_id:
          current_line_count += 1
          decoded_so_far = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
          lines_generated = decoded_so_far.split("\n")
          if len(lines_generated) >= 2:
            # Get the most recently completed line (the one before the newline)
            completed_line = lines_generated[-2].strip()
            if completed_line:
              last_word = completed_line.split()[-1]
              letter = rhyme_pattern[current_line_count - 1]
              # If this is the first occurrence for this rhyme letter, store the target ending.
              if letter not in rhyme_map:
                rhyme_map[letter] = last_word
          # If we have reached the 14-line limit, stop generation.
          if current_line_count >= max_lines:
            break
        # -----------------------------------------------
      generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
      return token_ids, generated_output

  def get_rhyme_token_ids(self, target_word):
    """
    Dummy implementation to obtain token ids for words that rhyme with target_word.
    In practice, this should use a rhyming dictionary or phonetic lookup.
    For demonstration purposes, we simply return the token ids for target_word.
    """
    return self.tokenizer.encode(target_word, add_special_tokens=False)

  def _beam_search_generate(self, input_ids, temperature, top_k, beam_width, max_length):
    """
    Performs beam search generation. At each step, for every candidate sequence the top_k continuations are 
    considered and the beams are updated. A length normalization penalty (exponent 0.7) is applied.
    """
    beams = [(input_ids.to(self.get_device()), 0.0)]  # Each beam: (sequence, cumulative log probability)
    completed_beams = []
    norm_exponent = 0.7
    for _ in range(max_length):
      new_beams = []
      for seq, score in beams:
        # If EOS reached, retain the beam
        if seq[0, -1].item() == self.tokenizer.eos_token_id:
          completed_beams.append((seq, score))
          continue
        attention_mask = torch.ones(seq.shape, dtype=torch.int64).to(self.get_device())
        logits = self.forward(seq, attention_mask)
        logits_last = logits[:, -1, :] / temperature
        probs = torch.softmax(logits_last, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=top_k)
        for i in range(top_k):
          token_prob = topk_probs[0, i].item()
          token_id = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # Shape: (1,1)
          new_seq = torch.cat([seq, token_id], dim=1)
          new_score = score + math.log(token_prob + 1e-8)  # Accumulate log probability
          new_beams.append((new_seq, new_score))
      if not new_beams:
        break
      # Sort new beams with length normalization to avoid overly short sequences.
      new_beams = sorted(new_beams, key=lambda x: x[1] / (x[0].shape[1] ** norm_exponent), reverse=True)
      beams = new_beams[:beam_width]
      if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _ in beams):
        completed_beams.extend(beams)
        break
    if completed_beams:
      best_seq, best_score = max(completed_beams, key=lambda x: x[1] / (x[0].shape[1] ** norm_exponent))
    else:
      best_seq, best_score = beams[0]
    return best_seq


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
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
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
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

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