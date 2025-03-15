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

def compute_log_likelihood(model, prompt, continuation, device):
  # Concatenate prompt and continuation.
  full_text = prompt + continuation
  inputs = model.tokenizer(full_text, return_tensors='pt')
  inputs = {k: v.to(device) for k, v in inputs.items()}
  # Forward pass (no dropout since model is in eval mode).
  outputs = model(inputs['input_ids'], inputs['attention_mask'])
  logits = outputs  # logits shape: [1, seq_len, vocab_size]
  log_probs = torch.log_softmax(logits, dim=-1)
  # Determine the number of tokens in the prompt.
  prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
  input_ids = inputs['input_ids'][0]
  # Only compute likelihood for tokens in the continuation.
  ll = 0.0
  # We start predicting from the position right after the prompt.
  for i in range(len(prompt_ids), input_ids.shape[0] - 1):
    target_token = input_ids[i+1]  # target token at position i+1
    ll += log_probs[0, i, target_token]
  return ll


# DPO Extension: A simple heuristic to generate a negative continuation by swapping two random lines.
def generate_negative(continuation):
    lines = continuation.strip().split('\n')
    if len(lines) > 1:
        # Swap one randomly chosen pair of adjacent lines.
        idx = random.randint(0, len(lines) - 2)
        lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]
        return '\n'.join(lines)
    else:
        words = continuation.split()
        if len(words) > 3:
            # Swap one pair of adjacent words.
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            return ' '.join(words)
        else:
            return continuation


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True
    """
    for name, param in self.gpt.named_parameters():
      if "h." in name:
        # Extract the block index from the parameter name (e.g., "h.3.attn.c_attn.weight")
        block_index = int(name.split('.')[1])
        if block_index < args.l - args.unfrozen_blocks:  # Freeze all blocks except the last transformer block.
          param.requires_grad = False
        else:
          param.requires_grad = True
      else:
        # Always fine-tune parameters not in the transformer blocks (e.g. embeddings, layer norms, output projection).
        param.requires_grad = True
    """

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    # Get the hidden states for all tokens in the sequence
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs["last_hidden_state"]
    hidden_states = F.dropout(hidden_states, p=0.2, training=self.training)
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    This implementation includes awareness of sonnet structure:
    - Ensures we generate a complete sonnet (14 lines total, including the provided first 3)
    - Uses slightly different temperature for different parts of the sonnet
    - Potentially adjusts sampling parameters based on position in the sonnet
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    prompt_len = token_ids.shape[1]
    
    # Count initial number of newlines to track how many lines we're starting with
    initial_text = self.tokenizer.decode(token_ids[0])
    initial_newlines = initial_text.count('\n')
    
    # Keep track of the line count as we generate
    current_newlines = initial_newlines
    line_count = 3  # We assume we start with the first 3 lines provided
    
    # Adjust temperatures for different parts of the sonnet
    middle_quatrains_temp = temperature * 0.95  # Lower for middle quatrains
    final_couplet_temp = temperature * 0.9    # Even lower for final couplet
    
    # Generate new tokens
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      
      # Adjust temperature based on position in sonnet
      current_temp = temperature
      if line_count >= 4 and line_count < 12:
        current_temp = middle_quatrains_temp
      elif line_count >= 12:
        current_temp = final_couplet_temp
      
      logits_last_token = logits_sequence[:, -1, :] / current_temp  # Apply temperature scaling
      
      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)
      
      # Top-p sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
      
      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)
      
      # Stop generating if 14 lines reached or EOS token generated
      if sampled_token.item() == self.tokenizer.eos_token_id or current_newlines >= initial_newlines + 14:
        break
      
      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )
      
      # Update line count
      token_str = self.tokenizer.decode([sampled_token.item()])
      if '\n' in token_str:
        current_newlines += 1
        if current_newlines > initial_newlines:
          line_count += 1
    
    # Return generated sonnet
    #generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    generated_output = self.tokenizer.decode(token_ids[0, prompt_len:].cpu().numpy().tolist())


    return token_ids, generated_output


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


"""def train(args):
  Train GPT-2 for paraphrase detection on the Quora dataset.
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

  #DPO Extenstion
  if args.dpo_mode:
    ref_model = SonnetGPT(args)
    ref_model = ref_model.to(device)
    for param in ref_model.parameters():
      param.requires_grad = False
    ref_model.eval()

  for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0.0
    num_samples = 0

    if args.dpo_mode:
      # DPO training loop.
      for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch} (DPO)', disable=TQDM_DISABLE):
        optimizer.zero_grad()
        batch_loss = 0.0
        # Process each sample in the batch individually.
        b_ids, _ = batch['token_ids'], batch['attention_mask']
        for sample_ids in b_ids:
          # Decode sample to text.
          full_text = model.tokenizer.decode(sample_ids)
          lines = full_text.strip().split('\n')
          if len(lines) < 4:
            continue  # Skip if not enough lines.
          prompt = '\n'.join(lines[:3]) + '\n'
          winning = '\n'.join(lines[3:])
          losing = generate_negative(winning)
          # Compute log-likelihoods.
          LL_theta_win = compute_log_likelihood(model, prompt, winning, device)
          LL_theta_loss = compute_log_likelihood(model, prompt, losing, device)
          LL_ref_win = compute_log_likelihood(ref_model, prompt, winning, device)
          LL_ref_loss = compute_log_likelihood(ref_model, prompt, losing, device)
          diff = (LL_theta_win - LL_ref_win) - (LL_theta_loss - LL_ref_loss)
          sample_loss = -torch.log(torch.sigmoid(args.beta * diff) + 1e-8)
          batch_loss += sample_loss
          num_samples += 1
        if num_samples > 0:
          batch_loss = batch_loss / num_samples
          batch_loss.backward()
          optimizer.step()
          epoch_loss += batch_loss.item()
      avg_loss = epoch_loss / (len(sonnet_dataloader) if len(sonnet_dataloader) > 0 else 1)
      print(f"Epoch {epoch} (DPO): avg loss = {avg_loss:.3f}.")
    else:
      # Original cross-entropy training loop.
      for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_samples += 1
      avg_loss = epoch_loss / num_samples
      print(f"Epoch {epoch}: train loss = {avg_loss:.3f}.")

    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')"""

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                   collate_fn=sonnet_dataset.collate_fn)
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    # DPO Extension: load reference model.
    if args.dpo_mode:
        ref_model = SonnetGPT(args)
        ref_model = ref_model.to(device)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        # If DPO mode is enabled and we are in the staged period, run pure CE training.
        if args.dpo_mode and epoch < args.staged_dpo_epochs:
            for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch} (Staged CE)', disable=TQDM_DISABLE):
                b_ids, b_mask = batch['token_ids'], batch['attention_mask']
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                optimizer.zero_grad()
                logits = model(b_ids, b_mask)
                logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
                labels = b_ids[:, 1:].contiguous().flatten()
                loss = F.cross_entropy(logits, labels, reduction='mean')
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_samples += 1
            avg_loss = epoch_loss / num_samples
            print(f"Epoch {epoch} (Staged CE): train loss = {avg_loss:.3f}.")
        elif args.dpo_mode:
            # Combined DPO + CE training.
            dpo_loss_total = 0.0
            ce_loss_total = 0.0
            sample_count = 0
            for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch} (Combined)', disable=TQDM_DISABLE):
                optimizer.zero_grad()
                batch_dpo_loss = 0.0
                # Compute CE loss over the batch.
                b_ids, b_mask = batch['token_ids'], batch['attention_mask']
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                logits = model(b_ids, b_mask)
                logits_ce = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
                labels = b_ids[:, 1:].contiguous().flatten()
                ce_loss = F.cross_entropy(logits_ce, labels, reduction='mean')
                ce_loss_total += ce_loss.item()

                # Compute DPO loss sample-by-sample.
                for sample_ids in b_ids:
                    full_text = model.tokenizer.decode(sample_ids)
                    lines = full_text.strip().split('\n')
                    if len(lines) < 4:
                        continue  # Skip if not enough lines.
                    prompt = '\n'.join(lines[:3]) + '\n'
                    winning = '\n'.join(lines[3:])
                    losing = generate_negative(winning)
                    LL_theta_win = compute_log_likelihood(model, prompt, winning, device)
                    LL_theta_loss = compute_log_likelihood(model, prompt, losing, device)
                    LL_ref_win = compute_log_likelihood(ref_model, prompt, winning, device)
                    LL_ref_loss = compute_log_likelihood(ref_model, prompt, losing, device)
                    diff = (LL_theta_win - LL_ref_win) - (LL_theta_loss - LL_ref_loss)
                    sample_loss = -torch.log(torch.sigmoid(args.beta * diff) + 1e-8)
                    batch_dpo_loss += sample_loss
                    sample_count += 1
                if sample_count > 0:
                    batch_dpo_loss = batch_dpo_loss / sample_count
                else:
                    batch_dpo_loss = 0.0
                dpo_loss_total += batch_dpo_loss.item()

                # Combine losses.
                combined_loss = args.mix_ratio * ce_loss + (1 - args.mix_ratio) * batch_dpo_loss
                combined_loss.backward()
                optimizer.step()
            avg_ce_loss = ce_loss_total / len(sonnet_dataloader)
            avg_dpo_loss = dpo_loss_total / len(sonnet_dataloader)
            print(f"Epoch {epoch} (Combined): avg CE loss = {avg_ce_loss:.3f}, avg DPO loss = {avg_dpo_loss:.3f}.")
        else:
            # Original cross-entropy training loop (if dpo_mode is not enabled).
            for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask = batch['token_ids'], batch['attention_mask']
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                optimizer.zero_grad()
                logits = model(b_ids, b_mask)
                logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
                labels = b_ids[:, 1:].contiguous().flatten()
                loss = F.cross_entropy(logits, labels, reduction='mean')
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_samples += 1
            avg_loss = epoch_loss / num_samples
            print(f"Epoch {epoch}: train loss = {avg_loss:.3f}.")

        print('Generating several output sonnets...')
        model.eval()
        for batch in held_out_sonnet_dataset:
            encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
            output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
            print(f'{batch[1]}{output[1]}\n\n')
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
  parser.add_argument("--epochs", type=int, default=8)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.0)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.95)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  #Below added for finetuning
  parser.add_argument("--unfrozen_blocks", type=int, help="Number of transformer blocks to fine-tune"
                      " (from the end)", default=2)
  #DPO Extension
  parser.add_argument("--dpo_mode", action='store_true', help="Enable Direct Preference Optimization training.")
  parser.add_argument("--beta", type=float, default=1.0, help="Beta scaling parameter for DPO loss.")
  parser.add_argument("--mix_ratio", type=float, default=0.5, help="Mix ratio between CE loss and DPO loss. 1.0 = pure CE; 0.0 = pure DPO.")
  parser.add_argument("--staged_dpo_epochs", type=int, default=2, help="Number of epochs to run pure CE loss before mixing in DPO loss.")

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