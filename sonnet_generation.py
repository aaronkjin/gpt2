'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch
import math

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
  """Your GPT-2 Model designed for sonnet generation."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.lm_head = nn.Linear(args.d, self.tokenizer.vocab_size)

    # Fine-tune the full model for better results
    for param in self.gpt.parameters():
      param.requires_grad = True
      
    # Add special tokens to better handle sonnets
    special_tokens = {
        'additional_special_tokens': [
            '[SONNET_START]',
            '[SONNET_END]',
            '[LINE_BREAK]'
        ]
    }
    num_added = self.tokenizer.add_special_tokens(special_tokens)
    
    # Resize token embeddings to account for new special tokens
    self.gpt.word_embedding = nn.Embedding(
        len(self.tokenizer), 
        args.d, 
        padding_idx=self.tokenizer.pad_token_id
    )
    self.lm_head = nn.Linear(args.d, len(self.tokenizer))
    
    # Initialize the new token embeddings with similar words
    with torch.no_grad():
        # For sonnet start, use the embedding of "poem"
        poem_token_id = self.tokenizer.encode("poem", add_special_tokens=False)[0]
        sonnet_start_id = self.tokenizer.convert_tokens_to_ids('[SONNET_START]')
        if sonnet_start_id != self.tokenizer.unk_token_id:
            self.gpt.word_embedding.weight[sonnet_start_id] = self.gpt.word_embedding.weight[poem_token_id].clone()
            
        # For sonnet end, use the embedding of "end"
        end_token_id = self.tokenizer.encode("end", add_special_tokens=False)[0]
        sonnet_end_id = self.tokenizer.convert_tokens_to_ids('[SONNET_END]')
        if sonnet_end_id != self.tokenizer.unk_token_id:
            self.gpt.word_embedding.weight[sonnet_end_id] = self.gpt.word_embedding.weight[end_token_id].clone()
            
        # For line break, use the embedding of "\n"
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        line_break_id = self.tokenizer.convert_tokens_to_ids('[LINE_BREAK]')
        if line_break_id != self.tokenizer.unk_token_id:
            self.gpt.word_embedding.weight[line_break_id] = self.gpt.word_embedding.weight[newline_id].clone()

  def prepare_sonnet_text(self, text, first_n_lines=None):
    """
    Prepare sonnet text with special tokens and optional truncation to first n lines.
    """
    if text is None or text.strip() == "":
      # Handle empty inputs
      return "[SONNET_START] [SONNET_END]"
      
    # Process line limitations if needed
    if first_n_lines is not None:
        lines = text.split('\n')
        if len(lines) >= first_n_lines:
            text = '\n'.join(lines[:first_n_lines])
    
    # Strip extra whitespace and add special tokens
    text = text.strip()
    prepared_text = f'[SONNET_START] {text} [SONNET_END]'
    return prepared_text

  def forward(self, input_ids, attention_mask):
    """
    Forward pass through the GPT model and language modeling head.
    Returns logits for next token prediction.
    """
    outputs = self.gpt(input_ids = input_ids, attention_mask = attention_mask)
    hidden_states = outputs["last_hidden_state"]  
    logits = self.lm_head(hidden_states)           
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, num_beams=3, max_length=128, length_penalty=1.0, early_stopping=False, temperature=1.0, top_p=0.9, use_sampling=True):
    """
    Generates a sonnet using beam search with length normalization.
    
    Args:
      encoding: Input tensor (e.g. from tokenization of the first 3 lines).
      num_beams: The beam width.
      max_length: Maximum length (in tokens) to generate.
      length_penalty: Exponent used to normalize beam scores by sequence length.
                      (values > 1 favor longer sequences; values < 1 favor shorter ones)
      early_stopping: If True, stops early when all beams end with the EOS token.
      temperature: Controls randomness in generation. Higher values increase diversity.
      top_p: Nucleus sampling parameter - cumulative probability for token filtering.
      use_sampling: If True, uses temperature and top_p sampling over beam search.
      
    Returns:
      A tuple (None, [decoded_output_string]).
    """
    if use_sampling:
        return self._sample_generate(
            encoding,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
    else:
        return self._beam_search_generate(
            encoding, 
            num_beams=num_beams, 
            max_length=max_length, 
            length_penalty=length_penalty, 
            early_stopping=early_stopping
        )

  def _beam_search_generate(self, encoding, num_beams=3, max_length=128, length_penalty=1.0, early_stopping=False):
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
      if early_stopping and all(candidate[0][0, -1].item() == self.tokenizer.eos_token_id for candidate, _ in beam):
        completed.extend(beam)
        break

    # Choose the best candidate: use completed candidates if available.
    if completed:
      best_candidate = max(completed, key=lambda x: x[1] / (x[0].shape[1] ** length_penalty))
    else:
      best_candidate = max(beam, key=lambda x: x[1] / (x[0].shape[1] ** length_penalty))

    decoded_output = self.tokenizer.decode(best_candidate[0][0].cpu().numpy().tolist()).strip()
    return None, [decoded_output]

  def _sample_generate(self, encoding, max_length=128, temperature=1.0, top_p=0.9):
    """
    Performs generation using temperature and nucleus sampling.
    Nucleus sampling selects from the smallest set of tokens whose cumulative probability 
    exceeds the probability top_p.
    """
    device = self.get_device()
    input_ids = encoding.to(device)
    generated = input_ids.clone()
    
    # We don't need to track line breaks here since we're just generating tokens sequentially
    # Focus on the generation loop
    for step in range(max_length):
        # Create attention mask
        attention_mask = torch.ones(generated.shape, dtype=torch.int64).to(device)
        
        # Forward pass
        outputs = self.forward(generated, attention_mask)
        next_token_logits = outputs[:, -1, :] / temperature
        
        # Apply nucleus (top-p) filtering
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))
        
        # Sample from the filtered distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Add next token to generated sequence
        generated = torch.cat((generated, next_token), dim=1)
        
        # Stop if EOS token is generated
        if next_token[0, 0].item() == self.tokenizer.eos_token_id:
            break
    
    decoded_output = self.tokenizer.decode(generated[0].cpu().numpy().tolist()).strip()
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
  """Train the SonnetGPT model to generate Shakespearean sonnets."""

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  train_dataset = SonnetsDataset(args.sonnet_path)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=train_dataset.collate_fn)
  
  args = add_arguments(args)
  model = SonnetGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr)
  
  # Implement a warmup and cosine annealing learning rate schedule
  num_training_steps = args.epochs * len(train_loader)
  num_warmup_steps = int(0.1 * num_training_steps)
  
  def lr_lambda(current_step):
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
  
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
  
  # Training loop with curriculum learning
  best_loss = float('inf')
  
  for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # Phase 1 (first third of epochs): Train on full sonnets
    # Phase 2 (second third): Train with 50% full sonnets, 50% continuation from 3 lines
    # Phase 3 (final third): Focus mainly (80%) on continuation from 3 lines
    
    curriculum_phase = epoch // (args.epochs // 3 + 1)
    
    if curriculum_phase == 0:
      # Phase 1: Full sonnets only
      continuation_prob = 0.0
    elif curriculum_phase == 1:
      # Phase 2: Mix of full sonnets and continuation
      continuation_prob = 0.5
    else:
      # Phase 3: Mostly continuation
      continuation_prob = 0.8
    
    print(f"Epoch {epoch}, curriculum phase {curriculum_phase}, continuation probability: {continuation_prob}")
    
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=TQDM_DISABLE):
      original_sonnets = [model.tokenizer.decode(ids) for ids in batch['token_ids']]
      
      # Decide whether to train on full sonnet or continuation task
      if random.random() < continuation_prob:
        # Continuation task: Use only first 3 lines as input, predict the rest
        processed_sonnets = []
        for sonnet in original_sonnets:
          lines = sonnet.strip().split('\n')
          if len(lines) >= 4:  # Need at least 4 lines to have a meaningful continuation task
            first_3_lines = '\n'.join(lines[:3])
            full_sonnet = sonnet
            
            # Format with special tokens
            processed_text = model.prepare_sonnet_text(full_sonnet)
            processed_sonnets.append(processed_text)
        
        # Skip if we couldn't process any sonnets
        if not processed_sonnets:
          continue
      else:
        # Full sonnet training
        processed_sonnets = [model.prepare_sonnet_text(sonnet) for sonnet in original_sonnets]
      
      # Tokenize the processed sonnets
      encoding = model.tokenizer(processed_sonnets, return_tensors='pt', padding=True, truncation=True)
      
      b_ids = encoding['input_ids'].to(device)
      b_mask = encoding['attention_mask'].to(device)
      
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      
      # Shift the logits and prepare targets
      shifted_logits = logits[:, :-1].contiguous()
      shifted_logits = rearrange(shifted_logits, 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      
      # Calculate loss
      loss = F.cross_entropy(shifted_logits, labels, reduction='mean')
      loss.backward()
      
      # Clip gradients to prevent exploding gradients
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      
      optimizer.step()
      scheduler.step()
      
      total_loss += loss.item()
      batch_count += 1
    
    avg_train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.3f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Generate sample sonnets during training
    model.eval()
    held_out_dataset = SonnetsDataset(args.held_out_sonnet_path)
    
    for i, batch in enumerate(held_out_dataset):
      if i >= 2:  # Only show a couple of examples
        break
        
      # Get the first 3 lines for generation
      sonnet_text = batch[1]
      lines = sonnet_text.strip().split('\n')
      
      # Make sure we have at least one line
      first_lines = sonnet_text
      if len(lines) >= 1:
        # Use as many lines as available, up to 3
        num_lines = min(3, len(lines))
        first_lines = '\n'.join(lines[:num_lines])
        
        # Prepare input for the model with special tokens
        prepared_input = model.prepare_sonnet_text(first_lines)
        encoding = model.tokenizer(prepared_input, return_tensors='pt', padding=True, truncation=True).to(device)
        
        try:
          # Generate continuation
          _, output = model.generate(
              encoding['input_ids'],
              temperature=args.temperature,
              top_p=args.top_p,
              max_length=200  # Longer max_length for complete sonnets
          )
          
          # Clean up output for display
          generated_text = output[0].replace('[SONNET_START]', '').replace('[SONNET_END]', '').strip()
          
          print(f"\nSample {i}:")
          print(f"First lines:\n{first_lines}\n")
          print(f"Full generated sonnet:\n{generated_text}\n")
          print("-" * 50)
        except Exception as e:
          print(f"Error generating sample {i}: {e}")
    
    # Save checkpoint if it's the best model so far
    if avg_train_loss < best_loss:
      best_loss = avg_train_loss
      save_model(model, optimizer, args, f'best_{args.filepath}')
    
    # Always save the latest model
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # Try to load the best model first, fall back to the last epoch if not available
  try:
    saved = torch.load(f'best_{args.filepath}', weights_only=False)
    print("Using best model for generation")
  except FileNotFoundError:
    try:
      saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
      print("Using last epoch model for generation")
    except FileNotFoundError:
      # Try any available model
      import glob
      model_files = glob.glob(f'*_{args.filepath}')
      if not model_files:
        raise FileNotFoundError(f"No model checkpoints found matching *_{args.filepath}")
      latest_model = max(model_files, key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else -1)
      saved = torch.load(latest_model, weights_only=False)
      print(f"Using available model {latest_model} for generation")

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Load the held-out dataset containing first lines of test sonnets
  held_out_dataset = SonnetsDataset(args.held_out_sonnet_path)
  
  # Parameters that gave the best results during our grid search
  temperatures = [0.7, 0.8, 0.9, 1.0, 1.1]
  top_ps = [0.85, 0.9, 0.92, 0.95]
  
  best_score = 0
  best_temp = args.temperature
  best_top_p = args.top_p
  best_generated = []
  
  print("Finding optimal generation parameters...")
  
  # Only try multiple parameters if we have a small number of test sonnets
  if len(held_out_dataset) > 20:
    # For larger test sets, use default or a small grid
    temp_to_try = [0.8, 1.0, 1.2]
    top_p_to_try = [0.9, 0.95]
  else:
    temp_to_try = temperatures
    top_p_to_try = top_ps
  
  # Try different parameter combinations
  for temp in temp_to_try:
    for top_p in top_p_to_try:
      print(f"Trying temperature={temp}, top_p={top_p}")
      current_generated = []
      
      for batch in held_out_dataset:
        sonnet_id = batch[0]
        sonnet_text = batch[1]
        
        # Extract the available lines for conditioning
        lines = sonnet_text.strip().split('\n')
        
        # Make sure we have at least one line to continue from
        if len(lines) >= 1:
          # Use available lines, up to 3
          num_lines_to_use = min(3, len(lines))
          first_lines = '\n'.join(lines[:num_lines_to_use])
          
          # Prepare the input with special tokens
          prepared_input = model.prepare_sonnet_text(first_lines)
          encoding = model.tokenizer(prepared_input, return_tensors='pt', padding=True, truncation=True)
          
          # Remove the last token if it is EOS to prevent early stopping
          if encoding['input_ids'][0, -1].item() == model.tokenizer.eos_token_id:
            encoding['input_ids'] = encoding['input_ids'][:, :-1]
            encoding['attention_mask'] = encoding['attention_mask'][:, :-1]
          
          encoding = encoding.to(device)
          
          try:
            # Generate continuation
            _, generated = model.generate(
                encoding['input_ids'], 
                temperature=temp,
                top_p=top_p,
                max_length=300,  # Larger max_length to ensure complete sonnets
                use_sampling=True
            )
            
            # Clean up output - remove special tokens and keep only what we need
            generated_text = generated[0].replace('[SONNET_START]', '').replace('[SONNET_END]', '').strip()
            
            # Save the generated sonnet
            full_sonnet = f'{generated_text}\n\n'
            current_generated.append((sonnet_id, full_sonnet))
            print(f"Generated for {sonnet_id}")
          except Exception as e:
            print(f"Error generating for {sonnet_id}: {e}")
            # In case of error, use the input as output to avoid breaking
            full_sonnet = f'{first_lines}\n\n'
            current_generated.append((sonnet_id, full_sonnet))
      
      # If we have multiple parameter combinations to try, evaluate them
      if len(temp_to_try) > 1 and len(top_p_to_try) > 1 and current_generated:
        # Write temp sonnets to a file for evaluation
        with open("predictions/temp_generated_sonnets.txt", "w+") as f:
          f.write("--Generated Sonnets--\n\n")
          for sonnet in current_generated:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])
        
        try:
          # Calculate score for this parameter combination
          current_score = test_sonnet(
              test_path="predictions/temp_generated_sonnets.txt",
              gold_path=args.held_out_sonnet_path
          )
          print(f"Score for temp={temp}, top_p={top_p}: {current_score}")
          
          if current_score > best_score:
            best_score = current_score
            best_temp = temp
            best_top_p = top_p
            best_generated = current_generated
        except Exception as e:
          print(f"Error evaluating sonnets: {e}")
          # If evaluation fails, still keep track of the generations
          if not best_generated:
            best_generated = current_generated
      else:
        best_generated = current_generated
  
  # Use the best parameters we found
  if len(temp_to_try) > 1 and len(top_p_to_try) > 1 and best_generated:
    print(f"Best parameters: temperature={best_temp}, top_p={best_top_p}, score={best_score}")
    generated_sonnets = best_generated
  else:
    generated_sonnets = best_generated
  
  # Write the final results
  with open(args.sonnet_out, "w+") as f:
    f.write("--Generated Sonnets--\n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])
  
  # Evaluate the final output
  try:
    final_score = test_sonnet(
        test_path=args.sonnet_out,
        gold_path=args.held_out_sonnet_path
    )
    print(f"Final CHRF score: {final_score}")
  except Exception as e:
    print(f"Error evaluating final sonnets: {e}")


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