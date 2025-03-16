# GPT-2

An autocomplete language model using GPT-2 architecture. A CS 224N final project.

## Getting Started

1. Set up a Conda environment

```bash
# get setup
conda env create -f env.yml
conda activate cs224n_dfp

# install dependencies
pip install -r requirements.txt
```

2. Featured use-cases:

- Sentiment analysis
- Paraphrase detection
- Sonnet generation

```bash
# run training for sentiment classification (full model)
python3 classifier.py –fine-tune-mode full-model –batch_size 128 –lr 1e-5 hidden_dropout_prob=0.1 –epochs=10

# run training for sentiment classification (last linear layer)
python3 classifier.py –fine-tune-mode last-linear-layer –batch_size 64 –lr 1e-3 hidden_dropout_prob=0.1 –epochs=10

# run training for paraphrase detection (via cloze-style classification)
temp

# run training for sonnet generation (via autoregressive LM)
python3 sonnet_generation.py --held_out_sonnet_path data/sonnets_held_out_dev.txt --sonnet_out predictions/generated_sonnets_dev.txt
```

## Background

GPT-2 is a large, transformer-based language model that generates text via predicting the next word given context. We focus on building a smaller version of GPT-2 from scratch, focusing on its architecture (e.g. multi-head self-attention, position-wise feed-forward networks, byte-pair encoding for tokenization).

## Developers

[Aaron Jin](https://github.com/aaronkjin)

[Brandon Bui](https://github.com/brandonbui5)

[Eli Wandless](https://github.com/elidbc)
