# TamilGPT 

A learning repository to understand and run series of experiments on the GPT architecture with Tamil.

<p align=center><img src="assets/cover.jpg" width="200px"></p>


## Folder Structure

```
gpt2-tamil/
├── trainer.py
├── inference.py
├── layers/
│   ├── attention.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   └── gpt.py
├── tokenization/
│   └── train_tokenizer.py
├── utils/
│   └── data_loader.py
```


## Currently supports

- ✅ A lazy data loader to avoid all data into RAM during dataset creation.
- ✅ Flexible GPT-2 architecture blocks.
- ✅ A sentencepiece tokenizer with bpe.
- ✅ Flexible training loop with checkpoint saving and resuming.
- ✅ Wandb logging.

## More to come

- ⏳ kv-cache
- ⏳ ROPE encoding
- ⏳ sliding attention
- ⏳ More sampling methods

## How to run?

1. Install the packages

```bash
poetry install
```

2. Train the tokenizer

```bash
python train_tokenizer.py
```

3. Run the Trainer class

```bash
python trainer.py
```

4. Run the inference class

```
python inference.py
```
