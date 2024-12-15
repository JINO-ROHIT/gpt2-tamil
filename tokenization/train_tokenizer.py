import sentencepiece as spm

import os

options = dict(
  input="data/ta_dedup.txt",
  input_format="text",
  model_prefix="tok32000", # output filename prefix
  model_type="bpe",
  vocab_size=32000,
  normalization_rule_name="identity",
  remove_extra_whitespaces=False,
  max_sentence_length=4192, 
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.9995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,

  num_threads=os.cpu_count(),
)

spm.SentencePieceTrainer.train(**options)