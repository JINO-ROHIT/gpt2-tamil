import os
import random
from typing import Iterator, Optional, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, IterableDataset


class LazyTamilDataset(IterableDataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: spm.SentencePieceProcessor,
                 max_length: int,
                 stride: int,
                 split: Optional[str] = None,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 debug: bool = False):
        """
        Lazy loading dataset for large text files with memory-efficient tokenization and train/val split.
        
        Args:
            file_path (str): Path to the text file.
            tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer.
            max_length (int): Maximum sequence length.
            stride (int): Stride for overlapping chunks.
            split (str, optional): 'train' or 'val'. None uses the full dataset.
            train_ratio (float): Proportion of data to use for training.
            seed (int): Random seed for reproducibility.
            debug (bool): Enable debug mode.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.split = split
        self.train_ratio = train_ratio
        self.debug = debug

        random.seed(seed)
        torch.manual_seed(seed)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def _generate_chunks(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generator method to yield tokenized chunks lazily with train/val splitting.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            chunk_counter = 0
            while True:
                chunk = f.read(1024 * 1024)  # Read 1MB at a time
                if not chunk:
                    break

                token_ids = self.tokenizer.encode(chunk)

                for i in range(0, len(token_ids) - self.max_length, self.stride):
                    if self.split is not None:
                        # Deterministic split based on chunk counter
                        is_train = random.random() < self.train_ratio

                        if (self.split == 'train' and not is_train) or \
                           (self.split == 'val' and is_train):
                            chunk_counter += 1
                            continue

                    input_chunk = token_ids[i:i + self.max_length]
                    target_chunk = token_ids[i + 1: i + self.max_length + 1]

                    yield (
                        torch.tensor(input_chunk, dtype=torch.long),
                        torch.tensor(target_chunk, dtype=torch.long)
                    )

                    chunk_counter += 1

    def __iter__(self):
        """
        Make the dataset iterable for lazy loading.
        """
        return self._generate_chunks()

def create_lazy_split_dataloader(
    file_path: str,
    batch_size: int,
    max_length: int,
    stride: int,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create lazy loading train and validation DataLoaders with a split.
    
    Args:
        file_path (str): Path to the text file.
        batch_size (int): Batch size for dataloaders.
        max_length (int): Maximum sequence length.
        stride (int): Stride for overlapping chunks.
        train_ratio (float): Proportion of data to use for training.
        num_workers (int): Number of worker processes for data loading.
        seed (int): Random seed for reproducibility.
    
    Returns:
        Tuple of train and validation DataLoaders.
    """

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('models/tok32000.model')


    train_dataset = LazyTamilDataset(
        file_path = file_path,
        tokenizer = tokenizer,
        max_length = max_length,
        stride = stride,
        split = 'train',
        train_ratio = train_ratio,
        seed = seed
    )

    val_dataset = LazyTamilDataset(
        file_path = file_path,
        tokenizer = tokenizer,
        max_length = max_length,
        stride = stride,
        split = 'val',
        train_ratio = train_ratio,
        seed = seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers
    )

    return train_dataloader, val_dataloader


import multiprocessing
from typing import List, Tuple
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

class TamilDataset(Dataset):
    def __init__(self, path: str, tokenizer: spm.SentencePieceProcessor, max_length: int, stride: int, debug: bool = False):
        """
        PyTorch Dataset for tokenized Tamil text.
         
        Args:
            path (str): Path to the text file.
            tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer.
            max_length (int): Maximum sequence length.
            stride (int): Stride for overlapping chunks.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.debug = debug
        self.input_ids = []
        self.target_ids = []
        
        # Process text line by line
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in tqdm(enumerate(f)):
                token_ids = self.tokenizer.encode(line.strip())
                for i in range(0, len(token_ids) - max_length, stride):
                    input_chunk = token_ids[i:i + max_length]
                    target_chunk = token_ids[i + 1: i + max_length + 1]
                    self.input_ids.append(torch.tensor(input_chunk))
                    self.target_ids.append(torch.tensor(target_chunk))
                del token_ids
                del line
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# tokenizer = spm.SentencePieceProcessor()
# tokenizer.load('models/tok32000.model')

# dataset = TamilDataset('data/ta_dedup.txt', tokenizer, 128, 1, debug=True)

# if __name__ == '__main__':
#     train_dataloader, val_dataloader = create_lazy_split_dataloader(
#         file_path = 'data/sample.txt',
#         batch_size = 32,
#         max_length = 256,
#         stride = 1,
#         train_ratio = 0.8,
#         num_workers = multiprocessing.cpu_count() // 2,
#         seed = 42
#     )


#     print("Training Data:")
#     for batch_inputs, batch_targets in train_dataloader:
#         print(f"Train batch inputs shape: {batch_inputs.shape}")
#         print(f"Train batch targets shape: {batch_targets.shape}")
#         break

#     print("\nValidation Data:")
#     for batch_inputs, batch_targets in val_dataloader:
#         print(f"Val batch inputs shape: {batch_inputs.shape}")
#         print(f"Val batch targets shape: {batch_targets.shape}")
#         break
