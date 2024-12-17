import multiprocessing
from typing import List, Tuple
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset, random_split

def get_line_offsets(path: str, chunk_size: int = 2 ** 20) -> List[int]:
    """
    Get line offsets from a file for fast random access.
    """
    offsets = [0]
    with open(path, "rb") as file:
        chunk = file.readlines(chunk_size)
        while chunk:
            for line in chunk:
                offsets.append(offsets[-1] + len(line))
            print(f"Lines found: {len(offsets)}", end='\r')
            chunk = file.readlines(chunk_size)
    return offsets

class TamilDataset(Dataset):
    def __init__(self, path: str, offset_dict: List[int], tokenizer: spm.SentencePieceProcessor, max_length: int, stride: int, debug: bool = False):
        """
        PyTorch Dataset for tokenized Tamil text.
         
        Args:
            path (str): Path to the text file.
            offset_dict (List[int]): List of line offsets.
            tokenizer (spm.SentencePieceProcessor): SentencePiece tokenizer.
            max_length (int): Maximum sequence length.
            stride (int): Stride for overlapping chunks.
        """
        self.path = path
        self.offset_dict = offset_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.debug = debug
    
    def __len__(self) -> int:
        return len(self.offset_dict)
    
    def __getitem__(self, idx):
        """
        Retrieves tokenized input and target sequences for a given index.
        """
        offset = self.offset_dict[idx]
        
        with open(self.path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline().strip()
        
        token_ids = self.tokenizer.encode(line)
        
        input_chunks = []
        target_chunks = []
        
        for i in range(0, max(0, len(token_ids) - self.max_length), self.stride):
            input_chunk = token_ids[i:i + self.max_length]
            target_chunk = token_ids[i + 1: i + self.max_length + 1]
            
            input_chunks.append(torch.tensor(input_chunk, dtype=torch.long))
            target_chunks.append(torch.tensor(target_chunk, dtype=torch.long))
        
        if self.debug:
            return line, input_chunks, target_chunks
        return input_chunks, target_chunks

def create_dataloader(
    path: str, 
    batch_size: int, 
    max_length: int, 
    stride: int, 
    shuffle: bool, 
    drop_last: bool, 
    num_workers: int,
    train_split: float = 0.75
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with a specified split.
    
    Args:
        path (str): Path to the text file.
        batch_size (int): Batch size for dataloaders.
        max_length (int): Maximum sequence length.
        stride (int): Stride for overlapping chunks.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of worker processes for data loading.
        train_split (float, optional): Proportion of data to use for training. Defaults to 0.75.
    
    Returns:
        Tuple of train and validation DataLoaders.
    """

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('models/tok32000.model')
    
    offsets = get_line_offsets(path)
    
    dataset = TamilDataset(path, offsets, tokenizer, max_length, stride, debug=False)
    
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    train_dataloader, val_dataloader = create_dataloader(
        'data/ta_dedup.txt', 
        batch_size=1, 
        max_length=256, 
        stride=1, 
        shuffle=True, 
        drop_last=False, 
        num_workers=multiprocessing.cpu_count() // 2
    )
    
    train_iter = iter(train_dataloader)
    val_iter = iter(val_dataloader)
    
    first_train_batch = next(train_iter)
    first_val_batch = next(val_iter)

    # print("Train batch first chunk shapes:", first_train_batch[0][0].shape, first_train_batch[0][1].shape)
    # print("Validation batch first chunk shapes:", first_val_batch[0][0].shape, first_val_batch[0][1].shape)

    # TODO: need to fix this empty results in some cases

    print(first_train_batch)
    
    print(f"Train dataset size: {len(train_dataloader.dataset)}")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")