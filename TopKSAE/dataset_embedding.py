from torch.utils.data import Dataset
import torch
import glob
from typing import List, Dict, Any

class EmbeddingDataset(Dataset):
    def __init__(self, path, file_pattern, use_files = 40):
        # Load all `.pt` files based on the pattern
        self.file_path = path
        self.files = sorted(glob.glob(path+"/"+file_pattern))
        print(f"Num of files found : {len(self.files)}")
        print(f"Num of files used : {len(self.files[:use_files])}")
        self.data = []

        # Read and store all embeddings from all files
        for file in self.files[:use_files]:
            batch_data = torch.load(file)
            # Extract embeddings and flatten them into a list
            self.data.extend([item["embedding"] for item in batch_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# class EfficientEmbeddingDataset(Dataset):
#     def __init__(self, path: str, file_pattern: str):
#         self.file_path = path
#         self.files = sorted(glob.glob(path + "/" + file_pattern))
#         print(f"Num of files found : {len(self.files)}")
        
#         # Instead of loading all data, we'll create a mapping of indices to file locations
#         self.index_mapping: List[Dict[str, Any]] = []
#         total_items = 0
        
#         # Build index mapping
#         for file_idx, file in enumerate(self.files):
#             # Load just the length information
#             batch_data = torch.load(file)
#             batch_size = len(batch_data)
            
#             # Store the mapping information
#             for batch_idx in range(batch_size):
#                 self.index_mapping.append({
#                     'file_idx': file_idx,
#                     'batch_idx': batch_idx
#                 })
            
#             total_items += batch_size
            
#         print(f"Total number of embeddings: {total_items}")

#     def __len__(self) -> int:
#         return len(self.index_mapping)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         # Get the file and batch index for this item
#         file_info = self.index_mapping[idx]
#         file_path = self.files[file_info['file_idx']]
        
#         # Load the specific file
#         batch_data = torch.load(file_path)
        
#         # Return the specific embedding
#         return batch_data[file_info['batch_idx']]['embedding']
    
#     def get_batch_indices(self, start_idx: int, batch_size: int) -> List[int]:
#         """
#         Optional: Helper method to get indices that are likely in the same file
#         Useful for optimizing batch loading
#         """
#         end_idx = min(start_idx + batch_size, len(self))
#         return list(range(start_idx, end_idx))

# class CachedEfficientEmbeddingDataset(EfficientEmbeddingDataset):
#     """
#     Optional: Version with LRU cache for frequently accessed files
#     """
#     def __init__(self, path: str, file_pattern: str, cache_size: int = 5):
#         super().__init__(path, file_pattern)
#         from functools import lru_cache
#         self.cache_size = cache_size
#         self.load_file = lru_cache(maxsize=cache_size)(self._load_file)
    
#     def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
#         return torch.load(file_path)
    
#     def __getitem__(self, idx: int) -> torch.Tensor:
#         file_info = self.index_mapping[idx]
#         file_path = self.files[file_info['file_idx']]
#         batch_data = self.load_file(file_path)
#         return batch_data[file_info['batch_idx']]['embedding']