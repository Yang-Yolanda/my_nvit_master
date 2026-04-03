
import torch
import numpy as np
import sys
from pathlib import Path

# Add root to path to import smpl_topology
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from smpl_topology import get_kinematic_chain_order, get_smpl_adjacency_matrix
from hmr2.datasets import ImageDataset

class BioMambaDataset(torch.utils.data.Dataset):
    """
    Bio-Mimetic Dataset Wrapper.
    
    Organizes standard data into the "5-Branch" Kinematic Toplogy format:
    1. Returns 'kinematic_chain': Indices for Center-Out Mamba scanning.
    2. Returns 'gcn_adj': Directed Adjacency Matrix for Global Refinement.
    """
    def __init__(self, cfg, dataset_file, img_dir, train=True, **kwargs):
        # Initialize standard HMR2 dataset
        self.base_dataset = ImageDataset(cfg, dataset_file=dataset_file, img_dir=img_dir, train=train, **kwargs)
        
        # Precompute Topology Artifacts
        # 1. Kinematic Order (for Mamba)
        # Flattened list of joint indices sorted topologicaly (Pelvis -> Center -> Extremities)
        self.kinematic_chain = torch.tensor(get_kinematic_chain_order(), dtype=torch.long)
        
        # 2. Adjacency Matrix (for GCN)
        # Directed: Parent -> Child
        self.gcn_adj = get_smpl_adjacency_matrix(directed=True, add_self_loops=True)
        
        print(f"🧬 BioMambaDataset initialized: {len(self.base_dataset)} samples.")
        print(f"   > Kinematic Chain Length: {len(self.kinematic_chain)}")
        print(f"   > GCN Adjacency Shape: {self.gcn_adj.shape}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get standard batch
        batch = self.base_dataset[idx]
        
        # Inject Structural Priors
        # Note: These are static for the skeleton, but we verify consistency.
        # If the model expects them as input (e.g. for dynamic masking), we pass them.
        
        batch['kinematic_chain'] = self.kinematic_chain
        batch['gcn_adj'] = self.gcn_adj
        
        return batch

    def __iter__(self):
        """Make map-style dataset iterable for mixing."""
        for i in range(len(self)):
            yield self[i]

class BioWebDataset(torch.utils.data.IterableDataset):
    """
    Bio-Mimetic Wrapper for WebDatasets (Iterative).
    """
    def __init__(self, wds_dataset):
        self.wds_dataset = wds_dataset
        
        # Precompute Topology Artifacts (Same as BioMambaDataset)
        self.kinematic_chain = torch.tensor(get_kinematic_chain_order(), dtype=torch.long)
        self.gcn_adj = get_smpl_adjacency_matrix(directed=True, add_self_loops=True)
        
    def __iter__(self):
        # Iterate over the underlying WebDataset and inject structural priors
        for batch in self.wds_dataset:
            batch.pop('__key__', None) # Remove WDS metadata for consistency
            batch['kinematic_chain'] = self.kinematic_chain
            batch['gcn_adj'] = self.gcn_adj
            yield batch
