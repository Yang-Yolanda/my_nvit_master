#!/home/yangz/.conda/envs/4D-humans/bin/python
import os
import sys
import glob
import scipy.io
import numpy as np
import torch
from pathlib import Path

# Target Format matches datasets_3dpw.py expectation
# {
#   'img_name': str,
#   'jointPositions': (N, 24, 3), # World
#   'extrinsics': (N, 4, 4),
#   'intrinsics': (1, 3, 3),
#   'poses': (N, 72) optional
#   'betas': (N, 10) optional
# }

def convert_mpi_inf_3dhp(source_root, target_root):
    source_path = Path(source_root)
    target_path = Path(target_root) / "sequenceFiles" / "train" # MPI usually train
    os.makedirs(target_path, exist_ok=True)
    
    print(f"🚀 Converting MPI-INF-3DHP from {source_path} to {target_path}")

    # Subjects S1 to S8
    subjects = sorted([d.name for d in source_path.iterdir() if d.is_dir() and d.name.startswith('S')])
    
    for subj in subjects:
        subj_dir = source_path / subj
        sequences = sorted([d.name for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith('Seq')])
        
        for seq in sequences:
            seq_dir = subj_dir / seq
            annot_file = seq_dir / "annot.mat"
            
            if not annot_file.exists():
                print(f"⚠️ Missing annotation: {annot_file}")
                continue
                
            print(f"Processing {subj}/{seq}...")
            
            try:
                mat = scipy.io.loadmat(str(annot_file))
                
                # annot2/3 structure is usually cell array of shape (14, 1) for 14 cameras
                # Each cell contains (N_frames, 28, 2/3) or similar
                # We need to flatten this: One NPZ per (Subject_Sequence_Camera)
                
                n_cams = 14
                
                for cam_idx in range(n_cams):
                    # Check if data exists for this camera
                    # Note: MPI-INF structure is tricky. annot3 is a cell array
                    # mat['annot3'][cam_idx, 0] is the data
                    
                    joints3d_univ = mat['univ_annot3'][cam_idx, 0] # Universal frame (usually preferred)
                    if joints3d_univ.size == 0: continue
                    
                    # Reshape to (N, J, 3). MPI has 28 joints usually.
                    # We need to map MPI joints to SMPL 24 joints if possible, or keep as is and let dataset loader handle?
                    # 4d_humans loader expects 24 joints for 'jointPositions' (SMPL)
                    # Mapping MPI -> SMPL is non-trivial without a regressor.
                    # FOR NOW: We will save AS IS, and update the dataset loader to handle mismatch if needed.
                    # BUT datasets_3dpw.py expects [:24], so we must ensure we have at least 24.
                    
                    n_frames = joints3d_univ.shape[0]
                    # MPI 3D is usually reshaped
                    joints3d = joints3d_univ.reshape(n_frames, -1, 3) 
                    
                    # Create Dummy Extrinsics/Intrinsics (MPI is usually in Universal frame already or Camera frame)
                    # If using univ_annot3, it's metric.
                    # We will assume Identity camera for now to satisfy loader schema, 
                    # OR we need real camera params from 'cameras.h5' or 'annot.mat'
                    
                    # Create Output NPZ
                    out_name = f"{subj}_{seq}_cam{cam_idx}.npz"
                    out_path = target_path / out_name
                    
                    # Mocking SMPL params (since MPI doesn't have ground truth SMPL parameters usually)
                    # We fill with zeros so the loader doesn't crash, but 'has_smpl' will be False
                    dummy_poses = np.zeros((n_frames, 72))
                    dummy_betas = np.zeros((n_frames, 10))
                    
                    # Mock Camera (Identity)
                    extrinsics = np.repeat(np.eye(4)[None, ...], n_frames, axis=0)
                    intrinsics = np.eye(3)[None, ...]
                    
                    # Image Names
                    # Images are in S1/Seq1/imageSequence/video_{cam_idx}.avi or extracted frames
                    # We assume frames are extracted to S1/Seq1/imageSequence/video_{cam_idx}/frame_{xxxxx}.jpg
                    # The loader needs 'img_name' to point to directory?
                    # datasets_3dpw.py: img_path = self.image_dir / seq_name / img_name
                    # So we need to match this structure.
                    
                    # Save
                    np.savez_compressed(
                        out_path,
                        img_name=f"{subj}_{seq}_cam{cam_idx}", # Needs to match directory in imageFiles
                        seq_name=f"{subj}_{seq}_cam{cam_idx}",
                        jointPositions=joints3d, # (N, 28, 3) -> Loader takes [:24]
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        poses=dummy_poses,
                        betas=dummy_betas,
                        norm_poses=dummy_poses # Flag for loader
                    )
                    
            except Exception as e:
                print(f"❌ Error processing {subj}/{seq}: {e}")

if __name__ == "__main__":
    SRC = "/home/yangz/4D-Humans/data/mpi_inf_3dhp/dataset"
    DST = "/home/yangz/4D-Humans/data/MPI_INF_PROCESSED" # Staging area
    convert_mpi_inf_3dhp(SRC, DST)
