#!/home/yangz/.conda/envs/4D-humans/bin/python
import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

def rot6d_to_axis_angle(rot6d):
    # Not needed if output is 3x3 matrices
    pass

def matrix_to_axis_angle(rot_matrices):
    """
    Convert batch of rotation matrices (N, 3, 3) to axis-angle vectors (N, 3)
    """
    N = rot_matrices.shape[0]
    r_vecs = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        R = rot_matrices[i]
        # cv2.Rodrigues handles 3x3 -> 3x1
        vec, _ = cv2.Rodrigues(R)
        r_vecs[i] = vec.flatten()
    return r_vecs

def run_inference_video(video_path, onnx_path, output_npz):
    # 1. Init ONNX Session
    print(f"Loading ONNX Model: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
        
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {num_frames} frames...")
    
    all_poses = [] # Will store flattened axis-angle
    all_joints3d = []
    
    # 3. Loop
    for _ in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess: Resize to 256x256, Normalize
        # HMR2 expects BGR or RGB? Usually RGB, normalized [0,1].
        # ViT usually expects ImageNet mean/std normalize.
        # Let's assume standard HMR preprocessing: Resize -> /255. -> Normalize
        
        input_img = cv2.resize(frame, (256, 256))
        input_img = input_img[:, :, ::-1] # BGR -> RGB
        input_img = input_img.astype(np.float32) / 255.0
        
        # Mean/Std for ViT (ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_img = (input_img - mean) / std
        
        # HWC -> CHW, Batch dim
        input_tensor = input_img.transpose(2, 0, 1)[None, :, :, :]
        
        # Inference
        outputs = session.run(None, {input_name: input_tensor})
        
        # Output map based on check_onnx_io
        # [0]: body_pose (1, 23, 3, 3)
        # [4]: pred_joints (1, 44, 3) (or similar)
        
        body_pose_mat = outputs[0] # (1, 23, 3, 3)
        pred_joints = outputs[4]   # (1, 44, 3)
        
        # Prepare SMPL params for metrics (Axis-Angle)
        # Flatten batch
        bp_mat_flat = body_pose_mat.reshape(-1, 3, 3) # (23, 3, 3)
        bp_vec_flat = matrix_to_axis_angle(bp_mat_flat) # (23, 3)
        
        # Global orient (outputs[2]) is (1, 1, 3, 3) -> 1 axis angle
        glob_orient_mat = outputs[2].reshape(1, 3, 3)
        glob_vec = matrix_to_axis_angle(glob_orient_mat) # (1, 3)
        
        # Concatenate Global(3) + Body(69) = 72
        full_pose = np.concatenate([glob_vec, bp_vec_flat], axis=0).flatten() # (72,)
        
        all_poses.append(full_pose)
        all_joints3d.append(pred_joints[0]) ## Assumes this is Model Joints.
        
    cap.release()
    
    # 4. Save
    all_poses = np.array(all_poses)
    all_joints3d = np.array(all_joints3d)
    
    # Construct dict for gait_metrics
    # gait_metrics needs: {'smpl_params': {'body_pose': ..., 'global_orient': ...}}
    # But current gait_metrics.py reads:
    #   if 'smpl_params' in data: params = data['smpl_params'].item()
    #   self.poses = params.get('body_pose', ... ) or merged full pose
    #
    # Wait, my previous `gait_metrics.py` logic:
    # "lk_idx = 4 * 3" -> It accesses index 12..15 of `self.poses`.
    # Standard SMPL `body_pose` (23 joints) usually starts from Joint 1 (Left Hip).
    # Joint 0 (Pelvis) is `global_orient`.
    # If `gait_metrics` uses `lk_idx=4*3`, it expects `poses` to include Pelvis at 0?
    # SMPL Joint 4 is Left Knee.
    # If `poses` is ONLY `body_pose` (23 joints, 1..23), index 4 is Right Hip (Joint 5)?
    # No. 
    # Standard: 0=Pelvis, 1=L_Hip, 2=R_Hip, 3=Spine1, 4=L_Knee...
    # `body_pose` usually omits Pelvis. So `body_pose[0]` is L_Hip.
    # So `body_pose[3]` is L_Knee.
    # IF `gait_metrics` accesses index 12 (4*3), it implies it expects Pelvis(0), L_Hip(1), R_Hip(2), Spine(3), L_Knee(4).
    # So I MUST concatenate Global Orient + Body Pose.
    # My script does: `np.concatenate([glob_vec, bp_vec_flat])`.
    # This creates a 24-joint pose vector. Indices match Standard SMPL.
    # Correct.

    # We save 'pose' key as the full 72-dim vector for simplicity
    np.savez(output_npz, 
             pose=all_poses,        # (N, 72)
             pred_joints=all_joints3d # (N, 44, 3)
    )
    print(f"Saved {output_npz} with {len(all_poses)} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    run_inference_video(args.video, args.onnx, args.out)
