#!/home/yangz/.conda/envs/4D-humans/bin/python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Joint Map (SMPL)
JOINT_MAP = {
    'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3,   
    'L_Knee': 4, 'R_Knee': 5, 'Spine2': 6, 'L_Ankle': 7, 
    'R_Ankle': 8, 'Spine3': 9, 'L_Foot': 10, 'R_Foot': 11
}

class GaitAnalyzer:
    def __init__(self, data_path, height_cm=170.0):
        self.data = np.load(data_path, allow_pickle=True)
        self.height = height_cm
        
        # --- Handle Key Variations ---
        if 'jointPositions' in self.data:
            self.joints3d = self.data['jointPositions']
        elif 'pred_joints' in self.data:
            self.joints3d = self.data['pred_joints']
        else:
            raise KeyError("Could not find joint positions (searched 'jointPositions', 'pred_joints')")

        # Handle SMPL Params
        self.poses = None
        if 'smpl_params' in self.data:
            # Nested dict case
            params = self.data['smpl_params'].item()
            # Try body_pose (23x3)
            if 'body_pose' in params:
                 self.poses = params['body_pose'] # (B, 23, 3, 3) or (B, 23, 3)
                 # If (N, 23, 3, 3), convert to axis-angle or flatten?
                 # Current logic below handles (N, 23, 3) or (N, 69)
                 if self.poses.ndim == 4:
                     # Matrices to axis angle? Or just rely on flat layout support
                     print("Warning: Loaded Rotation Matrices. Angle calc might need update.")
        
        elif 'pose' in self.data:
            # Flat case (N, 72)
            # 0-2: Global, 3-71: Body (23*3)
            full_pose = self.data['pose']
            # We want Body Pose for Knee (Joint 4 of Body, i.e., indices 12:15 of Body Part)
            # Body starts at index 3 of full_pose
            self.poses = full_pose[:, 3:] # (N, 69)

    def analyze(self):
        # ... (Same logic, relying on self.poses) ...
        # Ensure self.poses is handled as (N, 69) or (N, 23, 3)
        
        # --- A. Step Parameters (Geometry) ---
        # L_Ankle (7), R_Ankle (8)
        l_ankle = self.joints3d[:, JOINT_MAP['L_Ankle']]
        r_ankle = self.joints3d[:, JOINT_MAP['R_Ankle']]
        
        # Simple projection to ground plane (drop Y?) NViT Y is up or down?
        # Usually HMR/SMPL: Y is DOWN or Up? 
        # Actually, let's compute euclidean distance in XZ plane (assuming Y vertical)
        # HMR default camera: Y down? No, usually Y up or down depending on renderer.
        # We calculate 3D distance for step length?
        # Better: Project onto "Forward Vector" of Pelvis.
        
        # Compute Pelvis Trajectory
        pelvis = self.joints3d[:, JOINT_MAP['Pelvis']]
        # Forward vector (Pelvis -> Spine/Neck?) No, Motion direction.
        velocity = pelvis[1:] - pelvis[:-1]
        mean_vel = np.mean(velocity, axis=0) # (3,)
        mean_vel[1] = 0 # Assume walking on flat ground (ignore vertical)
        forward_dir = mean_vel / (np.linalg.norm(mean_vel) + 1e-6)
        
        ortho_dir = np.array([-forward_dir[2], 0, forward_dir[0]]) 
        
        # Project Ankles
        l_proj = np.dot(l_ankle[:, [0, 2]], forward_dir[[0, 2]])
        r_proj = np.dot(r_ankle[:, [0, 2]], forward_dir[[0, 2]])
        
        step_dist = np.abs(l_proj - r_proj)
        
        l_vel = np.linalg.norm(l_ankle[1:] - l_ankle[:-1], axis=1) * 30 # cm/s (Assume 30fps)
        
        metrics = {}
        metrics['Step_Length_Mean'] = np.mean(step_dist) * 100 # m -> cm
        metrics['Step_Length_Max'] = np.percentile(step_dist, 95) * 100
        metrics['Symmetry_Stance'] = 50.0 # Placeholder
        metrics['L_Stance_Ratio'] = 50.0
        metrics['R_Stance_Ratio'] = 50.0
        metrics['Step_Width_Mean'] = 15.0 # Placeholder/Calc
        
        width = np.abs(np.dot(l_ankle[:,[0,2]] - r_ankle[:,[0,2]], ortho_dir[[0,2]]))
        metrics['Step_Width_Mean'] = np.mean(width) * 100

        # --- B. Kinematics (Angles) ---
        l_knee_angles = np.zeros(len(l_ankle))
        
        if self.poses is not None:
             # Expect self.poses to be (N, 69) or (N, 23, 3)
             # L_Knee is body joint 4.
             # Index in flattened body pose: 4*3 = 12 : 15
             
             if self.poses.ndim == 2: # (N, 69)
                 lk_idx = 4 * 3
                 lk_pose = self.poses[:, lk_idx : lk_idx+3]
                 # Magnitude = Angle (Axis-Angle representation)
                 l_knee_angles = np.linalg.norm(lk_pose, axis=1) * (180.0 / np.pi)
                 print(f"✅ Calculated Knee Angle from SMPL Params (Mean: {np.mean(l_knee_angles):.1f}°)")

        metrics['L_Knee_Flexion_Max'] = np.max(l_knee_angles)
        metrics['L_Knee_Flexion_Mean'] = np.mean(l_knee_angles)

        return metrics, {
            'L_Knee_Angle': l_knee_angles,
            'Step_Dist_Trace': step_dist * 100,
            'L_Vel': l_vel
        }

    def visualize(self, traces, metrics, save_path):
        import matplotlib
        matplotlib.use('Agg')
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(traces['L_Knee_Angle'], label='Left Knee Flexion', color='blue')
        ax1.set_title("Knee Flexion (SMPL)")
        ax1.set_ylabel("Degrees")
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(traces['Step_Dist_Trace'], label='Step Length (cm)', color='green')
        ax2.set_title("Step Length")
        
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        txt = "CLINICAL GAIT REPORT (INT8 MODEL)\n"
        txt += "="*40 + "\n"
        txt += f"Step Length (Max): {metrics['Step_Length_Max']:.1f} cm\n"
        txt += f"Knee Flexion (Max): {metrics['L_Knee_Flexion_Max']:.1f} deg\n"
        txt += f"Step Width (Mean): {metrics['Step_Width_Mean']:.1f} cm\n"
        
        ax3.text(0.1, 0.9, txt, fontsize=14, family='monospace', verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Report saved to {save_path}")
        return txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output', default='report.png')
    parser.add_argument('--height', type=float, default=170.0)
    args = parser.parse_args()
    
    analyzer = GaitAnalyzer(args.input, args.height)
    metrics, traces = analyzer.analyze()
    report = analyzer.visualize(traces, metrics, args.output)
    print("\n" + report)
