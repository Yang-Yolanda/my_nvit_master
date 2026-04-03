
import argparse
import sys
import os
import torch
from pathlib import Path

# Path setup
sys.path.append(str(Path.cwd())) 
sys.path.append("/home/yangz/PHALP-master")
sys.path.append("/home/yangz/4D-Humans")
sys.path.append("/home/yangz/NViT-master/nvit/Code_Paper2_Implementation")

from phalp.visualize.visualizer import Visualizer
from phalp.trackers.PHALP import PHALP
from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from nvit2_models.guided_hmr2 import GuidedHMR2Module

class CustomPredictor(HMR2018Predictor):
    def __init__(self, cfg, checkpoint_path):
        super().__init__(cfg)
        print(f"Loading Custom Checkpoint: {checkpoint_path}")
        self.model = GuidedHMR2Module.load_from_checkpoint(checkpoint_path, strict=False)
        self.model.eval()
        self.model.to(torch.device('cuda'))

    def forward(self, x):
        # x is (B, 3, 256, 256) normalized
        # Standard PHALP/HMR flow
        hmar_out = self.hmar_old(x)
        
        batch = {'img': x} # Add mask if needed? GuidedHMR usually takes raw img.
        
        with torch.no_grad():
            model_out = self.model.forward_step(batch, train=False)
            
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
            'pred_keypoints_3d': model_out['pred_keypoints_3d']
        }
        return out

class VideoEvaluator(PHALP):
    def __init__(self, cfg, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        super().__init__(cfg)
        
    def setup_hmr(self):
        self.HMAR = CustomPredictor(self.cfg, self.checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to video or image folder')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='output/video_eval')
    parser.add_argument('--track_id', type=int, default=0)
    args = parser.parse_args()

    cfg = FullConfig()
    cfg.video.source = args.input
    cfg.video.output_dir = args.output
    cfg.render.enable = True
    
    evaluator = VideoEvaluator(cfg, args.checkpoint)
    evaluator.track()
    
    print(f"Tracking completed. Results in {args.output}")

if __name__ == '__main__':
    main()
