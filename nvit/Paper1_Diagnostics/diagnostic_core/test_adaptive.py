
import unittest
import numpy as np
from diagnostic_engine import ViTDiagnosticLab

class TestAdaptiveMasking(unittest.TestCase):
    def test_suggest_split(self):
        from unittest.mock import MagicMock
        mock_wrapper = MagicMock()
        mock_wrapper.model = MagicMock()
        lab = ViTDiagnosticLab(mock_wrapper, model_name="test_model")
        lab.total_layers = 32
        # Mock metrics: High entropy -> Dip (L8) -> Rise -> Dip (L24)
        lab.layer_metrics = {}
        for i in range(32):
            val = 10.0
            if i == 8: val = 2.0 # First dip
            if i == 24: val = 1.0 # Second dip
            if 8 < i < 24: val = 5.0 # Rise in middle
            lab.layer_metrics[i] = {'entropy': [val]}
            
        split1, split2 = lab.analyze_metrics_and_suggest_split()
        print(f"Splits: {split1}, {split2}")
        self.assertEqual(split1, 8)
        self.assertEqual(split2, 24)
        
        # Mock model name and model (None)
        from unittest.mock import MagicMock
        mock_wrapper = MagicMock()
        mock_wrapper.model = MagicMock()
        lab = ViTDiagnosticLab(mock_wrapper, model_name="test_model")
        # Manually set total_layers as it's usually derived from model name/config
        lab.total_layers = 32
        lab.add_adaptive_group(8, 24)
        group = lab.groups['Adaptive-8-24']
        # Correct assertion: No mask for first layers
        self.assertIsNone(group['layer_modes'].get(0))
        # My implementation:
        # if i < split_layer_1: pass (No mask? Default mode?)
        # Base engine runner logic handles 'hybrid' mode.
        # If 'hybrid', it looks up layer_modes. If key missing? Defaults to None (No Mask).
        # Let's check logic.
        
        self.assertEqual(group['layer_modes'].get(0), None)
        self.assertEqual(group['layer_modes'][10], 'soft')
        self.assertEqual(group['layer_modes'][25], 'hard')

if __name__ == '__main__':
    unittest.main()
