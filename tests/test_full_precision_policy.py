"""Regression policy: production training/evaluation must remain full precision."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FullPrecisionPolicyTest(unittest.TestCase):
    def test_production_pipeline_has_no_amp_enable_path(self):
        config = (ROOT / "oemseg/config.py").read_text()
        trainer = (ROOT / "oemseg/engine/trainer.py").read_text()
        evaluator = (ROOT / "oemseg/engine/evaluator.py").read_text()
        visualization = (ROOT / "oemseg/utils/visualization.py").read_text()
        kaggle = (ROOT / "scripts/kaggle_paper_repro.sh").read_text()

        self.assertNotIn('parser.add_argument("--mixed-precision"', config)
        self.assertIn('parser.set_defaults(mixed_precision="no")', config)
        self.assertIn('mixed_precision="no"', trainer)
        self.assertNotIn('mixed_precision=args.mixed_precision', trainer)
        self.assertNotIn('accelerator.autocast()', trainer)
        self.assertNotIn('accelerator.autocast()', evaluator)
        self.assertNotIn('accelerator.autocast()', visualization)
        self.assertNotIn('--mixed-precision', kaggle)


if __name__ == "__main__":
    unittest.main()
