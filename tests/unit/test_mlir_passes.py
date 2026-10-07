import unittest

from compiler.mlir import DEFAULT_PASS_PIPELINE


class TestMLIRPasses(unittest.TestCase):
    """Pass pipeline definitions for the compiler."""

    def test_default_pipeline_contains_expected_order(self):
        self.assertEqual(
            DEFAULT_PASS_PIPELINE.as_list(),
            [
                "model-to-linalg",
                "tile-linalg",
                "linalg-to-mgpu",
                "mgpu-to-isa",
            ],
        )

    def test_default_pipeline_has_descriptions(self):
        self.assertTrue(
            all(pass_.description for pass_ in DEFAULT_PASS_PIPELINE.passes)
        )


if __name__ == "__main__":
    unittest.main()
