import unittest

from compiler.model_ir import (
    DType,
    Graph,
    ModelIRError,
    Node,
    Op,
    Value,
    infer_node_outputs,
)


class TestModelIR(unittest.TestCase):
    """Unit tests for the unified Model IR."""

    def test_value_metadata(self):
        value = Value("x", DType.F32, (2, 3, 4))
        self.assertEqual(value.rank, 3)
        self.assertEqual(value.numel, 24)

    def test_value_rejects_invalid_shape(self):
        with self.assertRaises(ModelIRError):
            Value("x", DType.F32, (2, -1))

    def test_graph_rejects_undefined_input(self):
        graph = Graph(name="broken")
        graph.add_input("x", DType.F32, (4,))
        graph.nodes.append(
            Node(
                name="relu",
                op=Op.RELU,
                inputs=[Value("missing", DType.F32, (4,))],
                outputs=[Value("y", DType.F32, (4,))],
            )
        )

        with self.assertRaisesRegex(ModelIRError, "undefined value"):
            graph.validate()

    def test_infer_conv2d_shape(self):
        x = Value("x", DType.F32, (1, 3, 8, 8))
        weight = Value("w", DType.F32, (4, 3, 3, 3))

        outputs = infer_node_outputs(
            Op.CONV2D,
            [x, weight],
            {"strides": 1, "padding": 0, "dilation": 1, "groups": 1},
        )

        self.assertEqual(outputs, [(DType.F32, (1, 4, 6, 6))])

    def test_format_graph_prints_typed_values(self):
        graph = Graph(name="tiny")
        x = graph.add_input("x", DType.F32, (2, 3))
        y = graph.add_node(Op.RELU, [x])[0]
        graph.mark_outputs([y])

        text = graph.format()

        self.assertIn("graph @tiny(", text)
        self.assertIn("%x: f32<2x3>", text)
        self.assertIn('"relu"', text)


if __name__ == "__main__":
    unittest.main()
