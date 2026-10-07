import unittest

from compiler.frontends.manual_frontend import ManualFrontend
from compiler.model_ir import DType, ModelIRError


class TestManualFrontend(unittest.TestCase):
    """Build, print, and validate representative Model IR graphs."""

    def test_cnn_graph_builds_and_validates(self):
        frontend = ManualFrontend("cnn_block")

        x = frontend.add_input("x", DType.F32, (1, 3, 8, 8))
        conv_w = frontend.add_input("conv_w", DType.F32, (4, 3, 3, 3))
        conv_b = frontend.add_input("conv_b", DType.F32, (4,))
        fc_w = frontend.add_input("fc_w", DType.F32, (10, 36))
        fc_b = frontend.add_input("fc_b", DType.F32, (10,))

        y = frontend.conv2d(x, conv_w, conv_b)
        y = frontend.relu(y)
        y = frontend.max_pool2d(y, kernel=2, strides=2)
        y = frontend.reshape(y, (1, -1))
        logits = frontend.linear(y, fc_w, fc_b)
        output = frontend.sigmoid(logits)
        frontend.output(output)

        graph = frontend.build()
        text = graph.format()

        self.assertEqual(output.shape, (1, 10))
        self.assertIn('"conv2d"', text)
        self.assertIn('"max_pool2d"', text)
        self.assertIn('"linear"', text)
        self.assertIn('"sigmoid"', text)

    def test_transformer_graph_builds_and_validates(self):
        frontend = ManualFrontend("transformer_block")

        x = frontend.add_input("x", DType.F32, (1, 4, 8, 16))
        q_w = frontend.add_input("q_w", DType.F32, (16, 16))
        k_w = frontend.add_input("k_w", DType.F32, (16, 16))
        v_w = frontend.add_input("v_w", DType.F32, (16, 16))
        cos = frontend.add_input("cos", DType.F32, (1, 4, 8, 16))
        sin = frontend.add_input("sin", DType.F32, (1, 4, 8, 16))
        norm_w = frontend.add_input("norm_w", DType.F32, (16,))
        norm_b = frontend.add_input("norm_b", DType.F32, (16,))

        q = frontend.rope(frontend.linear(x, q_w), cos, sin)
        k = frontend.linear(x, k_w)
        v = frontend.linear(x, v_w)
        mask = frontend.causal_mask(query_len=8, key_len=8, batch=1)
        attn = frontend.masked_attention(q, k, v, mask)
        residual = frontend.add(x, attn)
        output = frontend.layer_norm(residual, norm_w, norm_b)
        frontend.output(output)

        graph = frontend.build()
        text = graph.format()

        self.assertEqual(output.shape, (1, 4, 8, 16))
        self.assertIn('"rope"', text)
        self.assertIn('"masked_attention"', text)
        self.assertIn('"layer_norm"', text)

    def test_shape_layout_ops_build_and_validate(self):
        frontend = ManualFrontend("shape_ops")

        x = frontend.add_input("x", DType.F32, (1, 3, 4, 4))
        y = frontend.transpose(x, (0, 2, 3, 1))
        y = frontend.reshape(y, (1, 48))
        y = frontend.unsqueeze(y, (0,))
        y = frontend.expand(y, (2, 1, 48))
        y = frontend.squeeze(y, (1,))
        y = frontend.slice(y, (0, 0), (2, 16), axes=(0, 1))
        y = frontend.pad(y, ((0, 0), (1, 2)))
        y = frontend.concat(y, y, axis=1)
        frontend.output(y)

        graph = frontend.build()

        self.assertEqual(y.shape, (2, 38))
        self.assertIn('"transpose"', graph.format())
        self.assertIn('"reshape"', graph.format())
        self.assertIn('"concat"', graph.format())

    def test_additional_cnn_and_transformer_ops(self):
        frontend = ManualFrontend("additional_ops")

        x = frontend.add_input("x", DType.F32, (1, 3, 4, 4))
        depthwise_w = frontend.add_input("dw", DType.F32, (3, 1, 3, 3))
        depthwise = frontend.depthwise_conv2d(x, depthwise_w)
        pooled = frontend.global_avg_pool(depthwise)
        scale = frontend.add_input("scale", DType.F32, (3,))
        bias = frontend.add_input("bias", DType.F32, (3,))
        mean = frontend.add_input("mean", DType.F32, (3,))
        variance = frontend.add_input("variance", DType.F32, (3,))
        norm_w = frontend.add_input("norm_w", DType.F32, (4,))
        batch_norm = frontend.batch_norm(depthwise, scale, bias, mean, variance)
        grouped = frontend.group_norm(
            frontend.relu(batch_norm), scale, bias, num_groups=1
        )
        normalized = frontend.rms_norm(frontend.reshape(grouped, (1, 3, 4)), norm_w)
        boxes = frontend.add_input("boxes", DType.F32, (5, 4))
        deltas = frontend.add_input("deltas", DType.F32, (5, 4))
        scores = frontend.add_input("scores", DType.F32, (5,))
        decoded = frontend.box_decode(boxes, deltas)
        kept = frontend.nms(decoded, scores)
        cache = frontend.add_input("cache", DType.F32, (1, 4, 8, 16))
        value = frontend.add_input("kv_value", DType.F32, (1, 4, 8, 16))
        positions = frontend.add_input("positions", DType.I64, (1, 4))
        appended = frontend.kv_cache_append(cache, value)
        gathered = frontend.kv_cache_gather(appended, positions)
        frontend.output(normalized, kept, gathered)

        graph = frontend.build()

        self.assertEqual(pooled.shape, (1, 3, 1, 1))
        self.assertEqual(gathered.shape, (1, 4, 4, 16))
        self.assertIn('"depthwise_conv2d"', graph.format())
        self.assertIn('"batch_norm"', graph.format())
        self.assertIn('"kv_cache_gather"', graph.format())

    def test_invalid_graph_fails_validation(self):
        frontend = ManualFrontend("bad_matmul")
        a = frontend.add_input("a", DType.F32, (2, 3))
        b = frontend.add_input("b", DType.F32, (4, 5))

        with self.assertRaises(ModelIRError):
            frontend.matmul(a, b)


if __name__ == "__main__":
    unittest.main()
