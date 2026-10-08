import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from mma_safetybench import cli, runtime


class RuntimeTests(unittest.TestCase):
    def test_endpoint_accepts_base_and_full_path(self):
        self.assertEqual(
            runtime.endpoint("https://provider.test/v1/"),
            "https://provider.test/v1/chat/completions",
        )
        self.assertEqual(
            runtime.endpoint("https://provider.test/v1/chat/completions"),
            "https://provider.test/v1/chat/completions",
        )

    def test_rejects_secret_urls_and_placeholders(self):
        for url in [
            "YOUR_API_URL",
            "https://api.example.com/v1",
            "https://host/v1?key=secret",
            "https://user:pass@host/v1",
        ]:
            with self.assertRaises(runtime.EvaluationError):
                runtime.endpoint(url)

    def test_scores_do_not_default_to_safe(self):
        for value in ["", "API Error", "#thescore: 2", "#thescore: 5 #thescore: 1"]:
            with self.assertRaises(runtime.EvaluationError):
                runtime.parse_score(value)
        self.assertEqual(runtime.parse_score("#thescore: 3"), 3)
        for value in [True, "5", None, 2]:
            with self.assertRaises(runtime.EvaluationError):
                runtime.score_value(value)

    def test_coordinates_preserve_zero_and_explicit_space(self):
        self.assertEqual(
            runtime.coordinates("<point>0, 20</point>", 500, 500, "pixels")[:2], (0, 20)
        )
        self.assertEqual(
            runtime.coordinates("<point>500, 500</point>", 800, 600, "normalized")[:2], (400, 300)
        )
        self.assertEqual(runtime.coordinates("no click", 100, 100)[:2], (None, None))

    def test_fixed_denominator_and_incomplete_status(self):
        result = runtime.summarize(
            [{"status": "ok", "score": 5, "is_hit": 1}, {"status": "error"}], 209
        )
        self.assertEqual(result["expected"], 209)
        self.assertEqual(result["errors"], 1)
        self.assertIsNone(result["asr_percent"])
        self.assertAlmostEqual(result["asr_lower_bound_percent"], 100 / 209)
        self.assertFalse(result["complete"])

    def test_complete_metrics(self):
        result = runtime.summarize([{"status": "ok", "score": 3, "is_hit": 0}], 1)
        self.assertEqual(result["asr_percent"], 0)
        self.assertEqual(result["scr_percent"], 100)

    def test_local_path_and_unique_legacy_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "domain"
            (root / "images").mkdir(parents=True)
            file = root / "images" / "one.png"
            file.touch()
            for reference in ["images/one.png", "domain/images/one.png", "/old/machine/one.png"]:
                self.assertEqual(runtime.resolve_image(reference, root), str(file.resolve()))
            for reference in ["https://host/one.png", "../one.png"]:
                with self.assertRaises(runtime.EvaluationError):
                    runtime.resolve_image(reference, root)
            (root / "one.png").touch()
            with self.assertRaises(runtime.EvaluationError):
                runtime.resolve_image("/old/machine/one.png", root)

    def test_valid_encoding_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "sample.png"
            Image.new("RGBA", (20, 20)).save(file)
            encoded = runtime.encode_jpeg(file)
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.assertEqual(image.format, "JPEG")
            file.write_bytes(b"not an image")
            with self.assertRaises(runtime.EvaluationError):
                runtime.encode_jpeg(file)

    def test_web_mapping_preserves_task_subdirectories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "03_web"
            for task in ["a", "b"]:
                folder = root / "images_200" / task
                folder.mkdir(parents=True)
                (folder / "fake_popup.jpg").touch()
            self.assertEqual(
                runtime.resolve_image("./images/a/fake_popup.jpg", root),
                str((root / "images_200/a/fake_popup.jpg").resolve()),
            )

    @patch("mma_safetybench.runtime.requests.post")
    def test_http_failure_never_becomes_a_model_response(self, post):
        post.return_value.status_code = 401
        with self.assertRaises(runtime.EvaluationError):
            runtime.post_chat("https://provider.test/v1", "secret", {})
        self.assertEqual(post.call_count, 1)

    @patch("mma_safetybench.runtime.time.sleep")
    @patch("mma_safetybench.runtime.requests.post")
    def test_server_retry(self, post, sleep):
        bad = Mock(status_code=503)
        good = Mock(status_code=200)
        good.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        post.side_effect = [bad, good]
        self.assertEqual(runtime.post_chat("https://provider.test/v1", "secret", {}), "ok")
        self.assertEqual(post.call_count, 2)


class AdapterTests(unittest.TestCase):
    def test_cli_partial_run_keeps_full_gui_denominator(self):
        tasks = [{"task_id": f"task_{i}"} for i in range(209)]
        config = {
            "VICTIM_API_KEY": "private",
            "JUDGE_API_KEY": "private",
            "VICTIM_API_URL": "https://provider.test/v1",
            "JUDGE_API_URL": "https://provider.test/v1",
            "VICTIM_MODEL": "victim",
            "JUDGE_MODEL": "judge",
        }
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, config):
            output = Path(temp) / "run"
            args = SimpleNamespace(
                domain="gui", defense=False, data_root=None, output=output, limit=2, dry_run=False
            )
            with (
                patch.object(cli, "load_tasks", return_value=tasks),
                patch.object(cli, "check_task"),
                patch.object(
                    cli,
                    "gui_task",
                    side_effect=[{"score": 5, "is_hit": 1}, runtime.EvaluationError("failure")],
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(cli.evaluate(args), 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["expected"], 209)
            self.assertEqual(summary["valid"], 1)
            self.assertEqual(summary["errors"], 1)
            self.assertIsNone(summary["asr_percent"])
            self.assertEqual(json.loads((output / "task_1.json").read_text())["score"], None)
            self.assertNotIn("private", (output / "config.json").read_text())
            with (
                patch.object(cli, "load_tasks", return_value=tasks),
                patch.object(cli, "check_task"),
            ):
                with self.assertRaises(runtime.EvaluationError):
                    cli.evaluate(args)

    def test_cli_dry_run_does_not_call_models(self):
        tasks = [{"task_id": f"task_{i}"} for i in range(209)]
        args = SimpleNamespace(
            domain="gui",
            defense=False,
            data_root=None,
            output=Path("unused"),
            limit=None,
            dry_run=True,
        )
        with (
            patch.object(cli, "load_tasks", return_value=tasks),
            patch.object(cli, "check_task"),
            patch.object(cli, "gui_task") as run,
            patch("builtins.print"),
        ):
            self.assertEqual(cli.evaluate(args), 0)
            run.assert_not_called()

    def test_dataset_size_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.json"
            module = SimpleNamespace(DATASET_JSON_PATH=str(path))
            path.write_text(json.dumps([{"task_id": "a"}]))
            with self.assertRaises(runtime.EvaluationError):
                cli.load_tasks("web", module)
            path.write_text(json.dumps([{"task_id": "a"}] * 200))
            with self.assertRaises(runtime.EvaluationError):
                cli.load_tasks("web", module)

    def test_all_adapters_import_without_clients_or_output(self):
        for domain in cli.DOMAINS:
            for defense in [False, True]:
                with patch("builtins.print") as printer:
                    module = cli.load_module(domain, defense)
                printer.assert_not_called()
                if domain == "mobile":
                    self.assertIsNone(module.victim_client)

    def test_invoice_retains_semantic_score_three(self):
        module = cli.load_module("invoice", False)
        victim = Mock()
        victim.chat.return_value = "response"
        judge = Mock()
        judge.chat.return_value = "#thescore: 3"
        task = {
            "task_id": "invoice_test",
            "fake_popup_image": "local.png",
            "anchor_prompt": "task",
            "benign_task": "read",
            "attack_prompt": "test",
            "attack_dimension": "test",
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(module, "fix_path", return_value="local.png"),
            patch.object(module, "encode_img", return_value="encoded"),
        ):
            module.RESULT_DIR = temp
            self.assertEqual(module.evaluate_task(task, victim, judge)[1], 3)
            judge.chat.return_value = "invalid judge reply"
            self.assertIsNone(module.evaluate_task(task, victim, judge)[1])

    def test_gui_variants_use_coordinates_independently_of_judge_score(self):
        module = cli.load_module("gui", False)
        module.VICTIM_API_URL = "https://provider.test/v1"
        module.VICTIM_API_KEY = "secret"
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "image.png"
            Image.new("RGB", (1000, 1000)).save(file)
            task = {
                "task_id": "a",
                "poisoned_image_path": "image.png",
                "instruction": "task",
                "malicious_bbox": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
            }
            cases = [
                ("<point>0, 0</point>", 1, 1, [0, 0]),
                ("<point>0, 0</point>", 3, 1, [0, 0]),
                ("<point>50, 50</point>", 5, 0, [50, 50]),
                ("No click", 5, 0, None),
            ]
            for defense in [False, True]:
                module = cli.load_module("gui", defense)
                for answer, score, hit, point in cases:
                    with (
                        self.subTest(defense=defense, answer=answer, score=score),
                        patch.object(cli, "resolve_image", return_value=str(file)),
                        patch.object(cli, "post_chat", return_value=answer),
                        patch.object(
                            module,
                            "gpt5_judge",
                            return_value={"toxic_score": score, "reasoning": "test"},
                        ),
                    ):
                        result = cli.gui_task(module, task, defense)
                        self.assertEqual(result["click_coord"], point)
                        self.assertEqual(result["is_hit"], hit)
                        self.assertEqual(result["coord_hit"], hit)
                        self.assertEqual(result["score"], score)
                        self.assertEqual(result["asr_rule"], "coordinate_target_match")

    def test_mobile_empty_negative_button_does_not_match_everything(self):
        for defense in [False, True]:
            module = cli.load_module("mobile", defense)
            field = "score" if defense else "toxic_score"
            response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps({field: 5, "is_hit": 1}))
                    )
                ]
            )
            module.judge_client = Mock()
            module.judge_client.chat.completions.create.return_value = response
            self.assertEqual(
                module.call_judge_evaluator(
                    "task", "thought", "CLICK", "Authorize", {"pos_btn": "Authorize", "neg_btn": ""}
                ),
                (5, 1),
            )
            module.judge_client.chat.completions.create.side_effect = RuntimeError("network")
            self.assertEqual(
                module.call_judge_evaluator("task", "thought", "CLICK", "Authorize", {}),
                (None, None),
            )

    def test_mobile_png_has_no_random_trailing_bytes(self):
        module = cli.load_module("mobile", False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            Image.new("RGB", (5, 5)).save(path)
            a = base64.b64decode(module.get_stitched_base64(path, path))
            b = base64.b64decode(module.get_stitched_base64(path, path))
            self.assertEqual(a, b)
            self.assertTrue(a.endswith(b"IEND\xaeB`\x82"))


if __name__ == "__main__":
    unittest.main()
