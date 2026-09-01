import io
import runpy
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


E2E_SCRIPT = Path(__file__).parents[1] / "src" / "test_e2e.py"


def _stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]]):
        self.rows = rows

    def select(self, indices) -> "_FakeDataset":
        return _FakeDataset([self.rows[index] for index in indices])

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


def _run_e2e(
    official_scores: list[float],
    format_scores: list[float],
    wrong_score: float,
) -> tuple[int | None, str, mock.Mock, mock.Mock]:
    partial_reward = mock.Mock(
        side_effect=(official_scores, [wrong_score])
    )
    binary_reward = mock.Mock(
        side_effect=(official_scores, [wrong_score])
    )
    format_reward = mock.Mock(return_value=format_scores)

    transformed = _FakeDataset(
        [
            {
                "task_id": f"HumanEval/{index}",
                "verification_info": {
                    "test_code": "def check(candidate): pass",
                    "entry_point": f"candidate_{index}",
                },
            }
            for index in range(3)
        ]
    )
    raw = _FakeDataset(
        [
            {
                "prompt": f"def candidate_{index}():\n",
                "canonical_solution": f"    return {index}\n",
            }
            for index in range(3)
        ]
    )
    stubs = {
        "data_prep": _stub_module(
            "data_prep", load_humaneval=lambda split: transformed
        ),
        "datasets": _stub_module(
            "datasets", load_dataset=lambda *args, **kwargs: raw
        ),
        "reward_funcs": _stub_module(
            "reward_funcs",
            code_reward_humaneval=binary_reward,
            code_reward_humaneval_partial=partial_reward,
            format_reward=format_reward,
        ),
    }

    output = io.StringIO()
    exit_code = None
    with mock.patch.object(sys, "path", sys.path.copy()):
        with mock.patch.dict(sys.modules, stubs):
            with redirect_stdout(output):
                try:
                    runpy.run_path(str(E2E_SCRIPT), run_name="__main__")
                except SystemExit as exc:
                    exit_code = exc.code

    return exit_code, output.getvalue(), partial_reward, binary_reward


class TestE2EContract(unittest.TestCase):
    def test_success_uses_training_reward_and_exits_zero(self):
        exit_code, output, partial_reward, binary_reward = _run_e2e(
            official_scores=[1.0, 1.0, 1.0],
            format_scores=[0.5, 0.5, 0.5],
            wrong_score=0.0,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(partial_reward.call_count, 2)
        binary_reward.assert_not_called()
        negative_completion = partial_reward.call_args_list[1].args[0][0]
        negative_code = negative_completion[-1]["content"]
        self.assertIn("raise RuntimeError", negative_code)
        self.assertIn("negative control", negative_code)
        self.assertIn("[OK] 端到端测试完成!", output)

    def test_any_failed_check_exits_nonzero_without_success_message(self):
        cases = (
            ("official reward", [1.0, 0.5, 1.0], [0.5, 0.5, 0.5], 0.0),
            ("missing official reward", [1.0, 1.0], [0.5, 0.5, 0.5], 0.0),
            ("format reward", [1.0, 1.0, 1.0], [0.5, 0.0, 0.5], 0.0),
            ("missing format reward", [1.0, 1.0, 1.0], [0.5, 0.5], 0.0),
            ("negative control", [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 0.5),
        )

        for name, official_scores, format_scores, wrong_score in cases:
            with self.subTest(check=name):
                exit_code, output, _, _ = _run_e2e(
                    official_scores=official_scores,
                    format_scores=format_scores,
                    wrong_score=wrong_score,
                )

                self.assertEqual(exit_code, 1)
                self.assertIn("[FAIL]", output)
                self.assertNotIn("[OK] 端到端测试完成!", output)


if __name__ == "__main__":
    unittest.main()
