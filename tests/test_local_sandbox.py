import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import local_sandbox


class TestSandboxInterpreter(unittest.TestCase):
    @patch("local_sandbox.subprocess.run")
    def test_run_one_test_uses_current_interpreter(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="3\n", stderr="")

        self.assertTrue(local_sandbox.run_one_test("print(3)", "", "3"))
        self.assertEqual(mock_run.call_args.args[0][:2], [sys.executable, "-c"])

    @patch("local_sandbox.subprocess.run")
    def test_run_humaneval_test_uses_current_interpreter(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")

        self.assertTrue(local_sandbox.run_humaneval_test("def f(): pass", "def check(candidate): pass", "f"))
        self.assertEqual(mock_run.call_args.args[0][:2], [sys.executable, "-c"])

    @patch("local_sandbox.subprocess.run")
    def test_partial_humaneval_reward_uses_current_interpreter(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout="__MINI_R1_PASSED_COUNT__=1/1\n",
            stderr="",
        )

        score = local_sandbox.compute_humaneval_pass_rate(
            "def f(): return 1",
            "def check(candidate):\n    assert candidate() == 1",
            "f",
        )

        self.assertEqual(score, 1.0)
        self.assertEqual(mock_run.call_args.args[0][:2], [sys.executable, "-c"])

    def test_partial_humaneval_reward_preserves_check_setup(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            "def f(): return 3",
            "def check(candidate):\n    expected = 3\n    assert candidate() == expected",
            "f",
        )

        self.assertEqual(score, 1.0)

    def test_partial_humaneval_reward_counts_loop_assertions(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            "def f(value): return value if value < 2 else -1",
            (
                "def check(candidate):\n"
                "    offset = 0\n"
                "    for value in range(3):\n"
                "        assert candidate(value) == value + offset"
            ),
            "f",
        )

        self.assertAlmostEqual(score, 2 / 3)

    def test_partial_humaneval_reward_ignores_candidate_stdout(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            'def f():\n    print("debug", end="")\n    return 1',
            "def check(candidate):\n    assert candidate() == 1",
            "f",
        )

        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
