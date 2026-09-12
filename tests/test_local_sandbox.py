import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import local_sandbox


class TestSandboxInterpreter(unittest.TestCase):
    def test_stdio_keeps_assertions_and_preserves_other_environment(self) -> None:
        code = (
            "import os\n"
            "assert int(input()) > 0\n"
            "print(os.environ['MINI_R1_TEST_ENV'])"
        )
        for level in ("1", "2"):
            with self.subTest(optimization=level):
                with patch.dict(os.environ, {
                    "PYTHONOPTIMIZE": level, "MINI_R1_TEST_ENV": "preserved"
                }):
                    self.assertTrue(local_sandbox.run_one_test(code, "1", "preserved"))
                    self.assertFalse(local_sandbox.run_one_test(code, "0", "preserved"))
                    self.assertEqual(os.environ["PYTHONOPTIMIZE"], level)
                    self.assertEqual(os.environ["MINI_R1_TEST_ENV"], "preserved")

    def test_binary_humaneval_keeps_assertions_when_parent_enables_optimization(self) -> None:
        test_code = (
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(2) == 2"
        )
        for level in ("1", "2"):
            with self.subTest(optimization=level):
                with patch.dict(os.environ, {"PYTHONOPTIMIZE": level}):
                    self.assertTrue(local_sandbox.run_humaneval_test(
                        "def f(value): return value", test_code, "f"
                    ))
                    self.assertFalse(local_sandbox.run_humaneval_test(
                        "def f(value): return 1", test_code, "f"
                    ))
                    self.assertEqual(os.environ["PYTHONOPTIMIZE"], level)

    def test_partial_humaneval_keeps_assertions_when_parent_enables_optimization(self) -> None:
        test_code = (
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(2) == 2"
        )
        for level in ("1", "2"):
            with self.subTest(optimization=level):
                with patch.dict(os.environ, {"PYTHONOPTIMIZE": level}):
                    self.assertEqual(local_sandbox.compute_humaneval_pass_rate(
                        "def f(value): return value", test_code, "f"
                    ), 1.0)
                    self.assertEqual(local_sandbox.compute_humaneval_pass_rate(
                        "def f(value): return 1", test_code, "f"
                    ), 0.5)
                    self.assertEqual(os.environ["PYTHONOPTIMIZE"], level)

    def test_stdio_timeout_does_not_skip_later_cases(self) -> None:
        # Finite sleep keeps a timeout regression from hanging the test suite.
        code = (
            "import time\n"
            "value = int(input())\n"
            "if value == 0:\n"
            "    time.sleep(2)\n"
            "print(value)"
        )
        score = local_sandbox.compute_pass_rate(
            code,
            [{"input": "0", "output": "0"}, {"input": "1", "output": "1"}],
            timeout=1,
        )
        self.assertEqual(score, 0.5)

    def test_binary_humaneval_times_out_then_scores_next_candidate(self) -> None:
        test_code = "def check(candidate):\n    assert candidate() == 1"
        slow_code = "import time\ndef f():\n    time.sleep(2)\n    return 1"
        self.assertFalse(
            local_sandbox.run_humaneval_test(slow_code, test_code, "f", timeout=1)
        )
        self.assertTrue(
            local_sandbox.run_humaneval_test("def f(): return 1", test_code, "f")
        )

    def test_partial_humaneval_timeout_discards_incomplete_score(self) -> None:
        test_code = (
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(1) == 1"
        )
        slow_code = (
            "import time\n"
            "def f(value):\n"
            "    if value == 1:\n"
            "        time.sleep(2)\n"
            "    return value"
        )
        self.assertEqual(
            local_sandbox.compute_humaneval_pass_rate(
                slow_code, test_code, "f", timeout=1
            ),
            0.0,
        )
        self.assertEqual(
            local_sandbox.compute_humaneval_pass_rate(
                "def f(value): return value", test_code, "f"
            ),
            1.0,
        )

    @patch("local_sandbox.subprocess.run")
    def test_run_one_test_uses_current_interpreter(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, stdout="3\n", stderr=""
        )

        self.assertTrue(local_sandbox.run_one_test("print(3)", "", "3"))
        self.assertEqual(mock_run.call_args.args[0][:2], [sys.executable, "-c"])

    @patch("local_sandbox.subprocess.run")
    def test_run_humaneval_test_uses_current_interpreter(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout="__MINI_R1_HUMANEVAL_SUCCESS__:0123456789abcdef\n",
            stderr="",
        )

        with patch(
            "local_sandbox.secrets.token_hex",
            return_value="0123456789abcdef",
        ):
            passed = local_sandbox.run_humaneval_test(
                "def f(): pass",
                "def check(candidate): pass",
                "f",
            )

        self.assertTrue(passed)
        self.assertEqual(mock_run.call_args.args[0][:2], [sys.executable, "-c"])

    def test_run_humaneval_test_rejects_exit_code_spoof(self):
        self.assertFalse(
            local_sandbox.run_humaneval_test(
                (
                    "import atexit\n"
                    "import os\n"
                    "atexit.register(lambda: os._exit(0))\n"
                    "def f(value):\n"
                    "    return 0"
                ),
                "def check(candidate):\n    assert candidate(1) == 1",
                "f",
            )
        )

    def test_run_humaneval_test_rejects_early_clean_exit(self):
        self.assertFalse(
            local_sandbox.run_humaneval_test(
                "raise SystemExit(0)",
                "def check(candidate):\n    assert candidate(1) == 1",
                "f",
            )
        )

    def test_run_humaneval_test_accepts_correct_candidate(self):
        self.assertTrue(
            local_sandbox.run_humaneval_test(
                "def f(value):\n    return value + 1",
                "def check(candidate):\n    assert candidate(1) == 2",
                "f",
            )
        )

    def test_run_humaneval_test_preserves_candidate_check_helper(self):
        self.assertTrue(
            local_sandbox.run_humaneval_test(
                (
                    "def check(value):\n"
                    "    return value + 1\n"
                    "def f(value):\n"
                    "    return check(value)"
                ),
                "def check(candidate):\n    assert candidate(1) == 2",
                "f",
            )
        )

    def test_run_humaneval_test_isolates_test_builtins(self):
        self.assertFalse(
            local_sandbox.run_humaneval_test(
                (
                    "import builtins\n"
                    "builtins.abs = lambda value: 0\n"
                    "def f(value):\n"
                    "    return 0"
                ),
                "def check(candidate):\n    assert candidate(-3) == abs(-3)",
                "f",
            )
        )

    def test_run_humaneval_test_does_not_copy_candidate_abs_to_tests(self):
        self.assertFalse(
            local_sandbox.run_humaneval_test(
                "abs = lambda value: 0\ndef f(value):\n    return 0",
                "def check(candidate):\n    assert candidate(-3) == abs(-3)",
                "f",
            )
        )

    def test_run_humaneval_test_restores_builtins_import_for_tests(self):
        self.assertFalse(
            local_sandbox.run_humaneval_test(
                (
                    "import builtins\n"
                    "builtins.abs = lambda value: 0\n"
                    "def f(value):\n"
                    "    return 0"
                ),
                (
                    "def check(candidate):\n"
                    "    import builtins\n"
                    "    assert candidate(-3) == builtins.abs(-3)"
                ),
                "f",
            )
        )

    @patch("local_sandbox.subprocess.run")
    def test_partial_humaneval_reward_uses_current_interpreter(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout="__MINI_R1_PASSED_COUNT__:0123456789abcdef=1/1\n",
            stderr="",
        )

        with patch(
            "local_sandbox.secrets.token_hex",
            return_value="0123456789abcdef",
        ):
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
            (
                "def check(candidate):\n"
                "    expected = 3\n"
                "    assert candidate() == expected"
            ),
            "f",
        )

        self.assertEqual(score, 1.0)

    def test_partial_humaneval_reward_preserves_candidate_check_helper(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "def check(value):\n"
                "    return value + 1\n"
                "def f(value):\n"
                "    return check(value)"
            ),
            "def check(candidate):\n    assert candidate(1) == 2",
            "f",
        )

        self.assertEqual(score, 1.0)

    def test_partial_humaneval_reward_isolates_test_builtins(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "import builtins\n"
                "builtins.abs = lambda value: 0\n"
                "def f(value):\n"
                "    return 0"
            ),
            "def check(candidate):\n    assert candidate(-3) == abs(-3)",
            "f",
        )

        self.assertEqual(score, 0.0)

    def test_partial_humaneval_reward_does_not_copy_candidate_abs_to_tests(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            "abs = lambda value: 0\ndef f(value):\n    return 0",
            "def check(candidate):\n    assert candidate(-3) == abs(-3)",
            "f",
        )

        self.assertEqual(score, 0.0)

    def test_partial_humaneval_reward_restores_builtins_import_for_tests(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "import builtins\n"
                "builtins.abs = lambda value: 0\n"
                "def f(value):\n"
                "    return 0"
            ),
            (
                "def check(candidate):\n"
                "    import builtins\n"
                "    assert candidate(-3) == builtins.abs(-3)"
            ),
            "f",
        )

        self.assertEqual(score, 0.0)

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

    def test_partial_humaneval_reward_counts_loop_case_errors(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "def f(value):\n"
                "    if value == 1:\n"
                "        raise ValueError('bad case')\n"
                "    return value"
            ),
            (
                "def check(candidate):\n"
                "    for value in range(3):\n"
                "        actual = candidate(value)\n"
                "        assert actual == value"
            ),
            "f",
        )

        self.assertAlmostEqual(score, 2 / 3)

    def test_partial_humaneval_reward_preserves_user_exception_handling(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "def f(value):\n"
                "    if value == 1:\n"
                "        raise ValueError('bad case')\n"
                "    return value"
            ),
            (
                "def check(candidate):\n"
                "    for value in range(3):\n"
                "        try:\n"
                "            actual = candidate(value)\n"
                "            assert actual == value\n"
                "        except ValueError:\n"
                "            assert value == 1"
            ),
            "f",
        )

        self.assertEqual(score, 1.0)

    def test_partial_humaneval_reward_fails_asserts_dependent_on_case_setup(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "def f(value):\n"
                "    if value == 1:\n"
                "        raise ValueError('bad case')\n"
                "    return value"
            ),
            (
                "def check(candidate):\n"
                "    for value in range(3):\n"
                "        actual = candidate(value)\n"
                "        assert actual == value\n"
                "        assert actual >= 0"
            ),
            "f",
        )

        self.assertAlmostEqual(score, 4 / 6)

    def test_partial_humaneval_reward_runs_independent_consecutive_asserts(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "def f(value):\n"
                "    if value == 1:\n"
                "        raise ValueError('bad case')\n"
                "    return value"
            ),
            (
                "def check(candidate):\n"
                "    for value in range(3):\n"
                "        actual = candidate(value)\n"
                "        assert actual == value\n"
                "        assert value >= 0"
            ),
            "f",
        )

        self.assertAlmostEqual(score, 5 / 6)

    @unittest.skipUnless(
        sys.version_info >= (3, 11),
        "requires except* syntax",
    )
    def test_partial_humaneval_reward_preserves_user_exception_groups(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "def f(value):\n"
                "    if value == 1:\n"
                "        raise ValueError('bad case')\n"
                "    return value"
            ),
            (
                "def check(candidate):\n"
                "    for value in range(3):\n"
                "        try:\n"
                "            actual = candidate(value)\n"
                "            assert actual == value\n"
                "        except* ValueError:\n"
                "            assert value == 1"
            ),
            "f",
        )

        self.assertEqual(score, 1.0)

    def test_partial_humaneval_reward_ignores_candidate_stdout(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            'def f():\n    print("debug", end="")\n    return 1',
            "def check(candidate):\n    assert candidate() == 1",
            "f",
        )

        self.assertEqual(score, 1.0)

    def test_partial_humaneval_reward_rejects_forged_pass_marker(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "import atexit\n"
                "atexit.register(lambda: print(\"__MINI_R1_PASSED_COUNT__=2/2\"))\n"
                "def f(value):\n"
                "    return 0"
            ),
            (
                "def check(candidate):\n"
                "    assert candidate(1) == 1\n"
                "    assert candidate(2) == 2"
            ),
            "f",
        )

        self.assertEqual(score, 0.0)

    @patch("local_sandbox.subprocess.run")
    def test_partial_humaneval_reward_rejects_duplicate_markers(self, mock_run):
        marker = "__MINI_R1_PASSED_COUNT__:0123456789abcdef="
        mock_run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout=f"{marker}0/2\n{marker}2/2\n",
            stderr="",
        )

        with patch(
            "local_sandbox.secrets.token_hex",
            return_value="0123456789abcdef",
        ):
            score = local_sandbox.compute_humaneval_pass_rate(
                "def f(value): return 0",
                "def check(candidate):\n    assert candidate(1) == 1",
                "f",
            )

        self.assertEqual(score, 0.0)

    def test_partial_humaneval_reward_ignores_candidate_str_override(self):
        score = local_sandbox.compute_humaneval_pass_rate(
            (
                "def str(value):\n"
                "    return '2'\n"
                "def f(value):\n"
                "    return 0"
            ),
            (
                "def check(candidate):\n"
                "    assert candidate(1) == 1\n"
                "    assert candidate(2) == 2"
            ),
            "f",
        )

        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
