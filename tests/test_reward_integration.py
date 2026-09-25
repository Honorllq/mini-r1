"""Offline reward-to-sandbox checks using real, finite Python subprocesses.

Synthetic fixtures verify integration, not model quality or the full training loop.
No reward, code extraction, or sandbox function is mocked.
"""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import reward_funcs


def _completion(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": text}]


def _code_completion(code: str) -> list[dict[str, str]]:
    return _completion(
        "<reasoning>Check the cases.</reasoning>\n"
        f"<answer>\n```python\n{code}\n```\n</answer>"
    )


class TestRewardSandboxIntegration(unittest.TestCase):
    def test_humaneval_batch_distinguishes_binary_and_partial_rewards(self) -> None:
        completions = [
            _code_completion("def identity(value): return value"),
            _code_completion("def partial(value): return value if value < 2 else -1"),
            _code_completion("def broken(: pass"),
            _completion("I cannot solve this."),
            _code_completion("def increment(value): return value + 1"),
        ]
        metadata = [
            {
                "entry_point": name,
                "test_code": (
                    "def check(candidate):\n"
                    "    for value in range(3):\n"
                    f"        assert candidate(value) == value + {offset}"
                ),
            }
            for name, offset in (
                ("identity", 0), ("partial", 0), ("broken", 0),
                ("missing", 0), ("increment", 1),
            )
        ]
        self.assertEqual(
            reward_funcs.code_reward_humaneval(completions, verification_info=metadata),
            [1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(
            reward_funcs.code_reward_humaneval_partial(
                completions, verification_info=metadata
            ),
            [1.0, 2 / 3, 0.0, 0.0, 1.0],
        )
        self.assertEqual(
            reward_funcs.format_reward(completions), [0.5, 0.5, 0.5, 0.0, 0.5]
        )

    def test_stdio_batch_uses_each_samples_cases_and_continues_after_errors(self) -> None:
        completions = [
            _code_completion("print(int(input()) + 1)"),
            _code_completion("print(int(input()) + 1)"),
            _code_completion("raise RuntimeError('negative control')"),
            _completion("No Python block."),
            _completion("```python\r\nprint(input())\r\n```"),
        ]
        cases = [
            [{"input": "1", "output": "2"}, {"input": "2", "output": "3"}],
            [{"input": "1", "output": "99"}, {"input": "2", "output": "3"}],
            [{"input": "", "output": ""}],
            [{"input": "", "output": ""}],
            [{"input": "你好 café", "output": "你好 café"}],
        ]
        self.assertEqual(
            reward_funcs.code_reward(
                completions, verification_info=[{"test_cases": value} for value in cases]
            ),
            [1.0, 0.5, 0.0, 0.0, 1.0],
        )

    def test_invalid_output_does_not_abort_the_humaneval_reward_batch(self) -> None:
        metadata = [{
            "test_code": "def check(candidate):\n    assert candidate() == 1",
            "entry_point": "solve",
        }] * 2
        for fd in (1, 2):
            completions = [
                _code_completion(f"import os\nos.write({fd}, b'\\xff')\ndef solve(): return 1"),
                _code_completion("def solve(): return 1"),
            ]
            for reward in (
                reward_funcs.code_reward_humaneval,
                reward_funcs.code_reward_humaneval_partial,
            ):
                with self.subTest(fd=fd, reward=reward.__name__):
                    self.assertEqual(reward(completions, verification_info=metadata), [0.0, 1.0])

    def test_stdio_reward_accepts_crlf_cases_without_changing_scores(self) -> None:
        completion = _code_completion("print(int(input()) + int(input()))")
        self.assertEqual(reward_funcs.code_reward(
            [completion],
            verification_info=[{"test_cases": [
                {"input": "10\n20\n", "output": "30"},
                {"input": "10\r\n20\r\n", "output": "30"},
                {"input": "10\r\n20\r\n", "output": "99"},
            ]}],
        ), [2 / 3])


if __name__ == "__main__":
    unittest.main()
