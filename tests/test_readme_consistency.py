"""Regression checks for the sandbox documentation in README.md."""

import unittest
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


class TestReadmeSandboxDocumentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README.read_text(encoding="utf-8")
        section_start = cls.readme.index("### 1. 本地沙箱")
        section_end = cls.readme.index("### 3. 训练配置", section_start)
        cls.sandbox_docs = cls.readme[section_start:section_end]

    def test_does_not_show_legacy_exit_code_only_sandbox(self) -> None:
        self.assertNotIn('full_script = f"{code}', self.sandbox_docs)
        self.assertNotIn('["python", "-c",', self.sandbox_docs)
        self.assertIn(
            '[sys.executable, "-c", full_script]', self.sandbox_docs
        )
        self.assertIn(
            "return success_lines == [success_marker]", self.sandbox_docs
        )

    def test_uses_current_partial_reward_design(self) -> None:
        self.assertNotIn("_extract_asserts", self.sandbox_docs)
        self.assertIn("_instrument_humaneval_tests", self.sandbox_docs)
        self.assertIn("单个子进程", self.sandbox_docs)

    def test_states_subprocess_security_boundary(self) -> None:
        self.assertIn(
            "不是对抗恶意 Python 的安全边界", self.sandbox_docs
        )


if __name__ == "__main__":
    unittest.main()
