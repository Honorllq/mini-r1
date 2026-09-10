import copy
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


DATA_PREP_SCRIPT = Path(__file__).parents[1] / "src" / "data_prep.py"


class TestHumanEvalDataPreparation(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = mock.Mock(spec=["map"])
        self.dataset.map.return_value = mock.sentinel.transformed_dataset
        self.load_dataset = mock.Mock(return_value=self.dataset)
        datasets_stub = types.ModuleType("datasets")
        datasets_stub.load_dataset = self.load_dataset
        with mock.patch.dict(sys.modules, {"datasets": datasets_stub}):
            self.module = runpy.run_path(str(DATA_PREP_SCRIPT))
        self.load_dataset.assert_not_called()

    def test_loader_forwards_split_and_returns_mapped_dataset(self) -> None:
        for arguments, expected_split in (((), "test"), (("test[:2]",), "test[:2]")):
            with self.subTest(split=expected_split):
                self.load_dataset.reset_mock()
                self.dataset.map.reset_mock()

                result = self.module["load_humaneval"](*arguments)

                self.load_dataset.assert_called_once_with(
                    "openai/openai_humaneval", split=expected_split
                )
                self.dataset.map.assert_called_once()
                self.assertTrue(callable(self.dataset.map.call_args.args[0]))
                self.assertIs(result, mock.sentinel.transformed_dataset)

    def test_transform_preserves_task_contract_without_mutating_input(self) -> None:
        self.module["load_humaneval"]()
        transform = self.dataset.map.call_args.args[0]
        for index, name in enumerate(("add", "问候")):
            example = {
                "task_id": f"HumanEval/{index}",
                "prompt": f'def {name}(value):\n    """Keep 空格 and newlines."""\n',
                "canonical_solution": "    return 'REFERENCE_ONLY'\n",
                "test": "def check(candidate):\n    assert candidate(1) == 2\n",
                "entry_point": name,
            }
            original = copy.deepcopy(example)
            with self.subTest(task_id=example["task_id"]):
                result = transform(example)

                self.assertEqual(
                    result,
                    {
                        "prompt": [
                            {"role": "system", "content": self.module["SYSTEM_PROMPT"]},
                            {
                                "role": "user",
                                "content": (
                                    "Complete the following Python function:\n\n"
                                    f"```python\n{example['prompt']}```"
                                ),
                            },
                        ],
                        "verification_info": {
                            "test_code": example["test"],
                            "entry_point": name,
                        },
                        "task_id": example["task_id"],
                    },
                )
                self.assertEqual(example, original)

    def test_reference_solution_and_tests_do_not_affect_model_prompt(self) -> None:
        self.module["load_humaneval"]()
        transform = self.dataset.map.call_args.args[0]
        example = {
            "task_id": "HumanEval/0",
            "prompt": "def solve():\n",
            "canonical_solution": "    return 'REFERENCE_SECRET_A'\n",
            "test": "def check(candidate):\n    assert candidate() == 'TEST_SECRET_A'\n",
            "entry_point": "solve",
        }
        changed = {
            **example,
            "canonical_solution": "    return 'REFERENCE_SECRET_B'\n",
            "test": "def check(candidate):\n    assert candidate() == 'TEST_SECRET_B'\n",
        }

        original_result = transform(example)
        changed_result = transform(changed)

        self.assertEqual(original_result["prompt"], changed_result["prompt"])
        for message in original_result["prompt"] + changed_result["prompt"]:
            self.assertNotIn("REFERENCE_SECRET", message["content"])
            self.assertNotIn("TEST_SECRET", message["content"])
        self.assertEqual(
            changed_result["verification_info"]["test_code"], changed["test"]
        )


if __name__ == "__main__":
    unittest.main()
