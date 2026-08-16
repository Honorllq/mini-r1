"""
本地 Python 子进程沙箱 —— 替换 Open-R1 的付费 E2B 云沙箱

这是你项目最核心的差异化贡献。核心能力:
  1. 接收一段 Python 代码和若干 test cases
  2. 在隔离子进程里执行代码
  3. 对比 stdout 和 expected output
  4. 返回 pass rate (0~1 float)

替换 Open-R1 rewards.py line 592 的 execution_provider.execute_scripts(...)

运行方式:
    python src/local_sandbox.py   # 自测
"""

import ast
import subprocess
import sys
from typing import List, Dict


def run_one_test(
    code: str,
    test_input: str,
    expected_output: str,
    timeout: int = 5,
) -> bool:
    """跑一个 test case，返回 True/False

    Args:
        code: Python 代码字符串 (从 stdin 读输入，print 到 stdout)
        test_input: 喂给 stdin 的文本
        expected_output: 期望的 stdout 输出
        timeout: 秒，超时则判为失败

    Returns:
        True = pass, False = fail 或 error 或 timeout
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],  # 使用当前 Python 环境执行代码
            input=test_input,          # 喂 stdin
            capture_output=True,       # 抓 stdout + stderr
            text=True,                 # 按字符串处理 (不是 bytes)
            timeout=timeout,           # 超时强杀
        )
    except subprocess.TimeoutExpired:
        return False                   # 死循环
    except Exception:
        return False                   # 其他异常（进程启动失败等）

    if result.returncode != 0:         # 代码报错 (SyntaxError, RuntimeError 等)
        return False

    # 对比 stdout 和 expected (去首尾空白)
    actual = result.stdout.strip()
    expected = expected_output.strip()
    return actual == expected


def compute_pass_rate(
    code: str,
    test_cases: List[Dict[str, str]],
    timeout: int = 5,
) -> float:
    """跑完所有 test cases，返回 pass rate

    Args:
        code: Python 代码
        test_cases: [{"input": "...", "output": "..."}, ...]
        timeout: 单个 case 的超时时间

    Returns:
        float in [0, 1], 通过的比例
    """
    if not test_cases:
        return 0.0

    passed = 0
    for case in test_cases:
        if run_one_test(code, case["input"], case["output"], timeout):
            passed += 1

    return passed / len(test_cases)


def run_humaneval_test(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> bool:
    """跑 HumanEval 风格的函数测试

    HumanEval 的测试不是 stdin/stdout, 而是一个 check() 函数里写一堆 assert.
    完整可执行脚本 = 模型代码 + check() 定义 + 调用 check(函数名)
    只要 subprocess 退出码为 0, 说明所有 assert 都通过.

    Args:
        code: 模型生成的代码 (必须包含 entry_point 定义的函数)
        test_code: HumanEval 提供的 check() 函数代码
        entry_point: 要测试的函数名, 例如 "add"
        timeout: 秒, 超时判为失败

    Returns:
        True = 全部通过, False = 失败/超时/报错
    """
    full_script = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

    return result.returncode == 0   # 0 = 所有 assert 通过


def _extract_asserts(test_code: str) -> List[str]:
    """从 HumanEval 的 check() 函数里抠出所有 assert 语句

    例如 test_code 里有:
        def check(candidate):
            assert candidate(1, 2) == 3
            assert candidate(5, 5) == 10

    返回: ["assert candidate(1, 2) == 3", "assert candidate(5, 5) == 10"]
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    asserts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    asserts.append(ast.unparse(stmt))
    return asserts


def compute_humaneval_pass_rate(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> float:
    """分段 pass rate: 把 check() 里的每个 assert 单独跑, 返回通过比例

    和 run_humaneval_test 的区别:
      - run_humaneval_test:          全过→1.0, 任一失败→0.0  (二元)
      - compute_humaneval_pass_rate: 7 个 assert 过 4 个 → 0.57   (连续)

    连续 reward 提供更丰富的梯度信号, 缓解 reward_std=0 的问题.

    Args:
        code: 模型生成的代码
        test_code: HumanEval 的 check() 函数代码
        entry_point: 函数名
        timeout: 总超时

    Returns:
        float in [0, 1], 通过的 assert 比例
    """
    asserts = _extract_asserts(test_code)
    if not asserts:
        # 抠不出 assert (可能格式特殊), 退回二元判断
        return 1.0 if run_humaneval_test(code, test_code, entry_point, timeout) else 0.0

    # 构造一个单一脚本, 每个 assert 独立 try/except, 避免一个失败连累后面
    script_parts = [
        code,                               # 模型的代码
        f"candidate = {entry_point}",       # 把函数 alias 成 candidate
        "passed_count = 0",
    ]
    for assert_stmt in asserts:
        # 把多行 assert 转成带 4 空格缩进的形式
        indented = "\n    ".join(assert_stmt.split("\n"))
        script_parts.append(
            f"try:\n    {indented}\n    passed_count += 1\nexcept:\n    pass"
        )
    script_parts.append("print(passed_count)")

    full_script = "\n".join(script_parts)

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0

    if result.returncode != 0:
        return 0.0

    try:
        passed = int(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0

    return passed / len(asserts)


# ========== 自测（python src/local_sandbox.py 就会跑） ==========
if __name__ == "__main__":

    # 测试用例 (和 Open-R1 的 verification_info 格式一致)
    test_cases = [
        {"input": "1 2", "output": "3"},
        {"input": "10 20", "output": "30"},
        {"input": "5 5", "output": "10"},
    ]

    # --- 场景 1: 正确代码 ---
    code_correct = """
import sys
a, b = map(int, sys.stdin.read().split())
print(a + b)
"""
    rate = compute_pass_rate(code_correct, test_cases)
    print(f"[正确代码] Pass rate: {rate:.2%}  (期望 100%)")

    # --- 场景 2: 逻辑错误 (减法) ---
    code_wrong = """
import sys
a, b = map(int, sys.stdin.read().split())
print(a - b)
"""
    rate = compute_pass_rate(code_wrong, test_cases)
    print(f"[逻辑错误] Pass rate: {rate:.2%}  (期望 0%)")

    # --- 场景 3: 语法错误 ---
    code_syntax_error = "print(hello world"
    rate = compute_pass_rate(code_syntax_error, test_cases)
    print(f"[语法错误] Pass rate: {rate:.2%}  (期望 0%)")

    # --- 场景 4: 死循环（测 timeout） ---
    code_hang = "while True: pass"
    rate = compute_pass_rate(code_hang, test_cases, timeout=2)
    print(f"[死循环]   Pass rate: {rate:.2%}  (期望 0%，应该 6 秒左右)")

    # --- 场景 5: 部分通过 (只对 1+2=3，其他算错) ---
    code_partial = """
import sys
a, b = map(int, sys.stdin.read().split())
if a == 1 and b == 2:
    print(3)
else:
    print(999)
"""
    rate = compute_pass_rate(code_partial, test_cases)
    print(f"[部分通过] Pass rate: {rate:.2%}  (期望 33.33%)")

    print("\n[OK] 如果 5 个场景都符合期望，说明沙箱工作正常！")
