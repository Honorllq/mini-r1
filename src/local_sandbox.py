"""
本地 Python 子进程沙箱 —— 替换 Open-R1 的付费 E2B 云沙箱

这是你项目最核心的差异化贡献。核心能力:
  1. 接收一段 Python 代码和若干 test cases
  2. 在隔离子进程里执行代码
  3. 对比 stdout 和 expected output
  4. 返回 pass rate (0~1 float)

替换 Open-R1 rewards.py line 592 的 execution_provider.execute_scripts(...)

安全边界:
    子进程用于限制崩溃和超时，不是对抗恶意 Python 的安全边界。
    对不可信代码应再使用容器、低权限账户或专用沙箱。

运行方式:
    python src/local_sandbox.py   # 自测
"""

import ast
import secrets
import subprocess
import sys
from typing import Dict, List, Optional


_PASS_COUNT_MARKER_PREFIX = "__MINI_R1_PASSED_COUNT__"
_SUCCESS_MARKER_PREFIX = "__MINI_R1_HUMANEVAL_SUCCESS__"
_PASSED_VAR = "__mini_r1_passed_count"
_TOTAL_VAR = "__mini_r1_total_count"


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
    success_marker = f"{_SUCCESS_MARKER_PREFIX}:{secrets.token_hex(16)}"
    success_payload = f"\n{success_marker}\n".encode("ascii")
    script_parts = [
        "import builtins as __mini_r1_builtins, os as __mini_r1_os",
        "__mini_r1_base_exception = __mini_r1_builtins.BaseException",
        "__mini_r1_compile = __mini_r1_builtins.compile",
        "__mini_r1_exec = __mini_r1_builtins.exec",
        "__mini_r1_write = __mini_r1_os.write",
        "__mini_r1_exit = __mini_r1_os._exit",
        "__mini_r1_pristine_builtins = __mini_r1_builtins.__dict__.copy()",
        (
            '__mini_r1_candidate_globals = {"__builtins__": '
            '__mini_r1_builtins, "__name__": "__main__"}'
        ),
        "try:",
        (
            f'    __mini_r1_exec(__mini_r1_compile({code!r}, "<candidate>", '
            '"exec"), __mini_r1_candidate_globals)'
        ),
        (
            "    __mini_r1_candidate = "
            f"__mini_r1_candidate_globals[{entry_point!r}]"
        ),
        "    for __mini_r1_name in __mini_r1_builtins.__dict__.copy():",
        "        if __mini_r1_name not in __mini_r1_pristine_builtins:",
        "            del __mini_r1_builtins.__dict__[__mini_r1_name]",
        (
            "    __mini_r1_builtins.__dict__.update("
            "__mini_r1_pristine_builtins)"
        ),
        "    __mini_r1_test_globals = __mini_r1_candidate_globals.copy()",
        "    for __mini_r1_name in __mini_r1_pristine_builtins:",
        "        __mini_r1_test_globals.pop(__mini_r1_name, None)",
        (
            '    __mini_r1_test_globals["__builtins__"] = '
            "__mini_r1_pristine_builtins.copy()"
        ),
        '    __mini_r1_test_globals["__name__"] = "__main__"',
        (
            f'    __mini_r1_exec(__mini_r1_compile({test_code!r}, "<tests>", '
            '"exec"), __mini_r1_test_globals)'
        ),
        '    __mini_r1_test_globals["check"](__mini_r1_candidate)',
        "except __mini_r1_base_exception:",
        "    __mini_r1_exit(1)",
        f"__mini_r1_write(1, {success_payload!r})",
        "__mini_r1_exit(0)",
    ]
    full_script = "\n".join(script_parts)

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

    success_lines = [
        line for line in result.stdout.splitlines() if line == success_marker
    ]
    return result.returncode == 0 and len(success_lines) == 1


class _AssertInstrumenter:
    """把 check() 中实际执行的每个 assert 转成独立计数测试。"""

    _ASSIGNMENT_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign)
    _NESTED_SCOPE_NODES = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
    )
    _TRY_NODES = (ast.Try,)
    if hasattr(ast, "TryStar"):
        _TRY_NODES += (getattr(ast, "TryStar"),)

    def __init__(self, candidate_name: str, existing_names):
        self.candidate_name = candidate_name
        self.existing_names = set(existing_names)
        self.assert_count = 0
        self.validity_count = 0

    def instrument_block(
        self,
        statements: List[ast.stmt],
        allow_setup_guard: bool = True,
    ) -> List[ast.stmt]:
        """Instrument direct candidate setup followed by related asserts."""
        transformed = []
        index = 0

        while index < len(statements):
            statement = statements[index]
            self._instrument_nested_blocks(statement, allow_setup_guard)

            if (
                allow_setup_guard
                and isinstance(statement, self._ASSIGNMENT_NODES)
                and getattr(statement, "value", None) is not None
            ):
                stored_names = self._stored_names(statement)
                loaded_names = self._assignment_loaded_names(statement)
                assert_end = index + 1
                assertions = []
                while (
                    assert_end < len(statements)
                    and isinstance(statements[assert_end], ast.Assert)
                ):
                    assertions.append(statements[assert_end])
                    assert_end += 1

                has_related_assert = any(
                    stored_names & self._loaded_names(assertion.test)
                    for assertion in assertions
                )
                if (
                    stored_names
                    and self.candidate_name in loaded_names
                    and has_related_assert
                ):
                    transformed.extend(
                        self._guard_case_setup(
                            statement,
                            assertions,
                            stored_names,
                        )
                    )
                    index = assert_end
                    continue

            if isinstance(statement, ast.Assert):
                transformed.extend(self._guard_assert(statement))
            else:
                transformed.append(statement)
            index += 1

        return transformed

    def _instrument_nested_blocks(
        self,
        node: ast.AST,
        allow_setup_guard: bool,
    ) -> None:
        """Recurse through control flow without entering nested scopes."""
        if isinstance(node, self._NESTED_SCOPE_NODES):
            return

        if isinstance(node, self._TRY_NODES):
            node.body = self.instrument_block(
                node.body,
                allow_setup_guard=False,
            )
            for handler in node.handlers:
                handler.body = self.instrument_block(
                    handler.body,
                    allow_setup_guard=allow_setup_guard,
                )
            node.orelse = self.instrument_block(
                node.orelse,
                allow_setup_guard=allow_setup_guard,
            )
            node.finalbody = self.instrument_block(
                node.finalbody,
                allow_setup_guard=allow_setup_guard,
            )
            return

        if isinstance(node, (ast.With, ast.AsyncWith)):
            node.body = self.instrument_block(
                node.body,
                allow_setup_guard=False,
            )
            return

        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                if all(isinstance(item, ast.stmt) for item in value):
                    setattr(
                        node,
                        field,
                        self.instrument_block(value, allow_setup_guard),
                    )
                else:
                    for item in value:
                        if isinstance(item, ast.AST):
                            self._instrument_nested_blocks(
                                item,
                                allow_setup_guard,
                            )
            elif isinstance(value, ast.AST):
                self._instrument_nested_blocks(value, allow_setup_guard)

    @staticmethod
    def _loaded_names(node: ast.AST):
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }

    @staticmethod
    def _stored_names(node: ast.AST):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]

        return {
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        }

    def _assignment_loaded_names(self, node: ast.AST):
        value = getattr(node, "value", None)
        loaded_names = self._loaded_names(value) if value is not None else set()
        if isinstance(node, ast.AugAssign):
            loaded_names.update(self._stored_names(node))
        return loaded_names

    def _new_validity_flag(self) -> str:
        while True:
            self.validity_count += 1
            name = f"__mini_r1_case_valid_{self.validity_count}"
            if name not in self.existing_names:
                self.existing_names.add(name)
                return name

    def _assert_check(self, node: ast.Assert) -> ast.Try:
        passed_increment = ast.AugAssign(
            target=ast.Name(id=_PASSED_VAR, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        return ast.Try(
            body=[node],
            handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])],
            orelse=[passed_increment],
            finalbody=[],
        )

    def _guard_assert(self, node: ast.Assert) -> List[ast.stmt]:
        self.assert_count += 1
        total_increment = ast.AugAssign(
            target=ast.Name(id=_TOTAL_VAR, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        return [total_increment, self._assert_check(node)]

    def _guard_case_setup(
        self,
        case_setup: ast.stmt,
        assertions: List[ast.Assert],
        stored_names,
    ) -> List[ast.stmt]:
        """Skip only assertions that depend on a failed case setup."""
        validity_flag = self._new_validity_flag()
        transformed = [
            ast.Assign(
                targets=[ast.Name(id=validity_flag, ctx=ast.Store())],
                value=ast.Constant(value=False),
            ),
            ast.Try(
                body=[case_setup],
                handlers=[
                    ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])
                ],
                orelse=[
                    ast.Assign(
                        targets=[ast.Name(id=validity_flag, ctx=ast.Store())],
                        value=ast.Constant(value=True),
                    )
                ],
                finalbody=[],
            ),
        ]

        for assertion in assertions:
            total_increment, guarded_assert = self._guard_assert(assertion)
            transformed.append(total_increment)
            if stored_names & self._loaded_names(assertion.test):
                transformed.append(
                    ast.If(
                        test=ast.Name(id=validity_flag, ctx=ast.Load()),
                        body=[guarded_assert],
                        orelse=[],
                    )
                )
            else:
                transformed.append(guarded_assert)

        return transformed


def _instrument_humaneval_tests(test_code: str) -> Optional[str]:
    """保留 check() 的准备语句和控制流，并为断言加入动态计数。"""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return None

    check_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "check"
        ),
        None,
    )
    if check_function is None or isinstance(check_function, ast.AsyncFunctionDef):
        return None

    positional_args = [
        *check_function.args.posonlyargs,
        *check_function.args.args,
    ]
    if not positional_args:
        return None

    argument_names = {
        argument.arg
        for argument in (
            *check_function.args.posonlyargs,
            *check_function.args.args,
            *check_function.args.kwonlyargs,
        )
    }
    if check_function.args.vararg is not None:
        argument_names.add(check_function.args.vararg.arg)
    if check_function.args.kwarg is not None:
        argument_names.add(check_function.args.kwarg.arg)

    existing_names = {
        node.id for node in ast.walk(check_function) if isinstance(node, ast.Name)
    } | argument_names
    instrumenter = _AssertInstrumenter(
        positional_args[0].arg,
        existing_names,
    )
    transformed_body = instrumenter.instrument_block(check_function.body)

    if instrumenter.assert_count == 0:
        return None

    check_function.body = [
        ast.Assign(
            targets=[ast.Name(id=_PASSED_VAR, ctx=ast.Store())],
            value=ast.Constant(value=0),
        ),
        ast.Assign(
            targets=[ast.Name(id=_TOTAL_VAR, ctx=ast.Store())],
            value=ast.Constant(value=0),
        ),
        *transformed_body,
        ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.Name(id=_PASSED_VAR, ctx=ast.Load()),
                    ast.Name(id=_TOTAL_VAR, ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            )
        ),
    ]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


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
    instrumented_tests = _instrument_humaneval_tests(test_code)
    if instrumented_tests is None:
        # 找不到可计数的 assert (可能格式特殊), 退回二元判断
        return 1.0 if run_humaneval_test(code, test_code, entry_point, timeout) else 0.0

    pass_count_marker = f"{_PASS_COUNT_MARKER_PREFIX}:{secrets.token_hex(16)}="
    pass_count_payload = f"\n{pass_count_marker}%d/%d\n".encode("ascii")

    # 测试命名空间复制候选 helper，但移除同名覆盖并使用独立的 builtins 快照；
    # 评分逻辑留在外层，并预先缓存依赖，阻止普通名称覆盖和退出钩子干扰结果。
    script_parts = [
        "import builtins as __mini_r1_builtins, os as __mini_r1_os",
        "__mini_r1_base_exception = __mini_r1_builtins.BaseException",
        "__mini_r1_compile = __mini_r1_builtins.compile",
        "__mini_r1_exec = __mini_r1_builtins.exec",
        "__mini_r1_write = __mini_r1_os.write",
        "__mini_r1_exit = __mini_r1_os._exit",
        "__mini_r1_pristine_builtins = __mini_r1_builtins.__dict__.copy()",
        (
            '__mini_r1_candidate_globals = {"__builtins__": '
            '__mini_r1_builtins, "__name__": "__main__"}'
        ),
        "try:",
        (
            f'    __mini_r1_exec(__mini_r1_compile({code!r}, "<candidate>", '
            '"exec"), __mini_r1_candidate_globals)'
        ),
        (
            "    __mini_r1_candidate = "
            f"__mini_r1_candidate_globals[{entry_point!r}]"
        ),
        "    for __mini_r1_name in __mini_r1_builtins.__dict__.copy():",
        "        if __mini_r1_name not in __mini_r1_pristine_builtins:",
        "            del __mini_r1_builtins.__dict__[__mini_r1_name]",
        (
            "    __mini_r1_builtins.__dict__.update("
            "__mini_r1_pristine_builtins)"
        ),
        "    __mini_r1_test_globals = __mini_r1_candidate_globals.copy()",
        "    for __mini_r1_name in __mini_r1_pristine_builtins:",
        "        __mini_r1_test_globals.pop(__mini_r1_name, None)",
        (
            '    __mini_r1_test_globals["__builtins__"] = '
            "__mini_r1_pristine_builtins.copy()"
        ),
        '    __mini_r1_test_globals["__name__"] = "__main__"',
        (
            f'    __mini_r1_exec(__mini_r1_compile({instrumented_tests!r}, '
            '"<tests>", "exec"), __mini_r1_test_globals)'
        ),
        (
            f"    {_PASSED_VAR}, {_TOTAL_VAR} = "
            '__mini_r1_test_globals["check"](__mini_r1_candidate)'
        ),
        "except __mini_r1_base_exception:",
        "    __mini_r1_exit(1)",
        (
            f"__mini_r1_write(1, {pass_count_payload!r} % "
            f"({_PASSED_VAR}, {_TOTAL_VAR}))"
        ),
        "__mini_r1_exit(0)",
    ]

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

    marker_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(pass_count_marker)
    ]
    if len(marker_lines) != 1:
        return 0.0

    try:
        passed_text, total_text = (
            marker_lines[0].removeprefix(pass_count_marker).split("/", 1)
        )
        passed = int(passed_text)
        total = int(total_text)
    except ValueError:
        return 0.0

    if total <= 0 or not 0 <= passed <= total:
        return 0.0

    return passed / total


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
