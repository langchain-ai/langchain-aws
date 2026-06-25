import concurrent.futures
import os
import uuid

import pytest
from langchain_core.runnables.config import RunnableConfig

from langchain_aws.tools.code_interpreter_toolkit import (
    CodeInterpreterToolkit,
    _extract_output_from_stream,
)

REGION = os.environ.get("CODE_INTERPRETER_INTEG_REGION", "us-west-2")


pytestmark = pytest.mark.skipif(
    not os.environ.get("AWS_DEFAULT_REGION")
    and not os.environ.get("AWS_REGION")
    and not os.environ.get("CODE_INTERPRETER_INTEG_REGION"),
    reason="AWS credentials/region not configured",
)


def _run_code(
    toolkit: CodeInterpreterToolkit, config: RunnableConfig, code: str
) -> str:
    interpreter = toolkit._get_or_create_interpreter(config)
    response = interpreter.invoke(
        method="executeCode", params={"language": "python", "code": code}
    )
    return _extract_output_from_stream(response)


def test_session_persists_across_checkpoint_namespaces() -> None:
    """Session state persists across differing checkpoint_ns in one thread."""
    toolkit = CodeInterpreterToolkit(region=REGION)
    thread_id = "integ-thread-1057"
    ns_write = f"tools:{uuid.uuid4()}"
    ns_read = f"tools:{uuid.uuid4()}"

    try:
        _run_code(
            toolkit,
            RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ns_write}
            ),
            "open('/tmp/persist_1057.txt', 'w').write('Hello World')",
        )
        output = _run_code(
            toolkit,
            RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ns_read}
            ),
            "print(open('/tmp/persist_1057.txt').read())",
        )

        assert list(toolkit._code_interpreters) == [thread_id]
        assert "Hello World" in output
    finally:
        for interpreter in toolkit._code_interpreters.values():
            try:
                interpreter.stop()
            except Exception:
                pass


def test_distinct_threads_get_distinct_sessions() -> None:
    """Different ``thread_id`` values must not share a code interpreter session."""
    toolkit = CodeInterpreterToolkit(region=REGION)

    try:
        _run_code(
            toolkit,
            RunnableConfig(configurable={"thread_id": "integ-thread-a"}),
            "open('/tmp/thread_scope.txt', 'w').write('from-a')",
        )
        output = _run_code(
            toolkit,
            RunnableConfig(configurable={"thread_id": "integ-thread-b"}),
            "import os; print(os.path.exists('/tmp/thread_scope.txt'))",
        )

        assert set(toolkit._code_interpreters) == {"integ-thread-a", "integ-thread-b"}
        assert "False" in output
    finally:
        for interpreter in toolkit._code_interpreters.values():
            try:
                interpreter.stop()
            except Exception:
                pass


def test_isolates_sessions_per_subagent() -> None:
    """Parallel subagents (nested checkpoint_ns) get isolated sessions."""
    toolkit = CodeInterpreterToolkit(region=REGION)
    thread_id = "integ-thread-subagents"
    sub_a = f"research-a:{uuid.uuid4()}|tools:{uuid.uuid4()}"
    sub_b = f"research-b:{uuid.uuid4()}|tools:{uuid.uuid4()}"

    try:
        _run_code(
            toolkit,
            RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": sub_a}
            ),
            "open('/tmp/subagent_scope.txt', 'w').write('from-a')",
        )
        output = _run_code(
            toolkit,
            RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": sub_b}
            ),
            "import os; print(os.path.exists('/tmp/subagent_scope.txt'))",
        )

        assert len(toolkit._code_interpreters) == 2
        assert "False" in output
    finally:
        for interpreter in toolkit._code_interpreters.values():
            try:
                interpreter.stop()
            except Exception:
                pass


def test_concurrent_tool_calls_share_one_session() -> None:
    """Concurrent tool calls in one thread reuse a single session safely."""
    toolkit = CodeInterpreterToolkit(region=REGION)
    thread_id = "integ-thread-concurrent"

    def _call(i: int) -> str:
        config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "checkpoint_ns": f"tools:{uuid.uuid4()}",
            }
        )
        return _run_code(toolkit, config, f"print({i} * {i})")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(_call, range(5)))

        assert list(toolkit._code_interpreters) == [thread_id]
        for i, output in enumerate(results):
            assert str(i * i) in output
    finally:
        for interpreter in toolkit._code_interpreters.values():
            try:
                interpreter.stop()
            except Exception:
                pass
