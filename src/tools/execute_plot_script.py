"""Tool: 执行适配后的绘图脚本。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import ExecutePlotScriptInput, ExecutionResult
from src.tools._utils import error_result


@tool(
    "execute_plot_script",
    args_schema=ExecutePlotScriptInput,
    description="使用 subprocess 执行 adapted_plot.py，禁止 shell=True，捕获 stdout/stderr 和生成文件。",
)
def execute_plot_script(
    script_path: str = "workspace/adapted_plot.py",
    data_path: str = "",
    output_directory: str = "outputs",
    timeout_seconds: int = 120,
    python_executable: str = "",
) -> dict:
    """执行绘图脚本并保存 execution_result.json。"""

    try:
        ExecutePlotScriptInput(
            script_path=script_path,
            data_path=data_path,
            output_directory=output_directory,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
    except ValidationError as error:
        return error_result(
            "validation_error", "脚本执行参数校验失败。", details={"errors": error.errors()}
        )

    script = Path(script_path)
    data = Path(data_path)
    out_dir = Path(output_directory)
    if not script.exists():
        return error_result(
            "script_not_found", "适配脚本不存在。", details={"script_path": str(script)}
        )
    if not data.exists():
        return error_result(
            "data_file_not_found", "用户数据文件不存在。", details={"data_path": str(data)}
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {
        p.resolve(): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in out_dir.glob("*")
        if p.is_file()
    }
    env = os.environ.copy()
    env["DATA_PATH"] = str(data.resolve())
    env["OUTPUT_DIR"] = str(out_dir.resolve())
    executable = python_executable or sys.executable
    start = time.time()
    try:
        completed = subprocess.run(
            [executable, str(script.resolve()), str(data.resolve()), str(out_dir.resolve())],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
            env=env,
            cwd=str(script.parent.resolve()),
        )
        error_type = "" if completed.returncode == 0 else "script_failed"
    except subprocess.TimeoutExpired as error:
        completed = None
        error_type = "script_timeout"
        stdout = error.stdout or ""
        stderr = error.stderr or str(error)
        return_code = None
    else:
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
    generated = []
    for path in out_dir.glob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        stat = path.stat()
        if resolved not in before or before[resolved] != (stat.st_mtime_ns, stat.st_size):
            generated.append(str(path))
    result = ExecutionResult(
        success=return_code == 0 and bool(generated),
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        generated_files=generated,
        execution_time=round(time.time() - start, 4),
        error_type=error_type,
    )
    Path("workspace").mkdir(exist_ok=True)
    Path("workspace/execution_result.json").write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"success": result.success, "execution_result": result.model_dump()}
