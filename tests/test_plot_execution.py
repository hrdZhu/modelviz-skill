from src.tools import execute_plot_script


def test_execute_plot_script_counts_overwritten_output_as_generated(tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("x,y\n1,2\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    chart_path = output_dir / "chart.png"
    chart_path.write_bytes(b"old")
    script_path = tmp_path / "plot.py"
    script_path.write_text(
        """
import sys
from pathlib import Path

output_dir = Path(sys.argv[2])
(output_dir / "chart.png").write_bytes(b"new-content")
""",
        encoding="utf-8",
    )

    result = execute_plot_script.invoke(
        {
            "script_path": str(script_path),
            "data_path": str(data_path),
            "output_directory": str(output_dir),
        }
    )

    assert result["success"] is True
    assert str(chart_path) in result["execution_result"]["generated_files"]
