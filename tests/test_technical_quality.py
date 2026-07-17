from PIL import Image

from src.tools import collect_plot_warnings, inspect_generated_image, validate_output_artifacts


def test_validate_output_artifacts_and_inspect_image(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    image_path = out_dir / "chart.png"
    image = Image.new("RGB", (400, 300), "white")
    for x in range(50, 200):
        for y in range(50, 200):
            image.putpixel((x, y), (255, 0, 0))
    image.save(image_path)

    artifacts = validate_output_artifacts.invoke(
        {"generated_files": [str(image_path)], "output_directory": str(out_dir)}
    )
    assert artifacts["success"] is True
    assert artifacts["generated_formats"] == ["png"]

    inspected = inspect_generated_image.invoke({"image_path": str(image_path)})
    assert inspected["success"] is True
    assert inspected["image_width"] == 400
    assert inspected["possible_blank_image"] is False


def test_inspect_generated_image_missing_corrupted_and_blank(tmp_path):
    missing = inspect_generated_image.invoke({"image_path": str(tmp_path / "missing.png")})
    assert missing["success"] is False
    assert missing["error_type"] == "image_not_found"

    corrupted = tmp_path / "bad.png"
    corrupted.write_text("not png", encoding="utf-8")
    bad = inspect_generated_image.invoke({"image_path": str(corrupted)})
    assert bad["success"] is False
    assert bad["error_type"] == "image_corrupted"

    blank = tmp_path / "blank.png"
    Image.new("RGBA", (400, 300), (255, 255, 255, 255)).save(blank)
    blank_result = inspect_generated_image.invoke({"image_path": str(blank)})
    assert blank_result["success"] is False
    assert blank_result["possible_blank_image"] is True


def test_collect_plot_warnings():
    result = collect_plot_warnings.invoke(
        {
            "stdout": "",
            "stderr": "ModuleNotFoundError: No module named 'foo'\nGlyph 123 missing from current font\ntight_layout warning",
            "return_code": 1,
        }
    )

    issue_types = {issue["issue_type"] for issue in result["issues"]}
    assert {"runtime_error", "missing_dependency", "font_warning", "layout_warning"} <= issue_types
