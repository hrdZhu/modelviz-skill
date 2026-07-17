# Stage 5-6 Skill Workflow

阶段 5 和阶段 6 的 Skill 执行流程：

1. 读取 `workspace/final_template_selection.json`。
2. 根据 `docs/template_catalog.yaml` 读取选定模板代码和元数据。
3. 使用 AST 识别模板 `import`，合并 catalog 中的 `dependencies`。
4. 排除 Python 标准库和项目内部模块。
5. 根据 `docs/package_name_mapping.yaml` 将 import 名转换为 pip 包名。
6. 检查当前 Python 环境中的依赖可用性。
7. 自动安装确认缺失且合法的第三方依赖。
8. 安装后再次检查依赖。
9. 读取 `prompts/template_adaptation.md` 对应的阶段五 Prompt，生成 `workspace/adaptation_plan.json`。
10. 生成并保存 `workspace/adapted_plot.py` 和 `workspace/adaptation_result.json`。
11. 执行 `adapted_plot.py`，保存 `workspace/execution_result.json`。
12. 执行程序化技术检查，保存 `workspace/technical_quality_report.json`。
13. 读取 `prompts/visual_quality_check.md` 对应的视觉检查 Prompt。
14. 必要时读取 `prompts/code_repair.md` 对应的修复 Prompt。
15. 检查修复代码是否请求新增依赖，并最多补全两轮依赖。
16. 最多修复三次。
17. 返回最终图表、代码、依赖清单和质量报告。

注意：

- 不把长 Prompt 复制进 `SKILL.md`。
- 不修改 `templates/` 下的原始模板。
- 不覆盖用户原始数据。
- 依赖安装必须使用当前 Python 解释器和 `subprocess.run(..., shell=False)`。
- 大语言模型不得直接决定安装任意包名；包名必须通过普通代码校验。
