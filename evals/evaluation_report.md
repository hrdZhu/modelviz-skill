# ModelViz Skill Evaluation Report

本报告由 `evals/run_evaluation.py` 实际运行生成。模型相关步骤使用 Mock 结构化输出；程序化解析、候选召回、Pydantic 校验、依赖检查、代码执行、图片检查和修复循环调用现有实现。

## Case Counts

- Requirement parsing cases: 20
- Template recall cases: 15
- Final selection cases: 10
- End-to-end cases: 8

## Core Metrics

- End-to-end task success rate: 87.50%
- Candidate template Recall@5: 100.00%
- Candidate template Recall@8: 100.00%
- Final template selection accuracy: 100.00%
- First execution success rate: 62.50%
- Final execution success rate: 87.50%
- Average image quality score: 4.50 / 5
- Automatic repair success rate: 1.0
- Average repair attempts: 1.00

## Guardrail Metrics

- Data column hallucination rate: 0.00%
- Candidate scope violation rate: 0.00%
- Original template modification rate: 0.00%
- User data modification rate: 0.00%

## Test Coverage

- Unit tests cover Pydantic schemas, scoring logic, dependency mapping/install safety, output artifact inspection, image inspection, warning parsing, repair limits, and original-file protection.
- Integration tests cover stage 2 to stage 3, stage 3 to stage 4, stage 4 to stage 5, and stage 5 to stage 6 handoffs with Mock structured models.
- End-to-end evaluation covers user request + data through final template, adapted code, generated image, quality report, and repair loop.
- Latest verification run: `python -m pytest -q` -> 66 passed; `python -m ruff check src schemas tests evals\run_evaluation.py evals\__init__.py` -> passed; `python -m ruff format --check src schemas tests evals\run_evaluation.py evals\__init__.py` -> passed.

## Main Failure Reasons

- end_to_end:e2e_pca failed at recheck_dependencies

## Current Limitations

- Mock 模型评估验证流程衔接和护栏，不代表真实在线模型的语义选择能力。
- 图片质量分为评估脚本基于技术/视觉通过状态给出的 1-5 分；真实使用时应接入视觉模型或人工抽检。
- 端到端代码生成使用简单 Matplotlib Mock 代码，主要验证执行、输出和修复闭环。
- 全量模板脚本本身未纳入 Ruff 格式质量评估，因为 `templates/` 是原始模板库。

## Optimization Log

- Before fixes: end-to-end success was 0.00%, Recall@5 was 93.33%, final selection accuracy was 90.00%, final execution success was 0.00%, and guardrails showed data-column/candidate-scope violations from downstream failures.
- Fix 1: made package-name reverse mapping case-insensitive in `src/tools/dependency_utils.py`, so catalog dependency `pillow` resolves to import module `PIL`.
- Fix 2: added a category-alias score in `src/tools/match_candidate_templates.py`, improving coarse recall when functional keywords map directly to catalog categories.
- Fix 3: changed `src/tools/execute_plot_script.py` to count overwritten output files as generated artifacts, which is required during rerun and repair.
- Fix 4: narrowed `src/tools/collect_plot_warnings.py` so ordinary `savefig` warning lines are not treated as save failures.
- After fixes: end-to-end success is 87.50%, Recall@5 is 100.00%, final selection accuracy is 100.00%, and final execution success is 87.50%.
- Remaining failure: `e2e_pca` stops at dependency recheck because the selected PCA template needs an additional dependency while this evaluation runs stage 5 with `auto_install=false`.
