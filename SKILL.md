---
name: modelviz-skill
description: Scientific visualization workflow for mathematical modeling competitions. Use when a user wants to turn their own CSV/XLSX/XLS data and a natural-language plotting request into a publication-style scientific figure by selecting from the existing template catalog, adapting the selected Python template without modifying originals, executing the plot, checking technical and visual quality, and repairing clear issues when needed.
---

# ModelViz Skill

Use this skill to create competition-report or paper-ready scientific visualizations from user data and the local template library. The workflow chooses an existing template, preserves its layout and style, adapts it to real data, runs the generated script, and returns the final chart, code, dependency list, and quality reports.

Do not use this skill for explaining math concepts without plotting, modifying user data, running arbitrary non-plotting code, or forcing an unsupported chart type when no suitable template exists.

## Inputs

- Natural-language plotting request from the user.
- User data file. Current code supports `csv`, `xlsx`, and `xls` through `src/tools/prepare_dataset_context.py`.
- Optional chart type, style, negative requirements, output preferences, and Excel `sheet_name`.
- A chat model or mock runnable capable of structured output for the LLM stages.

Ask a clarification question before continuing if the request is too vague, the required sheet is unclear, the data cannot support the request, no candidate template fits, or several columns have ambiguous meaning.

## Core Rule

Keep this responsibility split:

- Program code handles file I/O, Pydantic validation, deterministic candidate recall, data sampling facts, dependency inspection/check/install, script execution, artifact checks, image statistics, warning collection, repair limits, and JSON reports.
- The language model handles natural-language requirement parsing, semantic data interpretation, final template choice from candidates, template style understanding, column mapping, template code adaptation, visual judgment, and localized code repair.

Do not introduce an Agent, RAG, embeddings, or a vector database. Do not bypass tools by inventing files, installed dependencies, candidate templates, generated images, or quality status.

## Implementation Map

Catalog and workflow references:

- `docs/template_catalog.yaml`
- `docs/template_index.csv`
- `docs/stage5_stage6_skill_workflow.md`

Service entry files:

- `src/services/requirement_parser.py`
- `src/services/candidate_matching_pipeline.py`
- `src/services/final_template_selection_pipeline.py`
- `src/services/template_adaptation_pipeline.py`
- `src/services/plot_quality_pipeline.py`

Tool files used by the main flow:

- `src/tools/load_requirement.py`
- `src/tools/load_catalog.py`
- `src/tools/match_candidate_templates.py`
- `src/tools/rank_candidate_templates.py`
- `src/tools/save_candidate_templates.py`
- `src/tools/prepare_dataset_context.py`
- `src/tools/load_selected_template.py`
- `src/tools/inspect_template_dependencies.py`
- `src/tools/check_python_dependencies.py`
- `src/tools/install_python_dependencies.py`
- `src/tools/save_adapted_script.py`
- `src/tools/execute_plot_script.py`
- `src/tools/validate_output_artifacts.py`
- `src/tools/inspect_generated_image.py`
- `src/tools/collect_plot_warnings.py`

## Main Workflow

### 1. Parse User Requirement

Use `src.services.requirement_parser.parse_and_save_requirement`.

It combines:

- Prompt: `src/prompts/requirement_parser.py`
- Schema: `schemas/requirement_schema.py`
- Vocabulary: `docs/requirement_vocabulary.yaml`
- Output: `workspace/user_requirement.json`

The schema is `PlotRequirement` with fields `original_request`, `goal`, `functional_keywords`, `chart_types`, `style_keywords`, `use_case`, `negative_requirements`, `explicit_template`, `is_ambiguous`, and `clarification_question`.

Rules:

- Preserve `original_request`.
- If the user did not specify a chart type, keep `chart_types=[]`.
- Put style words such as `科研风`, `简洁`, and `低饱和` in `style_keywords`, not `functional_keywords`.
- Put exclusions such as `不要雷达图` or `不要三维` in `negative_requirements`.
- If `is_ambiguous=true`, ask `clarification_question` and stop before template recall.

### 2. Recall Candidate Templates

Use `src.services.candidate_matching_pipeline.run_candidate_matching_pipeline`.

It calls the stage-3 tools in this order:

1. `load_user_requirement`
2. `load_template_catalog`
3. `match_candidate_templates`
4. `rank_candidate_templates`
5. `save_candidate_templates`

Inputs:

- `workspace/user_requirement.json`
- `docs/template_catalog.yaml`

Output:

- `workspace/candidate_templates.json`

The template catalog currently contains 89 templates across 12 categories. Candidate recall is deterministic and does not read user data. Default `top_k=8`, and `top_k` is constrained to 5-10. `min_score` is constrained to 0-1.

Scoring in `src/tools/match_candidate_templates.py` prioritizes:

- Functional keyword matches against template keywords.
- Explicit chart type matches.
- Original user wording against template synonyms.
- Goal and use-case matches.
- Style matches with low weight.
- User negative requirements as exclusion.
- Template `negative_keywords` as penalties.

Do not treat the top candidate as the final answer. Stage 3 only recalls 5-10 plausible candidates.

### 3. Select Final Template With Data

Use `src.services.final_template_selection_pipeline.run_final_template_selection_pipeline`.

It combines:

- Data tool: `prepare_dataset_context`
- Prompt: `src/prompts/final_template_selector.py`
- Schema: `src/schemas/final_template_selection.py`
- Inputs: `workspace/user_requirement.json`, `workspace/candidate_templates.json`, and the user data file
- Outputs: `workspace/dataset_context.json`, `workspace/final_template_selection.json`

`prepare_dataset_context` only returns facts: file name, file type, sheet, row count, column count, real column names, head records, sampled records, sampling status, and warnings. It must not decide which chart is best.

The model must analyze the real data meaning and choose only from `candidate_templates.json`. Python validation then rejects:

- Template IDs outside the candidate list.
- Template names inconsistent with selected IDs.
- Data columns not present in the file.
- Missing `clarification_question` when `needs_clarification=true`.

If `selected_template_id` is null, stop and return the model's reason or question.

### 4. Inspect and Complete Template Dependencies

Use the first part of `src.services.template_adaptation_pipeline.run_template_adaptation_pipeline`.

It calls:

- `load_selected_template`
- `inspect_template_dependencies`
- `check_python_dependencies`
- `install_python_dependencies`
- `check_python_dependencies` again after initial install

Files:

- Catalog: `docs/template_catalog.yaml`
- Package mapping: `docs/package_name_mapping.yaml`
- Reports: `workspace/dependency_report.json`, `workspace/dependency_install_result.json`, `workspace/requirements.generated.txt`

Dependency rules:

- Inspect imports with Python AST and merge catalog `dependencies`.
- Convert submodules to top-level imports.
- Exclude standard-library modules and project-internal modules.
- Map known import names to pip package names using `docs/package_name_mapping.yaml`.
- Stop if required dependencies cannot be mapped.
- Install only validated ordinary package names with the current Python executable.
- Never execute installation commands from template code.
- Never use `shell=True`, `sudo`, URLs, Git addresses, local paths, or extra pip arguments.
- Do not overwrite project `requirements.txt`; write task dependencies to `workspace/requirements.generated.txt`.

### 5. Generate Adaptation Plan and Code

Continue with `run_template_adaptation_pipeline`.

Planning:

- Prompt: `src/prompts/template_adaptation_plan.py`
- Schema: `src/schemas/adaptation_plan.py`
- Output: `workspace/adaptation_plan.json`

Code generation:

- Prompt: `src/prompts/template_code_adapter.py`
- Schema: `src/schemas/adaptation_result.py`
- Outputs: `workspace/adaptation_result.json`, `workspace/adapted_plot.py`

The model must preserve core template layout, plotting library, color style, axes/legend conventions, and report structure where possible. It must replace template demo data with real data from the user file and use only real column names.

Python validation rejects:

- Nonexistent data columns.
- Dependencies not confirmed available.
- Empty `adapted_code`.
- Attempts to save into `templates/`.

Original templates under `templates/` are read-only. Do not edit or overwrite user data.

If the code model requests additional dependencies, the current pipeline allows one extra installation pass through `install_python_dependencies`. If that fails, stop.

### 6. Execute the Adapted Script

`run_template_adaptation_pipeline` saves and executes the script with:

- Tool: `save_adapted_script`
- Tool: `execute_plot_script`
- Script: `workspace/adapted_plot.py`
- Output directory: `outputs`
- Execution report: `workspace/execution_result.json`

`execute_plot_script` uses `subprocess.run([...], shell=False)`, passes the data path and output directory as arguments, sets `DATA_PATH` and `OUTPUT_DIR`, captures stdout/stderr, and records generated files.

Stop if the script times out, the user data path does not exist, the script fails, or no output file is generated.

### 7. Run Technical and Visual Quality Checks

Use `src.services.plot_quality_pipeline.run_plot_quality_pipeline`.

Technical tools:

- `execute_plot_script`
- `validate_output_artifacts`
- `inspect_generated_image`
- `collect_plot_warnings`

Technical outputs:

- `workspace/technical_quality_report.json`
- `workspace/execution_result.json`

Checks include output location, extension, size, missing files, PNG readability, image dimensions, pixel variance, near-blank risk, Python runtime errors, dependency errors, font warnings, layout warnings, legend warnings, data warnings, and save failures.

Visual check:

- Prompt: `src/prompts/visual_quality_checker.py`
- Schema: `src/schemas/visual_quality_report.py`
- Output: `workspace/visual_quality_report.json`

The visual model should compare the generated image with the request, final selection, adaptation plan, technical report, and template preview when available. The current pipeline passes image paths in the prompt; when using a multimodal model, the caller should provide the actual generated image and preview image as visual inputs.

### 8. Repair Only Clear Problems

If either technical or visual checks require repair, `run_plot_quality_pipeline` uses:

- Prompt: `src/prompts/plot_code_repair.py`
- Schema: `src/schemas/repair_result.py`
- Backups: `workspace/repair_versions/adapted_plot_v{n}.py`
- History: `workspace/repair_history.json`
- Final report: `workspace/final_quality_report.json`

Repair rules:

- Fix only issues named in the technical or visual reports.
- Do not redesign the chart from scratch.
- Do not modify `templates/` or user data.
- Preserve data meaning and template style.
- Prefer existing dependencies and plotting logic.
- New dependencies must go through `install_python_dependencies`.

Limits:

- `max_repair_attempts=3`
- `max_dependency_install_rounds=2`

Stop when all checks pass, maximum repairs are reached, the same issues repeat, repaired code is unchanged, dependency-install rounds are exhausted, dependency installation fails, the model returns `can_retry=false`, or user data cannot satisfy the requirement.

## Prompt Files

Use the Python `ChatPromptTemplate` modules as the executable prompt sources:

- `src/prompts/requirement_parser.py`
- `src/prompts/final_template_selector.py`
- `src/prompts/template_adaptation_plan.py`
- `src/prompts/template_code_adapter.py`
- `src/prompts/visual_quality_checker.py`
- `src/prompts/plot_code_repair.py`

Markdown prompt notes exist at `prompts/template_adaptation.md`, `prompts/visual_quality_check.md`, and `prompts/code_repair.md`. Treat them as supporting references, not a second execution path.

## Files and Outputs

Write intermediate JSON to `workspace/` using UTF-8 and unescaped Chinese. Write final chart files to `outputs/`. Keep debug logs separate from final JSON reports. Do not claim a file exists until the path has been created and checked.

Typical outputs:

- `workspace/user_requirement.json`
- `workspace/candidate_templates.json`
- `workspace/dataset_context.json`
- `workspace/final_template_selection.json`
- `workspace/dependency_report.json`
- `workspace/dependency_install_result.json`
- `workspace/requirements.generated.txt`
- `workspace/adaptation_plan.json`
- `workspace/adaptation_result.json`
- `workspace/adapted_plot.py`
- `workspace/execution_result.json`
- `workspace/technical_quality_report.json`
- `workspace/visual_quality_report.json`
- `workspace/repair_history.json`
- `workspace/final_quality_report.json`
- `outputs/chart.png` or other generated files reported by `execute_plot_script`

## Failure Handling

Return a structured failure with the failed stage, `error_type`, message, details, and suggestion when possible. Stop rather than pretending success for:

- Missing or invalid `user_requirement.json`, `template_catalog.yaml`, candidate file, final selection file, template source, or user data file.
- Unsupported data format or missing Excel sheet.
- Ambiguous requirement not yet clarified.
- Empty candidate list.
- Model output outside schema.
- Model-selected template outside candidates.
- Model-created data columns.
- Unmapped or uninstalled key dependencies.
- Illegal package names or failed dependency installation.
- Empty or unsafe adapted code.
- Script timeout or failure.
- Missing, corrupt, or blank output image.
- Visual model failure when visual checking is required.
- Maximum repair attempts reached.

## Current Limitations

- The final workflow requires caller-supplied structured-output model objects; tests use mock runnables.
- Visual inspection quality depends on giving the visual model the actual image content, not only the path string.
- `README.md`, `pyproject.toml`, and `scripts/` are not present; use the Python service entry points directly.
- The older recommendation pipeline (`run_template_matching_pipeline`) still exists for compatibility, but the Skill flow should use `run_candidate_matching_pipeline` and `candidate_templates.json`.
