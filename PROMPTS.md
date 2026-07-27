# Prompt Design

## DeepSeek Stage 1

Task: Classify each code change into one of eight change categories such as `Bug Fix`, `Feature`, `Refactor`, `Test`, and `Documentation`

System Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L96](data_preprocessing/deepseek_analyze/prompt_templates.py#L96)

User Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L158](data_preprocessing/deepseek_analyze/prompt_templates.py#L158)

Execution Path:
- [data_preprocessing/deepseek_analyze/stage_one.py](data_preprocessing/deepseek_analyze/stage_one.py)

Key Design:
- Uses a senior-code-reviewer framing
- Relies on commit message, file path, original file, and diff
- Enforces exactly one label from a fixed 8-class taxonomy
- Requires strict JSON output with `id`, `reason`, and `change_category`

## DeepSeek Stage 2

Task: Determine whether a bug fix is quantum-specific and classify it into a lifecycle stage and submodule

System Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L257](data_preprocessing/deepseek_analyze/prompt_templates.py#L257)

User Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L345](data_preprocessing/deepseek_analyze/prompt_templates.py#L345)

Execution Path:
- [data_preprocessing/deepseek_analyze/stage_two.py](data_preprocessing/deepseek_analyze/stage_two.py)

Key Design:
- Encodes a 7-stage quantum software lifecycle directly in the system prompt
- Separates `lifecycle_stage`, `submodule`, and `quantum_specific`
- Requires short rationales for both lifecycle assignment and quantum specificity
- Uses original file plus function-level diff as the main evidence
- Outputs strict JSON

## DeepSeek Stage 3

Task: Batch-classify samples with missing submodules and propose new submodules when necessary

System Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L388](data_preprocessing/deepseek_analyze/prompt_templates.py#L388)

User Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L458](data_preprocessing/deepseek_analyze/prompt_templates.py#L458)

Execution Path:
- [data_preprocessing/deepseek_analyze/stage_three.py](data_preprocessing/deepseek_analyze/stage_three.py)

Key Design:
- Operates on batches within the same lifecycle stage
- Reuses Stage 2 `lifecycle_reason` as structured context
- Forces every assigned submodule to be either predefined or newly declared
- Encourages minimal and non-overlapping new submodules
- Outputs strict JSON with `new_submodules` and `classifications`

## DeepSeek Stage 4

Task: Merge static and dynamic submodules into a cleaner lifecycle-stage hierarchy

System Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L533](data_preprocessing/deepseek_analyze/prompt_templates.py#L533)

User Prompt Location: [data_preprocessing/deepseek_analyze/prompt_templates.py#L573](data_preprocessing/deepseek_analyze/prompt_templates.py#L573)

Execution Path:
- [data_preprocessing/deepseek_analyze/stage_four.py](data_preprocessing/deepseek_analyze/stage_four.py)

Key Design:
- Treats static submodules as baseline structure and dynamic submodules as candidate refinements
- Pushes small submodules to be merged into broader categories
- Targets a compact, orthogonal final hierarchy
- Requires every source submodule to map into exactly one merged output
- Outputs strict JSON with `merged_submodules`

## LLM Bug Detection

Task: Decide whether a target code file contains a definite bug

System Prompt Location: [llm/detection_evaluator.py#L166](llm/detection_evaluator.py#L166)

User Prompt Location: [llm/detection_evaluator.py#L194](llm/detection_evaluator.py#L194)

Prompt Assembly Location: [llm/detection_evaluator.py#L294](llm/detection_evaluator.py#L294)

Key Design:
- Uses a high-evidence-bar code-review prompt
- Explicitly distinguishes definite bugs from smells or uncertain issues
- Adds quantum-domain awareness to avoid false positives on valid quantum idioms
- Uses the full target Python file as input
- Outputs strict JSON with `label` and `reason`

## LLM Bug Repair

Task: Generate bug fixes as `SEARCH/REPLACE` edit blocks over a full buggy file

System Prompt Location: [llm/repair_evaluator.py#L199](llm/repair_evaluator.py#L199)

User Prompt Location: [llm/repair_evaluator.py#L233](llm/repair_evaluator.py#L233)

Prompt Assembly Location: [llm/repair_evaluator.py#L451](llm/repair_evaluator.py#L451)

Key Design:
- Forces a constrained edit format instead of free-form explanations
- Emphasizes exact search matching, uniqueness, and minimal edits
- Uses the full buggy file as context
- Disallows JSON, unified diff, and natural-language reasoning in outputs
- Optimizes for directly applicable local edits
