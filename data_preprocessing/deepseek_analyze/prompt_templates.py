"""Prompt template utilities used by DeepSeek analysis scripts.

Each main prompt is split into system_prompt (static) and user_prompt (dynamic)
to optimize API cache hit rate and improve maintainability.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Set


class PromptTemplate:
    """Centralized prompt builders for the DeepSeek analysis pipeline."""

    # =========================================================================
    # Lifecycle Definitions (Structured)
    # =========================================================================
    
    LIFECYCLE_DEFINITIONS = {
        "Problem Modeling": {
            "id": 1,
            "desc": "Formulating classical problems as quantum-compatible mathematical models.",
            "submodules": [
                ("Problem Formulation", "Casting problem statements into quantum-representable physical systems"),
                ("Hamiltonian Construction", "Encoding objective functions (e.g., energy, cost) as Hamiltonian operators"),
                ("Model Validation", "Ensuring operators satisfy physical constraints (e.g., Hermiticity, unitarity, trace=1)")
            ]
        },
        "Data Encoding": {
            "id": 2,
            "desc": "Embedding classical data into Hilbert space via parameterized quantum circuits.",
            "submodules": [
                ("Input Validation", "Verifying data dimensions, normalization, and value ranges for encoding"),
                ("Feature Mapping", "Selecting encoding schemes (e.g., basis, angle, amplitude, IQP)"),
                ("State Preparation", "Building gate sequences (e.g., Ansatz) to prepare the target quantum state")
            ]
        },
        "Circuit Design": {
            "id": 3,
            "desc": "Designing hardware-agnostic quantum circuits that define state evolution and measurement.",
            "submodules": [
                ("Register Allocation", "Declaring quantum/classical registers and establishing qubit indexing"),
                ("Parameter Definition", "Creating symbolic variables for variational circuit parameters"),
                ("Circuit Construction", "Assembling gates (e.g., H, CX, T), control flow, and subroutines (e.g., Oracles)"),
                ("Measurement Configuration", "Specifying measurement bases and qubit-to-classical-bit mappings")
            ]
        },
        "Transpilation": {
            "id": 4,
            "desc": "Transpiling logical circuits into hardware-executable form under device constraints.",
            "submodules": [
                ("Backend Constraints", "Fetching coupling maps, native gates, and calibration data from backends"),
                ("Gate Decomposition", "Breaking down abstract unitaries into native gate sets"),
                ("Qubit Routing", "Mapping logical qubits to physical qubits; inserting SWAPs for connectivity"),
                ("Circuit Optimization", "Reducing depth via gate fusion, commutation, and cancellation")
            ]
        },
        "Execution": {
            "id": 5,
            "desc": "Running quantum programs on real hardware or simulators and managing job lifecycles.",
            "submodules": [
                ("Backend Setup", "Establishing sessions and setting shots, error mitigation options"),
                ("Job Management", "Submitting jobs, tracking status, and handling failures"),
                ("Noise Modeling", "(Simulation) Applying error channels (e.g., T1/T2 relaxation, depolarization)"),
                ("Result Retrieval", "Fetching and validating raw measurement outcomes")
            ]
        },
        "Post-processing": {
            "id": 6,
            "desc": "Interpreting measurement data and driving classical optimization loops.",
            "submodules": [
                ("Result Parsing", "Decoding bitstrings, fixing endianness, formatting counts"),
                ("Metrics Computation", "Computing expectation values, fidelity, and probability distributions"),
                ("Optimizer Control", "(Variational) Updating parameters via gradient descent or other optimizers")
            ]
        },
        "Infrastructure": {
            "id": 7,
            "desc": "Cross-cutting utilities that support quantum workflows without involving quantum logic directly.\nCovers classical software engineering essentials for quantum projects.",
            "submodules": [
                ("Visualization", "Rendering circuit diagrams, molecular structures, and result charts"),
                ("File Processing", "Handling quantum object I/O (e.g., QASM, JSON, QPY formats)"),
                ("Logging", "Providing logging, custom exceptions, and error management"),
                ("Version Migration", "Adapting to API changes and deprecated functions across versions")
            ]
        }
    }

    # =========================================================================
    # Stage 1: Change Category Classification
    # =========================================================================
    
    @staticmethod
    def build_change_category_system_prompt() -> str:
        """Stage 1 system prompt - role + classification criteria + output format (static, cacheable)"""
        return """# Code Change Classification

You are a senior software engineer specializing in code review. Your goal is to categorize each code change into exactly one of the predefined categories based on the commit message, file path, and diff provided.

---

## Category Definitions

> **Important**: These samples were pre-filtered by keywords like "bug" or "fix" in the commit message. However, **do not rely solely on the commit message**—examine the actual diff to determine whether the change truly fixes a defect or serves another purpose (e.g., adding a feature, refactoring).

Choose exactly ONE category from the following 8 options:

1. **Bug Fix**: Fixes a **defect** that causes incorrect output, runtime errors, or crashes.
   - Key indicator: The code produced **wrong results** or **failed to run** before the change.
   - Exclude: Fixes to test code should be classified as Test.
2. **Feature**: Adds **new** functionality that did not exist before, or extends capabilities beyond the original scope.
   - Key indicator: The code was **working as designed** before; this change adds something new.
3. **Refactor**: Restructures code or improves performance **without changing external behavior**.
   - Key indicator: Input/output behavior remains identical; only internal implementation changes.
4. **Style**: Formatting-only changes (whitespace, indentation) with no logic changes.
5. **Test**: Changes only to tests, without touching production code.
6. **Build/Chore**: Changes to build scripts, dependencies, or CI/CD pipelines.
7. **Documentation**: Changes only to comments, README, or docs.
8. **Other**: Does not fit any category above.

> **Tip**: The file path often provides classification hints. For example, files under `tests/` typically indicate **Test**; `setup.py`, `requirements.txt`, or CI config files suggest **Build/Chore**; files under `docs/` point to **Documentation**.

---

## Output Format

**Important**: Output your analysis as a **valid JSON object** with these fields:

- **id**: Sample ID (use the value provided in the input)
- **reason**: A concise explanation (1–2 sentences) of why you chose this category.
- **change_category**: One of the 8 categories above.

### Examples

**Example 1 (Bug Fix):**
```json
{
  "id": "sample_1",
  "reason": "The diff shows matrix multiplication order was fixed (A @ B -> B @ A), correcting a computational error that produced wrong results.",
  "change_category": "Bug Fix"
}
```

**Example 2 (Feature):**
```json
{
  "id": "sample_2",
  "reason": "The diff adds a new `timeout` parameter to the function signature and implements retry logic, extending the existing functionality.",
  "change_category": "Feature"
}
```

**Note**: Output only the JSON object. No extra text or markdown."""

    @staticmethod
    def build_change_category_user_prompt(sample: Any) -> str:
        """Stage 1 user prompt - sample data (dynamic)"""
        return f"""## Input

### 1. Sample ID
`{sample.id}`

### 2. Commit Message
"{sample.commit_message}"
*This message describes the entire commit and may span multiple files. Use it to understand the developer's intent, but base your classification primarily on the diff below.*

### 3. File Path
`{sample.file_path}`

### 4. Original File (before change)
```python
{sample.buggy_file}
```

### 5. Diff
```diff
{sample.diff}
```"""

    @staticmethod
    def get_change_category_prompts(sample: Any) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt) tuple for Stage 1"""
        return (
            PromptTemplate.build_change_category_system_prompt(),
            PromptTemplate.build_change_category_user_prompt(sample)
        )

    @staticmethod
    def build_change_category_prompt(sample: Any) -> str:
        """[Legacy] Returns combined full prompt for backward compatibility"""
        system = PromptTemplate.build_change_category_system_prompt()
        user = PromptTemplate.build_change_category_user_prompt(sample)
        return f"{system}\n\n---\n\n{user}"



    @staticmethod
    def _get_lifecycle_submodule_description(
        dynamic_submodules: Optional[List[Dict]] = None,
        target_stage: Optional[str] = None
    ) -> str:
        """Generates lifecycle stage and submodule descriptions.
        
        If dynamic_submodules are provided, they are integrated into their respective stages.
        If target_stage is provided, only generates description for that specific stage.
        """
        output = []
        
        # Group dynamic submodules by stage
        dynamic_by_stage = {}
        if dynamic_submodules:
            for mod in dynamic_submodules:
                stage = mod.get('lifecycle_stage')
                if stage:
                    if stage not in dynamic_by_stage:
                        dynamic_by_stage[stage] = []
                    dynamic_by_stage[stage].append(mod)

        # Iterate definitions and generate descriptions
        for stage_name, data in PromptTemplate.LIFECYCLE_DEFINITIONS.items():
            # Filter for specific stage
            if target_stage and stage_name != target_stage:
                continue
                
            stage_id = data["id"]
            desc = data["desc"]
            
            output.append(f"## Stage {stage_id}: {stage_name}")
            output.append(f"{desc}\n")
            output.append("**Submodules:**")
            
            # 1. Static submodules
            for sub_name, sub_desc in data["submodules"]:
                output.append(f"- **{sub_name}**: {sub_desc}")
            
            # 2. Dynamic submodules (if any)
            if stage_name in dynamic_by_stage:
                for dyn_mod in dynamic_by_stage[stage_name]:
                    d_name = dyn_mod.get('name', 'Unnamed')
                    d_desc = dyn_mod.get('description', 'No description')
                    logging.debug(f"Adding dynamic submodule to prompt context: [{stage_name}] {d_name}")
                    output.append(f"- **{d_name}**: {d_desc}")
            
            output.append("")  # Empty line between stages

        return "\n".join(output)



    # =========================================================================
    # Stage 2: Quantum Relevance Analysis
    # =========================================================================

    @staticmethod
    def build_analysis_system_prompt() -> str:
        """Stage 2 system prompt - role + lifecycle + analysis workflow + output format (static, cacheable)"""
        # Stage 2 currently does not use dynamic submodules (or uses accumulated results from previous rounds if needed)
        lifecycle_desc = PromptTemplate._get_lifecycle_submodule_description()
        return rf"""# Bug-Fix Classification

You are a quantum computing software engineering researcher. Your task is to:
1. Determine whether a bug fix is **quantum-specific**.
2. Classify the bug fix into one of the 7 **lifecycle stages**, and further identify a **submodule** within that stage (or "None" if no submodule fits).

## Quantum Software Lifecycle
Below are the 7 stages of the quantum software lifecycle and their submodule definitions (submodules may not be exhaustive):
{lifecycle_desc}

### Field Descriptions

1. **id**: Sample ID (use the value provided in the input)
2. **lifecycle_stage**: Select the most appropriate stage (Stage 1–7) based on the **core functionality** of the change.
3. **submodule**: Within the selected `lifecycle_stage`, find a matching submodule.
   - If the change **fully matches** a submodule's description under that stage, provide the submodule name.
   - Otherwise, use `"None"` (while keeping the parent stage).
   - **Important**: Submodules are tightly bound to their stages. For example, "Visualization" exists only in Stage 7; if the change belongs to Stage 6, you cannot use that submodule.
4. **quantum_specific**: 
   - `true`: We consider a bug to be quantum-specific if the mistake is in handling quantum-specific concepts, which typically implies that understanding and fixing the bug requires knowledge of the quantum programming domain.
   - `false`: All other bugs are classical—they could occur in any non-quantum project and do not require quantum knowledge to fix.
5. **lifecycle_reason**: In 30–60 words, explain **why you selected this lifecycle stage and submodule**. If submodule is "None", explain why it does not fit any specific submodule.
6. **quantum_reason**: In 30–60 words, explain **why quantum_specific is true or false**.

### Analysis Order
Follow this sequence when analyzing:
1. First, determine lifecycle_stage: Based on the core functionality of the change, select the most appropriate stage from the 7 options.
2. Then, determine submodule: Within the chosen stage, find a matching submodule.
3. Fallback check: If no submodule matches in Step 2, check whether the submodules under "Stage 7: Infrastructure" (Visualization, File Processing, Logging, Version Migration) better describe the change. If so, update lifecycle_stage to "Infrastructure" and assign the corresponding submodule; otherwise, keep the original stage and set submodule to "None".
4. Finally, determine quantum_specific: Apply the criteria defined above.

### Output Examples
**Important**: Output your analysis strictly as a **valid JSON object** in the following format:

**Example 1: Quantum-specific bug **
```json
{{
  "id": "single_1",
  "lifecycle_reason": "The fix addresses confusion between PauliTerm and PauliSum during Hamiltonian construction, which involves encoding objective functions as quantum operators.",
  "quantum_reason": "Understanding this bug requires knowledge of Pauli operator algebra and quantum mechanical operator theory, making it quantum-specific.",
  "lifecycle_stage": "Problem Modeling",
  "submodule": "Hamiltonian Construction",
  "quantum_specific": true
}}
```

**Example 2: Within quantum lifecycle but a classical software engineering issue**
```json
{{
  "id": "single_2",
  "lifecycle_reason": "The fix occurs in the backend connection module, involving session management and resource cleanup for quantum backend services.",
  "quantum_reason": "The bug is a memory leak caused by unreleased resources after connection timeout—a general resource management issue that does not require quantum knowledge to fix.",
  "lifecycle_stage": "Execution",
  "submodule": "Backend Setup",
  "quantum_specific": false
}}
```

**Example 3: Within quantum lifecycle stage but no specific submodule**
```json
{{
  "id": "single_3",
  "lifecycle_reason": "The fix involves overall control flow for quantum circuit construction, falling under the circuit design stage but not matching any specific submodule.",
  "quantum_reason": "Understanding this bug requires knowledge of quantum circuit concepts and construction workflows, making it quantum-specific.",
  "lifecycle_stage": "Circuit Design",
  "submodule": "None",
  "quantum_specific": true
}}
```

**Example 4: Infrastructure stage, no specific submodule**
```json
{{
  "id": "single_4",
  "lifecycle_reason": "The fix involves script logic for auto-generating quantum gate documentation, falling under infrastructure but not matching Visualization, File Processing, Logging, or Version Migration submodules.",
  "quantum_reason": "The bug only involves docstring template concatenation errors and does not require quantum knowledge to fix.",
  "lifecycle_stage": "Infrastructure",
  "submodule": "None",
  "quantum_specific": false
}}
```
**Note**: Output only the JSON object. Do not add any extra text or markdown."""

    @staticmethod
    def build_analysis_user_prompt(sample: Any) -> str:
        """Stage 2 user prompt - sample data"""
        return f"""## Input

### Sample Metadata
- **id**: `{sample.id}`

### Original File (before change)
```python
{sample.buggy_file}
```

### Diff (function-level change)
```diff
{sample.diff}
```
"""

    @staticmethod
    def get_analysis_prompts(sample: Any) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt) tuple for Stage 2"""
        return (
            PromptTemplate.build_analysis_system_prompt(),
            PromptTemplate.build_analysis_user_prompt(sample)
        )

    @staticmethod
    def build_analysis_prompt(sample: Any) -> str:
        """[Legacy] Returns combined full prompt for backward compatibility"""
        system = PromptTemplate.build_analysis_system_prompt()
        user = PromptTemplate.build_analysis_user_prompt(sample)
        return f"{system}\n\n---\n\n{user}"

    @staticmethod
    def build_custom_prompt(sample: Any, template: str) -> str:
        """Fill a custom template using the BugSample dataclass fields."""
        return template.format(**asdict(sample))

    # =========================================================================
    # Stage 3: Batch Submodule Classification
    # =========================================================================

    @staticmethod
    def build_stage3_system_prompt(
        current_stage: str = "Unknown",
        dynamic_submodules: Optional[List[Dict]] = None
    ) -> str:
        """Stage 3 system prompt - batch submodule classification"""
        lifecycle_desc = PromptTemplate._get_lifecycle_submodule_description(
            dynamic_submodules, 
            target_stage=current_stage
        )
        
        return rf"""# Batch Submodule Classification

You are a quantum software engineering researcher. Your task is to assign submodules to Bug Fix samples that have already been placed in a lifecycle stage but lack a specific submodule.

## Submodule Definitions for This Stage
> **Note**: Only submodules for the **current lifecycle stage** are listed below. 

{lifecycle_desc}

## Background

In the previous step, these samples were assigned to a lifecycle stage but their submodule was set to **"None"**—either because their features were ambiguous or because no existing submodule was a good match.

Follow these three steps:

1. **Match to existing submodules first**: Review each sample's `lifecycle_reason` and code diff. If it fits an existing submodule (including any discovered ones above), assign it directly.
2. **Find common patterns among remaining samples**: Group samples that cannot be matched to existing submodules and look for shared functionality or recurring problem types.
3. **Propose new submodules sparingly**: Create the minimum number of new submodules to cover unmatched samples.

## Critical Rules

**Rule 1**: Every submodule name in `classifications` must either:
- Already exist in the submodule list above, OR
- Be declared in `new_submodules` in this output

**Rule 2**: Every `id` in `classifications` must be a valid sample ID from the input. Do not fabricate IDs.

## Output Format

```json
{{
  "new_submodules": [
    {{
      "name": "New Submodule Name",
      "description": "Brief description (10-30 words)"
    }}
  ],
  "classifications": [
    {{
      "id": "sample_id",
      "submodule": "Assigned submodule name",
      "reason": "Brief explanation (20-40 words)"
    }}
  ]
}}
```

## Constraints

1. **Group similar samples**: Avoid one-sample-per-submodule fragmentation.
2. **Quality requirements**:
   - **Orthogonality**: New submodules must not overlap with existing ones.
   - **Abstraction**: Submodules represent functional categories (e.g., "Parameter Validation"), not specific fixes.
3. **Stay within the stage**: Do not use submodules from other lifecycle stages.
4. **Naming**: Use concise English names (2-4 words) with clear technical meaning.
5. **Completeness**: Every input sample must appear in `classifications`.

**Note**: Output only the JSON object. No extra text or markdown."""

    @staticmethod
    def build_stage3_user_prompt(
        samples: List[Any],
        stage_results: List[Dict],
        lifecycle_hierarchy: Dict[str, List[str]]
    ) -> str:
        """
        Stage 3 user prompt - batch sample data (same stage)
        
        Args:
            samples: List of BugSample (all from the same stage)
            stage_results: Stage 2 results (containing lifecycle_stage)
            lifecycle_hierarchy: Lifecycle-submodule hierarchy definition
        """
        id_to_result = {r.get('sample_id'): r for r in stage_results}
        
        first_result = id_to_result.get(samples[0].id, {}) if samples else {}
        current_stage = first_result.get('lifecycle_stage', 'Unknown')
        
        prompt_parts = ["## Input\n"]
        
        prompt_parts.append(f"### Current Lifecycle Stage: {current_stage}\n")
        existing_submodules = lifecycle_hierarchy.get(current_stage, [])
        existing_submodules = [s for s in existing_submodules if s != "None"]
        
        if existing_submodules:
            prompt_parts.append(f"**Predefined Submodules**: {', '.join(existing_submodules)}")
        else:
            prompt_parts.append(f"**Predefined Submodules**: (none)")
        prompt_parts.append("")
        
        valid_ids = [str(sample.id) for sample in samples]
        prompt_parts.append(f"\n### Valid Sample IDs ({len(samples)} total, only use these IDs)")
        prompt_parts.append(f"**{valid_ids}**\n")
        
        prompt_parts.append(f"### Sample Details\n")
        
        for idx, sample in enumerate(samples, 1):
            result = id_to_result.get(sample.id, {})
            lifecycle_reason = result.get('lifecycle_reason', 'N/A')
            
            prompt_parts.append(f"#### Sample {idx}: `{sample.id}`")
            prompt_parts.append(f"- **lifecycle_reason** (from Stage 2): {lifecycle_reason}")
            prompt_parts.append(f"- **buggy_file**:")
            prompt_parts.append(f"```python\n{sample.buggy_file}\n```")
            prompt_parts.append(f"- **diff**:")
            prompt_parts.append(f"```diff\n{sample.diff}\n```")
            prompt_parts.append("")
        
        return "\n".join(prompt_parts)

    @staticmethod
    def get_stage3_prompts(
        samples: List[Any],
        stage_results: List[Dict],
        lifecycle_hierarchy: Dict[str, List[str]],
        dynamic_submodules: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt) tuple for Stage 3"""
        # Get current stage
        id_to_result = {r.get('sample_id'): r for r in stage_results}
        first_result = id_to_result.get(samples[0].id, {}) if samples else {}
        current_stage = first_result.get('lifecycle_stage', 'Unknown')

        return (
            PromptTemplate.build_stage3_system_prompt(current_stage, dynamic_submodules),
            PromptTemplate.build_stage3_user_prompt(
                samples, stage_results, lifecycle_hierarchy
            )
        )

    # =========================================================================
    # Stage 4: Submodule Merging
    # =========================================================================

    @staticmethod
    def build_stage4_system_prompt(target_stage: str, min_count: int) -> str:
        """Stage 4 system prompt - submodule merging"""
        return f"""# Submodule Merging and Optimization

You are a quantum software architect. Your task is to consolidate all submodules for the lifecycle stage **{target_stage}** into a clean, orthogonal, and non-redundant hierarchy.

## Input Description

You will receive two types of submodules:

1. **Static Submodules (Baseline)**: Expert-defined categories that are well-generalized and orthogonal. Preserve these as the core structure unless there is strong reason to merge them.
2. **Dynamic Submodules (Discovered)**: Algorithm-discovered patterns that fill gaps not covered by static submodules. They may be redundant or fragmented. Valuable discoveries should be kept; trivial ones should be merged into broader categories.

Dynamic submodules complement static ones—they are not simply absorbed.

## Merging Principles

- Submodules with n < 10 samples MUST be merged into modules with n ≥ 10. The final merged module must have at least 10 samples total.
- Use concise 2-4 word names. Avoid "and" or "or" in names—pick the dominant concept instead.
- Merge subsets into parent modules; combine semantically similar ones.
- Target 3–6 output modules per lifecycle stage.
- Every input must map to exactly one output. No omissions.

## Output Format

```json
{{
  "merged_submodules": [
    {{
      "name": "Concise module name",
      "description": "Brief description covering merged functionalities",
      "source_submodules": ["Original A", "Original B", "Original C"]
    }}
  ]
}}
```

**Note**: Output only the JSON object. No extra text or markdown."""

    @staticmethod
    def build_stage4_user_prompt(
        stage: str, 
        static_subs: List[Tuple], 
        dynamic_subs: List[Dict],
        submodule_counts: Optional[Dict[str, int]] = None
    ) -> str:
        """Stage 4 user prompt - submodules to be merged
        
        Args:
            stage: Lifecycle stage name
            static_subs: List of (name, description) tuples for static submodules
            dynamic_subs: List of dicts with 'name' and 'description' for dynamic submodules
            submodule_counts: Optional dict mapping submodule name -> sample count
        """
        prompt = [f"## Submodules to Merge — Stage: {stage}\n"]
        
        # Calculate total for percentage if counts provided
        total_count = sum(submodule_counts.values()) if submodule_counts else 0
        
        prompt.append(f"### 1. Static Submodules ({len(static_subs)} total)")
        if static_subs:
            for name, desc in static_subs:
                if submodule_counts and name in submodule_counts:
                    count = submodule_counts[name]
                    pct = (count / total_count * 100) if total_count > 0 else 0
                    prompt.append(f"- **{name}** (n={count}, {pct:.1f}%): {desc}")
                else:
                    prompt.append(f"- **{name}**: {desc}")
        else:
            prompt.append("(none)")
            
        prompt.append(f"\n### 2. Dynamic Submodules ({len(dynamic_subs)} total)")
        if dynamic_subs:
            for sub in dynamic_subs:
                name = sub.get('name', 'Unnamed')
                desc = sub.get('description', 'No description')
                if submodule_counts and name in submodule_counts:
                    count = submodule_counts[name]
                    pct = (count / total_count * 100) if total_count > 0 else 0
                    prompt.append(f"- **{name}** (n={count}, {pct:.1f}%): {desc}")
                else:
                    prompt.append(f"- **{name}**: {desc}")
        else:
            prompt.append("(none)")
            
        return "\n".join(prompt)

    @staticmethod
    def get_stage4_prompts(
        stage: str, 
        static_subs: List[Tuple], 
        dynamic_subs: List[Dict],
        submodule_counts: Optional[Dict[str, int]] = None
    ) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt) tuple for Stage 4"""
        min_count = len(static_subs)
        # If static submodules are few (0 or 1), set a reasonable minimum (3) to avoid merging into 1
        if min_count < 3:
            min_count = 3
            
        return (
            PromptTemplate.build_stage4_system_prompt(stage, min_count),
            PromptTemplate.build_stage4_user_prompt(stage, static_subs, dynamic_subs, submodule_counts)
        )


__all__ = ["PromptTemplate"]
