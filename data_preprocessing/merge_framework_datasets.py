#!/usr/bin/env python3
"""Merge per-framework method-level samples into one dataset."""

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import config
except ImportError:
    print("Failed to import config")
    sys.exit(1)


def load_framework_data(framework: str) -> list:
    """Load samples for one framework."""
    json_path = config.EXTRACTED_DIR / framework / "method_level_single.json"
    
    if not json_path.exists():
        print(f"  Missing file: {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Failed to read file: {e}")
        return []
    
    samples = data.get('samples', [])
    
    for sample in samples:
        sample['framework'] = framework
        if 'buggy_file_path' in sample:
            sample['parent_file_path'] = sample.pop('buggy_file_path')
    
    return samples


def get_sample_key(sample: dict) -> str:
    """Build a stable key for intra- and cross-framework deduplication."""
    commit_sha = sample.get('commit_sha', '')
    function_name = sample.get('function_name', '')
    buggy_code = sample.get('buggy_code', '')
    return f"{commit_sha}|{function_name}|{hash(buggy_code)}"


def merge_datasets(output_path: Path):
    """Merge all framework datasets using qiskit > cirq > pennylane priority."""
    print("=" * 60)
    print("🔀 Merge multi-framework datasets")
    print("=" * 60)
    
    framework_priority = ['qiskit', 'cirq', 'pennylane']
    
    all_samples = []
    framework_counts = {}
    framework_kept = {}
    duplicates_removed = 0
    seen_keys = set()
    
    for framework in framework_priority:
        if framework not in config.QUANTUM_FRAMEWORKS:
            continue
            
        print(f"\n📂 Loading {framework}...")
        samples = load_framework_data(framework)
        framework_counts[framework] = len(samples)
        
        kept_count = 0
        for sample in samples:
            key = get_sample_key(sample)
            if key not in seen_keys:
                seen_keys.add(key)
                all_samples.append(sample)
                kept_count += 1
            else:
                duplicates_removed += 1
        
        framework_kept[framework] = kept_count
        print(f"  ✅ Loaded {len(samples)} samples, kept {kept_count}")
    
    if not all_samples:
        print("\n❌ No samples found")
        return
    
    if duplicates_removed > 0:
        print(f"\n🔍 Dedup removed {duplicates_removed} duplicate samples")
    
    print(f"\n🔢 Assigning sample IDs...")
    field_order = [
        "id", "framework", "repository", "commit_sha", "commit_url",
        "commit_message", "parent_file_path", "function_name",
        "buggy_code", "fixed_code", "diff", "buggy_file", "fixed_file"
    ]
    samples_with_id = []
    for idx, sample in enumerate(all_samples, start=1):
        new_sample = {'id': idx}
        for field in field_order[1:]:
            if field in sample:
                new_sample[field] = sample[field]
        samples_with_id.append(new_sample)
    all_samples = samples_with_id
    print(f"  ✅ Assigned {len(all_samples)} IDs (1-{len(all_samples)})")
    
    merged_data = {
        "metadata": {
            "total_samples": len(all_samples),
            "framework_samples": framework_kept
        },
        "samples": all_samples
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Merge complete")
    except Exception as e:
        print(f"\n❌ Failed to save merged dataset: {e}")
        return
    
    print("\n" + "=" * 60)
    print("📊 Merge summary")
    print("=" * 60)
    for fw in framework_priority:
        if fw in framework_counts:
            loaded = framework_counts[fw]
            kept = framework_kept.get(fw, 0)
            removed = loaded - kept
            percentage = kept / len(all_samples) * 100 if all_samples else 0
            if removed > 0:
                print(f"  {fw}: {kept}/{loaded} ({percentage:.1f}%) [dedup {removed}]")
            else:
                print(f"  {fw}: {kept} ({percentage:.1f}%)")
    print(f"  {'─' * 30}")
    if duplicates_removed > 0:
        print(f"  Dedup removed: {duplicates_removed}")
    print(f"  Total: {len(all_samples)}")
    print(f"\n📁 Output file: {output_path}")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge per-framework method-level datasets")
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: data/03_extracted/method_level_single_merged.json)')
    args = parser.parse_args()
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config.EXTRACTED_DIR / "method_level_single_merged.json"
    
    merge_datasets(output_path)


if __name__ == "__main__":
    main()
