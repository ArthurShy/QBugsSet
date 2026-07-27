#!/usr/bin/env python3
"""DeepSeek Analyze CLI entry point."""

import os
import sys
import json
import time
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

script_dir = Path(__file__).resolve().parent
code_dir = script_dir.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

if str(script_dir) in sys.path:
    sys.path.remove(str(script_dir))



from deepseek_analyze.core import (
    BugSample,
    AnalysisResult,
    DataLoader,
    ResultSaver,
    ErrorLogger,
)
from deepseek_analyze.prompt_templates import PromptTemplate
from deepseek_analyze.stage_one import StageOneAnalyzer
from deepseek_analyze.stage_two import StageTwoAnalyzer
from deepseek_analyze.stage_three import StageThreeAnalyzer

try:
    from api_clients import DeepSeekClient
    from api_clients.deepseek_client import (
        DEFAULT_MODEL,
        DEFAULT_TEMPERATURE,
        DEFAULT_MAX_OUTPUT_TOKENS,
        DEFAULT_SEED,
        MODEL_CONTEXT_WINDOW,
        DEEPSEEK_MODELS,
    )
except ImportError:
    DEFAULT_MODEL = "deepseek-chat"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_OUTPUT_TOKENS = 8000
    DEFAULT_SEED = 42
    MODEL_CONTEXT_WINDOW = {}
    DEEPSEEK_MODELS = {}
    DeepSeekClient = None


from deepseek_analyze.stage_three import StageThreeAnalyzer, DEFAULT_BATCH_SIZE
from deepseek_analyze.stage_four import StageFourAnalyzer
from deepseek_analyze.validators import VALID_LIFECYCLE_STAGES


def setup_logging(log_file: Optional[Path] = None):
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_test_labels_csv(csv_path: Path) -> List[Dict]:
    test_samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get('sample_id') or row.get('id')
            if sample_id:
                test_samples.append(row)
    
    valid_types = {'Bug Fix', 'Feature', 'Refactor', 'Style', 'Test', 'Build/Chore', 'Documentation', 'Other'}
    invalid_type_samples = {}
    
    for sample in test_samples:
        sample_type = (
            sample.get('type') 
            or sample.get('code-pair type')
            or sample.get('code_pair_type')
            or sample.get('codepair_type')
        )
        if sample_type:
            sample_type = sample_type.strip()
            if sample_type not in valid_types:
                if sample_type not in invalid_type_samples:
                    invalid_type_samples[sample_type] = []
                invalid_type_samples[sample_type].append(sample.get('sample_id') or sample.get('id'))
    
    if invalid_type_samples:
        logging.warning(f"\n{'='*80}")
        logging.warning(f"Invalid type field values found:")
        for invalid_type, sample_ids in invalid_type_samples.items():
            logging.warning(f"  - '{invalid_type}' ({len(sample_ids)} samples)")
        logging.warning(f"{'='*80}\n")
    
    return test_samples


def filter_samples_by_csv(samples: List[BugSample], csv_samples: List[Dict]) -> List[BugSample]:
    id_to_sample = {}
    for s in samples:
        id_to_sample[s.id] = s
        if s.id.startswith('single_'):
            id_to_sample[s.id.replace('single_', '')] = s
    
    filtered = []
    for csv_sample in csv_samples:
        sample_id = csv_sample.get('sample_id') or csv_sample.get('id')
        if sample_id:
            sample = id_to_sample.get(sample_id) or id_to_sample.get(f"single_{sample_id}")
            if sample:
                filtered.append(sample)
    
    return filtered


def extract_predictions_for_evaluation(test_samples: List[Dict], results: List[Dict]) -> Tuple[List, List, List, Dict]:
    id_to_result = {}
    for r in results:
        sid = r['sample_id']
        id_to_result[sid] = r
        if sid.startswith('single_'):
            num = sid.replace('single_', '')
            id_to_result[num] = r
    
    y_true = []
    y_pred = []
    sample_ids = []
    skipped_samples = []
    error_details = {
        'false_positive': [],
        'false_negative': []
    }
    
    for sample in test_samples:
        sample_id = sample.get('sample_id') or sample.get('id')
        
        if not sample_id:
            continue
        
        label_str = sample.get('label', '')
        try:
            true_label = int(label_str)
        except (ValueError, TypeError):
            continue
        
        result = id_to_result.get(sample_id)
        if result is None:
            result = id_to_result.get(f"single_{sample_id}")
        
        if result is None:
            skipped_samples.append(sample_id)
            continue
        
        quantum_specific = result.get('quantum_specific')
        
        if quantum_specific is None:
            pred_label = 0
        else:
            pred_label = 1 if quantum_specific else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        sample_ids.append(sample_id)
        
        if true_label == 0 and pred_label == 1:
            actual_change_category = (
                sample.get('type')
                or sample.get('code-pair type')
                or sample.get('code_pair_type')
                or sample.get('codepair_type')
            )
            if actual_change_category:
                actual_change_category = actual_change_category.strip()
            
            error_details['false_positive'].append({
                'sample_id': sample_id,
                'actual_change_category': actual_change_category,
                'predicted_change_category': result.get('change_category'),
                'submodule': result.get('submodule') or result.get('bug_type'),
                'reason': result.get('reason')
            })
        elif true_label == 1 and pred_label == 0:
            actual_change_category = (
                sample.get('type')
                or sample.get('code-pair type')
                or sample.get('code_pair_type')
                or sample.get('codepair_type')
            )
            if actual_change_category:
                actual_change_category = actual_change_category.strip()
            
            error_details['false_negative'].append({
                'sample_id': sample_id,
                'actual_change_category': actual_change_category,
                'predicted_change_category': result.get('change_category'),
                'submodule': result.get('submodule') or result.get('bug_type'),
                'reason': result.get('reason')
            })
    
    logging.info(f"Matched {len(y_true)} samples for evaluation")
    if skipped_samples:
        logging.warning(f"{len(skipped_samples)} samples skipped (no analysis result)")
    
    return y_true, y_pred, sample_ids, error_details


def extract_change_category_predictions(test_samples: List[Dict], results: List[Dict]) -> Tuple[List, List, List, Dict]:
    id_to_result = {}
    for r in results:
        sid = r['sample_id']
        id_to_result[sid] = r
        if sid.startswith('single_'):
            num = sid.replace('single_', '')
            id_to_result[num] = r
    
    y_true = []
    y_pred = []
    sample_ids = []
    skipped_samples = []
    error_details = {
        'false_positive': [],
        'false_negative': []
    }
    
    for sample in test_samples:
        sample_id = sample.get('sample_id') or sample.get('id')
        if not sample_id:
            continue
        
        commit_label = (
            sample.get('type')
            or sample.get('code-pair type')
            or sample.get('code_pair_type')
            or sample.get('codepair_type')
        )
        if commit_label is None:
            skipped_samples.append(sample_id)
            continue
        
        result = id_to_result.get(sample_id)
        if result is None:
            result = id_to_result.get(f"single_{sample_id}")
        
        if result is None:
            skipped_samples.append(sample_id)
            continue
        
        change_category_pred = result.get('change_category')
        if change_category_pred is None:
            skipped_samples.append(sample_id)
            continue
        
        true_label = 1 if commit_label.strip().lower() == 'bug fix' else 0
        pred_label = 1 if change_category_pred == 'Bug Fix' else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        sample_ids.append(sample_id)
        
        if true_label == 0 and pred_label == 1:
            error_details['false_positive'].append({
                'sample_id': sample_id,
                'actual_type': commit_label.strip(),
                'predicted_type': change_category_pred,
                'quantum_specific': result.get('quantum_specific'),
                'submodule': result.get('submodule') or result.get('bug_type')
            })
        elif true_label == 1 and pred_label == 0:
            error_details['false_negative'].append({
                'sample_id': sample_id,
                'actual_type': commit_label.strip(),
                'predicted_type': change_category_pred,
                'quantum_specific': result.get('quantum_specific'),
                'submodule': result.get('submodule') or result.get('bug_type')
            })
    
    logging.info(f"Matched {len(y_true)} samples for change category evaluation")
    if skipped_samples:
        logging.warning(f"{len(skipped_samples)} samples skipped in change category evaluation")
    
    return y_true, y_pred, sample_ids, error_details


def calculate_metrics_manual(y_true: List, y_pred: List) -> Dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    total = len(y_true)
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'total_samples': total,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'positive_samples': tp + fn,
        'negative_samples': tn + fp
    }


def print_evaluation_results(metrics: Dict, title: str = "Evaluation Results", error_details: Optional[Dict] = None) -> None:
    if "Bug Fix" in title or "commit" in title.lower() or "type" in title.lower():
        positive_label = "Bug Fix"
        negative_label = "Non-Bug Fix"
        pred_positive = "Pred Bug Fix"
        pred_negative = "Pred Non-Bug Fix"
        actual_positive = "Actual Bug Fix"
        actual_negative = "Actual Non-Bug Fix"
    else:
        positive_label = "Quantum"
        negative_label = "Non-Quantum"
        pred_positive = "Pred Quantum"
        pred_negative = "Pred Non-Quantum"
        actual_positive = "Actual Quantum"
        actual_negative = "Actual Non-Quantum"
    
    logging.info(f"\n{'='*80}")
    logging.info(f"📊 {title}")
    logging.info(f"{'='*80}")
    
    logging.info(f"\nMetrics:")
    logging.info(f"   Total: {metrics['total_samples']}")
    logging.info(f"   - {positive_label}: {metrics['positive_samples']}")
    logging.info(f"   - {negative_label}: {metrics['negative_samples']}")
    logging.info(f"")
    logging.info(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logging.info(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    logging.info(f"   Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    logging.info(f"   F1-Score: {metrics['f1_score']:.4f}")
    
    logging.info(f"\nConfusion Matrix:")
    logging.info(f"                 {pred_negative}  {pred_positive}")
    logging.info(f"   {actual_negative}:      {metrics['true_negative']:4d}      {metrics['false_positive']:4d}")
    logging.info(f"   {actual_positive}:        {metrics['false_negative']:4d}      {metrics['true_positive']:4d}")
    
    logging.info(f"{'='*80}\n")


def save_evaluation_report(
    output_path: Path,
    quantum_metrics: Dict,
    test_count: int,
    commit_metrics: Optional[Dict] = None,
    quantum_error_details: Optional[Dict] = None,
    commit_error_details: Optional[Dict] = None
) -> None:
    summary = {
        'label': {
            'test_samples': test_count,
            'evaluated_samples': quantum_metrics['total_samples'],
            'accuracy': quantum_metrics['accuracy'],
            'precision': quantum_metrics['precision'],
            'recall': quantum_metrics['recall'],
            'f1_score': quantum_metrics['f1_score']
        }
    }
    
    if commit_metrics:
        summary['type'] = {
            'evaluated_samples': commit_metrics['total_samples'],
            'accuracy': commit_metrics['accuracy'],
            'precision': commit_metrics['precision'],
            'recall': commit_metrics['recall'],
            'f1_score': commit_metrics['f1_score']
        }
    
    report = {
        'summary': summary,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    error_samples = {}
    if quantum_error_details:
        error_samples['quantum_related'] = {
            'false_positive': quantum_error_details.get('false_positive', []),
            'false_negative': quantum_error_details.get('false_negative', [])
        }
    if commit_error_details:
        error_samples['change_category'] = {
            'false_positive': commit_error_details.get('false_positive', []),
            'false_negative': commit_error_details.get('false_negative', [])
        }
    
    if error_samples:
        report['error_samples'] = error_samples
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Evaluation report saved: {output_path}")


def save_stage1_report(
    output_path: Path,
    stage1_results: List[Dict],
    total_samples: int,
    valid_count: int = 0,
    test_count: int = 0,
    csv_samples: Optional[List[Dict]] = None
) -> None:
    change_category_stats = {}
    change_category_examples = {}
    
    for result in stage1_results:
        change_category = result.get('change_category', 'Unknown')
        reason = result.get('reason', '')
        sample_id = result.get('sample_id', '')
        
        change_category_stats[change_category] = change_category_stats.get(change_category, 0) + 1
        
        if change_category not in change_category_examples:
            change_category_examples[change_category] = []
        if len(change_category_examples[change_category]) < 3 and reason:
            change_category_examples[change_category].append({
                'sample_id': sample_id,
                'reason': reason
            })
    
    change_category_percentages = {}
    for ct, count in change_category_stats.items():
        change_category_percentages[ct] = {
            'count': count,
            'percentage': round(count / len(stage1_results) * 100, 2) if len(stage1_results) > 0 else 0,
            'examples': change_category_examples.get(ct, [])
        }
    
    report = {
        'summary': {
            'total_samples': total_samples,
            'analyzed_samples': len(stage1_results),
            'change_category_distribution': change_category_percentages
        },
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if valid_count > 0 or test_count > 0:
        report['summary']['data_sources'] = {
            'valid_set': valid_count,
            'test_set': test_count
        }
    
    if csv_samples:
        y_true, y_pred, sample_ids, error_details = extract_change_category_predictions(csv_samples, stage1_results)
        if len(y_true) > 0:
            metrics = calculate_metrics_manual(y_true, y_pred)
            report['evaluation'] = {
                'bug_fix_classification': {
                    'description': 'Bug Fix (1) vs Other (0)',
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'confusion_matrix': {
                        'true_negative': metrics['true_negative'],
                        'false_positive': metrics['false_positive'],
                        'false_negative': metrics['false_negative'],
                        'true_positive': metrics['true_positive']
                    }
                }
            }
            if error_details:
                report['error_samples'] = {
                    'false_positive': error_details.get('false_positive', []),
                    'false_negative': error_details.get('false_negative', [])
                }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Stage 1 report saved: {output_path}")


def get_model_suffix(api_type: str, model: str) -> str:
    if model == 'deepseek-chat':
        return ''
    elif model == 'deepseek-reasoner':
        return '-reasoner'
    else:
        return f'-{model}'


def run_evaluation_mode(
    mode_name: str,
    csv_path: Optional[Path],
    samples: List[BugSample],
    stage_one_analyzer: StageOneAnalyzer,
    stage_two_analyzer: StageTwoAnalyzer,
    project_root: Path,
    no_incremental: bool,
    csv_path_2: Optional[Path] = None,
    parallel: bool = False,
    max_workers: int = 3,
    batch_save: bool = False,
    stage_one_only: bool = False,
    stage_two_only: bool = False,
    model: str = DEFAULT_MODEL
) -> None:
    is_evaluation_mode = csv_path is not None
    is_mix_mode = csv_path_2 is not None
    
    if is_evaluation_mode:
        logging.info(f"\n{'='*80}")
        logging.info(f"Mode: {mode_name} - Two-stage analysis with evaluation")
        logging.info(f"{'='*80}")
        
        if not csv_path.exists():
            logging.error(f"Label file not found: {csv_path}")
            return
        
        if is_mix_mode and not csv_path_2.exists():
            logging.error(f"Second label file not found: {csv_path_2}")
            return
        
        if is_mix_mode:
            valid_samples_csv = load_test_labels_csv(csv_path)
            test_samples_csv_data = load_test_labels_csv(csv_path_2)
            test_samples_csv = valid_samples_csv + test_samples_csv_data
            logging.info(f"Merged: {len(test_samples_csv)} samples")
        else:
            test_samples_csv = load_test_labels_csv(csv_path)
            logging.info(f"CSV contains {len(test_samples_csv)} samples")
        
        filtered_samples = filter_samples_by_csv(samples, test_samples_csv)
    else:
        logging.info(f"\n{'='*80}")
        logging.info(f"Mode: Full dataset - Two-stage analysis")
        logging.info(f"{'='*80}")
        
        filtered_samples = samples
        test_samples_csv = None
    
    if not filtered_samples:
        logging.error(f"No samples available")
        return
    
    logging.info(f"Analyzing {len(filtered_samples)} samples")
    
    model_suffix = get_model_suffix('deepseek', model)
    
    if is_evaluation_mode:
        prefix = mode_name
        stage1_path = project_root / f'data/04_analyzed/stage1/{prefix}_stage1{model_suffix}.json'
        stage2_path = project_root / f'data/04_analyzed/stage2/{prefix}_stage2{model_suffix}.json'
    else:
        stage1_path = project_root / f'data/04_analyzed/stage1/all_stage1{model_suffix}.json'
        stage2_path = project_root / f'data/04_analyzed/stage2/all_stage2{model_suffix}.json'
    
    if stage_two_only:
        if not stage1_path.exists():
            logging.error(f"Stage 1 result not found: {stage1_path}")
            return
        logging.info(f"Skipping Stage 1, proceeding to Stage 2")
    else:
        logging.info(f"\n{'='*80}")
        logging.info("Stage 1: Code change classification")
        logging.info(f"{'='*80}\n")
        
        stage1_saver = ResultSaver(stage1_path, load_existing=not no_incremental)
        stage_one_analyzer.result_saver = stage1_saver
        
        if parallel:
            stage_one_analyzer.analyze_batch_parallel(
                filtered_samples, 
                save_incremental=not no_incremental,
                max_workers=max_workers,
                realtime_save=not batch_save
            )
        else:
            stage_one_analyzer.analyze_batch(
                filtered_samples, 
                save_incremental=not no_incremental
            )
        
        logging.info(f"Stage 1 done, saved to: {stage1_path}")
        
        if stage_one_only:
            logging.info(f"\n{'='*80}")
            logging.info(f"Stage 1 only completed")
            logging.info(f"{'='*80}")
            return
    
    logging.info(f"\n{'='*80}")
    logging.info("Stage 2: Quantum relevance and lifecycle classification")
    logging.info(f"{'='*80}\n")
    
    with open(stage1_path, 'r', encoding='utf-8') as f:
        stage1_results = json.load(f).get('results', [])
    
    stage2_saver = ResultSaver(stage2_path, load_existing=not no_incremental)
    
    stage_two_analyzer.result_saver = stage2_saver
    
    if parallel:
        stage_two_analyzer.analyze_batch_parallel(
            filtered_samples, 
            stage1_results, 
            save_incremental=not no_incremental,
            max_workers=max_workers,
            realtime_save=not batch_save
        )
    else:
        stage_two_analyzer.analyze_batch(
            filtered_samples, 
            stage1_results, 
            save_incremental=not no_incremental
        )
    logging.info(f"Stage 2 done, saved to: {stage2_path}")
    
    if not is_evaluation_mode:
        logging.info(f"\n{'='*80}")
        logging.info(f"{mode_name} completed")
        logging.info(f"{'='*80}")
        return
    
    logging.info(f"\n{'='*80}")
    logging.info("Step 3: Evaluation")
    logging.info(f"{'='*80}\n")
    
    with open(stage1_path, 'r', encoding='utf-8') as f:
        stage1_results = json.load(f).get('results', [])
    with open(stage2_path, 'r', encoding='utf-8') as f:
        stage2_results = json.load(f).get('results', [])
    
    stage2_map = {r['sample_id']: r for r in stage2_results}
    merged_results = []
    for r1 in stage1_results:
        merged_results.append(stage2_map.get(r1['sample_id'], r1))
    
    y_true, y_pred, sample_ids, quantum_error_details = extract_predictions_for_evaluation(test_samples_csv, merged_results)
    if len(y_true) == 0:
        logging.error("No matching samples found for evaluation")
        return
    
    quantum_metrics = calculate_metrics_manual(y_true, y_pred)
    print_evaluation_results(quantum_metrics, title=f"Quantum Relevance Evaluation", error_details=quantum_error_details)
    
    commit_true, commit_pred, commit_sample_ids, commit_error_details = extract_change_category_predictions(test_samples_csv, merged_results)
    commit_metrics = None
    if len(commit_true) > 0:
        commit_metrics = calculate_metrics_manual(commit_true, commit_pred)
        print_evaluation_results(commit_metrics, title=f"Bug Fix Classification Evaluation", error_details=commit_error_details)
    
    report_name = f"{mode_name}_report{model_suffix}"
    eval_report_path = project_root / f'data/04_analyzed/reports/{report_name}.json'
    save_evaluation_report(
        eval_report_path, 
        quantum_metrics, 
        len(test_samples_csv), 
        commit_metrics,
        quantum_error_details,
        commit_error_details
    )
    
    logging.info(f"\n{'='*80}")
    logging.info(f"{mode_name} mode completed")
    logging.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek API analyzer for quantum bug fix samples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, default='data/03_extracted/method_level_single_merged.json', help='Input JSON file path')
    parser.add_argument('--output', type=str, default='data/04_analyzed/stage2/all_stage2.json', help='Output JSON file path')
    parser.add_argument('--log', type=str, default='data_preprocessing/log/deepseek_analyze.log', help='Log file path')
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--model', type=str, default=None, help='Model to use')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help=f'Temperature (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help=f'Max output tokens (default: {DEFAULT_MAX_OUTPUT_TOKENS})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--limit', type=int, help='Limit number of samples')
    parser.add_argument('--no-incremental', action='store_true', help='Disable incremental saving')
    parser.add_argument('--debug', action='store_true', help='Debug mode: process only 3 samples')
    parser.add_argument('--stage-one', action='store_true', help='Stage 1: Code change classification')
    parser.add_argument('--stage-two', action='store_true', help='Stage 2: Quantum relevance and lifecycle classification')
    parser.add_argument('--stage-three', action='store_true', help='Stage 3: Batch submodule classification')
    parser.add_argument('--stage-four', action='store_true', help='Stage 4: Submodule merging and optimization')
    parser.add_argument('--batch-size', type=int, default=None, help='Stage 3 batch size')
    parser.add_argument(
        '--stage',
        type=str,
        default=None,
        help='Specify lifecycle stage for Stage 3 (e.g. "Problem Mapping & Model Construction" or number 1-7)'
    )
    parser.add_argument('--stage-one-output', type=str, default='data/04_analyzed/stage1/all_stage1.json', help='Stage 1 output file path')
    parser.add_argument('--train', action='store_true', help='Training set mode')
    parser.add_argument('--train-json', type=str, default='data/bugfix_by_project/train_bugfix_by_project.json', help='Training set JSON path')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=5, help='Max parallel workers')
    parser.add_argument('--batch-save', action='store_true', help='Use batch saving in parallel mode')
    
    args = parser.parse_args()
    
    if args.parallel and args.max_workers < 1:
        print("Error: --max-workers must be > 0")
        sys.exit(1)
    
    if args.debug:
        log_path = project_root / 'data_preprocessing/log/deepseek_analyze_debug.log'
    else:
        log_path = project_root / args.log if args.log else None
    setup_logging(log_path)
    
    logging.info("=" * 80)
    if args.debug:
        logging.info("DeepSeek API Analyzer - Debug Mode")
    else:
        logging.info("DeepSeek API Analyzer")
    logging.info("=" * 80)
    
    try:
        if args.debug:
            logging.info("Debug mode: processing 3 samples only\n")
            sample_limit = 3
            train_limit = None
        elif args.train:
            sample_limit = None
            train_limit = args.limit
        elif args.stage_three:
            sample_limit = None
            train_limit = None
        else:
            sample_limit = args.limit
            train_limit = None
        
        input_path = project_root / args.input if not Path(args.input).is_absolute() else Path(args.input)
        silent_load = args.train
        samples = DataLoader.load_samples(input_path, limit=sample_limit, silent=silent_load)
        
        if not samples:
            logging.error("No samples found")
            return
        
        prompt_template = PromptTemplate()
        
        if args.model is None:
            if args.stage_four:
                model = "deepseek-reasoner"
            else:
                model = DEFAULT_MODEL
        else:
            model = args.model
        
        model_suffix = get_model_suffix('deepseek', model)
        
        logging.info(f"Using API: deepseek")
        logging.info(f"Using model: {model}")
        
        if DeepSeekClient is None:
            logging.error("DeepSeekClient not imported")
            return
        client = DeepSeekClient(
            api_key=args.api_key,
            model=model,
            temperature=args.temperature,
            max_output_tokens=args.max_tokens,
            seed=args.seed
        )
        
        if hasattr(client, 'is_available'):
            logging.info("Testing API connection...")
            if not client.is_available():
                logging.error("API connection failed")
                return
            logging.info("API connection OK")
        
        output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
        load_existing = not args.no_incremental
        result_saver = ResultSaver(output_path, load_existing=load_existing)
        
        error_log_path = output_path.parent / f"{output_path.stem}_errors.json"
        error_logger = ErrorLogger(error_log_path)
        
        stage_one_analyzer = StageOneAnalyzer(
            client=client,
            prompt_template=prompt_template,
            result_saver=result_saver,
            error_logger=error_logger,
            test_mode=args.debug
        )
        
        stage_two_analyzer = StageTwoAnalyzer(
            client=client,
            prompt_template=prompt_template,
            result_saver=result_saver,
            error_logger=error_logger,
            test_mode=args.debug
        )
        
        final_batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
        if args.batch_size is None:
             logging.info(f"Using default batch size: {final_batch_size}")
        else:
            logging.info(f"Using specified batch size: {final_batch_size}")

        stage_three_analyzer = StageThreeAnalyzer(
            client=client,
            prompt_template=prompt_template,
            result_saver=result_saver,
            error_logger=error_logger,
            test_mode=args.debug,
            batch_size=final_batch_size
        )
        
        if args.stage_three:
            logging.info(f"\n{'='*80}")
            logging.info("Stage 3: Batch submodule classification")
            logging.info(f"{'='*80}\n")
            
            stage2_path = project_root / 'data/04_analyzed/stage2/all_stage2.json'
            if not stage2_path.exists():
                 stage2_path = project_root / f'data/04_analyzed/stage2/all_stage2{model_suffix}.json'
            
            if not stage2_path.exists():
                logging.error(f"Stage 2 result not found: {stage2_path}")
                return
            
            with open(stage2_path, 'r', encoding='utf-8') as f:
                stage2_data = json.load(f)
                stage2_results = stage2_data.get('results', [])
            
            stage3_output_path = project_root / f'data/04_analyzed/stage3/all_stage3{model_suffix}.json'
            stage3_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            target_stage = args.stage
            if target_stage and target_stage.isdigit():
                try:
                    idx = int(target_stage) - 1
                    if 0 <= idx < len(VALID_LIFECYCLE_STAGES):
                        target_stage = VALID_LIFECYCLE_STAGES[idx]
                        logging.info(f"Mapping stage '{args.stage}' to: {target_stage}")
                    else:
                        logging.error(f"Invalid stage number: {args.stage}. Valid: 1-{len(VALID_LIFECYCLE_STAGES)}")
                        return
                except ValueError:
                    pass
            stage3_result = stage_three_analyzer.run(
                samples=samples,
                stage_two_results=stage2_results,
                output_path=str(stage3_output_path),
                limit=args.limit,
                target_stage=target_stage
            )
            
            logging.info(f"Stage 3 done, saved to: {stage3_output_path}")
            return

        if args.stage_four:
            logging.info(f"\n{'='*80}")
            logging.info("Stage 4: Submodule merging and optimization")
            logging.info(f"{'='*80}\n")
            
            target_stage = args.stage
            if target_stage and target_stage.isdigit():
                try:
                    idx = int(target_stage) - 1
                    if 0 <= idx < len(VALID_LIFECYCLE_STAGES):
                        target_stage = VALID_LIFECYCLE_STAGES[idx]
                        logging.info(f"Mapping stage '{args.stage}' to: {target_stage}")
                    else:
                        logging.error(f"Invalid stage number: {args.stage}. Valid: 1-{len(VALID_LIFECYCLE_STAGES)}")
                        return
                except ValueError:
                    pass
            
            # StageFourAnalyzer expects the base analyzed directory (e.g. data/04_analyzed)
            # data_preprocessing/deepseek_analyze -> data_preprocessing -> quantum -> data/04_analyzed
            # output_path is .../data/04_analyzed/stage1/all_stage1.json
            # output_path.parent -> stage1
            # output_path.parent.parent -> 04_analyzed (CORRECT)
            analyzer = StageFourAnalyzer(client, str(output_path.parent.parent))

            if target_stage:
                stages_to_run = [target_stage]
                analyzer.run(target_stage)
            else:
                stages_to_run = [s for s in VALID_LIFECYCLE_STAGES if s != "None"]
                logging.info(f"Running all {len(stages_to_run)} stages in parallel")
                
                with ThreadPoolExecutor(max_workers=min(len(stages_to_run), 7)) as executor:
                    future_to_stage = {executor.submit(analyzer.run, stage): stage for stage in stages_to_run}
                    for future in as_completed(future_to_stage):
                        stage = future_to_stage[future]
                        try:
                            result = future.result()
                            if result.get("success"):
                                logging.info(f"Stage [{stage}] completed")
                            else:
                                logging.error(f"Stage [{stage}] failed: {result.get('error')}")
                        except Exception as exc:
                            logging.error(f"Stage [{stage}] exception: {exc}")
                
            return
        
        if args.train:
            train_json_path = project_root / args.train_json if not Path(args.train_json).is_absolute() else Path(args.train_json)
            
            if not train_json_path.exists():
                logging.error(f"Training file not found: {train_json_path}")
                return
            
            with open(train_json_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            train_sample_ids = [s['sample_id'] for s in train_data['results']]
            total_train_samples = len(train_sample_ids)
            
            if train_limit is not None and train_limit > 0:
                train_sample_ids = train_sample_ids[:train_limit]
            
            train_samples = [s for s in samples if s.id in train_sample_ids]
            
            model_suffix = get_model_suffix('deepseek', model)
            output_path = project_root / f'data/bugfix_by_project/train_stage2{model_suffix}.json'
            
            load_existing = not args.no_incremental
            result_saver = ResultSaver(output_path, load_existing=load_existing)
            error_log_path = output_path.parent / f"{output_path.stem}_errors.json"
            error_logger = ErrorLogger(error_log_path)
            
            processed_count = len(result_saver.get_processed_sample_ids())
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Training set mode")
            logging.info(f"Submodule pool: {result_saver._count_unique_submodules()} types")
            logging.info(f"Training set: {len(train_sample_ids)}/{total_train_samples} samples")
            logging.info(f"Processed: {processed_count}")
            logging.info(f"Pending: {len(train_sample_ids) - processed_count}")
            logging.info(f"{'='*80}\n")
            
            stage_two_analyzer_train = StageTwoAnalyzer(
                client=client,
                prompt_template=prompt_template,
                result_saver=result_saver,
                error_logger=error_logger,
                test_mode=args.debug
            )
            
            stage_one_results = [{'sample_id': sid, 'change_category': 'Bug Fix', 'success': True} for sid in train_sample_ids]
            results = stage_two_analyzer_train.analyze_batch(
                samples=samples,
                stage_one_results=stage_one_results,
                save_incremental=True
            )
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Training set analysis completed")
            logging.info(f"Successful: {len([r for r in results if r.success])}/{len(results)} samples")
            logging.info(f"{'='*80}")
            return
        
        run_evaluation_mode(
            mode_name="all",
            csv_path=None,
            samples=samples,
            stage_one_analyzer=stage_one_analyzer,
            stage_two_analyzer=stage_two_analyzer,
            project_root=project_root,
            no_incremental=args.no_incremental,
            csv_path_2=None,
            parallel=args.parallel,
            max_workers=args.max_workers,
            batch_save=args.batch_save,
            stage_one_only=args.stage_one,
            stage_two_only=args.stage_two,
            model=model
        )
        
    except KeyboardInterrupt:
        logging.warning("User interrupted")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
