#!/usr/bin/env python3
"""Data models and shared constants for analysis stages."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

REQUEST_DELAY = 0.5
INTRA_MERGE_INTERVAL = 10
INTER_MERGE_INTERVAL = 30

@dataclass
class BugSample:
    """Method-level bug-fix sample."""
    id: str
    repository: str
    commit_sha: str
    commit_url: str
    commit_message: str
    file_path: str
    function_name: str
    buggy_code: str
    buggy_file: str = ""
    diff: str = ""
    framework: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BugSample':
        """Build a sample from serialized data."""
        file_path = data.get('parent_file_path', '') or data.get('buggy_file_path', '') or data.get('file_path', '')
        sample_id = data.get('id', '')
        if isinstance(sample_id, int):
            sample_id = str(sample_id)
        
        return cls(
            id=sample_id,
            repository=data.get('repository', ''),
            commit_sha=data.get('commit_sha', ''),
            commit_url=data.get('commit_url', ''),
            commit_message=data.get('commit_message', ''),
            file_path=file_path,
            function_name=data.get('function_name', ''),
            buggy_code=data.get('buggy_code', ''),
            buggy_file=data.get('buggy_file', ''),
            diff=data.get('diff', ''),
            framework=data.get('framework', '')
        )


@dataclass
class AnalysisResult:
    """Normalized result for one analysis sample."""
    sample_id: str
    success: bool
    change_category: Optional[str] = None
    lifecycle_stage: Optional[str] = None
    bug_type: Optional[str] = None
    quantum_specific: Optional[bool] = None
    bug_reason: Optional[str] = None
    reason: Optional[str] = None
    quantum_reason: Optional[str] = None
    lifecycle_reason: Optional[str] = None
    commit_message: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    api_response_time: Optional[float] = None
    token_usage: Optional[Dict] = None
    parent_file_path: Optional[str] = None
    repository: Optional[str] = None
    commit_sha: Optional[str] = None
    framework: Optional[str] = None
    _sort_index: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Serialize using the field order expected by downstream stages."""
        simplified_token_usage = None
        if self.token_usage:
            simplified_token_usage = {
                "prompt_cache_hit_tokens": self.token_usage.get('prompt_cache_hit_tokens', 0) or 0,
                "prompt_cache_miss_tokens": self.token_usage.get('prompt_cache_miss_tokens', 0) or 0,
            }
        
        return {
            "id": self.sample_id,
            "framework": self.framework,
            "parent_file_path": self.parent_file_path,
            "commit_message": self.commit_message,
            "change_category": self.change_category,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "success": self.success,
            "lifecycle_stage": self.lifecycle_stage,
            "submodule": self.bug_type,
            "quantum_specific": self.quantum_specific,
            "quantum_reason": self.quantum_reason,
            "lifecycle_reason": self.lifecycle_reason,
            "error": self.error,
            "api_response_time": self.api_response_time,
            "token_usage": simplified_token_usage,
            "bug_reason": self.bug_reason or self.reason,
        }

    @staticmethod
    def from_dict(data: Dict) -> 'AnalysisResult':
        """Rebuild an ``AnalysisResult`` from serialized data."""
        quantum_specific = data.get('quantum_specific')
        if quantum_specific is None:
            quantum_specific = data.get('is_quantum_related')
        if quantum_specific is None:
            classical_specific = data.get('classical_specific')
            if classical_specific is not None:
                quantum_specific = not classical_specific

        bug_reason = data.get('bug_reason')
        reason = data.get('reason')
        if bug_reason is None and reason is not None:
            bug_reason = reason
        
        return AnalysisResult(
            sample_id=data.get('sample_id', ''),
            success=data.get('success', False),
            change_category=data.get('change_category'),
            lifecycle_stage=data.get('lifecycle_stage'),
            bug_type=data.get('submodule') or data.get('bug_type'),
            quantum_specific=quantum_specific,
            bug_reason=bug_reason,
            reason=reason,
            quantum_reason=data.get('quantum_reason'),
            lifecycle_reason=data.get('lifecycle_reason'),
            commit_message=data.get('commit_message'),
            raw_response=data.get('raw_response'),
            error=data.get('error'),
            api_response_time=data.get('api_response_time'),
            token_usage=data.get('token_usage'),
            parent_file_path=data.get('parent_file_path'),
            repository=data.get('repository'),
            commit_sha=data.get('commit_sha'),
            framework=data.get('framework'),
            _sort_index=data.get('_sort_index')
        )
