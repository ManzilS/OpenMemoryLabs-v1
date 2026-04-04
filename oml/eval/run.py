import json
import time
from datetime import datetime
from pathlib import Path
from typing import Type

from oml.eval.base import EvalTask, EvalResult
from oml.eval.models import get_model

# Registry pattern
_TASK_REGISTRY: dict[str, Type[EvalTask]] = {}

def register_task(name: str):
    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator

def get_task(name: str) -> Type[EvalTask]:
    if name not in _TASK_REGISTRY:
        raise ValueError(f"Task '{name}' not found. Available: {list(_TASK_REGISTRY.keys())}")
    return _TASK_REGISTRY[name]

def run_task(task_name: str, model_name: str = "dummy", config: dict = None) -> EvalResult:
    """
    Orchestrates the running of a single task.
    """
    if config is None:
        config = {}

    # 1. Load Model
    model = get_model(model_name)

    # 2. Load Task
    task_cls = get_task(task_name)
    task = task_cls()

    # 3. Run
    print(f"Running task: {task_name} with model: {model_name}...")
    start_time = time.time()
    result = task.run(model, config)
    duration = time.time() - start_time
    
    # 4. Save Artifacts
    _save_report(result, duration, model_name)
    
    return result

def _save_report(result: EvalResult, duration: float, model_name: str):
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON Report
    report_data = {
        "timestamp": timestamp,
        "task": result.task_name,
        "model": model_name,
        "score": result.score,
        "duration_seconds": duration,
        "details": result.details
    }
    
    json_path = reports_dir / f"run_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    # Markdown Summary
    md_path = reports_dir / f"summary_{timestamp}.md"
    md_content = f"""# Evaluation Report
**Task**: {result.task_name}
**Model**: {model_name}
**Date**: {timestamp}
**Duration**: {duration:.2f}s

## Score: {result.score:.2f}

## Details
```json
{json.dumps(result.details, indent=2)}
```
"""
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print(f"Reports saved to:\n  {json_path}\n  {md_path}")
