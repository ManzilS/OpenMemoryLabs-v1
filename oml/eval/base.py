from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

class ModelInterface(ABC):
    """Abstract base class for models used in evaluation."""
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a text completion for the given prompt."""
        pass

@dataclass
class EvalResult:
    """Standardized result tracking for any evaluation task."""
    task_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

class EvalTask(Protocol):
    """Protocol that all evaluation tasks must follow."""
    name: str
    
    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        ...
