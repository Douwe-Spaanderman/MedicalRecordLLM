from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseAdapter(ABC):
    """Abstract base class for all adapters"""
    
    @abstractmethod
    def prepare_inputs(self) -> List[str]:
        """Prepare input texts for processing"""
        pass
    
    @abstractmethod
    def format_outputs(self, results: List[Dict[str, Any]]) -> Any:
        """Format processed results back to original format"""
        pass