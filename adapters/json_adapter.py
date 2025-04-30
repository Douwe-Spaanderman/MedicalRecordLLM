from typing import List, Dict, Any, Union, Tuple
import json
from pathlib import Path
from .base_adapter import BaseAdapter

class JsonAdapter(BaseAdapter):
    """Adapter for JSON file input."""

    def __init__(
        self,
        input_path: Union[str, Path],
        text_key: str = "text",
        id_key: str = "id"
    ):
        self.input_path = Path(input_path)
        self.text_key = text_key
        self.id_key = id_key
        self.original_data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and validate JSON data."""
        with open(self.input_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON data should be an array of objects")

        return data

    def prepare_inputs(self) -> Tuple[List[str], List[Any]]:
        """Extract texts and IDs from JSON objects."""
        texts = []
        ids = []

        for item in self.original_data:
            if self.text_key not in item:
                raise ValueError(f"Missing text key '{self.text_key}' in item: {item}")
            if self.id_key not in item:
                raise ValueError(f"Missing ID key '{self.id_key}' in item: {item}")

            texts.append(str(item[self.text_key]) if isinstance(item[self.text_key], str) else "")
            ids.append(item[self.id_key])

        return texts, ids

    def format_outputs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge results back into original JSON structure."""
        if len(results) != len(self.original_data):
            raise ValueError("Results length doesn't match input length")

        return [
            {**original, **result}
            for original, result in zip(self.original_data, results)
        ]
