import pandas as pd
from typing import Optional, List, Dict, Any
from .base_adapter import BaseAdapter

class DataFrameAdapter(BaseAdapter):
    """Adapter for pandas DataFrame input"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        report_type_column: str = "reportType",
        patient_id_column: str = "patientId",
        text_column: str = "text",
        report_type_filter: Optional[str] = None
    ):
        self.df = df.copy()
        self.report_type_column = report_type_column
        self.patient_id_column = patient_id_column
        self.text_column = text_column
        self.report_type_filter = report_type_filter

    def prepare_inputs(self) -> tuple[List[str], List[Any]]:
        """Extract and prepare reports and patient IDs for processing."""
        df = self.df

        if self.patient_id_column not in df.columns:
            raise ValueError(f"Missing required patient ID column: '{self.patient_id_column}'")

        if self.text_column not in df.columns:
            raise ValueError(f"Missing required text column: '{self.text_column}'")
        
        if self.report_type_filter and self.report_type_column in df.columns:
            df = df[df[self.report_type_column] == self.report_type_filter].reset_index(drop=True)

        df = df[~df[self.text_column].isna()].reset_index(drop=True)
        self.df = df

        texts = df[self.text_column].apply(lambda x: str(x) if isinstance(x, str) else "").tolist()
        patient_ids = df[self.patient_id_column].tolist()
        return texts, patient_ids

    def format_outputs(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Merge processed results back into the DataFrame."""
        if len(results) != len(self.df):
            raise ValueError("Mismatch between number of results and DataFrame rows.")

        results_df = pd.DataFrame(results)
        return pd.concat([self.df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
