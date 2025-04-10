import pandas as pd
from typing import Optional
from typing import List, Dict, Any
from .base_adapter import BaseAdapter

class DataFrameAdapter(BaseAdapter):
    """Adapter for pandas DataFrame input"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        report_type_column: str = "reportType",
        text_column: str = "text",
        report_type_filter: Optional[str] = None
    ):
        self.df = df.copy()
        self.report_type_column = report_type_column
        self.text_column = text_column
        self.report_type_filter = report_type_filter
        
    def prepare_inputs(self) -> List[str]:
        """Extract and prepare reports for processing"""
        df = self.df
        
        if self.report_type_filter is not None and self.report_type_column in df.columns:
            df = df[df[self.report_type_column] == self.report_type_filter]
            df = df.reset_index(drop=True)
            self.df = df.copy()
            
        if self.text_column not in df.columns:
            raise ValueError(f"DataFrame missing required text column: {self.text_column}")
            
        return df[self.text_column].apply(
            lambda x: str(x) if isinstance(x, str) else ""
        ).tolist()
    
    def format_outputs(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Merge processing results back into DataFrame"""
        if len(results) != len(self.df):
            raise ValueError("Results length doesn't match DataFrame length")
            
        results_df = pd.DataFrame(results)
        return pd.concat([self.df, results_df], axis=1)