import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Pattern, Tuple
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ThroughputRecord:
    time: str
    prompt_throughput: float
    generation_throughput: float
    running: int
    waiting: int
    kv_cache_usage: float
    prefix_cache_hit_rate: float

@dataclass
class ExperimentData:
    model_name: str
    strategy: str
    records: List[ThroughputRecord]

class LogParser:
    def __init__(self, error_log_path: str, output_log_path: str):
        self.error_log_path = Path(error_log_path).expanduser()
        self.output_log_path = Path(output_log_path).expanduser()
        
        # Compile regex patterns
        self.model_pattern = re.compile(r"Model:\s*(.+)")
        self.prompt_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*generating prompt using\s+(\w+)"
        )
        self.throughput_pattern = re.compile(
            r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Avg prompt throughput: ([\d.]+) tokens/s, "
            r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
            r"GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
        )
        
        self.model_changes: List[Dict] = []
        self.strategy_changes: List[Dict] = []
        self.experiment_data: Dict[str, Dict[str, List[ThroughputRecord]]] = defaultdict(lambda: defaultdict(list))

    def parse_error_log(self) -> None:
        """Parse the error log to extract model name and strategy change timestamps"""
        with open(self.error_log_path) as f:
            for line in f:
                if "Model:" in line:
                    self._parse_model_line(line)
                elif "generating prompt using" in line:
                    self._parse_strategy_line(line)

    def _parse_model_line(self, line: str) -> None:
        """Extract model name and timestamp from line"""
        match = self.model_pattern.search(line)
        if match:
            timestamp_str = line.split(",")[0]
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            self.model_changes.append({
                "timestamp": timestamp,
                "model": match.group(1).strip()
            })

    def _parse_strategy_line(self, line: str) -> None:
        """Extract strategy change timestamp and name from line"""
        match = self.prompt_pattern.search(line)
        if match:
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            strategy = match.group(2)
            self.strategy_changes.append({
                "timestamp": timestamp,
                "strategy": strategy
            })

    def parse_output_log(self) -> None:
        """Parse the output log to extract throughput metrics"""
        if not self.strategy_changes:
            return
            
        # Get year from first model change or current year
        year = self.model_changes[0]["timestamp"].year if self.model_changes else datetime.now().year
        
        with open(self.output_log_path) as f:
            for line in f:
                if "throughput" in line:
                    self._parse_throughput_line(line, year)

    def _parse_throughput_line(self, line: str, year: int) -> None:
        """Parse a throughput line and add to experiment data"""
        match = self.throughput_pattern.search(line)
        if not match:
            return
            
        month_day, time_str = match.group(1).split(" ", 1)
        month, day = month_day.split("-")
        timestamp = datetime.strptime(f"{year}-{month}-{day} {time_str}", "%Y-%m-%d %H:%M:%S")
        
        current_model = self._get_current_model(timestamp)
        current_strategy = self._get_current_strategy(timestamp)
        if not current_model or not current_strategy:
            return
            
        record = ThroughputRecord(
            time=timestamp.isoformat(),
            prompt_throughput=float(match.group(2)),
            generation_throughput=float(match.group(3)),
            running=int(match.group(4)),
            waiting=int(match.group(5)),
            kv_cache_usage=float(match.group(6)),
            prefix_cache_hit_rate=float(match.group(7))
        )
        
        self.experiment_data[current_model][current_strategy].append(record)

    def _get_current_model(self, timestamp: datetime) -> Optional[str]:
        """Find the active model at the given timestamp"""
        for change in reversed(self.model_changes):
            if change["timestamp"] <= timestamp:
                return change["model"]
        return None

    def _get_current_strategy(self, timestamp: datetime) -> Optional[str]:
        """Find the active strategy at the given timestamp"""
        for change in reversed(self.strategy_changes):
            if change["timestamp"] <= timestamp:
                return change["strategy"]
        return None

    def save_results(self, output_path: str = "experiment_data.json") -> None:
        """Save parsed data to JSON file with model-specific metrics"""
        serializable_data = {
            model: {
                strategy: [vars(record) for record in records]
                for strategy, records in strategies.items()
            }
            for model, strategies in self.experiment_data.items()
        }

        with open(output_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Data written to {output_path}")

    def run(self, output_path: str = "experiment_data.json") -> None:
        """Run the complete parsing process"""
        self.parse_error_log()
        self.parse_output_log()
        self.save_results(output_path)

class PerformanceAnalyzer:
    def __init__(self, experiment_data: Dict[str, Dict[str, List[ThroughputRecord]]]):
        self.data = experiment_data
        self.summary_stats = self._calculate_summary_stats()
    
    def _calculate_summary_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate summary statistics for each model-strategy combination"""
        stats = {}
        
        for model, strategies in self.data.items():
            stats[model] = {}
            for strategy, records in strategies.items():
                if not records:
                    continue
                
                prompt_tput = [r.prompt_throughput for r in records]
                gen_tput = [r.generation_throughput for r in records]
                kv_cache = [r.kv_cache_usage for r in records]
                cache_hit = [r.prefix_cache_hit_rate for r in records]
                
                stats[model][strategy] = {
                    'count': len(records),
                    'prompt_throughput_mean': statistics.mean(prompt_tput),
                    'prompt_throughput_median': statistics.median(prompt_tput),
                    'prompt_throughput_stdev': statistics.stdev(prompt_tput) if len(prompt_tput) > 1 else 0,
                    'generation_throughput_mean': statistics.mean(gen_tput),
                    'generation_throughput_median': statistics.median(gen_tput),
                    'generation_throughput_stdev': statistics.stdev(gen_tput) if len(gen_tput) > 1 else 0,
                    'avg_kv_cache_usage': statistics.mean(kv_cache),
                    'avg_cache_hit_rate': statistics.mean(cache_hit),
                    'total_requests': sum(r.running + r.waiting for r in records)
                }
        
        return stats

    def compare_strategies(self, model: str = None) -> pd.DataFrame:
        """Compare performance across strategies for a specific model or all models"""
        if model:
            return pd.DataFrame(self.summary_stats.get(model, {}))
        else:
            all_data = []
            for model, strategies in self.summary_stats.items():
                for strategy, stats in strategies.items():
                    stats['model'] = model
                    stats['strategy'] = strategy
                    all_data.append(stats)
            return pd.DataFrame(all_data)

    def get_best_strategy(self, metric: str = 'generation_throughput_mean') -> Tuple[str, str]:
        """Identify the best performing strategy based on a chosen metric"""
        best_value = -1
        best_model = None
        best_strategy = None
    
        for model, strategies in self.summary_stats.items():
            for strategy, stats in strategies.items():
                if stats[metric] > best_value:
                    best_value = stats[metric]
                    best_model = model
                    best_strategy = strategy
                
        return best_model, best_strategy, best_value

    def plot_throughput_comparison(self, output_file: str = None) -> None:
        """Generate comparison plots of throughput metrics"""
        df = self.compare_strategies()
        if df.empty:
            print("No data to plot")
            return
    
        plt.figure(figsize=(15, 6))
    
        # Prompt Throughput
        plt.subplot(1, 2, 1)
        sns.barplot(data=df, x='strategy', y='prompt_throughput_mean', hue='model')
        plt.title('Average Prompt Throughput by Strategy')
        plt.ylabel('Tokens/s')
        plt.xticks(rotation=45)
    
        # Generation Throughput
        plt.subplot(1, 2, 2)
        sns.barplot(data=df, x='strategy', y='generation_throughput_mean', hue='model')
        plt.title('Average Generation Throughput by Strategy')
        plt.ylabel('Tokens/s')
        plt.xticks(rotation=45)
    
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

    def plot_timeseries(self, model: str, strategy: str, output_file: str = None) -> None:
        """Plot timeseries data for a specific model-strategy combination"""
        if model not in self.data or strategy not in self.data[model]:
            print(f"No data for {model} - {strategy}")
            return
    
        records = self.data[model][strategy]
        if not records:
            return
    
        df = pd.DataFrame([vars(r) for r in records])
        df['time'] = pd.to_datetime(df['time'])
    
        plt.figure(figsize=(12, 8))
    
        # Throughput metrics
        plt.subplot(2, 1, 1)
        plt.plot(df['time'], df['prompt_throughput'], label='Prompt Throughput')
        plt.plot(df['time'], df['generation_throughput'], label='Generation Throughput')
        plt.ylabel('Tokens/s')
        plt.title(f'Throughput Over Time - {model} - {strategy}')
        plt.legend()
    
        # System metrics
        plt.subplot(2, 1, 2)
        plt.plot(df['time'], df['kv_cache_usage'], label='KV Cache Usage')
        plt.plot(df['time'], df['prefix_cache_hit_rate'], label='Prefix Cache Hit Rate')
        plt.ylabel('Percentage')
        plt.legend()
    
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse LLM experiment logs")
    parser.add_argument("--error-log", required=True, help="Path to error log file")
    parser.add_argument("--output-log", required=True, help="Path to output log file")
    parser.add_argument("--output", default="experiment_data.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    parser = LogParser(args.error_log, args.output_log)
    parser.run(Path(args.output) / "log.json")
    
    analyzer = PerformanceAnalyzer(parser.experiment_data)

    summary = analyzer.compare_strategies()
    print("Summary Statistics:")
    print(summary.to_string())

    best_model, best_strategy, best_value = analyzer.get_best_strategy()
    print(f"\nBest performing strategy: {best_strategy} with model {best_model}")
    print(f"Average generation throughput: {best_value:.2f} tokens/s")

    analyzer.plot_throughput_comparison(Path(args.output) / "throughput_comparison.png")
