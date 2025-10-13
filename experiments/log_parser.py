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
from adjustText import adjust_text
from matplotlib.patches import Circle, Patch

models = {
    'deepseek-ai/DeepSeek-R1-0528': {'category': 'large', 'size': 685, 'name': 'DeepSeek R1 0528', 'performance': 0.763104, 'GPU': '16x H100'},
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct': {'category': 'large', 'size': 402, 'name': 'Llama-4 Maverick 17B 128E Instruct', 'performance': 0.810003, 'GPU': '16x H100'},
    'Qwen/Qwen3-235B-A22B': {'category': 'large', 'size': 235, 'name': 'Qwen3 235B A22B', 'performance': 0.772821, 'GPU': '16x H100'},
    'm42-health/Llama3-Med42-70B': {'category': 'specialized', 'size': 70, 'name': 'Llama-3 Med42 70B', 'performance': 0.750560, 'GPU': '4x H100'}, 
    'nvidia/Llama-3_3-Nemotron-Super-49B-v1': {'category': 'medium', 'size': 49, 'name': 'Llama-3.3 Nemotron Super 49B v1', 'performance': 0.776537, 'GPU': '4x H100'},
    'meta-llama/Llama-4-Scout-17B-16E': {'category': 'medium', 'size': 109, 'name': 'Llama-4 Scout 17B 16E', 'performance': 0.760603, 'GPU': '4x H100'}, 
    'Qwen/Qwen2.5-72B-Instruct': {'category': 'medium', 'size': 72, 'name': 'Qwen2.5 72B Instruct', 'performance': 0.786064, 'GPU': '4x H100'}, 
    'aaditya/Llama3-OpenBioLLM-70B': {'category': 'specialized', 'size': 70, 'name': 'Llama-3 OpenBioLLM 70B', 'performance': 0.711920, 'GPU': '4x H100'}, 
    'google/medgemma-27b-it': {'category': 'specialized', 'size': 27, 'name': 'MedGemma 27B IT', 'performance': 0.751876, 'GPU': '4x H100'}, 
    'google/gemma-3-27b-it': {'category': 'small', 'size': 27, 'name': 'Gemma 3 27B IT', 'performance': 0.740504, 'GPU': '2x H100'},
    'mistralai/Mistral-Small-3.1-24B-Instruct-2503': {'category': 'small', 'size': 24, 'name': 'Mistral Small 3.1 24B Instruct 2503', 'performance': 0.775343, 'GPU': '2x H100'}, 
    'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': {'category': 'small', 'size': 8, 'name': 'DeepSeek R1 0528 Qwen3 8B', 'performance': 0.748488, 'GPU': '2x H100'},
    'google/gemma-3-4b-it': {'category': 'tiny', 'size': 4, 'name': 'Gemma 3 4B IT', 'performance': 0.678262, 'GPU': '2x H100'},
    'Qwen/Qwen3-1.7B': {'category': 'tiny', 'size': 1.7, 'name': 'Qwen3 1.7B', 'performance': 0.668524, 'GPU': '2x H100'},
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': {'category': 'tiny', 'size': 1.5, 'name': 'DeepSeek R1 Distill Qwen 1.5B', 'performance': 0.419071, 'GPU': '2x H100'},
}


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

        output_path = Path(output_path)
        if output_path.exists():
            try:
                existing_data = json.loads(output_path.read_text())
            except json.JSONDecodeError:
                existing_data = {}
        else:
            existing_data = {}

        existing_data.update(serializable_data)

        output_path.write_text(json.dumps(existing_data, indent=2))

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
            num_gpus = self.parse_num_gpus(models.get(model, {}).get('GPU', '1x H100'))
            for strategy, records in strategies.items():
                if not records:
                    continue
                
                prompt_tput = [r.prompt_throughput for r in records]
                gen_tput = [r.generation_throughput for r in records]
                kv_cache = [r.kv_cache_usage for r in records]
                cache_hit = [r.prefix_cache_hit_rate for r in records]

                # Per timestep: per-GPU throughput and cache-adjusted throughput
                per_gpu_tputs = [t / num_gpus for t in gen_tput]
                corrected_tputs = [
                    (t / num_gpus) / max(k / 100.0, 1e-3)  # clip denominator at 0.1%
                    for t, k in zip(gen_tput, kv_cache)
                ]

                # Total time in seconds
                times = [r.time for r in records]
                time_required = (max(pd.to_datetime(times)) - min(pd.to_datetime(times))).total_seconds()
                per_gpu_time_required = time_required * num_gpus
                
                stats[model][strategy] = {
                    'count': len(records),
                    'time_required_sec': time_required,
                    'time_required_per_gpu_sec': per_gpu_time_required,
                    'prompt_throughput_mean': statistics.mean(prompt_tput),
                    'prompt_throughput_median': statistics.median(prompt_tput),
                    'prompt_throughput_stdev': statistics.stdev(prompt_tput) if len(prompt_tput) > 1 else 0,
                    'generation_throughput_mean': statistics.mean(gen_tput),
                    'generation_throughput_median': statistics.median(gen_tput),
                    'generation_throughput_stdev': statistics.stdev(gen_tput) if len(gen_tput) > 1 else 0,
                    'avg_kv_cache_usage': statistics.mean(kv_cache),
                    'avg_cache_hit_rate': statistics.mean(cache_hit),
                    'total_requests': sum(r.running + r.waiting for r in records),
                    'generation_throughput_per_gpu_mean': statistics.mean(per_gpu_tputs),
                    'generation_throughput_per_gpu_median': statistics.median(per_gpu_tputs),
                    'generation_throughput_per_gpu_stdev': statistics.stdev(per_gpu_tputs) if len(per_gpu_tputs) > 1 else 0,
                    'cache_corrected_tput_per_gpu_mean': statistics.mean(corrected_tputs),
                    'cache_corrected_tput_per_gpu_median': statistics.median(corrected_tputs),
                    'cache_corrected_tput_per_gpu_stdev': statistics.stdev(corrected_tputs) if len(corrected_tputs) > 1 else 0,
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
        
        # Add the model name for better readability
        df['model_name'] = df['model'].apply(lambda m: models.get(m, {}).get('name', m))
    
        plt.figure(figsize=(15, 6))
    
        # Prompt Throughput
        plt.subplot(1, 2, 1)
        sns.barplot(data=df, x='strategy', y='prompt_throughput_mean', hue='model_name')
        plt.title('Average Prompt Throughput by Strategy')
        plt.ylabel('Tokens/s')
        plt.xticks(rotation=45)
    
        # Generation Throughput
        plt.subplot(1, 2, 2)
        sns.barplot(data=df, x='strategy', y='generation_throughput_mean', hue='model_name')
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

    def parse_num_gpus(self, gpu_str: str) -> int:
        """Extract number of GPUs from a string like '16x H100'. Fall back to 1 if not found."""
        if not isinstance(gpu_str, str):
            return 1
        match = re.match(r"\s*(\d+)\s*x", gpu_str, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 1

    def plot_model_size_vs_performance(self, output_file: str = None) -> None:
        """
        Two-panel plot (FewShot only):
        - left:  raw throughput per GPU (tokens/s)
        - right: cache-adjusted potential throughput per GPU (tokens/s)
        Bubble size = Number of parameters (in billions).
        """
        df = self.compare_strategies()
        if df.empty:
            print("No data to plot")
            return

        # Filter for FewShot only
        df = df[df['strategy'] == 'FewShot']
        if df.empty:
            print("No FewShot data to plot")
            return

        # Add model metadata
        df = df.copy()
        df['model_size'] = df['model'].apply(lambda m: models.get(m, {}).get('size', 0))  # in billions
        df['Model category'] = df['model'].apply(lambda m: models.get(m, {}).get('category', 'unknown')) + " (" + df['model'].apply(lambda m: models.get(m, {}).get('GPU', 'unknown')) + ")"
        df['model_name'] = df['model'].apply(lambda m: models.get(m, {}).get('name', m))
        df['performance'] = df['model'].apply(lambda m: models.get(m, {}).get('performance', None))

        # Use the precomputed per-GPU metrics (from _calculate_summary_stats)
        if 'generation_throughput_per_gpu_mean' not in df.columns:
            print("Missing per-GPU metrics in DataFrame — did you run the updated _calculate_summary_stats?")
            return

        df['throughput_per_gpu'] = df['generation_throughput_per_gpu_mean']
        df['potential_throughput_per_gpu'] = df['cache_corrected_tput_per_gpu_mean']

        # Column used for bubble size and its displayed name
        df['Number of parameters'] = df['model_size']  # in billions

        # Bubble size mapping: seaborn will map the numbers to marker sizes between sizes=(min, max)
        min_marker, max_marker = 50, 1000

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)

        # Common scatter kwargs
        scatter_kwargs = dict(
            alpha=0.75,
            edgecolor='black',
            linewidth=0.5,
            sizes=(min_marker, max_marker),
            palette='tab10',
            legend='brief',
        )

        # Left: raw throughput per GPU
        sns.scatterplot(
            data=df,
            x='throughput_per_gpu',
            y='performance',
            hue='Model category',
            size='Number of parameters',
            ax=ax1,
            **scatter_kwargs
        )
        ax1.set_xscale('log')
        ax1.set_xlabel('Raw Throughput per GPU (tokens/s)')
        ax1.set_title('Measured Throughput per GPU')

        # Right: cache-adjusted potential throughput per GPU
        sns.scatterplot(
            data=df,
            x='potential_throughput_per_gpu',
            y='performance',
            hue='Model category',
            size='Number of parameters',
            ax=ax2,
            
            **scatter_kwargs
        )
        ax2.set_xscale('log')
        ax2.set_xlabel('Cache-adjusted Potential Throughput per GPU (tokens/s)')
        ax2.set_title('Potential Throughput per GPU (cache-adjusted)')

        # Shared y label
        ax1.set_ylabel('Performance (Mean Score)')
        ax2.set_ylabel('') 

        # Create text labels for each subplot and adjust them separately
        texts_left, texts_right = [], []
        for _, row in df.iterrows():
            label = row['model_name']
            texts_left.append(ax1.text(row['throughput_per_gpu'], row['performance'], label, fontsize=8))
            texts_right.append(ax2.text(row['potential_throughput_per_gpu'], row['performance'], label, fontsize=8))

        # Adjust texts with arrows to reduce overlaps
        plt.sca(ax1)
        adjust_text(texts_left, arrowprops=dict(arrowstyle='-', color='gray', lw=0.4))
        plt.sca(ax2)
        adjust_text(texts_right, arrowprops=dict(arrowstyle='-', color='gray', lw=0.4))

        # Tidy up legends
        ax1.legend_.remove()
        handles, labels = ax2.get_legend_handles_labels()
        model_category_handles = []
        model_category_labels = []

        # The first entry is the title "Model category", then categories, then "Number of parameters", then sizes
        found_size_title = False
        for i, (handle, label) in enumerate(zip(handles, labels)):
            if label == 'Number of parameters':
                found_size_title = True
                continue
            if found_size_title:
                continue  # skip size handles
            else:
                if label != 'Model category':  # Skip the title itself
                    model_category_handles.append(handle)
                    model_category_labels.append(label)

        # Order legend entries by category_order
        category_order = ['large', 'medium', 'small', 'specialized', 'tiny']
        ordered_handles_labels = sorted(
            zip(model_category_handles, model_category_labels),
            key=lambda x: category_order.index(x[1].split(" ")[0]) if x[1].split(" ")[0] in category_order else len(category_order)
        )
        model_category_handles, model_category_labels = zip(*ordered_handles_labels)

        # Remove the original legend
        ax2.legend_.remove()

        # Create two separate legends
        legend1 = ax2.legend(model_category_handles, model_category_labels,
                            bbox_to_anchor=(1.02, 0.7), loc='center left',
                            frameon=False)

        size_breaks = [1, 50, 100, 500]  # billions
        size_labels = [f"{s}B" for s in size_breaks]

        # Map through same scaling Seaborn uses (linear between min/max)
        def scale_size(val, vmin, vmax, smin, smax):
            return smin + (smax - smin) * (val - vmin) / (vmax - vmin)

        size_handles = [
            plt.scatter([], [], s=scale_size(s, df['Number of parameters'].min(),
                                            df['Number of parameters'].max(),
                                            min_marker, max_marker),
                        color='gray', edgecolor='black', alpha=0.7)
            for s in size_breaks
        ]

        legend2 = ax2.legend(size_handles, size_labels,
                            bbox_to_anchor=(1.01, 0.3), loc='center left',
                            handletextpad=2.0,
                            labelspacing=1.4,
                            frameon=False)
        # Add both legends to the plot
        ax2.add_artist(legend1)
        ax2.text(1.04, 0.83, 'Model category', transform=ax2.transAxes,
                fontsize='medium', fontweight='bold', va='center', ha='left')

        ax2.text(1.04, 0.45, 'Parameters (B)', transform=ax2.transAxes,
                fontsize='medium', fontweight='bold', va='center', ha='left')

        plt.tight_layout(rect=[0, 0, 0.94, 1])  # leave space on right for legends

        if output_file:
            plt.savefig(output_file, dpi=300)
        else:
            plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse LLM experiment logs")
    parser.add_argument("--error-log", default=None, help="Path to error log file")
    parser.add_argument("--output-log", default=None, help="Path to output log file")
    parser.add_argument("--output", default="experiment_data.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    if not args.error_log or not args.output_log:
        print("Error log and output log paths are not provided so only analyzing existing data.")
        output = Path(args.output) / "log.json"
        data = json.loads(output.read_text())
        # Load as ExperimentData
        data = {
            model: {
                strategy: [ThroughputRecord(**record) for record in records]
                for strategy, records in strategies.items()
            }
            for model, strategies in data.items()
        }
        analyzer = PerformanceAnalyzer(data)
    else:     
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
    analyzer.plot_model_size_vs_performance(Path(args.output) / "model_size_vs_performance.png")

