import time
import json
import threading
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from collections import deque
from datetime import datetime

@dataclass
class MonitoringMetrics:
    timestamp: float
    throughput: float = 0.0
    queue_size: int = 0
    running_requests: int = 0
    avg_request_latency: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0

class AsyncLLMMonitor:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1/",
        api_key: Optional[str] = "DummyAPIKey",
        update_interval: float = 1,
        max_history: int = 3600,
        verbose: bool = False
    ):
        """
        Initialize the async monitor for LLM performance metrics using vLLM endpoints.
        
        Args:
            base_url: Base URL for the LLM API (default is http://localhost:8000/v1/)
            api_key: API key for authentication (default is DummyAPIKey)
            update_interval: Interval in seconds for collecting metrics (default is 1)
            max_history: Maximum number of metrics to keep in history (default is 3600)
            verbose: Whether to enable verbose logging (default is False)
        """
        self.vllm_api_url = base_url.rstrip('/')
        self.update_interval = update_interval
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Set up logging
        self.logger = logging.getLogger("LLMMonitor")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    def start_monitoring(self):
        """Start the async monitoring thread."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._run_monitoring_loop,
                daemon=True
            )
            self._monitor_thread.start()
            self.logger.info(f"Started monitoring thread with {self.update_interval}s interval")

    def stop_monitoring(self):
        """Stop the async monitoring thread."""
        self._monitoring_active = False
        if self._monitor_thread is not None:
            self._monitor_thread.join()
            self.logger.info("Stopped monitoring thread")

    def get_metrics_history(self) -> List[MonitoringMetrics]:
        """Get the collected metrics history."""
        with self._lock:
            return list(self.metrics_history)
        
    def get_latest_metrics(self) -> Optional[MonitoringMetrics]:
        """Get the latest collected metrics."""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None
        
    def save_metrics(self, file_path: str = "metrics_history.json"):
        """Save the metrics history to a JSON file."""
        with self._lock:
            with open(file_path, 'w') as f:
                json.dump([m.__dict__ for m in self.metrics_history], f, indent=4)
            self.logger.info(f"Metrics history saved to {file_path}")

    def _run_monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self._monitoring_active:
            try:
                metrics = self._get_vllm_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
            time.sleep(self.update_interval)

    def _get_vllm_metrics(self) -> Optional[Dict]:
        """Fetch metrics from vLLM's metrics endpoint."""
        try:
            response = requests.get(f"{self.vllm_api_url}/metrics", )
            if response.status_code == 200:
                return self._parse_vllm_metrics(response.text)
        except Exception as e:
            self.logger.warning(f"Failed to fetch vLLM metrics: {e}")
        return None

    def _parse_vllm_metrics(self, metrics_text: str) -> Dict:
        """Parse vLLM's Prometheus format metrics."""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            try:
                name, value = line.split(' ', 1)
                metrics[name] = float(value)
            except ValueError:
                continue
        
        return MonitoringMetrics(
            timestamp=datetime.now().timestamp(),
            throughput=metrics.get('vllm:generation_throughput_token_per_sec', 0),
            queue_size=metrics.get('vllm:num_requests_waiting', 0),
            running_requests=metrics.get('vllm:num_requests_running', 0),
            avg_request_latency=metrics.get('vllm:avg_request_latency_seconds', 0),
            gpu_utilization=metrics.get('vllm:gpu_utilization_percentage', 0),
            gpu_memory_used=metrics.get('vllm:gpu_memory_usage_percentage', 0),
            total_tokens=metrics.get('vllm:total_num_tokens_generated', 0)
        )

    def __enter__(self):
        """Context manager entry point."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop_monitoring()

        if self.metrics_history:
            self.save_metrics()