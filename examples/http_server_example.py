"""
Example: HTTP server with health checks and metrics endpoints.

Demonstrates how to expose health checks and Prometheus metrics via HTTP.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from pathlib import Path

from config.production_config import ProductionConfig
from config.logging.structured_logger import configure_from_config
from config.monitoring import configure_metrics, configure_health_checks
from config.monitoring.shutdown import configure_graceful_shutdown


class HealthMetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks and metrics."""

    def log_message(self, format, *args):
        """Override to use structured logging."""
        self.server.logger.info(
            format % args,
            context={
                "client": self.client_address[0],
                "method": self.command,
                "path": self.path,
            }
        )

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_startup()
        elif self.path == "/health/live":
            self._handle_liveness()
        elif self.path == "/health/ready":
            self._handle_readiness()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self.send_error(404, "Not Found")

    def _handle_startup(self):
        """Handle startup probe."""
        result = self.server.health.startup_probe()
        self._send_json_response(result)

    def _handle_liveness(self):
        """Handle liveness probe."""
        result = self.server.health.liveness_probe()
        status_code = 200 if result['status'] == 'healthy' else 503
        self._send_json_response(result, status_code)

    def _handle_readiness(self):
        """Handle readiness probe."""
        result = self.server.health.readiness_probe()
        status_code = 200 if result['status'] == 'healthy' else 503
        self._send_json_response(result, status_code)

    def _handle_metrics(self):
        """Handle metrics endpoint."""
        try:
            metrics_data = self.server.metrics.get_metrics()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write(metrics_data)
        except Exception as e:
            self.send_error(500, f"Failed to generate metrics: {e}")

    def _send_json_response(self, data: dict, status_code: int = 200):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())


class ProductionHTTPServer(HTTPServer):
    """HTTP server with health and metrics support."""

    def __init__(self, config: ProductionConfig):
        """
        Initialize server.

        Args:
            config: Production configuration
        """
        self.config = config

        # Setup logging
        self.logger = configure_from_config(config)

        # Setup metrics
        self.metrics = configure_metrics(config)

        # Setup health checks
        self.health = configure_health_checks(config)

        # Initialize HTTP server
        server_address = ('0.0.0.0', config.health_check.port)
        super().__init__(server_address, HealthMetricsHandler)

        # Setup graceful shutdown
        self.shutdown_handler = configure_graceful_shutdown(
            config,
            shutdown_hooks=[self.cleanup]
        )

        self.logger.info(
            f"HTTP server listening on port {config.health_check.port}",
            context={"port": config.health_check.port}
        )

    def cleanup(self):
        """Cleanup on shutdown."""
        self.logger.info("Shutting down HTTP server")
        self.shutdown()

    def serve_forever_with_health(self):
        """Serve forever and mark startup complete."""
        # Mark startup complete
        self.health.mark_startup_complete()

        # Start resource metrics collection
        import threading
        def update_metrics():
            while not self.shutdown_handler.is_shutdown_requested():
                self.metrics.update_resource_metrics()
                import time
                time.sleep(self.config.metrics.collect_interval_seconds)

        metrics_thread = threading.Thread(target=update_metrics, daemon=True)
        metrics_thread.start()

        # Serve requests
        self.logger.info("Server ready to accept requests")
        self.serve_forever()


def main():
    """Main entry point."""
    # Load configuration
    config = ProductionConfig.from_env()
    config.validate_and_raise()

    # Create and start server
    server = ProductionHTTPServer(config)

    try:
        server.serve_forever_with_health()
    except KeyboardInterrupt:
        server.logger.info("Received interrupt signal")
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()
