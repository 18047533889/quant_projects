"""
Example: Production service with full observability.

Demonstrates how to use logging, metrics, health checks, and graceful shutdown.
"""

import time
from pathlib import Path

from config.production_config import ProductionConfig
from config.logging.structured_logger import configure_from_config, get_logger
from config.monitoring import configure_metrics, configure_health_checks
from config.monitoring.shutdown import configure_graceful_shutdown


class ProductionService:
    """
    Example production service with enterprise features.
    """

    def __init__(self, config: ProductionConfig):
        """
        Initialize service.

        Args:
            config: Production configuration
        """
        self.config = config
        self.running = True

        # Setup logging
        self.logger = configure_from_config(config)
        self.logger.info(
            "Service initializing",
            context={
                "environment": config.environment,
                "version": config.version,
            }
        )

        # Setup metrics
        self.metrics = configure_metrics(config)

        # Setup health checks
        self.health = configure_health_checks(config)

        # Setup graceful shutdown
        self.shutdown_handler = configure_graceful_shutdown(
            config,
            shutdown_hooks=[self.cleanup]
        )

        self.logger.info("Service initialized successfully")

    def cleanup(self):
        """Cleanup resources on shutdown."""
        self.logger.info("Starting cleanup")
        self.running = False
        # Close connections, flush caches, etc.
        self.logger.info("Cleanup completed")

    def start(self):
        """Start the service."""
        self.logger.info("Service starting")

        try:
            # Simulate startup tasks
            self._initialize_components()

            # Mark startup complete
            self.health.mark_startup_complete()
            self.logger.info("Service ready to accept traffic")

            # Main service loop
            self._run_main_loop()

        except Exception as e:
            self.logger.error(
                "Service failed to start",
                context={"error": str(e)},
                exc_info=True
            )
            raise

    def _initialize_components(self):
        """Initialize service components."""
        components = ["database", "cache", "data_loader"]

        for component in components:
            self.logger.info(f"Initializing {component}")
            time.sleep(0.5)  # Simulate initialization
            self.logger.info(f"{component} initialized")

    def _run_main_loop(self):
        """Main service loop."""
        iteration = 0

        while self.running and not self.shutdown_handler.is_shutdown_requested():
            iteration += 1

            # Update resource metrics every 10 iterations
            if iteration % 10 == 0:
                self.metrics.update_resource_metrics()

            # Simulate work
            self._process_work()

            time.sleep(1)

    def _process_work(self):
        """Process work with metrics and logging."""
        # Track operation
        with self.metrics.track_operation("process_batch"):
            try:
                # Simulate factor computation
                start = time.time()
                self._compute_factors()
                duration = time.time() - start

                # Record metrics
                self.metrics.record_factor_evaluation("momentum", duration)
                self.metrics.record_data_points("timeseries", 1000)

                # Log success
                self.logger.info(
                    "Batch processed successfully",
                    context={
                        "duration_ms": round(duration * 1000, 2),
                        "factors": 10,
                        "data_points": 1000,
                    }
                )

            except Exception as e:
                # Record error
                self.metrics.record_error("process_batch", type(e).__name__)

                # Log error
                self.logger.error(
                    "Batch processing failed",
                    context={"error_type": type(e).__name__},
                    exc_info=True
                )

    def _compute_factors(self):
        """Simulate factor computation."""
        time.sleep(0.1)  # Simulate work


def main():
    """Main entry point."""
    # Load configuration from environment
    config = ProductionConfig.from_env()

    # Validate configuration
    config.validate_and_raise()

    # Create and start service
    service = ProductionService(config)
    service.start()


if __name__ == "__main__":
    main()
