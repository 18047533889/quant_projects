#!/usr/bin/env python3
"""
Quick start script to run production HTTP server with health checks and metrics.

Usage:
    python scripts/run_production_server.py
    python scripts/run_production_server.py --config config/examples/production.yaml
    python scripts/run_production_server.py --env development
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.production_config import ProductionConfig
from examples.http_server_example import ProductionHTTPServer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run production HTTP server with health checks and metrics"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--env",
        choices=["production", "staging", "development"],
        default="production",
        help="Environment (defaults to environment variables)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override health check port"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="Override metrics port"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override log level"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        import yaml
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = ProductionConfig(config_dict)
    else:
        print("Loading configuration from environment variables")
        config = ProductionConfig.from_env()

    # Apply overrides
    if args.port:
        config.health_check.port = args.port
    if args.metrics_port:
        config.metrics.export_port = args.metrics_port
    if args.log_level:
        config.logging.level = args.log_level

    # Validate configuration
    try:
        config.validate_and_raise()
        print("Configuration validated successfully")
    except Exception as e:
        print(f"Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print configuration summary
    print("\n=== Configuration Summary ===")
    print(f"Environment: {config.environment}")
    print(f"Service: {config.service_name}")
    print(f"Version: {config.version}")
    print(f"Log Level: {config.logging.level}")
    print(f"Log Format: {config.logging.format}")
    print(f"Health Check Port: {config.health_check.port}")
    print(f"Metrics Port: {config.metrics.export_port}")
    print(f"Graceful Shutdown: {config.enable_graceful_shutdown}")
    print("=" * 30)

    # Create and start server
    print("\nStarting production HTTP server...")
    print(f"Health checks: http://localhost:{config.health_check.port}/health")
    print(f"Metrics: http://localhost:{config.metrics.export_port}/metrics")
    print("\nPress Ctrl+C to stop\n")

    server = ProductionHTTPServer(config)

    try:
        server.serve_forever_with_health()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        server.cleanup()
        print("Server stopped")


if __name__ == "__main__":
    main()
