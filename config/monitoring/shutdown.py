"""
Graceful shutdown handler for production deployments.

Handles SIGTERM, SIGINT signals and ensures clean shutdown.
"""

import signal
import sys
import threading
import logging
from typing import List, Callable, Optional
import time


class ShutdownHandler:
    """
    Handles graceful shutdown of the application.

    Registers signal handlers and manages shutdown sequence.
    """

    def __init__(self, timeout_seconds: int = 30, service_name: str = "quant-platform"):
        """
        Initialize shutdown handler.

        Args:
            timeout_seconds: Maximum time to wait for shutdown
            service_name: Service name for logging
        """
        self.timeout_seconds = timeout_seconds
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        self.shutdown_hooks: List[Callable] = []
        self.shutdown_event = threading.Event()
        self.is_shutting_down = False

    def register_hook(self, hook: Callable):
        """
        Register a shutdown hook.

        Hooks are called in reverse registration order (LIFO).

        Args:
            hook: Callable to execute during shutdown

        Example:
            def cleanup():
                db.close()
                cache.close()

            handler.register_hook(cleanup)
        """
        self.shutdown_hooks.append(hook)

    def signal_handler(self, signum: int, frame):
        """
        Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")

        if self.is_shutting_down:
            self.logger.warning("Already shutting down, ignoring signal")
            return

        self.is_shutting_down = True
        self.shutdown_event.set()

        # Run shutdown hooks
        self._execute_shutdown()

    def _execute_shutdown(self):
        """Execute all shutdown hooks."""
        self.logger.info(f"Starting shutdown sequence for {self.service_name}")
        start_time = time.time()

        # Execute hooks in reverse order (LIFO)
        for i, hook in enumerate(reversed(self.shutdown_hooks)):
            hook_name = getattr(hook, '__name__', f'hook_{i}')

            try:
                elapsed = time.time() - start_time
                remaining = self.timeout_seconds - elapsed

                if remaining <= 0:
                    self.logger.warning(
                        f"Shutdown timeout exceeded, skipping remaining hooks"
                    )
                    break

                self.logger.info(f"Running shutdown hook: {hook_name}")
                hook()
                self.logger.info(f"Shutdown hook completed: {hook_name}")

            except Exception as e:
                self.logger.error(
                    f"Error in shutdown hook '{hook_name}': {e}",
                    exc_info=True
                )

        elapsed = time.time() - start_time
        self.logger.info(
            f"Shutdown sequence completed in {elapsed:.2f}s"
        )

        # Exit
        sys.exit(0)

    def setup(self):
        """
        Setup signal handlers.

        Registers handlers for SIGTERM and SIGINT.
        """
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        self.logger.info(f"Graceful shutdown handler configured for {self.service_name}")

    def wait_for_shutdown(self):
        """
        Block until shutdown is signaled.

        Use this in the main thread to keep the application running.
        """
        self.shutdown_event.wait()

    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown has been requested.

        Returns:
            True if shutdown is in progress
        """
        return self.is_shutting_down


class GracefulShutdownMixin:
    """
    Mixin for adding graceful shutdown support to services.

    Example:
        class MyService(GracefulShutdownMixin):
            def __init__(self):
                self.setup_shutdown_handler()

            def cleanup(self):
                # Custom cleanup logic
                pass

            def run(self):
                self.register_shutdown_hook(self.cleanup)
                # Main service logic
    """

    def setup_shutdown_handler(self, timeout_seconds: int = 30):
        """
        Setup shutdown handler.

        Args:
            timeout_seconds: Shutdown timeout
        """
        self._shutdown_handler = ShutdownHandler(
            timeout_seconds=timeout_seconds,
            service_name=self.__class__.__name__
        )
        self._shutdown_handler.setup()

    def register_shutdown_hook(self, hook: Callable):
        """
        Register a shutdown hook.

        Args:
            hook: Callable to execute during shutdown
        """
        if not hasattr(self, '_shutdown_handler'):
            raise RuntimeError("Shutdown handler not initialized. Call setup_shutdown_handler() first.")
        self._shutdown_handler.register_hook(hook)

    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown is in progress.

        Returns:
            True if shutdown requested
        """
        if not hasattr(self, '_shutdown_handler'):
            return False
        return self._shutdown_handler.is_shutdown_requested()


def configure_graceful_shutdown(config, shutdown_hooks: Optional[List[Callable]] = None) -> ShutdownHandler:
    """
    Configure graceful shutdown from ProductionConfig.

    Args:
        config: ProductionConfig instance
        shutdown_hooks: Optional list of shutdown hooks to register

    Returns:
        Configured ShutdownHandler

    Example:
        def cleanup_db():
            db.close()

        def cleanup_cache():
            cache.close()

        handler = configure_graceful_shutdown(
            config,
            shutdown_hooks=[cleanup_cache, cleanup_db]
        )
    """
    if not config.enable_graceful_shutdown:
        # Return no-op handler
        class NoOpHandler:
            def setup(self):
                pass
            def register_hook(self, hook):
                pass
            def is_shutdown_requested(self):
                return False
        return NoOpHandler()

    handler = ShutdownHandler(
        timeout_seconds=config.shutdown_timeout_seconds,
        service_name=config.service_name
    )

    # Register provided hooks
    if shutdown_hooks:
        for hook in shutdown_hooks:
            handler.register_hook(hook)

    handler.setup()
    return handler
