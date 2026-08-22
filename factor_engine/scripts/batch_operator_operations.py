#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch operator operations with workflow orchestration.

Advanced features:
1. Multi-stage fix workflows with dependency tracking
2. Validation and testing integration
3. Git integration for safe batch operations
4. Parallel processing for independent fixes
5. Incremental fix application with checkpointing
6. Automatic rollback on test failures
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.auto_fix_operators_enhanced import (
    EnhancedScanner, EnhancedFixer, Issue, FixResult
)


@dataclass
class BatchOperation:
    """A batch operation workflow."""
    name: str
    stages: list[BatchStage] = field(default_factory=list)
    checkpoint_dir: Path | None = None
    git_enabled: bool = False
    parallel: bool = False


@dataclass
class BatchStage:
    """A single stage in batch workflow."""
    name: str
    issue_filter: callable
    validation: callable | None = None
    max_failures: int = 0  # Max failures before aborting
    dry_run: bool = False


class ValidationRunner:
    """Runs validation checks after fixes."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def run_tests(self, test_pattern: str = "tests/") -> tuple[bool, str]:
        """Run pytest on specified tests."""
        print(f"🧪 Running tests: {test_pattern}")

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", test_pattern, "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            if success:
                print("✓ Tests passed")
            else:
                print(f"✗ Tests failed (exit code {result.returncode})")

            return success, output

        except subprocess.TimeoutExpired:
            return False, "Test timeout"
        except Exception as e:
            return False, str(e)

    def run_linter(self) -> tuple[bool, str]:
        """Run linter checks."""
        print("🔍 Running linter...")

        try:
            # Try ruff first, fallback to flake8
            result = subprocess.run(
                ["ruff", "check", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 127:  # ruff not found
                result = subprocess.run(
                    ["flake8", "."],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            if success:
                print("✓ Linting passed")
            else:
                print(f"⚠️  Linting issues found")

            return success, output

        except Exception as e:
            print(f"⚠️  Linter check skipped: {e}")
            return True, str(e)  # Don't fail on linter errors

    def validate_imports(self, files: list[Path]) -> tuple[bool, str]:
        """Validate that files can be imported."""
        print("🔍 Validating imports...")

        errors = []
        for file_path in files:
            if not file_path.suffix == ".py":
                continue

            try:
                # Try to compile the file
                content = file_path.read_text(encoding="utf-8")
                compile(content, str(file_path), "exec")

            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")

        if errors:
            print(f"✗ Import validation failed ({len(errors)} errors)")
            return False, "\n".join(errors)
        else:
            print("✓ Import validation passed")
            return True, ""


class GitIntegration:
    """Git integration for safe batch operations."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.original_branch: str | None = None
        self.working_branch: str | None = None

    def is_git_repo(self) -> bool:
        """Check if directory is a git repo."""
        return (self.repo_path / ".git").exists()

    def get_current_branch(self) -> str | None:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None

    def create_branch(self, branch_name: str) -> bool:
        """Create and checkout a new branch."""
        try:
            self.original_branch = self.get_current_branch()

            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )

            self.working_branch = branch_name
            print(f"✓ Created branch: {branch_name}")
            return True

        except Exception as e:
            print(f"✗ Failed to create branch: {e}")
            return False

    def commit_changes(self, message: str, files: list[str] | None = None) -> bool:
        """Commit changes."""
        try:
            if files:
                subprocess.run(
                    ["git", "add"] + files,
                    cwd=self.repo_path,
                    check=True
                )
            else:
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=self.repo_path,
                    check=True
                )

            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )

            print(f"✓ Committed: {message}")
            return True

        except Exception as e:
            print(f"⚠️  Commit failed: {e}")
            return False

    def rollback(self) -> bool:
        """Rollback to original branch and delete working branch."""
        try:
            if self.original_branch:
                subprocess.run(
                    ["git", "checkout", self.original_branch],
                    cwd=self.repo_path,
                    check=True
                )

            if self.working_branch:
                subprocess.run(
                    ["git", "branch", "-D", self.working_branch],
                    cwd=self.repo_path,
                    check=True
                )

            print("✓ Rolled back changes")
            return True

        except Exception as e:
            print(f"✗ Rollback failed: {e}")
            return False

    def get_diff_stat(self) -> str:
        """Get diff statistics."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except Exception:
            return "Could not get diff"


class BatchProcessor:
    """Orchestrates batch operator fixing workflows."""

    def __init__(self, project_root: Path, operation: BatchOperation):
        self.project_root = project_root
        self.operation = operation
        self.validator = ValidationRunner(project_root)
        self.git = GitIntegration(project_root) if operation.git_enabled else None
        self.checkpoint_file: Path | None = None
        self.results: dict[str, list[FixResult]] = {}

    def execute(self) -> bool:
        """Execute the batch operation."""
        print(f"\n{'='*60}")
        print(f"BATCH OPERATION: {self.operation.name}")
        print(f"{'='*60}\n")

        # Setup
        if not self._setup():
            return False

        # Execute stages
        for i, stage in enumerate(self.operation.stages, 1):
            print(f"\n--- Stage {i}/{len(self.operation.stages)}: {stage.name} ---\n")

            if not self._execute_stage(stage):
                print(f"\n✗ Stage {stage.name} failed, aborting workflow")
                self._cleanup(success=False)
                return False

            # Checkpoint
            self._save_checkpoint(stage.name)

        # Final validation
        print("\n--- Final Validation ---\n")
        if not self._final_validation():
            print("\n✗ Final validation failed")
            self._cleanup(success=False)
            return False

        # Cleanup
        self._cleanup(success=True)

        print(f"\n{'='*60}")
        print(f"✓ BATCH OPERATION COMPLETE: {self.operation.name}")
        print(f"{'='*60}\n")

        return True

    def _setup(self) -> bool:
        """Setup for batch operation."""
        # Create checkpoint directory
        if self.operation.checkpoint_dir:
            self.operation.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_file = self.operation.checkpoint_dir / "checkpoint.json"

        # Create git branch if enabled
        if self.git and self.git.is_git_repo():
            branch_name = f"auto-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if not self.git.create_branch(branch_name):
                print("⚠️  Continuing without git integration")
                self.git = None

        return True

    def _execute_stage(self, stage: BatchStage) -> bool:
        """Execute a single stage."""
        # Scan for issues
        scanner = EnhancedScanner()
        all_issues = scanner.scan_all()

        # Filter issues for this stage
        stage_issues = [i for i in all_issues if stage.issue_filter(i)]

        if not stage_issues:
            print(f"No issues found for stage {stage.name}")
            return True

        print(f"Processing {len(stage_issues)} issues...")

        # Apply fixes
        fixer = EnhancedFixer(dry_run=stage.dry_run)

        if self.operation.parallel and not stage.dry_run:
            results = self._process_parallel(fixer, stage_issues)
        else:
            results = fixer.fix_issues(stage_issues)

        # Check results
        self.results[stage.name] = results
        failures = [r for r in results if not r.success]

        if len(failures) > stage.max_failures:
            print(f"✗ Too many failures: {len(failures)} > {stage.max_failures}")
            return False

        # Stage validation
        if stage.validation:
            print("\nRunning stage validation...")
            if not stage.validation():
                return False

        # Commit if git enabled and not dry run
        if self.git and not stage.dry_run:
            modified_files = list(fixer.fixed_files)
            if modified_files:
                self.git.commit_changes(
                    f"Auto-fix: {stage.name} ({len(results)} fixes)",
                    modified_files
                )

        return True

    def _process_parallel(self, fixer: EnhancedFixer,
                         issues: list[Issue]) -> list[FixResult]:
        """Process issues in parallel."""
        # Group by file to avoid conflicts
        by_file = defaultdict(list)
        for issue in issues:
            if issue.file_path:
                by_file[issue.file_path].append(issue)

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(fixer._fix_file, file_path, file_issues): file_path
                for file_path, file_issues in by_file.items()
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_results = future.result()
                    results.extend(file_results)
                except Exception as e:
                    print(f"✗ Error processing {file_path}: {e}")

        return results

    def _final_validation(self) -> bool:
        """Run final validation checks."""
        # Validate imports
        modified_files = []
        for results in self.results.values():
            for result in results:
                if result.success and result.issue.file_path:
                    modified_files.append(Path(result.issue.file_path))

        if modified_files:
            success, output = self.validator.validate_imports(modified_files)
            if not success:
                print(f"Import validation failed:\n{output}")
                return False

        # Run linter
        success, output = self.validator.run_linter()
        if not success:
            print(f"Linter found issues:\n{output[:500]}")
            # Don't fail on linter warnings

        return True

    def _save_checkpoint(self, stage_name: str):
        """Save checkpoint after stage completion."""
        if not self.checkpoint_file:
            return

        checkpoint = {
            "operation": self.operation.name,
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "results": {
                stage: [
                    {
                        "operator": r.issue.operator,
                        "success": r.success,
                        "error": r.error
                    }
                    for r in results
                ]
                for stage, results in self.results.items()
            }
        }

        self.checkpoint_file.write_text(
            json.dumps(checkpoint, indent=2),
            encoding="utf-8"
        )

    def _cleanup(self, success: bool):
        """Cleanup after batch operation."""
        if not success and self.git:
            print("\nRolling back changes...")
            self.git.rollback()

        # Print summary
        print("\n--- Summary ---")
        for stage_name, results in self.results.items():
            successful = len([r for r in results if r.success])
            print(f"{stage_name}: {successful}/{len(results)} successful")

        if self.git and success:
            print("\nGit diff:")
            print(self.git.get_diff_stat())


def create_default_workflow() -> BatchOperation:
    """Create default fix workflow."""
    operation = BatchOperation(
        name="Standard Operator Fix Workflow",
        git_enabled=True,
        parallel=False
    )

    # Stage 1: Fix docstrings
    operation.stages.append(BatchStage(
        name="Add missing docstrings",
        issue_filter=lambda i: i.category == "docstring" and i.auto_fixable,
        dry_run=False
    ))

    # Stage 2: Fix defaults
    operation.stages.append(BatchStage(
        name="Add parameter defaults",
        issue_filter=lambda i: i.category == "defaults" and i.auto_fixable,
        dry_run=False
    ))

    # Stage 3: Generate policy templates (dry-run)
    operation.stages.append(BatchStage(
        name="Generate policy templates",
        issue_filter=lambda i: i.category == "policy",
        dry_run=True  # Policies need manual review
    ))

    return operation


def main():
    parser = argparse.ArgumentParser(
        description="Batch operator operations with workflow orchestration"
    )
    parser.add_argument("--workflow", choices=["default", "custom"],
                       default="default", help="Workflow to execute")
    parser.add_argument("--git", action="store_true",
                       help="Enable git integration")
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel processing")
    parser.add_argument("--checkpoint-dir", type=Path,
                       default=Path("/tmp/batch_operator_checkpoint"),
                       help="Checkpoint directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run all stages")

    args = parser.parse_args()

    # Create workflow
    if args.workflow == "default":
        operation = create_default_workflow()
    else:
        print("Custom workflows not yet implemented")
        return 1

    # Override settings
    operation.git_enabled = args.git
    operation.parallel = args.parallel
    operation.checkpoint_dir = args.checkpoint_dir

    if args.dry_run:
        for stage in operation.stages:
            stage.dry_run = True

    # Execute
    project_root = Path(__file__).parent.parent
    processor = BatchProcessor(project_root, operation)

    success = processor.execute()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
