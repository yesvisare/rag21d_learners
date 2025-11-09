"""
Tests for the portfolio scaffolder CLI.

These are smoke tests to ensure the scaffolder CLI works correctly.
"""

import subprocess
import sys
from pathlib import Path

import pytest


# Get the workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent
SCAFFOLDER_SCRIPT = WORKSPACE_ROOT / "m4_3_portfolio_scaffold.py"


def test_scaffolder_script_exists():
    """Test that the scaffolder script exists."""
    assert SCAFFOLDER_SCRIPT.exists(), f"Scaffolder not found at {SCAFFOLDER_SCRIPT}"


def test_scaffolder_help():
    """Test that scaffolder --help works."""
    result = subprocess.run(
        [sys.executable, str(SCAFFOLDER_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0, "Help command should exit with code 0"
    assert "usage:" in result.stdout.lower(), "Help should show usage"
    assert "--dry-run" in result.stdout, "Help should mention --dry-run flag"
    assert "--force" in result.stdout, "Help should mention --force flag"
    assert "--no-frontend" in result.stdout, "Help should mention --no-frontend flag"
    assert "--no-ci" in result.stdout, "Help should mention --no-ci flag"


def test_scaffolder_dry_run():
    """Test that scaffolder --dry-run works without creating files."""
    result = subprocess.run(
        [sys.executable, str(SCAFFOLDER_SCRIPT), "TestProject", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0, f"Dry-run should succeed. stderr: {result.stderr}"

    # Check for dry-run indicators in output
    assert "DRY RUN" in result.stdout, "Output should indicate DRY RUN mode"
    assert "Would create" in result.stdout, "Output should show planned actions"
    assert "directories" in result.stdout, "Output should mention directories"
    assert "files" in result.stdout, "Output should mention files"

    # Verify no actual directory was created
    test_project_path = Path("testproject")
    assert not test_project_path.exists(), "Dry-run should NOT create directory"


def test_scaffolder_dry_run_with_no_frontend():
    """Test that --no-frontend flag works in dry-run mode."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCAFFOLDER_SCRIPT),
            "BackendOnlyTest",
            "--dry-run",
            "--no-frontend"
        ],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0, f"Dry-run with --no-frontend should succeed. stderr: {result.stderr}"
    assert "DRY RUN" in result.stdout
    assert "DISABLED (--no-frontend)" in result.stdout or "Frontend: DISABLED" in result.stdout


def test_scaffolder_dry_run_with_no_ci():
    """Test that --no-ci flag works in dry-run mode."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCAFFOLDER_SCRIPT),
            "NoCITest",
            "--dry-run",
            "--no-ci"
        ],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0, f"Dry-run with --no-ci should succeed. stderr: {result.stderr}"
    assert "DRY RUN" in result.stdout
    assert "DISABLED (--no-ci)" in result.stdout or "CI/CD: DISABLED" in result.stdout


def test_scaffolder_dry_run_all_flags():
    """Test dry-run with all flags enabled."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCAFFOLDER_SCRIPT),
            "MinimalTest",
            "--dry-run",
            "--no-frontend",
            "--no-ci"
        ],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0, "Dry-run with all flags should succeed"
    assert "DRY RUN" in result.stdout
    # Should show both disabled features
    output_upper = result.stdout.upper()
    assert "FRONTEND" in output_upper and "DISABLED" in output_upper
    assert "CI" in output_upper or "CI/CD" in output_upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
