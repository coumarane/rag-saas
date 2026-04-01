"""Shared pytest configuration."""

import pytest


# pytest-asyncio configuration: all async tests run automatically without
# needing the @pytest.mark.asyncio decorator on each one.
pytest_plugins = ["pytest_asyncio"]
