# FlowFigTabMiner Test Suite

## Directory Structure

```
tests/
├── __init__.py              # Test package init
├── conftest.py              # Pytest fixtures and configuration
├── README.md                # This file
├── unit/                    # Unit tests for individual modules
│   ├── test_table_pipeline.py
│   ├── test_figure_processor.py
│   ├── test_molecule_processor.py
│   └── test_llm_engine.py
├── integration/             # End-to-end tests
│   ├── test_e2e_table.py
│   └── test_e2e_figure.py
└── fixtures/                # Test data
    ├── sample_table.png
    ├── sample_figure.png
    └── sample.pdf
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest --cov=src tests/
```

### Run specific test file
```bash
pytest tests/unit/test_table_pipeline.py
```

### Run with verbose output
```bash
pytest -v tests/
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Mock external dependencies (models, API calls)
- Fast execution (<1s per test)
- High coverage of edge cases

### Integration Tests (`tests/integration/`)
- Test complete pipelines end-to-end
- Use real models (CPU mode for CI)
- Slower execution (may take minutes)
- Verify system behavior

## Writing Tests

### Example Unit Test

```python
import pytest
from src.extraction.table.pipeline import TablePipeline

def test_table_pipeline_initialization(mock_config):
    """Test TablePipeline initializes correctly."""
    pipeline = TablePipeline(sequential_mode=True)
    assert pipeline.sequential_mode is True
    assert pipeline.filter is None  # Lazy loading
```

### Example Integration Test

```python
def test_end_to_end_table_extraction(sample_table_image, test_output_dir):
    """Test full table extraction pipeline."""
    pipeline = TablePipeline()
    result = pipeline.process_table(str(sample_table_image), str(test_output_dir))

    assert result['is_valid'] is True
    assert 'csv_path' in result
    assert result['num_extracted'] > 0
```

## Mocking Strategy

- **Models**: Use `unittest.mock.Mock` to simulate model outputs
- **File I/O**: Use pytest `tmp_path` fixture for temporary files
- **API Calls**: Mock LLM API responses with sample JSON

## Continuous Integration

Tests are designed to run in CI environments:
- All unit tests must pass before merge
- Integration tests can be skipped in CI with `--no-integration` flag
- Coverage threshold: 70% minimum

## Adding New Tests

1. Create test file in appropriate directory (`unit/` or `integration/`)
2. Name file `test_<module_name>.py`
3. Use descriptive test names: `test_<functionality>_<scenario>`
4. Add fixtures to `conftest.py` if needed
5. Update this README if adding new test categories
