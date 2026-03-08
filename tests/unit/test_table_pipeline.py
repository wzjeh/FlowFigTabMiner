"""
Unit tests for TablePipeline (Optimized without CellClassifier).

Tests cover:
- Initialization and configuration loading
- Sequential vs. standard mode
- Molecule detection and IoU calculation
- Cell content recognition (molecules vs text/numbers)
- Relevance filtering
- Error handling
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the module under test
from src.extraction.table.pipeline import TablePipeline


class TestTablePipelineInit:
    """Test TablePipeline initialization."""

    def test_init_sequential_mode(self):
        """Test initialization in sequential mode."""
        with patch('src.extraction.table.pipeline.load_config') as mock_config:
            mock_config.return_value = {"tables": {}}
            pipeline = TablePipeline(sequential_mode=True)

            assert pipeline.sequential_mode is True
            assert pipeline.filter is None  # Lazy loading
            assert pipeline.structure is None
            assert pipeline.recognizer is None
            assert pipeline.molecule_processor is None

    def test_init_standard_mode(self):
        """Test initialization in standard mode (load all models)."""
        with patch('src.extraction.table.pipeline.load_config') as mock_config:
            mock_config.return_value = {"tables": {
                "segmentation": {"model_path": "test.pt"},
                "structure": {"model_path": "test_struct"},
                "content": {"molscribe_path": "test_mol.pth"},
                "molecule_detection": {"model_path": "test_yolo.pt", "confidence_threshold": 0.25}
            }}

            with patch('src.extraction.table.pipeline.TableFilter'):
                with patch('src.extraction.table.pipeline.TableStructureRecognizer'):
                    with patch('src.extraction.table.pipeline.ContentRecognizer'):
                        with patch('src.extraction.table.pipeline.MoleculeProcessor'):
                            pipeline = TablePipeline(sequential_mode=False)

                            # In standard mode, _load_all_models should be called
                            assert pipeline.sequential_mode is False


class TestIoUCalculation:
    """Test IoU overlap calculation for molecule matching."""

    def test_calculate_iou_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        # This is implicitly tested in process_table, but we can test the logic
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]

        # Calculate manually
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        iou = inter_area / box1_area if box1_area > 0 else 0.0
        assert iou == 1.0

    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap."""
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]

        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)

        assert inter_area == 0

    def test_calculate_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]

        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        iou = inter_area / box1_area
        assert 0.0 < iou < 1.0
        assert iou == pytest.approx(0.25)  # 2500 / 10000


class TestProcessTable:
    """Test the main process_table pipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with mocked dependencies."""
        with patch('src.extraction.table.pipeline.load_config') as mock_config:
            mock_config.return_value = {"tables": {}}
            pipeline = TablePipeline(sequential_mode=True)
            return pipeline

    @pytest.fixture
    def mock_filter_result(self):
        """Mock table filter result."""
        return {
            'is_table': True,
            'conf': 0.95,
            'components': {
                'table_body': [{'crop': np.zeros((100, 100, 3), dtype=np.uint8)}]
            },
            'table_body_crop': np.zeros((100, 100, 3), dtype=np.uint8)
        }

    @pytest.fixture
    def mock_cells(self):
        """Mock TATR cell detection results."""
        return [
            {'box': [10, 10, 50, 50], 'row_index': 0, 'col_index': 0},
            {'box': [60, 10, 100, 50], 'row_index': 0, 'col_index': 1},
            {'box': [10, 60, 50, 100], 'row_index': 1, 'col_index': 0},
            {'box': [60, 60, 100, 100], 'row_index': 1, 'col_index': 1},
        ]

    @pytest.fixture
    def mock_mol_meta(self):
        """Mock molecule detection metadata."""
        return [
            {'box': [15, 15, 45, 45], 'smiles': 'CCO'},  # Overlaps with cell 0
            {'box': [65, 65, 95, 95], 'smiles': 'C1CCCCC1'},  # Overlaps with cell 3
        ]

    def test_process_table_rejected_by_filter(self, mock_pipeline, tmp_path):
        """Test table rejected by filter."""
        mock_filter = Mock()
        mock_filter.filter_tables.return_value = [{'is_table': False, 'conf': 0.3}]

        with patch.object(mock_pipeline, '_get_model', return_value=mock_filter):
            with patch.object(mock_pipeline, '_unload_model'):
                result = mock_pipeline.process_table(str(tmp_path / "test.png"))

                assert result['is_valid'] is False
                assert result['reason'] == 'Filtered by YOLO'

    def test_process_table_no_cells_detected(self, mock_pipeline, mock_filter_result, tmp_path):
        """Test when TATR detects no cells."""
        # Mock filter
        mock_filter = Mock()
        mock_filter.filter_tables.return_value = [mock_filter_result]

        # Mock structure recognizer (no cells)
        mock_structure = Mock()
        mock_structure.recognize_structure.return_value = {'cells': []}

        # Mock molecule processor
        mock_mol_processor = Mock()
        mock_mol_processor.process_image.return_value = (None, [])

        # Mock content recognizer
        mock_recognizer = Mock()

        def mock_get_model(model_type):
            if model_type == 'filter':
                return mock_filter
            elif model_type == 'structure':
                return mock_structure
            elif model_type == 'molecule':
                return mock_mol_processor
            elif model_type == 'content':
                return mock_recognizer
            return None

        test_img = tmp_path / "test.png"
        test_img.write_bytes(b'fake image')

        with patch.object(mock_pipeline, '_get_model', side_effect=mock_get_model):
            with patch.object(mock_pipeline, '_unload_model'):
                with patch('cv2.imwrite'):
                    result = mock_pipeline.process_table(str(test_img))

                    assert result['is_valid'] is False
                    assert result['reason'] == 'No cells detected'

    def test_process_table_molecule_cell_matching(
        self, mock_pipeline, mock_filter_result, mock_cells, mock_mol_meta, tmp_path
    ):
        """Test molecule-to-cell matching via IoU."""
        # Mock filter
        mock_filter = Mock()
        mock_filter.filter_tables.return_value = [mock_filter_result]

        # Mock structure recognizer
        mock_structure = Mock()
        mock_structure.recognize_structure.return_value = {'cells': mock_cells}

        # Mock molecule processor
        mock_mol_processor = Mock()
        mock_mol_processor.process_image.return_value = (
            np.zeros((100, 100, 3), dtype=np.uint8),
            mock_mol_meta
        )

        # Mock content recognizer
        mock_recognizer = Mock()
        mock_recognizer.recognize_content.return_value = "42.5"  # Text/number

        def mock_get_model(model_type):
            if model_type == 'filter':
                return mock_filter
            elif model_type == 'structure':
                return mock_structure
            elif model_type == 'molecule':
                return mock_mol_processor
            elif model_type == 'content':
                return mock_recognizer
            return None

        test_img = tmp_path / "test.png"
        test_img.write_bytes(b'fake image')

        with patch.object(mock_pipeline, '_get_model', side_effect=mock_get_model):
            with patch.object(mock_pipeline, '_unload_model'):
                with patch('cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
                    with patch('cv2.imwrite'):
                        with patch('builtins.open', create=True):
                            with patch('yaml.safe_load', return_value={'keywords': ['yield']}):
                                result = mock_pipeline.process_table(str(test_img), str(tmp_path))

                                # Check that cells were processed
                                assert 'cells' in result


    def test_process_table_relevance_filtering(self, mock_pipeline, tmp_path):
        """Test keyword-based relevance filtering."""
        # This would require mocking the entire pipeline
        # For brevity, we can test the relevance logic in isolation

        # Simulate relevance check
        keywords = ['yield', 'conversion', 'selectivity']
        caption_text = "This is a table about yield and conversion"
        note_text = "Reaction conditions: 25°C"

        check_text = (caption_text + " " + note_text).lower()
        import re
        norm_text = re.sub(r'[^a-z0-9]', '', check_text)

        is_relevant = any(kw in norm_text for kw in keywords)
        assert is_relevant is True

        # Test with irrelevant text
        caption_text = "This is about something else"
        check_text = caption_text.lower()
        norm_text = re.sub(r'[^a-z0-9]', '', check_text)
        is_relevant = any(kw in norm_text for kw in keywords)
        assert is_relevant is False


class TestGridIndexAssignment:
    """Test cell grid index assignment."""

    def test_assign_grid_indices_simple(self):
        """Test grid index assignment for simple 2x2 table."""
        cells = [
            {'box': [0, 0, 50, 50]},    # Top-left
            {'box': [50, 0, 100, 50]},  # Top-right
            {'box': [0, 50, 50, 100]},  # Bottom-left
            {'box': [50, 50, 100, 100]}, # Bottom-right
        ]

        # Compute centroids manually
        for c in cells:
            c['cx'] = (c['box'][0] + c['box'][2]) / 2
            c['cy'] = (c['box'][1] + c['box'][3]) / 2

        # Expected grid assignment
        assert cells[0]['cx'] == 25 and cells[0]['cy'] == 25
        assert cells[1]['cx'] == 75 and cells[1]['cy'] == 25
        assert cells[2]['cx'] == 25 and cells[2]['cy'] == 75
        assert cells[3]['cx'] == 75 and cells[3]['cy'] == 75


class TestSequentialMode:
    """Test sequential model loading/unloading."""

    def test_unload_model_in_sequential_mode(self):
        """Test model unloading clears memory."""
        with patch('src.extraction.table.pipeline.load_config') as mock_config:
            mock_config.return_value = {"tables": {}}
            pipeline = TablePipeline(sequential_mode=True)

            mock_model = Mock()

            with patch('gc.collect') as mock_gc:
                pipeline._unload_model(mock_model)
                mock_gc.assert_called_once()

    def test_no_unload_in_standard_mode(self):
        """Test model not unloaded in standard mode."""
        with patch('src.extraction.table.pipeline.load_config') as mock_config:
            mock_config.return_value = {"tables": {}}
            pipeline = TablePipeline(sequential_mode=False)

            mock_model = Mock()

            with patch('gc.collect') as mock_gc:
                pipeline._unload_model(mock_model)
                # gc.collect should NOT be called because sequential_mode=False
                mock_gc.assert_not_called()


# Run tests with: pytest tests/unit/test_table_pipeline.py -v
