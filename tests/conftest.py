from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_cross_encoder():
    """Mock CrossEncoder to avoid downloading HuggingFace models during tests."""
    mock_class = MagicMock()
    mock_instance = MagicMock()
    # Predict returns a list of scores, one for each input pair
    mock_instance.predict.return_value = [0.9] * 50
    mock_class.return_value = mock_instance
    
    with patch("sentence_transformers.CrossEncoder", mock_class):
        yield mock_class
