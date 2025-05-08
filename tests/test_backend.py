import numpy as np
import pytest
from main import ImageSearchApp

def test_load_images(search_engine):
    """Test that images are loaded correctly"""
    assert len(search_engine.images) == 3
    assert len(search_engine.filenames) == 3
    assert search_engine.images[0].shape == (224, 224, 3)

def test_feature_extraction(search_engine):
    """Test feature extraction for all descriptor types"""
    for desc_type in ['color_histogram', 'vgg16']:
        features = search_engine.features[desc_type]
        assert features is not None
        assert len(features) == 3
        assert features[0].shape[0] > 0  # Features should have some dimension

def test_search_similar(search_engine, query_image):
    """Test that search returns correct similar images"""
    results = search_engine.search_similar(query_image, 'color_histogram')
    
    assert len(results) == 3
    # The most similar should be test0 (red dominant)
    assert 'test0' in results[0]['filename']
    assert results[0]['similarity'] > 0.9
    
    # Verify similarity scores are in descending order
    assert results[0]['similarity'] >= results[1]['similarity']
    assert results[1]['similarity'] >= results[2]['similarity']

def test_invalid_descriptor(search_engine, query_image):
    """Test error handling for invalid descriptor"""
    with pytest.raises(ValueError, match="Descriptor invalid_desc not initialized"):
        search_engine.search_similar(query_image, 'invalid_desc')

def test_empty_database():
    """Test error handling for empty database"""
    engine = ImageSearchApp()
    with pytest.raises(ValueError, match="No images loaded"):
        engine.build_indexes()