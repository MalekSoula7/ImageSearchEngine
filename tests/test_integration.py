import os
import pytest
from main import ImageSearchApp

def test_end_to_end_workflow(sample_images, query_image):
    """Test complete workflow from loading to searching"""
    # Initialize
    engine = ImageSearchApp()
    
    # Load images
    count = engine.load_images(sample_images)
    assert count == 3
    
    # Build indexes
    engine.build_indexes()
    
    # Search
    results = engine.search_similar(query_image)
    assert len(results) == 3
    assert results[0]['similarity'] > results[1]['similarity']

def test_different_descriptors(search_engine, query_image):
    """Test that all descriptor types return results"""
    for desc_type in ['color_histogram', 'vgg16']:
        results = search_engine.search_similar(query_image, desc_type)
        assert len(results) == 3
        assert all(0 <= r['similarity'] <= 1 for r in results)