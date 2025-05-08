import pytest
import numpy as np
import cv2
import os
from main import ImageSearchApp

@pytest.fixture
def sample_images(tmp_path):
    """Create sample images for testing"""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()
    
    # Create 3 simple images
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, i] = 255  # Each image has a different dominant color
        img_path = str(img_dir / f"test{i}.jpg")
        cv2.imwrite(img_path, img)
    
    return str(img_dir)

@pytest.fixture
def search_engine(sample_images):
    """Initialize search engine with test images"""
    engine = ImageSearchApp()
    engine.load_images(sample_images)
    engine.build_indexes()
    return engine

@pytest.fixture
def query_image():
    """Create a simple query image (similar to test0)"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # Mostly red
    return img