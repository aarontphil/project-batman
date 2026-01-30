import cv2
import numpy as np
import unittest
from parameters.vehicle_size import extract_vehicle_size
from parameters.vehicle_color import extract_vehicle_color
from parameters.plate_text import detect_plate_text

class TestVehicleParameters(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy frame: 1920x1080, black
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Draw a red rectangle (vehicle)
        # BGR for Red is (0, 0, 255)
        cv2.rectangle(self.frame, (100, 100), (300, 300), (0, 0, 255), -1)
        self.bbox_red = [100, 100, 300, 300]
        
        # Draw a small blue rectangle (small vehicle)
        cv2.rectangle(self.frame, (400, 400), (450, 450), (255, 0, 0), -1)
        self.bbox_blue_small = [400, 400, 450, 450]

    def test_vehicle_size(self):
        print("\nTesting Vehicle Size...")
        # Red box: 200x200
        result = extract_vehicle_size(self.bbox_red, self.frame.shape)
        print(f"Red Box Size: {result}")
        self.assertEqual(result['bbox_area'], 40000)
        self.assertGreater(result['size_ratio'], 0)
        
        # Blue box: 50x50
        result_small = extract_vehicle_size(self.bbox_blue_small, self.frame.shape)
        print(f"Blue Box Size: {result_small}")
        self.assertEqual(result_small['size_class'], 'small')

    def test_vehicle_color(self):
        print("\nTesting Vehicle Color...")
        # Red box
        result = extract_vehicle_color(self.frame, self.bbox_red)
        print(f"Red Box Color: {result}")
        self.assertEqual(result['dominant_color'], 'red')
        
        # Blue box
        result_blue = extract_vehicle_color(self.frame, self.bbox_blue_small)
        print(f"Blue Box Color: {result_blue}")
        self.assertEqual(result_blue['dominant_color'], 'blue')

    def test_plate_text(self):
        print("\nTesting Plate Text (Mock)...")
        # Since we don't have a real plate and EasyOCR might be missing/slow
        # this basically tests that the function runs and returns the schema.
        result = detect_plate_text(self.frame, self.bbox_red)
        print(f"Plate Result: {result}")
        self.assertIn('plate_detected', result)
        self.assertIn('plate_text', result)
        if 'error' in result:
             print(f"Note: OCR Error (expected if not installed): {result['error']}")

if __name__ == '__main__':
    unittest.main()
