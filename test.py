import unittest
from app import app
import json

class FlaskTestCase(unittest.TestCase):
    
    # Test for the '/' route
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)  # Ensure the page loads successfully
        self.assertIn(b'<!DOCTYPE html>', response.data)  # Check if HTML is returned

    # Test for the '/predict' route (valid input)
    def test_predict_valid(self):
        tester = app.test_client(self)
        # Provide valid data in form of a POST request
        response = tester.post('/predict', data=dict(data="200,100,1,50"))
        self.assertEqual(response.status_code, 200)  # Check if the API response is OK
        data = json.loads(response.data)
        self.assertIn('prediction', data)  # Check if the response contains the 'prediction' field
        self.assertIn(data['prediction'], ["Sweet", "Savory"])  # Check if the prediction is valid

    # Test for the '/predict' route (invalid input)
    def test_predict_invalid(self):
        tester = app.test_client(self)
        # Provide invalid data (e.g., non-numeric input)
        response = tester.post('/predict', data=dict(data="invalid,data"))
        self.assertEqual(response.status_code, 200)  # Ensure the request goes through
        data = json.loads(response.data)
        self.assertIn('error', data)  # Check if the response contains an error message

if __name__ == '__main__':
    unittest.main()
