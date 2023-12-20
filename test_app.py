import json
from flask import Flask, render_template, request, jsonify
from unittest import TestCase
from unittest.mock import patch, MagicMock
from app import app

class TestApp(TestCase):
    def setUp(self):
        """
        Create a test client for the Flask app.
        """
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_index_get(self):
        """
        Test the home page endpoint.
        """    
        response = self.app.get('/')
        self.assertIn(b"Chat support", response.data)
        self.assertIn(b"chatbox", response.data)
        self.assertEqual(response.status_code, 200)

    @patch('app.get_response')
    def test_predict(self, mock_get_response):
        """
        Test the '/predict' endpoint.
        """
        mock_get_response.return_value = "Test response"
        input_data = {"message": "Hello"}
        response = self.app.post('/predict', json=input_data)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Test response")
