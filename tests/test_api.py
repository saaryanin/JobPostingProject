import requests

API_URL = "http://127.0.0.1:8000/predict"

def test_real_job():
    """Test API with a real job description."""
    job_description = {"text": "Looking for an experienced software engineer to join our team. Must have 5+ years experience in Python."}
    response = requests.post(API_URL, json=job_description)
    result = response.json()
    print("Real Job Test:", result)
    assert "label" in result and "confidence" in result
    assert result["label"] in ["Real Job Posting", "Fake Job Posting", "Uncertain - More Data Needed"]

def test_fake_job():
    """Test API with a fake job description."""
    job_description = {"text": "Earn $1000 per week by working 2 hours per day. No experience needed!"}
    response = requests.post(API_URL, json=job_description)
    result = response.json()
    print("Fake Job Test:", result)
    assert "label" in result and "confidence" in result
    assert result["label"] in ["Real Job Posting", "Fake Job Posting", "Uncertain - More Data Needed"]

if __name__ == "__main__":
    test_real_job()
    test_fake_job()
    print("✅ API Tests Completed Successfully!")
