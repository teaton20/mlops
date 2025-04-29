import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "post_text": "Hey! I'm loving this new Reddit thread. It's big slay."
}

response = requests.post(url, json=data)
print(response.json())