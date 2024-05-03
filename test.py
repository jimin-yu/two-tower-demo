import requests
import json
import weaviate
client = weaviate.Client("https://weaviate.dev.sinsang.market")
# client.is_ready()
url = 'https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny+vectors.json'
resp = requests.get(url)
data = json.loads(resp.text)
# Configure a batch process
client.batch.configure(batch_size=100)  # Configure batch
with client.batch as batch:
    # Batch import all Questions
    for i, d in enumerate(data):
        print(f"importing question: {i+1}")
        properties = {
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        }
        print(properties)
        # batch.add_data_object(properties, "Question", vector=d["Vector"])