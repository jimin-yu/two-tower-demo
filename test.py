from pymilvus import MilvusClient
client = MilvusClient(uri='http://milvus.dev.sinsang.market:19530')

client.create_collection(collection_name='test_collection', dim)
