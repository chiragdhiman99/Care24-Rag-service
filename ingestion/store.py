from qdrant_client import QdrantClient
from qdrant_client.models import Distance,PointStruct, VectorParams
from ingestion.embedder import embed
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "medical_knowledge"

def create_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print("Collection created")
    else:
        print("Collection already exists")

        

def store_embedding(chunks, embeddings):
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={"text": chunk}
        ))

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"Stored {i+len(batch)}/{len(points)}")



chunks =[]
for item in all_data:
    if(item["summary"]):
      chunks.append(item["title"] + " " + item["summary"])

embeddings = [embed(chunk) for chunk in chunks]

create_collection()
store_embedding(chunks, embeddings)