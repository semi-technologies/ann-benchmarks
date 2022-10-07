"""
ann-benchmarks interfaces for Weaviate.
"""
import logging
from typing import Optional
from time import timezone
from weaviate import Client

from ann_benchmarks.algorithms.base import BaseANN

def handle_errors(results: Optional[dict]) -> None:
    if results is not None:
        for result in results:
            if 'result' in result and 'errors' in result['result'] and 'error' in result['result']['errors']:
                for message in result['result']['errors']['error']:
                    print(message['message'])
                        
class WeaviateQuery(BaseANN):
    def __init__(self, dimension, method_param):
        self.dimension = dimension
        # self.method_param = method_param
        # self.param_string = "-".join(k + "-" + str(v) for k, v in self.method_param.items()).lower()
        self.name = f"weaviate"
        self.class_name = "Test"
        self.client = Client(
            url="http://localhost:8080",
        )
        self.client.batch.configure(
            batch_size=1000,
            callback=handle_errors,
            timeout_retries=3,
        )

    def fit(self, X):

        self.client.schema.delete_class(self.class_name)
        self.client.schema.create_class({
            "class": self.class_name,
            "vectorizer": "none",
            "vectorIndexConfig": {
                "ef": 128,
                "efConstruction": 128,
                "maxConnections": 32
            } 
        })

        print("Uploading data to the Index:", self.name)        
        with self.client.batch as batch:
            for i in range(1000):
                if i % 1000 == 0:
                    print(f"Writing record {i}/{X.shape[0]}")
                batch.add_data_object(
                    vector=X[i],
                    data_object={
                        'identifier': i
                    },
                    class_name=self.class_name
                )
        

    def query(self, q, n):
        res = self.client.query.get(self.class_name, ["identifier"]) \
            .with_near_vector({"vector": q}) \
            .with_limit(n).do()
        return [x["identifier"] for x in res['data']['Get'][self.class_name]]
