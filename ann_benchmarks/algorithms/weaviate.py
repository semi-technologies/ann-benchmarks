"""
ann-benchmarks interfaces for Weaviate.
"""
from __future__ import absolute_import
from typing import Optional
from weaviate import Client
from ann_benchmarks.algorithms.base import BaseANN


def handle_errors(results: Optional[dict]) -> None:
    if results is not None:
        for result in results:
            if 'result' in result and 'errors' in result['result'] and 'error' in result['result']['errors']:
                for message in result['result']['errors']['error']:
                    print(message['message'])

class WeaviateQuery(BaseANN):
    def __init__(self, metric: str, url: str, batch_size: int, vector_index_config: dict):
        self._metric = metric
        self.name = f"weaviate"
        self.class_name = "Test"
        self.vector_index_config = vector_index_config
        self.client = Client(
            url=url,
        )
        self.client.batch.configure(
            batch_size=batch_size,
            callback=handle_errors,
            timeout_retries=3,
        )
        if metric == "euclidean":
            self.vector_index_config["distance"] = "l2-squared"
        elif metric == "angular":
            self.vector_index_config["distance"] = "cosine"
        else:
            raise ValueError(f'metric: "{metric}" not added or not supported!')

    def fit(self, X):
        
        self.client.schema.delete_class(self.class_name)
        self.client.schema.create_class({
            "class": self.class_name,
            "vectorizer": "none",
            "vectorIndexConfig": self.vector_index_config,
            "distance": 
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

    def set_query_arguments(self, ef):
        self.client.schema.update_config('Index', {'vectorIndexConfig': {'ef': ef}})


    def query(self, q, n):

        res = self.client.query.get(self.class_name, ["identifier"]) \
            .with_near_vector({"vector": q}) \
            .with_limit(n).do()
        return [x["identifier"] for x in res['data']['Get'][self.class_name]]

    def __str__(self):
        return 'Weaviate(url=%s, batch=%d)' % (self.url, self.batch_size)

    def done(self):
        # remove all the objects from the Weaviate server
        self.client.schema.delete_all()