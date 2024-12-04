import os
import json
from bioext.elastic_utils import ElasticsearchSession
from elasticsearch import helpers

from dotenv import load_dotenv
load_dotenv()

project_name = "breast_brca_status"
base_dir = "./data"

project_dir = os.path.join(base_dir, project_name)
os.makedirs(project_dir, exist_ok=True)

es_session = ElasticsearchSession()

query = {
    "query": {
        "bool": {
            "must": [
                {"wildcard": {"text": "*brca*"}},
                {"wildcard": {"text": "*breast*"}}
            ]
        }
    }
}

try:
    results = helpers.scan(
        client=es_session.es,
        query=query,
        scroll='2m',
        index="brca_synth"
    )

    processed_count = 0
    for hit in results:
        doc_id = hit['_id']
        file_path = os.path.join(project_dir, f"{doc_id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(hit, f, indent=2)
        
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Up to {processed_count} docs...")

    print(f"\nTotal: {processed_count} docs")

except Exception as e:
    print(f"Error: {str(e)}")