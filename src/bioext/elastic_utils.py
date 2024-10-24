import os
from elasticsearch import Elasticsearch
from elastic_transport import RequestsHttpNode
import requests

# thanks @LAdams for implementing required http proxy
class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint = os.getenv("http_proxy")
        self.session.proxies = {"http": self.proxy_endpoint, "https":self.proxy_endpoint}

class ElasticsearchSession:
    def __init__(self, server=None):
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

        self.api_id = os.getenv("ELASTIC_API_ID")
        self.api_key = os.getenv("ELASTIC_API_KEY")
        self.es_server = server or os.getenv("ELASTIC_SERVER", "https://sv-pr-elastic01:9200") # set to GSTT server by default

        self.proxy_node = GsttProxyNode

        self.es = self.create_session()

    def create_session(self):
        return Elasticsearch(
            hosts=self.es_server,
            api_key=(self.api_id, self.api_key),
            node_class=self.proxy_node,
            verify_certs=False,
            ssl_show_warn=False
        )

    def list_indices(self):
        return self.es.indices.get_alias(index="*")
