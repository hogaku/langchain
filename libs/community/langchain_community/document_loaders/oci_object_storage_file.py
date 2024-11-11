import os
import tempfile
from typing import List
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_unstructured import UnstructuredLoader 
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import VsdxParser
from langchain_community.document_loaders.blob_loaders import Blob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCIObjectStorageFileLoader(BaseLoader):
    """Load from `OCI Object Storage` files."""

    def __init__(self, namespace: str, bucket_name: str, object_name: str, config: dict):
        """Initialize with OCI parameters, object name, and configuration."""
        self.namespace = namespace
        """Namespace for OCI Object Storage."""
        self.bucket_name = bucket_name
        """Bucket name."""
        self.object_name = object_name
        """Object name."""
        self.config = config
        """OCI configuration settings."""

    def load(self) -> List[Document]:
        """Load documents from OCI Object Storage."""
        try:
            import oci
        except ImportError as exc:
            raise ImportError(
                "Could not import oci package. "
                "Please install it with `pip install oci`."
            ) from exc

        client = oci.object_storage.ObjectStorageClient(config=self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, self.object_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            try:
                response = client.get_object(self.namespace, self.bucket_name, self.object_name)
                with open(file_path, "wb") as file:
                    file.write(response.data.content)
            except oci.exceptions.ServiceError as e:
                raise RuntimeError(f"Failed to download object '{self.object_name}': {e.message}")

            if file_path.endswith('.json'):
                logger.info('Using JSONLoader')
                jq_schema = "."
                loader = JSONLoader(file_path, jq_schema=jq_schema, text_content=False)
            elif file_path.endswith('.pdf'):
                logger.info('Using PyPDFLoader')
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.vsdx'):
                logger.info('Using VsdxParser')
                blob = Blob(path=file_path, source=self.object_name)
                parser = VsdxParser()
                return list(parser.parse(blob))
            else:
                logger.info('Using UnstructuredFileLoader')
                loader = UnstructuredLoader(file_path)

            return loader.load()
