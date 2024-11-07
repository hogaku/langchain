import os
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_unstructured import UnstructuredLoader

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

            loader = UnstructuredLoader(file_path)
            return loader.load()
