from typing import List, Optional, Dict, Any
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.oci_object_storage_file import OCIObjectStorageFileLoader
from oci.object_storage import ObjectStorageClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCIObjectStorageContainerLoader(BaseLoader):
    """Load documents from OCI Object Storage container."""

    def __init__(
        self,
        namespace: str,
        bucket_name: str,
        prefix: str = "",
        config: Optional[Dict[str, Any]] = None,
        oci_profile: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        max_objects: Optional[int] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the loader with OCI Object Storage parameters and configuration.

        Args:
            namespace (str): The namespace of the Object Storage in OCI.
            bucket_name (str): The name of the bucket to load from.
            prefix (str): Optional prefix for filtering objects by name.
            config (Optional[Dict[str, Any]]): A dictionary containing OCI configuration settings.
            oci_profile (Optional[str]): Optional profile for OCI configuration.
            metadata_keys (Optional[List[str]]): List of metadata keys to include in Document objects.
            max_objects (Optional[int]): Maximum number of objects to load.
            log_level (int): Logging level (default is INFO).

        Raises:
            ValueError: If namespace or bucket_name is empty.
        """
        if not namespace or not bucket_name:
            raise ValueError("Namespace and bucket_name cannot be empty.")

        self.namespace = namespace
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.config = config
        self.oci_profile = oci_profile
        self.metadata_keys = metadata_keys or []
        self.max_objects = max_objects

        # ロギングレベルの設定
        logging.getLogger().setLevel(log_level)

    def _initialize_client(self) -> ObjectStorageClient:
        try:
            import oci
        except ImportError as exc:
            raise ImportError(
                "Could not import oci package. "
                "Please install it with `pip install oci`."
            ) from exc

        try:
            if self.config:
                logger.info("Using provided OCI configuration for client initialization.")
                return ObjectStorageClient(config=self.config)
            else:
                logger.info("Using OCI profile for client initialization.")
                return ObjectStorageClient(
                    config=oci.config.from_file(profile_name=self.oci_profile)
                )
        except oci.exceptions.ConfigFileNotFound as e:
            logger.error("OCI configuration file not found: %s", e)
            raise RuntimeError("OCI configuration file not found.")
        except Exception as e:
            logger.error("Failed to initialize OCI client: %s", e)
            raise RuntimeError("Failed to initialize OCI client.")

    def _extract_metadata(self, obj) -> Dict[str, Any]:
        """Extract metadata from an OCI object."""
        metadata = {}
        for key in self.metadata_keys:
            metadata[key] = getattr(obj, key, None)
        return metadata

    def load(self) -> List[Document]:
        """Load documents from OCI Object Storage container.

        Returns:
            List[Document]: A list of Document objects loaded from the OCI Object Storage.
        """
        client = self._initialize_client()
        docs = []
        next_start_with = None
        loaded_count = 0

        while True:
            try:
                response = client.list_objects(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket_name,
                    prefix=self.prefix,
                    start=next_start_with,
                    limit=1000  # OCI allows a maximum of 1000 objects per call
                )
                objects = response.data.objects
                next_start_with = response.data.next_start_with
            except client.exceptions.ServiceError as e:
                logger.error("Failed to list objects from bucket '%s': %s", self.bucket_name, str(e))
                raise RuntimeError(f"Could not list objects from bucket '{self.bucket_name}'.")

            if not objects:
                break

            for obj in objects:
                if self.max_objects and loaded_count >= self.max_objects:
                    logger.info("Reached the maximum number of objects to load: %d", self.max_objects)
                    return docs

                file_loader = OCIObjectStorageFileLoader(
                    namespace=self.namespace,
                    bucket_name=self.bucket_name,
                    object_name=obj.name,
                    config=self.config
                )
                docs.extend(file_loader.load())
                loaded_count += 1

            if not next_start_with:
                break

        return docs
