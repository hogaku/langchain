from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
from langchain_community.document_loaders import OCIObjectStorageContainerLoader


@pytest.mark.parametrize("filename", ["object1.txt", "object2.txt"])
def test_oci_container_loader(filename: str) -> None:
    """Test OCIObjectStorageContainerLoader."""
    mock_client = MagicMock()
    mock_client.list_objects.return_value = MagicMock(
        data=MagicMock(
            objects=[
                MagicMock(name="object1.txt"),
                MagicMock(name="object2.txt")
            ],
            next_start_with=None
        )
    )

    mock_file_loader = MagicMock()
    mock_file_loader.load.return_value = [
        {"content": "Sample content", "metadata": {"filename": filename}}
    ]

    with patch(
        "langchain_community.document_loaders.oci_object_storage_container.ObjectStorageClient",
        return_value=mock_client
    ), patch(
        "langchain_community.document_loaders.oci_object_storage_container.OCIObjectStorageFileLoader",
        return_value=mock_file_loader
    ):
        loader = OCIObjectStorageContainerLoader(
            namespace="test_namespace",
            bucket_name="test_bucket",
            config={"mock": "config"},
            max_objects=2
        )

        docs = loader.load()

        assert len(docs) == 2
