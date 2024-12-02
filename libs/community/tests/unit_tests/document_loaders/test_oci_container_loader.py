import pytest
from unittest.mock import MagicMock, patch
from langchain_community.document_loaders import OCIObjectStorageContainerLoader


@pytest.mark.requires("oci")
def test_oci_container_loader() -> None:
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
    mock_file_loader.load.side_effect = [
        [{"content": "Document 1 content", "metadata": {"filename": "object1.txt"}}],
        [{"content": "Document 2 content", "metadata": {"filename": "object2.txt"}}]
    ]

    with patch(
        "langchain_community.document_loaders.oci_object_storage_container.ObjectStorageClient",
        return_value=mock_client
    ), patch(
        "langchain_community.document_loaders.oci_object_storage_container.OCIObjectStorageFileLoader",
        return_value=mock_file_loader
    ):
        loader = OCIObjectStorageContainerLoader(
            namespace="mock_namespace",
            bucket_name="mock_bucket",
            config={"mock": "config"},
            max_objects=2
        )
        
        docs = loader.load()

        assert len(docs) == 2  
        assert docs[0]["metadata"]["filename"] == "object1.txt"
        assert docs[1]["metadata"]["filename"] == "object2.txt"

        assert mock_file_loader.load.call_count == 2
