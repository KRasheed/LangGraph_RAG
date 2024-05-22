class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class CacheManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance.cached_data = None
        return cls._instance

    def set_data(self, documents):
        """Store the entire list of documents."""
        self.cached_data = documents

    def get_data(self):
        """Retrieve the stored list of documents."""
        return self.cached_data
