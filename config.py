import os

class Config:
    PDF_SOURCE_DIRECTORY: str = "data"
    PERSIST_DIRECTORY: str = "docs/chroma"

    CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 250

    EMBEDDING_MODEL_NAME: str = "all-MiniLm-L6-v2"

    CHAT_MODEL_NAME: str = "groq/compound-mini"

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. PDF documents should be placed in '{self.PDF_SOURCE_DIRECTORY}'.")

config = Config()