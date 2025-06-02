import os
from langchain_community.vectorstores import FAISS 


def load_vector_db(db_path, embbeding_handler, trust_source=True):
    if os.path.exists(db_path):
        vector_store = FAISS.load_local(folder_path=db_path, embeddings=embbeding_handler, allow_dangerous_deserialization=trust_source)
        print(f"{load_vector_db.__name__}: successfully loaded vector db from {db_path}")
        return vector_store
    else:
        raise ValueError(f"{load_vector_db.__name__}: can't load vector DB. Path '{db_path}' doesn't exist. Aborting!")
