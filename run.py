import logging
import sys
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.base import BaseEmbedding
from llama_index.bridge.pydantic import PrivateAttr

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from InstructorEmbedding import INSTRUCTOR
from angle_emb import AnglE

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Configuration
INPUT_DIR = "/home/ec2-user/SageMaker/VegRD_APD_Fruit_Phenotyping/scripts"
EMBEDDINGS_DEVICE = "cpu"  # Options: "auto", "cuda", "cpu"
LLM_DEVICE="auto"  # Options: "auto", "cuda", "cpu"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
EMBEDDING_CHOICES = ['st', 'bge', 'uae', 'instructor']

# Function Definitions
def get_meta(file_path):
    return {"file_name": file_path.split("/")[-1]}

# Embedding Classes
class UAEmbeddings(BaseEmbedding):
    
    _model: Any= PrivateAttr()
   
    def __init__(
        self,
        instructor_model_name: str = 'WhereIsAI/UAE-Large-V1',
        **kwargs: Any,
    ) -> None:
        
        super().__init__(**kwargs)  
        self._model=AnglE.from_pretrained(instructor_model_name, pooling_strategy='cls',device="cpu")
        #self._model.set_prompt(prompt=None)

        
    def _aget_query_embedding(self, query: str) -> List[float]:
        # Implementation of the abstract method
        embeddings = self._model.encode([ query])
        return embeddings[0]
        
    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([query])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([text])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [text for text in texts]
        )
        return embeddings

class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent code base to easily search from:",
        **kwargs: Any,
    ) -> None:
        self._model = INSTRUCTOR(instructor_model_name,device="cpu")
        self._instruction = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings

# Choose Embedding Model
def choose_embedding_model(choice):
    if choice == 'st':
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': DEVICE})
    elif choice == 'bge':
        return HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en", model_kwargs={'device': DEVICE})
    elif choice == 'uae':
        return UAEmbeddings()
    elif choice == 'instructor':
        return InstructorEmbeddings()
    else:
        raise ValueError(f"Invalid embedding choice: {choice}")

# Main Script Execution
def main(embedding_choice):
    reader = SimpleDirectoryReader(input_dir=INPUT_DIR, recursive=True, required_exts=[".py"])
    documents = reader.load_data()
    embed_model = choose_embedding_model(embedding_choice)

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=1000,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        system_prompt="You are a Q&A assistant...",
        query_wrapper_prompt="{query_str}",
        tokenizer_name=MODEL_NAME,
        model_name=MODEL_NAME,
        device_map=DEVICE,
        tokenizer_kwargs={"max_length": 4096},
        model_kwargs={"torch_dtype": torch.float32},
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("How is the curved backbone of hot peppers calculated?")
    print(response)

# Choose embedding model here ('hf', 'bge', 'uae', 'instructor')
if __name__ == "__main__":
    main(embedding_choice='hf')
