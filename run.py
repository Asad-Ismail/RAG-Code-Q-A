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
from angle_emb import AnglE, Prompts
import argparse

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument Parsing
parser = argparse.ArgumentParser(description="Script Configuration")
parser.add_argument("--input_dir", type=str, default="/home/ec2-user/SageMaker/VegRD_APD_Fruit_Phenotyping/scripts",help="Directory containing the scripts")
parser.add_argument("--embed_device", type=str, default="cpu", choices=["auto", "cuda", "cpu"], help="Device for embeddings ('auto', 'cuda', 'cpu')")
parser.add_argument("--llm_device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for LLM ('auto', 'cuda', 'cpu')")
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1",help="Name of the model to use")
parser.add_argument("--embedding_choice", type=str, default="st", choices=['st', 'bge', 'uae', 'instructor'],help="Embedding model to use ('st', 'bge', 'uae', 'instructor')")
args = parser.parse_args()

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
        #self._model.set_prompt(prompt=Prompts.C)
        logger.warn(f"Embedding device is {self._model.device}")


        
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
        self._model = INSTRUCTOR(instructor_model_name,device=args.embed_device)
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


def choose_embedding_model(choice):
    if choice == 'st':
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': args.embed_device})
    elif choice == 'bge':
        return HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en", model_kwargs={'device': args.embed_device})
    elif choice == 'uae':
        return UAEmbeddings()
    elif choice == 'instructor':
        return InstructorEmbeddings()
    else:
        raise ValueError(f"Invalid embedding choice: {choice}")

def build_RAG():
    logger.info(f"Loading Data!!")
    reader = SimpleDirectoryReader(input_dir=args.input_dir, recursive=True, required_exts=[".py"],file_metadata=get_meta)
    documents = reader.load_data()

    logger.info(f"Loading Emdedding Model!!")
    embed_model = choose_embedding_model(args.embedding_choice)

    logger.info(f"Loading LLM Model!!")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=1000,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        system_prompt="You are a Q&A assistant...",
        query_wrapper_prompt="{query_str}",
        tokenizer_name=args.model_name,
        model_name=args.model_name,
        device_map=args.llm_device,
        tokenizer_kwargs={"max_length": 4096},
        model_kwargs={"torch_dtype": torch.float16},
    )

    first_param = next(llm._model.parameters())
    llm_device = first_param.device
    llm_dtype = first_param.dtype
    logger.info(f"LLM device and dtype are: {llm_device}, {llm_dtype}")

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )
    logger.info(f"Building Index!!")
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    return query_engine


if __name__ == "__main__":

    query_engine=build_RAG()
    questions = [
    "How is curved length of hot peppers calculated?",
    "How to turn on/off visulaizations of detection and phenotypes?",]

    for question in questions:
        result = query_engine.query(question)
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result} \n")
