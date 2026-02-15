import os
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
    BitsAndBytesConfig,
)
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich import print as rprint
from rich.panel import Panel
import warnings

warnings.filterwarnings("ignore")

CACHE_DIR = "./models"
CHROMA_DB_DIR = "./chroma_db"

class BanglaRAGChain:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_model_id = None
        self.embed_model_id = None
        self.k = 4
        self.max_new_tokens = 512
        self.chunk_size = 500
        self.chunk_overlap = 150
        self.text_path = ""
        self.quantization = None
        self.temperature = 0.6
        self.top_p = 0.9
        self.hf_token = None
        self.tokenizer = None
        self.chat_model = None
        self._llm = None
        self._retriever = None
        self._db = None
        self._chain = None

    def load(self, args):
        self.chat_model_id = args.model
        self.embed_model_id = args.embed_model
        self.text_path = args.text_path
        self.quantization = args.quantize
        self.hf_token = args.hf_token

        if self.hf_token:
            os.environ["HF_TOKEN"] = str(self.hf_token)

        rprint(Panel(f"[bold green]Loading model: {self.chat_model_id}...", expand=False))
        self._load_models()

        rprint(Panel(f"[bold green]Loading text from {self.text_path}...", expand=False))
        self._create_document()

        rprint(Panel("[bold green]Updating Vector DB...", expand=False))
        self._update_chroma_db()

        self._get_retriever()
        self._get_llm()
        self._create_chain()

    def _load_models(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id, token=self.hf_token)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            bnb_config = None
            if self.quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                self.chat_model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=CACHE_DIR,
                token=self.hf_token
            )
        except Exception as e:
            rprint(Panel(f"[red]Error loading chat model: {e}", expand=False))
            raise e

    def _create_document(self):
        try:
            with open(self.text_path, "r", encoding="utf-8") as file:
                text_content = file.read()
            character_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "।", ".", " ", ""],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            self._documents = character_splitter.create_documents([text_content])
            print(f"Created {len(self._documents)} chunks.")
        except Exception as e:
            rprint(Panel(f"[red]Chunking failed: {e}", expand=False))
            raise e

    def _update_chroma_db(self):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model_id,
                model_kwargs={"device": self._device}
            )
            self._db = Chroma.from_documents(
                documents=self._documents,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )
        except Exception as e:
            rprint(Panel(f"[red]Vector DB failed: {e}", expand=False))
            raise e

    def _get_llm(self):
        try:
            pipe = pipeline(
                "text-generation",
                model=self.chat_model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            rprint(Panel(f"[red]LLM init failed: {e}", expand=False))
            raise e

    def _get_retriever(self):
        self._retriever = self._db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

    def _get_prompt_template(self):
        if "llama-3" in self.chat_model_id.lower():
            template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for Bangla language tasks. Answer based ONLY on the context provided. If unsure, say "I don't know" in Bangla.

Context:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        elif "qwen" in self.chat_model_id.lower():
            template = """<|im_start|>system
You are a helpful assistant for Bangla language tasks. Answer based ONLY on the context provided. If unsure, say "I don't know" in Bangla.

Context:
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        elif "mistral" in self.chat_model_id.lower():
             template = """<s>[INST] You are a helpful assistant for Bangla language tasks. Answer based ONLY on the context provided. If unsure, say "I don't know" in Bangla.

Context:
{context}

Question:
{question} [/INST]"""
        else:
            # Generic Alpaca fallback
            template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question based on the context.

### Context:
{context}

### Question:
{question}

### Response:
"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _create_chain(self):
        prompt = self._get_prompt_template()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self._chain = (
            {"context": self._retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )

    def get_response(self, query):
        try:
            return self._chain.invoke(query).strip()
        except Exception as e:
            rprint(Panel(f"[red]Generation failed: {e}", expand=False))
            return "Error."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--embed_model", type=str, default="l3cube-pune/bengali-bert")
    parser.add_argument("--text_path", type=str, default="test_data.txt")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    # Create dummy data if missing
    if not os.path.exists(args.text_path):
        with open(args.text_path, "w", encoding="utf-8") as f:
            f.write("BhashaLLM is a project to improve Bangla OCR and RAG pipelines. It uses Llama 3.1 as the core model. Qwen is also tested.")

    rag = BanglaRAGChain()
    rag.load(args)
    
    queries = [
        "What is BhashaLLM?",
        "Which model is the core engine?"
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        ans = rag.get_response(q)
        print(f"A: {ans}")
