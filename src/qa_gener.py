import json
import os
import sys
import re
from pathlib import Path
from typing import List


from openai import OpenAI
from openai.types.chat import ChatCompletion
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredPDFLoader,
    PDFMinerLoader,
)
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

os.environ["NLTK_DATA"] = "./lib/nltk_data"


client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def function_chat(
    function_name: str,
    messages: List[dict],
    use_stream=False,
) -> ChatCompletion:

    tools = [
        {
            "type": "function",
            "function": {
                "name": "interrogate",
                "description": "将给定文档尽可能多的转换为一系列的问题和答案。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions_and_answers": {
                            "type": "array",
                            "description": "问题和答案的列表。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "问题",
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "答案",
                                    },
                                },
                                "required": ["question", "answer"],
                            },
                        },
                    },
                    "required": ["questions_and_answers"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": function_name}},
        stream=use_stream,
        max_tokens=1000,
        temperature=0.7,
        presence_penalty=2.0,
        top_p=0.8,
    )
    if response:
        if use_stream:

            def gen_stream(result):
                for chunk in result:
                    yield chunk
                logger.info(f"chunk:{chunk}")

            return gen_stream(response)
        else:
            logger.info(f"response:{response}")
            return response
    else:
        logger.error(f"response:{response.status_code}")


def load_data(doc_path: str | Path, splitter=None, save_dir=None):

    docs = []

    if isinstance(doc_path, str):
        doc_path = Path(doc_path)

    if doc_path.is_file():
        if doc_path.name.endswith(".epub"):
            loader = UnstructuredEPubLoader(doc_path)
        elif doc_path.name.endswith(".pdf"):
            # loader = UnstructuredPDFLoader(doc_path, unstructured_kwargs={"lang":"chi_sim"})
            loader = PDFMinerLoader(doc_path)

        elif doc_path.name.endswith(".txt"):
            loader = TextLoader(doc_path, autodetect_encoding=True)
        else:
            loader = None

        if loader:
            if splitter:
                docs = loader.load_and_split(splitter)
            else:
                docs = loader.load()

            if doc_path.name.endswith(".pdf"):
                for doc in docs:
                    doc.page_content = re.sub("\x01|\x0c|\n|\s", "", doc.page_content)

            if save_dir:
                file_path = Path(save_dir, doc_path.name.split(".")[0] + ".txt")
                save_docs(docs, file_path)
        return docs

    if doc_path.is_dir():
        for file in doc_path.rglob("*.*"):
            if file.is_file():
                docs.extend(load_data(file, splitter, save_dir))

    return docs


def save_docs(docs: List[Document], out_put_path: str | Path):
    if isinstance(out_put_path, str):
        out_put_path = Path(out_put_path)
    if not out_put_path.parent.exists():
        out_put_path.mkdir(parents=True)
    with out_put_path.open("a", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.page_content)


def save_doc2txt(doc_dir="./data", save_dir="./data/text"):
    load_data(doc_dir, save_dir=save_dir)


def get_splited_docs(doc_dir="./data/text"):
    # embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # text_splitter = SemanticChunker(
    #     embeddings_model,
    #     sentence_split_regex=r"(?<=[.?!。？！…])\s+",
    #     breakpoint_threshold_amount=80,
    # )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    docs = load_data(doc_dir, text_splitter)

    return docs


def save_QA_json(doc_dir="./data/text", save_dir="./data/gen"):

    docs = get_splited_docs(doc_dir)

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    questions_and_answers = {}

    error_index = [0]

    last_source = docs[0].metadata["source"]
    for i, doc in enumerate(tqdm(docs)):

        source = doc.metadata["source"]

        if i not in error_index and i < error_index[-1]:
            continue

        retry = 1
        error_times = 0
        text = doc.page_content
        messages = [{"role": "user", "content": text}]

        while error_times < retry:
            # 获取文档问答
            responses = function_chat("interrogate", messages, False)
            arguments = responses.choices[0].message.tool_calls[0].function.arguments
            if arguments:
                try:
                    qa_json = json.loads(arguments)["questions_and_answers"]
                    if source not in questions_and_answers:
                        questions_and_answers[source] = qa_json
                    else:
                        questions_and_answers[source].extend(qa_json)
                    break
                except Exception as e:
                    error_times += 1
                    if error_times == 1:
                        logger.error(f"第{i}个文档错误{source}:\n{text}")
                    logger.error(f"原因{e},正在尝试第{error_times}次重试")

        # 保存json
        if source != last_source:
            if last_source in questions_and_answers:
                file_name = Path(last_source).name.split(".")[0]
                with open(f"{save_dir}/{file_name}.json", mode="w") as f:
                    json.dump(
                        questions_and_answers[last_source],
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )
            last_source = source
    # 保存最后一个json
    if last_source in questions_and_answers:
        file_name = Path(last_source).name.split(".")[0]
        with open(f"{save_dir}/{file_name}.json", mode="w") as f:
            json.dump(
                questions_and_answers[last_source],
                f,
                ensure_ascii=False,
                indent=4,
            )


if __name__ == "__main__":
    # save_doc2txt()
    save_QA_json(doc_dir="./data/text")
