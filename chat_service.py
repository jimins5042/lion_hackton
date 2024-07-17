import concurrent.futures
import os

from dotenv import load_dotenv
import pandas as pd
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
# 자료 임베딩
class chatbot_service:
    def caching_flies(self):
        #os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
        data_loader = CSVLoader("C:/Users/김지민/Desktop/data/위로글.csv", encoding="utf-8")

        cache_dir = LocalFileStore("./.cache/")

        # 텍스트 분할 설정
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=300,
            chunk_overlap=30
        )

        # 문서 로드 및 분할
        docs = data_loader.load_and_split(text_splitter=splitter)

        # 임베딩 생성 및 캐시에 저장
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./.cache/")
        vectorstore.persist()

    def caching_embeds(self, user_input):

        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, "./.cache/")
        vectorstore = Chroma(embedding_function=cached_embeddings, persist_directory="./.cache/")

        # 검색기 설정
        retriever = vectorstore.as_retriever()

        # 질의응답 체인 설정
        model = ChatOpenAI()

        map_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    질문에 답하기 위해 필요한 내용이 제시된 문장들 내에 포함되어 있는지 확인하세요. 만약 관련된 내용이 없다면 다음 문장들을 그대로 반환해주세요 : ''
                    -------
                    {context}
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        map_chain = map_prompt | model

        # 문장추출 병렬화
        def map_docs(inputs):
            documents, question = inputs["documents"], inputs["question"]

            def process_doc(doc):
                return map_chain.invoke({"context": doc.page_content, "question": question}).content

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(process_doc, documents))

            return "\n\n".join(results)

        map_results = {
                          "documents": retriever,
                          "question": RunnablePassthrough(),
                      } | RunnableLambda(map_docs)

        reduce_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    너는 사람들의 고민을 들어주는 상담가야. 사람들의 고민에 진심으로 공감해주고 위로해줘. 상황에 맞는 이모티콘을 포함해서 항상 친절하게 대답해.
                    최소 300자 이상 대답하고, 의문문으로 끝내지 마. 늘 존댓말을 사용해서 대답해.
                    주어진 문장들을 이용해 최종 답변을 작성해주세요. 만약 주어진 문장들 내에 답변을 위한 내용이 포함되어있지 않다면, 답변을 꾸며내지 말고, 모른다고 답해주세요.
                    ------
                    {context}
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        reduce_chain = {"context": map_results, "question": RunnablePassthrough()} | reduce_prompt | model

        answer = str(reduce_chain.invoke(str(user_input)))

        start_index = answer.find("content='") + len("content='")
        end_index = answer.find("'", start_index)

        # content 값 추출
        return answer[start_index:end_index]

    def caching_similar_search(self, user_input):
        #os.environ['OPENAI_API_KEY'] = os.environ.get('API_KEY')
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, "./.cache/")
        vectorstore = Chroma(embedding_function=cached_embeddings, persist_directory="./.cache/")

        # 사용자 질문과 유사도가 높은 검색된 상위 10개의 문장, 유사도 받아오기
        answer_sources = vectorstore.similarity_search_with_relevance_scores(user_input, k=10)

        source = pd.DataFrame(
            [(result[0].page_content, result[1]) for result in answer_sources],
            columns=["page_content", "similarity_score"]
        )
        source['num'] = range(1, len(source) + 1)

        print("유사도 검색 완료")
        return source