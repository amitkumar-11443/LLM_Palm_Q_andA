# to load info from .env

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


from langchain.llms import GooglePalm
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5)  # 0-1




instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vector_db_file_path='faiss_index'
# in order to not use vector db in momory as it taken longer time, for every streamlite launch
# so will save the vector db into the disc
# creat embedding and vector database


def create_vector_db():
    loader = CSVLoader(file_path='faqs.csv', source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vectordb.save_local(vector_db_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vector_db_file_path,instructor_embeddings) #Loading the vector db from the file, 2 arguments

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold = 0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",    # There are 2 type "Stuff" and "Map Reduce"
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)
    
    return chain
        

if __name__ == "__main__":
    #get_qa_chain()
    chain = get_qa_chain()
    #create_vector_db()   # while creating the vector db

    print(chain("near By schools?"))



            




