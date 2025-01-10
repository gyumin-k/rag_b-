import streamlit as st #
import tiktoken # 청크를 나눌 때 문자의 개수를 무엇을 기준으로 산정할 것인가? -> 토큰 토큰 개수를 세기 위한 라이브러리 

from langchain.chains import ConversationalRetrievalChain # 메모리를 가진 체인 사용 구현
from langchain.chat_models import ChatOpenAI # OpenAI의 LLM 모델 사용

from langchain.document_loaders import PyPDFLoader # 여러파일을 로드할 수 있는 로더들 
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter # 문맥에 따라 스플릿하게끔 
from langchain_openai import OpenAIEmbeddings # 다국어 모델 임베딩으로 

from langchain.memory import ConversationBufferMemory # 몇개까지의 대화를 메모리로 넣어줄 것인
from langchain.vectorstores import FAISS # 벡터스토어는 임시벡터저장소 사용 속도빠름. 

from langchain.callbacks import get_openai_callback # 메모리를 구현하는 데 필요한 라이브러리 1 
from langchain.memory import StreamlitChatMessageHistory # 2

def main():
    st.set_page_config(
    page_title="B+",
    page_icon=":books:")

    st.title("_Private Data :red[test B+]_ :books:")

    if "conversation" not in st.session_state: # session_state.conversation 미리 정의 해주기, 에러 방지용 
        st.session_state.conversation = None

    if "chat_history" not in st.session_state: # session_state.chat_history 미리 정의 해주기, 에러 방지용 
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state: # session_state.processComplete 미리 정의 해주기, 에러 방지용 
        st.session_state.processComplete = None

    with st.sidebar: # 구성요소 안에 하위 구성요소가 집행되어야 하는 경우 with 구문 사용 ex)사이드바 넣기 
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process: # process 버튼을 누르면 
        if not openai_api_key: # open_api가 입력이 안되있으면 
            st.info("Please add your OpenAI API key to continue.") # api 입력하세요 
            st.stop()
        files_text = get_text(uploaded_files) # 입력이 되어있으면 업로드된 파일을 텍스트로 변환
        text_chunks = get_text_chunks(files_text) # 텍스트를 청크로 나눈다.
        vetorestore = get_vectorstore(text_chunks) # 청크를 벡터화한다.
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) # 벡터스토어를 이용하여
# llm이 답변할 수 있도록 체인 구성
        st.session_state.processComplete = True

    if 'messages' not in st.session_state: # 스트림릿 페이지의 인사말 UI
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "B+은 맞자!"}] 

    for message in st.session_state.messages: # 메세지들 마다 with 구문으로 묶어준다.
        with st.chat_message(message["role"]): # 메세지의 롤에 따라서 
            st.markdown(message["content"]) # 마크다운으로 묶어줄 것이다.

    history = StreamlitChatMessageHistory(key="chat_messages") # 히스토리 부분을 정의해야 메모리를 가지고 컨텍스트를 고려

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base") # openai의 llm을 활용하기 때문에 cl100k_base사용 
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs): # 텍스트로 변환 

    doc_list = [] # 여러개의 파일을 처리해야함 list 생성 
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장. 처음엔 빈파일 일 것 
            file.write(doc.getvalue()) # 빈파일에다가 업로드된 밸류를 넣겠다. -> 업로드 된 파일에 대한 값을 저장할 수 있음.
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name: 
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=openai_api_key
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
