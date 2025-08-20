####################################################################
#                         import
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path

# Import openai and google_genai as main LLM services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

from langchain.schema import format_document

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Cohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere

# HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# Import streamlit
import streamlit as st

####################################################################
#              Config: LLM services, assistant language,...
####################################################################
list_LLM_providers = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
    ":hugging_face: **HuggingFace**",
]

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi heute?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

# Rutas base
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "vector_stores")

# 👉 Asegurar que existan los directorios base al iniciar
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#            Create app interface with streamlit
####################################################################
st.set_page_config(page_title="Chat With Your Data")
st.title("🤖 RAG chatbot")

# API keys (estado inicial)
st.session_state.openai_api_key = ""
st.session_state.google_api_key = ""
st.session_state.cohere_api_key = ""
st.session_state.hf_api_key = ""


def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "HuggingFace":
        st.session_state.hf_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.google_api_key = ""

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {LLM_provider} model", list_models
        )
        st.session_state.temperature = st.slider(
            "temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
        st.session_state.top_p = st.slider(
            "top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.05
        )


def sidebar_and_documentChooser():
    """UI: (1) crear vectorstore, (2) abrir vectorstore guardado (sin tkinter)."""
    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 LangChain, Cohere, OpenAI, Google Generative AI and 🤗"
        )
        st.write("")
        llm_chooser = st.radio(
            "Select provider",
            list_LLM_providers,
            captions=[
                "[OpenAI pricing page](https://openai.com/pricing)",
                "Rate limit: 60 requests per minute.",
                "**Free access.**",
            ],
        )

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                LLM_provider="OpenAI",
                text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
                list_models=[
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo",
                    "gpt-4-turbo-preview",
                ],
            )
        if llm_chooser == list_LLM_providers[1]:
            expander_model_parameters(
                LLM_provider="Google",
                text_input_API_key="Google API Key - [Get an API key](https://makersuite.google.com/app/apikey)",
                list_models=["gemini-pro"],
            )
        if llm_chooser == list_LLM_providers[2]:
            expander_model_parameters(
                LLM_provider="HuggingFace",
                text_input_API_key="HuggingFace API key - [Get an API key](https://huggingface.co/settings/tokens)",
                list_models=["mistralai/Mistral-7B-Instruct-v0.2"],
            )

        st.write("")
        st.session_state.assistant_language = st.selectbox(
            "Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retrievers")
        retrievers = list_retriever_types
        if st.session_state.selected_model == "gpt-3.5-turbo":
            # para gpt-3.5, evitamos el retriever base por tokens
            retrievers = list_retriever_types[:-1]
        st.session_state.retriever_type = st.selectbox("Select retriever type", retrievers)
        if st.session_state.retriever_type == list_retriever_types[0]:
            st.session_state.cohere_api_key = st.text_input(
                "Cohere API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                placeholder="insert your API key",
            )

        st.write("\n\n")
        st.write(
            f"ℹ _Your {st.session_state.LLM_provider} API key, '{st.session_state.selected_model}' parameters, "
            f"and {st.session_state.retriever_type} are only considered when loading or creating a vectorstore._"
        )

    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )

    # ---------- Tab 1: Crear vectorstore ----------
    with tab_new_vectorstore:
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Select documents**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )
        st.session_state.vector_store_name = st.text_input(
            label=(
                "**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). "
                "Please provide a valid dB name.**"
            ),
            placeholder="Vectorstore name",
        )
        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    # ---------- Tab 2: Abrir vectorstore existente ----------
    with tab_open_vectorstore:
        st.write(
            "Please select a Vectorstore directory (inside `data/vector_stores`) or paste a custom path."
        )

        # asegurar existencia y listar disponibles
        LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        available_vectorstores = [
            f.name for f in LOCAL_VECTOR_STORE_DIR.iterdir() if f.is_dir()
        ]

        st.session_state.selected_vectorstore_name = st.selectbox(
            "Choose a vectorstore",
            options=[""] + available_vectorstores,
            help="Vectorstores detectados en la carpeta local.",
        )

        custom_vectorstore_path = st.text_input(
            "Or paste a full path to a vectorstore",
            value="",
            placeholder="/absolute/path/to/your/vectorstore",
            help="Opcional: pega aquí una ruta absoluta si no está en la lista.",
        )

        if st.button("Load Vectorstore"):
            error_messages = []
            if (
                not st.session_state.openai_api_key
                and not st.session_state.google_api_key
                and not st.session_state.hf_api_key
            ):
                error_messages.append(
                    f"insert your {st.session_state.LLM_provider} API key"
                )
            if (
                st.session_state.retriever_type == list_retriever_types[0]
                and not st.session_state.cohere_api_key
            ):
                error_messages.append("insert your Cohere API key")

            # resolver ruta
            selected_vectorstore_path = None
            if st.session_state.selected_vectorstore_name:
                selected_vectorstore_path = LOCAL_VECTOR_STORE_DIR / st.session_state.selected_vectorstore_name
            elif custom_vectorstore_path.strip():
                selected_vectorstore_path = Path(custom_vectorstore_path.strip())

            if not selected_vectorstore_path or not selected_vectorstore_path.exists():
                error_messages.append("select a valid vectorstore path")

            if len(error_messages) == 1:
                st.session_state.error_message = "Please " + error_messages[0] + "."
                st.warning(st.session_state.error_message)
            elif len(error_messages) > 1:
                st.session_state.error_message = (
                    "Please "
                    + ", ".join(error_messages[:-1])
                    + ", and "
                    + error_messages[-1]
                    + "."
                )
                st.warning(st.session_state.error_message)
            else:
                with st.spinner("Loading vectorstore..."):
                    try:
                        embeddings = select_embeddings_model()
                        st.session_state.vector_store = Chroma(
                            embedding_function=embeddings,
                            persist_directory=selected_vectorstore_path.as_posix(),
                        )
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )
                        clear_chat_history()
                        st.session_state.selected_vectorstore_name = selected_vectorstore_path.name
                        st.info(f"**{st.session_state.selected_vectorstore_name}** is loaded successfully.")
                    except Exception as e:
                        st.error(e)


####################################################################
#        Process documents and create vectorstore (Chroma dB)
####################################################################
def delte_temp_files():
    """delete files from the './data/tmp' folder"""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def langchain_document_loader():
    """
    Create document loaders for PDF, TXT, CSV, DOCX.
    """
    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.csv",
        loader_cls=CSVLoader,
        show_progress=True,
        loader_kwargs={"encoding": "utf8"},
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents


def split_documents_to_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def select_embeddings_model():
    """Select embeddings model based on provider."""
    if st.session_state.LLM_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    if st.session_state.LLM_provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=st.session_state.google_api_key
        )
    if st.session_state.LLM_provider == "HuggingFace":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=st.session_state.hf_api_key, model_name="thenlper/gte-large"
        )
    return embeddings


def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="similarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v2.0",
    cohere_top_n=10,
):
    """
    Build the selected retriever on top of Chroma.
    """
    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever
    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings, base_retriever=base_retriever, k=compression_retriever_k
        )
        return compression_retriever
    elif retriever_type == "Cohere reranker":
        cohere_retriever = CohereRerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )
        return cohere_retriever
    else:
        return base_retriever


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever


def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    # 1) split
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")
    # 2) remove redundant
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    # 3) keep most relevant
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )
    # 4) reorder for long context
    reordering = LongContextReorder()
    # pipeline
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )
    return compression_retriever


def CohereRerank_retriever(
    base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=10
):
    compressor = CohereRerank(cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n)
    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever_Cohere


def chain_RAG_blocks():
    """Construye todo el pipeline: ingestión → vectorstore → retriever → chain + memory."""
    with st.spinner("Creating vectorstore..."):
        error_messages = []
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            error_messages.append(f"insert your {st.session_state.LLM_provider} API key")
        if st.session_state.retriever_type == list_retriever_types[0] and not st.session_state.cohere_api_key:
            error_messages.append("insert your Cohere API key")
        if not st.session_state.uploaded_file_list:
            error_messages.append("select documents to upload")
        if st.session_state.vector_store_name == "":
            error_messages.append("provide a Vectorstore name")

        if error_messages:
            if len(error_messages) == 1:
                st.session_state.error_message = "Please " + error_messages[0] + "."
            else:
                st.session_state.error_message = (
                    "Please " + ", ".join(error_messages[:-1]) + ", and " + error_messages[-1] + "."
                )
            return

        st.session_state.error_message = ""

        try:
            # 1) limpiar tmp
            delte_temp_files()
            # 2) asegurar tmp
            TMP_DIR.mkdir(parents=True, exist_ok=True)

            # 3) guardar archivos subidos en tmp
            if st.session_state.uploaded_file_list is not None:
                error_message = ""
                for uploaded_file in st.session_state.uploaded_file_list:
                    try:
                        temp_file_path = os.path.join(TMP_DIR.as_posix(), uploaded_file.name)
                        with open(temp_file_path, "wb") as temp_file:
                            temp_file.write(uploaded_file.read())
                    except Exception as e:
                        error_message += str(e)
                if error_message != "":
                    st.warning(f"Errors: {error_message}")

                # 4) cargar docs
                documents = langchain_document_loader()
                # 5) chunks
                chunks = split_documents_to_chunks(documents)
                # 6) embeddings
                embeddings = select_embeddings_model()

                # 7) crear carpeta de persistencia y vectorstore
                persist_path = LOCAL_VECTOR_STORE_DIR / st.session_state.vector_store_name
                persist_path.mkdir(parents=True, exist_ok=True)

                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_path.as_posix(),
                )
                st.info(f"Vectorstore **{st.session_state.vector_store_name}** is created successfully.")

                # 8) retriever
                st.session_state.retriever = create_retriever(
                    vector_store=st.session_state.vector_store,
                    embeddings=embeddings,
                    retriever_type=st.session_state.retriever_type,
                    base_retriever_search_type="similarity",
                    base_retriever_k=16,
                    compression_retriever_k=20,
                    cohere_api_key=st.session_state.cohere_api_key,
                    cohere_model="rerank-multilingual-v2.0",
                    cohere_top_n=10,
                )

                # 9) chain + memory
                (st.session_state.chain, st.session_state.memory) = create_ConversationalRetrievalChain(
                    retriever=st.session_state.retriever,
                    chain_type="stuff",
                    language=st.session_state.assistant_language,
                )

                # 10) limpiar historial
                clear_chat_history()

        except Exception as error:
            st.error(f"An error occurred: {error}")


####################################################################
#                       Create memory
####################################################################
def create_memory(model_name="gpt-3.5-turbo", memory_max_token=None):
    """SummaryMemory para gpt-3.5; BufferMemory para el resto."""
    if model_name == "gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key,
                temperature=0.1,
            ),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    return memory


####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################
def answer_template(language="english"):
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template


def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="english",
):
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))
    memory = create_memory(st.session_state.selected_model)

    if st.session_state.LLM_provider == "OpenAI":
        standalone_query_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
        response_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
        )
    if st.session_state.LLM_provider == "Google":
        standalone_query_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
            convert_system_message_to_human=True,
        )
        response_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            convert_system_message_to_human=True,
        )
    if st.session_state.LLM_provider == "HuggingFace":
        standalone_query_generation_llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={"temperature": 0.1, "top_p": 0.95, "do_sample": True, "max_new_tokens": 1024},
        )
        response_generation_llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "do_sample": True,
                "max_new_tokens": 1024,
            },
        )

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


def clear_chat_history():
    """clear chat history and memory."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        if st.session_state.LLM_provider == "HuggingFace":
            start = answer.find("\nAnswer: ")
            if start != -1:
                answer = answer[start + len("\nAnswer: ") :]

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("**Source documents**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Page: " + str(document.metadata.get("page")) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Source: " + str(document.metadata.get("source")) + page + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"
                st.markdown(documents_content)
    except Exception as e:
        st.warning(e)


####################################################################
#                         Chatbot
####################################################################
def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            st.info(f"Please insert your {st.session_state.LLM_provider} API key to continue.")
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()
