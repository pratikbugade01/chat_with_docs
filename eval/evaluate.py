from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
apikey = os.getenv("GROQ_API_KEY")

if not apikey:
    raise ValueError("GROQ_API_KEY not found in environment variables")


def get_pdf(name_of_pdf):
    loader = PyPDFLoader(name_of_pdf)
    return loader.load()


def split_document(pdf):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(pdf)


def create_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.7}
    )
    return retriever


def initialize_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=apikey
    )


def create_prompt():
    return PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_eval_pipeline(pdf_path):
    pdf       = get_pdf(pdf_path)
    chunks    = split_document(pdf)
    retriever = create_retriever(chunks)
    llm       = initialize_llm()
    prompt    = create_prompt()
    parser    = StrOutputParser()

    def run_and_collect(question: str):
        retrieved_docs = retriever.invoke(question)
        context_text   = format_docs(retrieved_docs)
        filled_prompt  = prompt.invoke({"context": context_text, "question": question})
        answer_obj     = llm.invoke(filled_prompt)
        answer_text    = parser.invoke(answer_obj)
        return {
            "answer":   answer_text,
            "contexts": [doc.page_content for doc in retrieved_docs]
        }

    return run_and_collect


TEST_QUESTIONS = [
    "How does LangChain handle data before retrieval?",
    "What is the role of LangSmith in LangChain?",
    "What are the security concerns with LangChain?",
]


def main():
    PDF_PATH = "langchain_info.pdf"

    print("Building pipeline...")
    pipeline = build_eval_pipeline(PDF_PATH)

    print("Running questions through RAG...\n")
    samples = []

    for question in TEST_QUESTIONS:
        print(f"  Q: {question}")
        result = pipeline(question)
        print(f"  A: {result['answer'][:80]}...\n")

        sample = SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=apikey,
            n=1
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=judge_llm,
        embeddings=ragas_embeddings,
    )

    print("\n========== RAGAS SCORES ==========")
    df = results.to_pandas()
    print(df[["user_input", "faithfulness"]])

    os.makedirs("eval", exist_ok=True)
    df.to_csv("eval/ragas_results.csv", index=False)
    print("\nResults saved to eval/ragas_results.csv")


if __name__ == "__main__":
    main()