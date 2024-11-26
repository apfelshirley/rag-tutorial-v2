import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
import time
import asyncio

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Du bist ein digitaler Assistent namens ePArat. Nachfolgend erhältst du einen Kontext.
Wenn Du eine Frage nicht beantworten kannst, verweise auf den Support (support@epa-cc.de) Hier ist der Kontext:

{context}

---

Beantworte die Frage basierend auf dem obenstehenden Kontext in höchstens 50 Wörtern in deutscher Sprache: {question}
"""


def main():
    # Create CLI.
    print("Willkommen! Ich bin der ePArat. Stelle mir eine Frage und ich werde versuchen, sie zu beantworten.")

    while True:
        query_text = str(input("Bitte gib deine Frage ein:\n\n"))
        if query_text.lower().endswith("ende"):
            print("Programm wird beendet. Auf Wiedersehen!")
            break
        answer, sources, contexts, duration = query_rag(query_text)
        print("\nANTWORT:\n", answer, "\n")
        print("Quellen:\n" + "\n".join(sources) + "\n\n")
        print(f"Dauer: {duration:.2f}")



async def query_rag(query_text: str):
    await asyncio.sleep(1)
    start = time.time()
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    query_vec = embedding_function.embed_query(query_text)


    # Search the DB.
    results = results = db.similarity_search_by_vector(query_vec, k=5)
    # results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = Ollama(model="mistral", temperature=0.5, top_k=20, top_p=0.5)
    model = Ollama(model="mistral", temperature=0.5)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc in results]
    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    contexts = [doc.page_content for doc in results]
    # contexts = [doc.page_content for doc, _score in results]
    stop = time.time()
    duration = stop-start
    return response_text, sources, contexts, duration


if __name__ == "__main__":
    main()
