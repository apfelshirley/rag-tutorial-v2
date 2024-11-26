import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
import time
import pandas as pd

CHROMA_PATH = "chroma"
LOG_DIR = r'C:\Users\LenaFrischen(epaCC)\source\datascience\llm\logs'

PROMPT_TEMPLATE = """
Beantworte die Frage basierend auf dem folgenden Kontext:

{context}

---

Beantworte die Frage basierend auf dem obenstehenden Kontext mit höchstens 60 Wörtern: {question}
"""
# questions = ["Was ist das epaSystem?", "Was ist epaSOLUTIONS Management?", "Was zeigt die Auswertung Sturzrisiko?", "Wie heißt du?", "What is epaSOLUTIONS Management?"]
questions = ["What is the dashboard?", "What is the SPI?", "What are valid cases?", "What does the continence history show?", "What is the diagnosis U50?"]
temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0]
# models = ["mistral", "llama3.2:1b", "llama3.2", "phi3.5"]
models = ["mistral", "llama3.2"]


def main():
    df = pd.DataFrame(columns=["Model", "Frage", "Antwort", "Quellen", "Dauer"])

    for i in range(len(questions)):
        for j in range(len(models)):
            query_text = questions[i]
            answer, sources, duration = query_rag(query_text, 0.5, models[j])
            df.loc[(i*len(models))+j] = [models[j], questions[i], answer, sources, duration]

    df.to_excel(str(LOG_DIR+"\different_models_english_only_Tooltip.xlsx"))



def query_rag(query_text: str, temp: float, model: str):
    start = time.time()
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=6)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model=model, temperature=temp)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    stop = time.time()
    duration = stop-start
    return response_text, sources, duration


if __name__ == "__main__":
    main()
