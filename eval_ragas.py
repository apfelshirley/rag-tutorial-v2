# https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a

from datasets import Dataset
import pandas as pd
from query_data import query_rag
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv
import os
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
import nest_asyncio
import asyncio
import tracemalloc

tracemalloc.start()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_TOKEN")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

qa_file_path = r'C:\Users\LenaFrischen(epaCC)\OneDrive - ePA-CC GmbH\Desktop\epa_Documents\word\Testdatenset.xlsm'
LOG_DIR = r'C:\Users\LenaFrischen(epaCC)\source\datascience\llm\logs'

qa_df = pd.read_excel(qa_file_path)[:]

questions = qa_df['Question'].tolist()
ground_truths = qa_df['Answer'].tolist()
answers = []
contexts = []

# Inference
async def process_questions(questions):
    for query in questions:
        answer, sources, context, duration = await query_rag(query)  # ÄNDERUNG: await hinzugefügt
        answers.append(answer)
        contexts.append(context)

# Starte die asynchrone Verarbeitung
loop = asyncio.get_event_loop()  # Hole die Event-Loop
loop.run_until_complete(process_questions(questions))

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Asynchronic calculation to suppress code exit
nest_asyncio.apply()

# metrics: context_recall, context_precision, faithfulness, answer_relevancy
result = evaluate(
    dataset = dataset,
    metrics=[
        context_precision,
        faithfulness
    ],
    llm = Ollama(model="mistral", temperature=0.5),
    embeddings = get_embedding_function()
)

df = result.to_pandas()

df.to_excel(LOG_DIR + r"\ragas_eval_chunking_separator_cs800_ol100_topk5_similarity_search_by vector.xlsx")