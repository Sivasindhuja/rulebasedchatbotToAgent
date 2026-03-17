import pandas as pd
from datasets import Dataset

from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall
)

from ragas import evaluate

# import your RAG pipeline
from rag import ask_question


# Load golden dataset
df = pd.read_csv("golden_dataset.csv")

questions = []
answers = []
contexts = []
ground_truths = []

print("Running RAG evaluation...\n")

for index, row in df.iterrows():

    question = row["question"]
    ground_truth = row["answer"]

    print(f"Processing {index+1}/{len(df)}")

    # run your RAG pipeline
    answer, docs = ask_question(question)

    retrieved_contexts = [doc.page_content for doc in docs]

    questions.append(question)
    answers.append(answer)
    contexts.append(retrieved_contexts)
    ground_truths.append(ground_truth)


# Prepare dataset for RAGAS
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(data)

# Run evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_correctness,
        context_precision,
        context_recall
    ]
)

print("\nEvaluation Results:\n")
print(result)