import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from datasets import Dataset
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    NoiseSensitivity,
    ResponseRelevancy,
)

from modules.llm.generation import LLM_API_URL, generate_response
from modules.llm.generation import MODEL_NAME as LLM_MODEL_NAME
from modules.llm.prompts import get_prompt
from modules.llm.retrieval import MODEL_NAME as EMBEDDER_MODEL_NAME
from modules.llm.retrieval import retrieve_k_most_similar_chunks

CHUNK_SEPARATOR = "\n\n" + "####" * 20 + "\n\n"
GENERATION_KWARGS = {"temperature": 0.2, "min_p": 0.1}


class CustomLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""
        response = requests.post(
            LLM_API_URL,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"


def load_golden_dataset(filepath):
    df = pd.read_json(filepath)
    return df.to_dict(orient="records")


def evaluate_chatbot_responses(golden_dataset):
    evaluator_llm = LangchainLLMWrapper(CustomLLM())
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        NoiseSensitivity(llm=evaluator_llm, max_retries=5),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    for item in golden_dataset:
        print("Loading chunks....")
        question = item["question"]
        ground_truth = item["answer"]
        chunks = retrieve_k_most_similar_chunks(question, k=5)
        joined_chunks = (
            "####" * 20 + "\n\n" + CHUNK_SEPARATOR.join(chunks) + "\n\n" + "####" * 20
        )

        prompt = get_prompt().format(
            user_question=question,
            chunks=joined_chunks,
        )
        print("Generating response....")

        answer = generate_response(
            prompt, model_name=LLM_MODEL_NAME, **GENERATION_KWARGS
        )["choices"][0]["message"]["content"]
        questions.append(question)
        answers.append(answer)
        contexts.append(chunks)
        ground_truths.append(ground_truth)

    data_samples = {
        "llm_for_generation": LLM_MODEL_NAME,
        "embedder_model": EMBEDDER_MODEL_NAME,
        "question": questions,
        "answer": answers,
        "ground_truth": ground_truths,
        "contexts": contexts,
        **GENERATION_KWARGS,
    }

    # save the dataset as json
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    with open(f"data/{timestamp}_evaluation_results.json", "w") as f:
        json.dump(data_samples, f)

    print("Evaluation results saved to disk.")

    dataset = Dataset.from_dict(data_samples)
    evaluation_results = evaluate(dataset, metrics=metrics)
    return evaluation_results


if __name__ == "__main__":
    golden_dataset = load_golden_dataset("data/golden_dataset.json")
    evaluation_results = evaluate_chatbot_responses(golden_dataset)
    print(evaluation_results)
    print("Evaluation completed successfully.")
