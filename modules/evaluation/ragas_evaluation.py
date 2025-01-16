from ragas import evaluate
import requests
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ContextPrecision,
    NoiseSensitivity,
    ResponseRelevancy,
)
from modules.llm.generation import LLM_API_URL, generate_response, MODEL_NAME
from modules.llm.prompts import get_prompt
from modules.llm.retrieval import retrieve_k_most_similar_chunks
import pandas as pd
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

CHUNK_SEPARATOR = "\n\n" + "####" * 20 + "\n\n"


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
                "model": MODEL_NAME,
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
        answer = generate_response(prompt)["choices"][0]["message"]["content"]
        questions.append(question)
        answers.append(answer)
        contexts.append(chunks)
        ground_truths.append(ground_truth)

    data_samples = {
        "question": questions,
        "answer": answers,
        "ground_truth": ground_truths,
        "contexts": contexts,
    }
    dataset = Dataset.from_dict(data_samples)
    evaluation_results = evaluate(dataset, metrics=metrics)
    return evaluation_results


if __name__ == "__main__":
    golden_dataset = load_golden_dataset("data/golden_dataset.json")
    evaluation_results = evaluate_chatbot_responses(golden_dataset)
    print(evaluation_results)
