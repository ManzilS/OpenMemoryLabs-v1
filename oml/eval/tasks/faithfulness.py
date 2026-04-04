from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple
from oml.eval.base import EvalTask, EvalResult, ModelInterface
from oml.eval.run import register_task

@register_task("faithfulness")
class FaithfulnessTask(EvalTask):
    name = "faithfulness"

    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        """
        Runs faithfulness check on a synthetic set of (question, answer, context).
        """
        # Keep this compact and deterministic for unit tests.
        # Expected verdicts: YES, NO, NO
        dataset = [
            {
                "q": "Where did Alan Turing work during World War II to break enemy codes?",
                "a": "Alan Turing worked at Bletchley Park to decipher German Enigma codes.",
                "c": (
                    "Alan Turing was a British mathematician who worked at Bletchley Park "
                    "during World War II. He led efforts to break the German Enigma cipher, "
                    "which encrypted Nazi military communications."
                ),
                "expected": 1.0,
            },
            {
                "q": "Who led the 1963 Birmingham Campaign demonstrations?",
                "a": "The 1963 Birmingham Campaign was led by Malcolm X and the Nation of Islam.",
                "c": (
                    "The Birmingham Campaign of 1963 was a strategic civil rights movement "
                    "to confront racial segregation in Birmingham, Alabama. Nonviolent "
                    "demonstrators were met with fire hoses and police dogs by city authorities."
                ),
                "expected": 0.0,
            },
            {
                "q": "What could Empusa transform into according to Greek mythology?",
                "a": (
                    "Empusa could transform into a donkey, a dog, or a beautiful woman, "
                    "and she was worshipped in all Greek city-states."
                ),
                "c": (
                    "In Greek mythology, Empusa was a shape-shifting spirit who could appear "
                    "as a donkey, a dog, or a beautiful woman. She was a servant of the "
                    "goddess Hecate."
                ),
                "expected": 0.0,
            },
        ]
        
        score_sum = 0.0
        details = {}

        # Run all dataset items concurrently — each makes one independent LLM call
        def _eval_item(args: Tuple[int, dict]) -> Tuple[int, str, str]:
            idx, item = args
            verdict, reasoning = self._evaluate_single(model, item["q"], item["a"], item["c"])
            return idx, verdict, reasoning

        workers = min(len(dataset), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_eval_item, enumerate(dataset)))

        for i, verdict, reasoning in sorted(results):
            item = dataset[i]
            numeric_verdict = 1.0 if verdict == "YES" else 0.0
            is_correct = (numeric_verdict == item["expected"])
            score_sum += 1.0 if is_correct else 0.0
            details[f"example_{i}"] = {
                "q": item["q"],
                "verdict": verdict,
                "reasoning": reasoning,
                "expected_verdict": "YES" if item["expected"] == 1.0 else "NO",
                "pass": is_correct,
            }

        final_score = score_sum / len(dataset)
        return EvalResult(task_name=self.name, score=final_score, details=details)

    def _evaluate_single(self, model: ModelInterface, question: str, answer: str, context: str) -> tuple[str, str]:
        prompt = f"""You are a strict faithfulness judge. Your only job is to check whether the ANSWER is supported by the CONTEXT.

CRITICAL RULES — you MUST follow all of them:
1. Base your verdict SOLELY on what is EXPLICITLY written in the CONTEXT below.
2. Do NOT use any world knowledge, common knowledge, or facts you learned during training.
3. Do NOT infer, assume, or extrapolate beyond what the CONTEXT literally says.
4. If the ANSWER states something that the CONTEXT does not explicitly state, the verdict is NO — even if the statement is true in the real world.
5. If the ANSWER introduces any information absent from the CONTEXT, the verdict is NO.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

OUTPUT FORMAT:
Write one sentence of reasoning that cites only the CONTEXT text.
Then on a new line write exactly: VERDICT: YES  or  VERDICT: NO"""
        response = model.generate(prompt)

        verdict = "NO"
        if "VERDICT: YES" in response.upper():
            verdict = "YES"

        return verdict, response.strip()
