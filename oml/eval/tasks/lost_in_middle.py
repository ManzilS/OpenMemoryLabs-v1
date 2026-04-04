from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple
from oml.eval.base import EvalTask, EvalResult, ModelInterface
from oml.eval.run import register_task

@register_task("lost-in-middle")
class LostInMiddleTask(EvalTask):
    name = "lost-in-middle"

    def run(self, model: ModelInterface, config: dict[str, Any]) -> EvalResult:
        """
        Runs the Lost in the Middle test.
        Config options:
          - context_length: int (length of haystack)
          - needle: str (the fact to find)
        """
        context_length = config.get("context_length", 1000)
        needle = config.get(
            "needle",
            "The historic Blue Mountains crossing by Blaxland, Lawson and Wentworth occurred in 1813.",
        )

        # Positions to test: 0% (start), 50% (middle), 100% (end)
        positions = [0.0, 0.5, 1.0]
        results = {}

        # Wikipedia-derived filler — unrelated topic (Islamic calendar + Canadian Forces + Ovambo)
        # so the needle about the 1813 Blue Mountains crossing stands out clearly.
        _WIKI_FILLER = (
            "The Islamic calendar, also known as the Hijri calendar, is a purely lunar calendar "
            "consisting of twelve months in a year of 354 days. Being a purely lunar calendar, "
            "it is not synchronised with the seasons. The months are Muharram, Safar, "
            "Rabi al-Awwal, Rabi al-Thani, Jumada al-Awwal, Jumada al-Thani, Rajab, Shaban, "
            "Ramadan, Shawwal, Dhul Qadah, and Dhul Hijjah. Each month begins with the sighting "
            "of the new crescent moon. Major religious observances including Ramadan and both "
            "Eid celebrations are timed according to this calendar. The Islamic year is "
            "approximately eleven days shorter than the solar year, so its months rotate through "
            "all seasons over a cycle of roughly 33 solar years. The Ovambo are the largest "
            "ethnic group in Namibia, comprising approximately 49 percent of the total population. "
            "They traditionally inhabit the northern regions of Namibia and southern Angola and "
            "speak Bantu languages belonging to the Niger-Congo family. The Canadian Armed Forces "
            "consist of the Maritime Command responsible for naval operations, the Air Command "
            "overseeing all air operations, and the Canadian Special Operations Forces Command "
            "which handles special operations worldwide. Empusa was a shape-shifting spirit in "
            "Greek mythology who served the goddess Hecate and could appear as a donkey, a dog, "
            "or a beautiful woman to deceive travellers on lonely roads. Otto Klemperer was a "
            "celebrated German conductor born in Breslau in 1885 who was renowned for his "
            "interpretations of the German classical and Romantic repertoire. "
        )
        filler = _WIKI_FILLER * 4  # ensure sufficient length at all context_length values
        filler = filler[:context_length]
        
        correct_count = 0

        def _probe(pos_pct: float) -> Tuple[float, bool, str]:
            insert_idx = int(len(filler) * pos_pct)
            prompt = filler[:insert_idx] + f"\n{needle}\n" + filler[insert_idx:]
            query = (
                "According to the context, in what year did Blaxland cross the Blue Mountains? "
                "Respond with only the year.\nContext:\n" + prompt + "\nAnswer:"
            )
            output = model.generate(query)
            return pos_pct, "1813" in output, output[:50]

        # All three position probes are independent — run concurrently
        with ThreadPoolExecutor(max_workers=len(positions)) as pool:
            probe_results = list(pool.map(_probe, positions))

        for pos_pct, is_correct, snippet in probe_results:
            results[f"pos_{int(pos_pct*100)}"] = {
                "correct": is_correct,
                "output_snippet": snippet,
            }
            if is_correct:
                correct_count += 1

        score = correct_count / len(positions)
        
        return EvalResult(
            task_name=self.name,
            score=score,
            details=results
        )
