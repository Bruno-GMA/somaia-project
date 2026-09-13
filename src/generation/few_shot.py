from collections import defaultdict

from src.models.question import RetrievedExample


class FewShotSelector:
    def __init__(
        self,
        max_per_level: int = 1,
        max_per_type: int = 1,
        min_distance_threshold: float = 0.8,
    ):
        self.max_per_level = max_per_level
        self.max_per_type = max_per_type
        self.min_distance_threshold = min_distance_threshold

    def select(self, examples: list[RetrievedExample], top_k: int) -> list[RetrievedExample]:
        if not examples:
            return []

        # Group by level and type
        by_level: dict[str, list[RetrievedExample]] = defaultdict(list)
        by_type: dict[str, list[RetrievedExample]] = defaultdict(list)

        for ex in examples:
            level = ex.question.level.value if ex.question.level else "Unknown"
            qtype = ex.question.question_type or "Unknown"
            by_level[level].append(ex)
            by_type[qtype].append(ex)

        # Sort each group by distance (ascending = more similar first)
        for group in by_level.values():
            group.sort(key=lambda x: x.distance)
        for group in by_type.values():
            group.sort(key=lambda x: x.distance)

        # Select diverse examples
        selected: list[RetrievedExample] = []
        seen_ids = set()

        # First pass: pick best from each level
        for level, group in by_level.items():
            if len(selected) >= top_k:
                break
            for ex in group[:self.max_per_level]:
                if ex.question.id not in seen_ids and len(selected) < top_k:
                    selected.append(ex)
                    seen_ids.add(ex.question.id)

        # Second pass: pick best from each type
        for qtype, group in by_type.items():
            if len(selected) >= top_k:
                break
            for ex in group[:self.max_per_type]:
                if ex.question.id not in seen_ids and len(selected) < top_k:
                    selected.append(ex)
                    seen_ids.add(ex.question.id)

        # Third pass: fill remaining with closest
        for ex in examples:
            if len(selected) >= top_k:
                break
            if ex.question.id not in seen_ids:
                selected.append(ex)
                seen_ids.add(ex.question.id)

        return selected[:top_k]


class CoTPromptBuilder:
    def __init__(self) -> None:
        self.system_prompt = (
            "Você é um especialista em matemática. "
            "Resolva a questão passo a passo, mostrando todo o raciocínio. "
            "Apresente a resposta final de forma clara e concisa no final. "
            "Seja preciso e rigoroso."
        )

    def build(self, question: str, examples: list[RetrievedExample]) -> str:
        if not examples:
            return f"{self.system_prompt}\n\nQuestão: {question}\n\nResposta:"

        examples_text = ""
        for i, ex in enumerate(examples, 1):
            q_text = ex.chunk_text or ex.question.text
            answer = ex.question.answer or ""
            examples_text += f"\n--- Exemplo {i} ---\nQuestão: {q_text}\nResposta: {answer}\n"

        return (
            f"{self.system_prompt}\n\n"
            f"A seguir, exemplos de questões semelhantes resolvidas passo a passo:{examples_text}\n"
            f"---\nAgora resolva a seguinte questão, seguindo o mesmo formato:\n"
            f"Questão: {question}\n\n"
            f"Raciocínio passo a passo:\n"
        )

    def build_simple(self, question: str, examples: list[RetrievedExample]) -> str:
        """Simpler prompt for basic arithmetic."""
        if not examples:
            return f"{self.system_prompt}\n\nQuestão: {question}\nResposta:"

        examples_text = ""
        for i, ex in enumerate(examples, 1):
            q_text = ex.chunk_text or ex.question.text
            answer = ex.question.answer or ""
            examples_text += f"\nExemplo {i}:\nQuestão: {q_text}\nResposta: {answer}\n"

        return (
            f"{self.system_prompt}\n\n"
            f"Exemplos:{examples_text}\n"
            f"Questão: {question}\nResposta:"
        )


def build_prompt_adaptive(
    question: str,
    examples: list[RetrievedExample],
    selector: FewShotSelector | None = None,
    top_k: int = 3,
    use_cot: bool = True,
) -> str:
    if selector:
        examples = selector.select(examples, top_k)

    builder = CoTPromptBuilder()
    if use_cot:
        return builder.build(question, examples)
    return builder.build_simple(question, examples)
