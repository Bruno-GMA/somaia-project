from src.models.question import RetrievedExample


def build_prompt(question: str, similar: list[RetrievedExample]) -> str:
    examples = ""
    for i, doc in enumerate(similar, 1):
        q_text = doc.chunk_text or doc.question.text
        examples += f"\nExemplo {i}:\nQuestão: {q_text}\nResposta: {doc.question.answer}\n"

    system = (
        "Você é um especialista em matemática. "
        "Resolva a questão passo a passo e apresente apenas a resposta final de forma clara. "
        "Seja preciso e conciso."
    )

    if examples:
        return (
            f"{system}\n\n"
            f"A seguir, alguns exemplos de questões semelhantes resolvidas:{examples}\n"
            f"Agora resolva:\nQuestão: {question}\nResposta:"
        )
    return f"{system}\n\nQuestão: {question}\nResposta:"
