from enum import StrEnum


class DifficultyLevel(StrEnum):
    SIMPLE = "Simples"
    INTERMEDIATE = "Intermediário"
    COMPLEX = "Complexo"
    UNKNOWN = "Unknown"

    @classmethod
    def from_str(cls, value: str) -> "DifficultyLevel":
        v = value.strip().lower()
        if v in ("simples", "simple"):
            return cls.SIMPLE
        if v in ("intermediario", "intermediário", "intermediate"):
            return cls.INTERMEDIATE
        if v in ("complexo", "complex"):
            return cls.COMPLEX
        return cls.UNKNOWN


class QuestionType(StrEnum):
    NON_CONTEXTUALIZED = "Non-contextualized"
    CONTEXTUALIZED = "Contextualized"
    UNKNOWN = "Unknown"

    @classmethod
    def from_str(cls, value: str) -> "QuestionType":
        v = value.strip().lower()
        if v in ("non-contextualized", "nao-contextualizada", "não-contextualizada"):
            return cls.NON_CONTEXTUALIZED
        if v in ("contextualized", "contextualizada"):
            return cls.CONTEXTUALIZED
        return cls.UNKNOWN

