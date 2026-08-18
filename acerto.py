import sys
from pathlib import Path

import pandas as pd

from src.evaluation import EvaluationMetrics, evaluate_predictions
from src.evaluation.validator import validate_answer
from src.models.question import Question
from src.models.enums import DifficultyLevel


def ler_dataset(csv_path: str = "dataset-600questoes_respondido.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def equalizar_resposta(df: pd.DataFrame) -> pd.DataFrame:
    import re

    def limpar_valor(texto):
        if pd.isna(texto):
            return texto

        texto = str(texto)
        texto = re.sub(r"[*#$]", "", texto)

        if "=" in texto:
            texto = texto.split("=")[-1]

        match = re.findall(r"\d+(?:[.,]\d+)?", texto)
        if match:
            return match[-1]
        return texto

    df["resposta"] = df["resposta"].apply(limpar_valor)
    return df


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "dataset-600questoes_respondido.csv"
    path = Path(csv_path)

    if not path.exists():
        print(f"Arquivo não encontrado: {csv_path}")
        return

    df = ler_dataset(str(path))
    df = equalizar_resposta(df)

    # Normalize for comparison
    df["Comparação_norm"] = pd.to_numeric(
        df["Comparação"].astype(str).str.replace(",", "."), errors="coerce"
    )
    df["resposta_norm"] = pd.to_numeric(
        df["resposta"].astype(str).str.replace(",", "."), errors="coerce"
    )

    print("Amostra dos dados normalizados:")
    print(df[["Comparação_norm", "resposta_norm"]].head(10))

    # Original exact match
    acertos = (df["Comparação_norm"] == df["resposta_norm"]).sum()
    total = len(df)
    porcentagem = (acertos / total) * 100

    print(f"\n=== Métrica Original (Exact Match Numérico) ===")
    print(f"Acertos: {acertos} de {total}")
    print(f"Porcentagem: {porcentagem:.2f}%")

    # New evaluation with tolerance
    questions = []
    predictions = []
    expected = []

    for _, row in df.iterrows():
        q = Question(
            id=int(row.get("ID", 0)) or None,
            text=str(row.get("Question", row.get("Questão", ""))),
            level=DifficultyLevel.from_str(str(row.get("Level", ""))),
            question_type=str(row.get("Type", "")),
            answer=str(row.get("resposta", "")),
        )
        questions.append(q)
        predictions.append(str(row.get("resposta", "")))
        expected.append(str(row.get("Comparação", "")))

    # Evaluate with validator (tolerance-based)
    metrics = EvaluationMetrics()
    for q, pred, exp in zip(questions, predictions, expected):
        metrics.add_result(q, pred, exp, latency=0.0, tolerance=0.01)

    print(metrics.summary())

    # Also show numeric tolerance breakdown
    df["numeric_match"] = df.apply(
        lambda r: abs(r["Comparação_norm"] - r["resposta_norm"]) / max(abs(r["Comparação_norm"]), 1e-9) <= 0.01
        if pd.notna(r["Comparação_norm"]) and pd.notna(r["resposta_norm"])
        else False,
        axis=1,
    )
    numeric_correct = df["numeric_match"].sum()
    print(f"\n=== Com Tolerância 1% (Numérico) ===")
    print(f"Acertos: {numeric_correct} de {total} ({numeric_correct/total:.2%})")

    # Per-level breakdown
    if "Level" in df.columns:
        print("\n--- Por Nível (Tolerância 1%) ---")
        for level in df["Level"].unique():
            subset = df[df["Level"] == level]
            acc = subset["numeric_match"].mean()
            print(f"  {level}: {subset['numeric_match'].sum()}/{len(subset)} ({acc:.2%})")


if __name__ == "__main__":
    main()