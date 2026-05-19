import pandas as pd
import csv
#regex para limpar a resposta
import re

def ler_dataset():
  # Colocar o dataset que foi respondido pelo modelo
  df = pd.read_csv('dataset-600questoes_respondido.csv')
  return df

def equalizar_resposta(df):
  def limpar_valor(texto):
      if pd.isna(texto):
          return texto

      texto = str(texto)
      texto = re.sub(r'[*#$]', '', texto)

      if '=' in texto:
          texto = texto.split('=')[-1]

      match = re.findall(r'\d+(?:[.,]\d+)?', texto)
      if match:
          return match[-1]
      return texto

  df['resposta'] = df['resposta'].apply(limpar_valor)
  return df

if __name__ == "__main__":
  df = ler_dataset()
  df = equalizar_resposta(df)

  df['Comparação_norm'] = pd.to_numeric(df['Comparação'].astype(str).str.replace(',', '.'), errors='coerce')
  df['resposta_norm'] = pd.to_numeric(df['resposta'].astype(str).str.replace(',', '.'), errors='coerce')

  print("Amostra dos dados normalizados:")
  print(df[['Comparação_norm', 'resposta_norm']].head(10))

  acertos = (df['Comparação_norm'] == df['resposta_norm']).sum()
  total = len(df)
  porcentagem = (acertos / total) * 100

  print(f'\nAcertos: {acertos} de {total}')
  print(f'\nPorcentagem de acertos: {porcentagem:.2f}%')
