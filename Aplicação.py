# Criar a aplicação para receber os dados
# organizar as classes necessárias na aplicação
# instaciar o modelo "modelo_xgb" (joblib)
# predict :)

# Libraries to import
import streamlit as st
import pandas as pd
import joblib
from joblib import load
from utils import MinMax  
from sklearn.pipeline import Pipeline

st.write('# Aplicação para prever evasão de estudantes 🔮')

st.warning('Preencha o formulário com todas as informações do estudante e clique em **ENVIAR** no final da página.')

# Criar um dicionário para armazenar os valores dos inputs
dados_estudante = {
    'Devedor': [],
    'TaxaDesemprego': [],
    'MensalidadesEmDia': [],
    'Bolsista': [],
    'UnidadesCurriculares1SemestreAprovado': [],
    'UnidadesCurriculares1SemestreAvaliacoes': [],
    'UnidadesCurriculares1SemestreGrau': [],
}

# Estudante Devedor
st.write('### O estudante é inadimplente?')
input_inadimplente = st.radio('O estudante está devendo débitos?', ['Sim', 'Não'])
dados_estudante['Devedor'].append(1 if input_inadimplente == 'Sim' else 0)

# Taxa de desemprego
st.write('### Qual é a taxa de desemprego do estudante? Selecione um valor de 0 até 20.')
input_taxa_desemprego = float(st.slider('Selecione a taxa', 0, 20))
dados_estudante['TaxaDesemprego'].append(input_taxa_desemprego)

# Mensalidades em dia
st.write('### O estudante possui as mensalidades em dia?')
input_mensalidade = st.radio('As mensalidades estão em dia?', ['Sim', 'Não'])
dados_estudante['MensalidadesEmDia'].append(1 if input_mensalidade == 'Sim' else 0)

# Estudante é bolsista?
st.write('### O estudante possui bolsa?')
input_bolsa = st.radio('Possui bolsa?', ['Sim', 'Não'])
dados_estudante['Bolsista'].append(1 if input_bolsa == 'Sim' else 0)

# Desempenho do 1º semestre
st.write('## Desempenho do 1º semestre: 📝')

# Número de unidades curriculares aprovadas no 1.º semestre
st.write('### Número de unidades curriculares aprovadas:')
input_unidades_curriculares_aprovadas = float(st.text_input('Digite o número de unidades curriculares aprovadas no 1º semestre e pressione ENTER para confirmar', 0))
dados_estudante['UnidadesCurriculares1SemestreAprovado'].append(input_unidades_curriculares_aprovadas)

# Número de avaliações no 1º semestre
st.write('### Número de unidades curriculares de avaliações:')
input_unidades_curriculares_avaliacao = float(st.text_input('Digite o número de unidades curriculares de avaliações no 1º semestre e pressione ENTER para confirmar', 0))
dados_estudante['UnidadesCurriculares1SemestreAvaliacoes'].append(input_unidades_curriculares_avaliacao)

# Média de notas no 1º semestre (entre 0 e 20)
st.write('### Média de notas no 1º semestre:')
input_media_notas = float(st.text_input('Digite a média de notas do 1º semestre e pressione ENTER para confirmar', 0))
dados_estudante['UnidadesCurriculares1SemestreGrau'].append(input_media_notas)

# Criar um DataFrame a partir do dicionário
df_estudante = pd.DataFrame(dados_estudante)

# Pipeline para preprocessar os dados
def pipeline(df):
    """
     Função para normalizar os dados.

     Argumentos:
         df: Insira um dataframe.

     Retorna:
         dataframe normalizado.
    """
    pipeline = Pipeline([
        ('min_max_scaler', MinMax())
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

#data_normalized = pipeline(df_estudante)

#Predições 
if st.button('Enviar'):
    model = joblib.load('modelo_xgb.joblib')
    final_pred = model.predict(df_estudante)
    if final_pred[-1] == 1:
        st.success('### Aluno propenso a se formar.')
        st.balloons()
    else:
        st.error('### Aluno propenso a evadir.')
