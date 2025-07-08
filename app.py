import os
import streamlit as st
import pandas as pd
from openai import OpenAI
import whisper
import yt_dlp
import tempfile
import re
from typing import List, Tuple
from dotenv import load_dotenv
from lightrag import LightRAG

load_dotenv(dotenv_path="config.env")
api_key = os.getenv("OPENROUTER_API_KEY")
senha = os.getenv("APP_SENHA")

print(api_key, senha)
# Configuração do OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Configuração de limites de tokens
MAX_TOKENS = 150000  # Limite seguro (deixa margem para resposta)
CHUNK_SIZE = 30000   # Tamanho dos chunks em caracteres (aprox 8000 tokens)
OVERLAP = 2000       # Sobreposição entre chunks

def count_tokens(text: str) -> int:
    """Estima tokens usando uma aproximação simples (4 caracteres ≈ 1 token)"""
    if not text:
        return 0
    
    # Remove espaços extras e conta caracteres
    clean_text = re.sub(r'\s+', ' ', text.strip())
    
    # Aproximação: 1 token ≈ 4 caracteres para português/inglês
    # Ajustamos para 3.5 para ser mais conservador
    estimated_tokens = len(clean_text) // 3.5
    
    return int(estimated_tokens)

def truncate_text(text: str, max_tokens: int) -> str:
    """Trunca texto para não exceder o limite de tokens"""
    if not text:
        return text
    
    current_tokens = count_tokens(text)
    if current_tokens <= max_tokens:
        return text
    
    # Calcula quantos caracteres manter (3.5 chars por token)
    max_chars = int(max_tokens * 3.5)
    
    # Trunca no limite de caracteres, mas tenta quebrar em palavra
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    
    # Tenta quebrar na última palavra completa
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:  # Se estiver próximo do fim
        truncated = truncated[:last_space]
    
    return truncated + "..."

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Divide texto em chunks menores com sobreposição (baseado em caracteres)"""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Se não é o último chunk, tenta quebrar em palavra
        if end < len(text):
            # Procura por quebra de linha primeiro
            last_newline = text.rfind('\n', start, end)
            if last_newline > start + chunk_size * 0.5:
                end = last_newline
            else:
                # Se não encontrar quebra de linha, procura espaço
                last_space = text.rfind(' ', start + chunk_size * 0.8, end)
                if last_space > start:
                    end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks

def summarize_text(text: str, max_summary_tokens: int = 1000) -> str:
    """Resume texto longo usando a API"""
    try:
        # Limita o texto de entrada para não causar erro na API
        max_input_tokens = 50000
        if count_tokens(text) > max_input_tokens:
            text = truncate_text(text, max_input_tokens)
        
        prompt = f"""Resuma o seguinte texto de forma concisa, mantendo as informações técnicas mais importantes:

{text}

Resumo:"""
        
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_summary_tokens
        )
        
        if response and response.choices:
            return response.choices[0].message.content
        # Fallback: trunca simplesmente
        return truncate_text(text, max_summary_tokens)
    except Exception as e:
        st.error(f"Erro ao resumir texto: {e}")
        return truncate_text(text, max_summary_tokens)

def get_text_stats(text: str) -> dict:
    """Retorna estatísticas básicas do texto"""
    if not text:
        return {"chars": 0, "words": 0, "tokens": 0, "lines": 0}
    
    chars = len(text)
    words = len(text.split())
    tokens = count_tokens(text)
    lines = len(text.split('\n'))
    
    return {
        "chars": chars,
        "words": words, 
        "tokens": tokens,
        "lines": lines
    }

def smart_text_summary(text: str, target_tokens: int) -> str:
    """Cria resumo inteligente priorizando informações técnicas"""
    if count_tokens(text) <= target_tokens:
        return text
    
    # Estratégia 1: Extrai parágrafos que contêm termos técnicos
    technical_terms = ['rede', 'sinal', 'antena', 'frequência', 'dbm', 'throughput', 
                      'latência', 'cobertura', 'interferência', 'handover', 'lte', 
                      '5g', '4g', 'rsrp', 'rsrq', 'sinr', 'cqi', 'site', 'bts']
    
    paragraphs = text.split('\n')
    technical_paragraphs = []
    
    for para in paragraphs:
        if any(term in para.lower() for term in technical_terms):
            technical_paragraphs.append(para)
    
    # Se encontrou parágrafos técnicos, usa eles
    if technical_paragraphs:
        summary = '\n'.join(technical_paragraphs)
        if count_tokens(summary) <= target_tokens:
            return summary
        else:
            # Trunca os parágrafos técnicos
            return truncate_text(summary, target_tokens)
    
    # Fallback: trunca o texto original
    return truncate_text(text, target_tokens)
def process_large_context(docs: List[str], question: str) -> str:
    """Processa contexto grande usando diferentes estratégias"""
    
    # Combina todos os documentos
    full_context = "\n".join(docs)
    context_tokens = count_tokens(full_context)
    question_tokens = count_tokens(question)
    
    # Reserva tokens para a pergunta e resposta
    available_tokens = MAX_TOKENS - question_tokens - 2000  # 2000 para resposta
    
    st.sidebar.markdown(f"📊 **Análise de Tokens:**")
    st.sidebar.markdown(f"- Contexto original: {context_tokens:,} tokens")
    st.sidebar.markdown(f"- Pergunta: {question_tokens} tokens")
    st.sidebar.markdown(f"- Limite disponível: {available_tokens:,} tokens")
    
    if context_tokens <= available_tokens:
        st.sidebar.success("✅ Contexto dentro do limite")
        return full_context
    
    # Estratégia 1: Usar resumo inteligente
    if context_tokens <= available_tokens * 3:
        st.sidebar.warning("⚠️ Usando resumo inteligente")
        return smart_text_summary(full_context, available_tokens)
    
    # Estratégia 2: Resumir usando API
    st.sidebar.warning("⚠️ Contexto muito grande - criando resumo via API")
    with st.spinner("Resumindo contexto..."):
        summary = summarize_text(full_context, available_tokens)
        summary_tokens = count_tokens(summary)
        st.sidebar.info(f"Resumo: {summary_tokens:,} tokens")
        return summary

def extract_relevant_chunks(docs: List[str], question: str, max_chunks: int = 5) -> List[str]:
    """Extrai chunks mais relevantes para a pergunta"""
    all_chunks = []
    
    # Processa cada documento
    for i, doc in enumerate(docs):
        doc_chunks = chunk_text(doc)
        # Adiciona identificador do documento
        labeled_chunks = [f"[Doc {i+1}] {chunk}" for chunk in doc_chunks]
        all_chunks.extend(labeled_chunks)
    
    # Palavras-chave da pergunta (remove stopwords simples)
    stopwords = ['o', 'a', 'os', 'as', 'de', 'da', 'do', 'em', 'na', 'no', 'para', 
                'por', 'com', 'sem', 'sobre', 'é', 'são', 'como', 'que', 'qual', 'quando']
    
    question_words = [word.lower() for word in question.split() 
                     if len(word) > 2 and word.lower() not in stopwords]
    
    # Pontuação dos chunks
    scored_chunks = []
    
    for chunk in all_chunks:
        chunk_lower = chunk.lower()
        score = 0
        
        # Conta ocorrências de palavras-chave
        for word in question_words:
            score += chunk_lower.count(word) * 2
        
        # Bonus para chunks com termos técnicos
        technical_terms = ['rede', 'sinal', 'antena', 'site', 'bts', 'lte', '5g', '4g']
        for term in technical_terms:
            if term in chunk_lower:
                score += 1
        
        scored_chunks.append((score, chunk))
    
    # Ordena por relevância e pega os top chunks
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    selected_chunks = [chunk for _, chunk in scored_chunks[:max_chunks]]
    
    return selected_chunks

st.set_page_config(page_title="RAG Otimizador Técnico", layout="wide")

# Proteção com senha
st.markdown("### 🔐 Acesso Restrito")
senha = st.text_input("Digite a senha para acessar:", type="password")

if senha != os.getenv("APP_SENHA"):
    st.warning("Acesso negado. Digite a senha correta para continuar.")
    st.stop()

st.title("📡 Otimizador Inteligente com RAG")
st.markdown("Envie planilhas, documentos ou links de vídeo e pergunte sobre sua rede.")

# Configurações na sidebar
st.sidebar.header("⚙️ Configurações")
processing_mode = st.sidebar.selectbox(
    "Modo de processamento:",
    ["Automático", "Truncar", "Resumir", "Chunks Relevantes"]
)

max_context_tokens = st.sidebar.slider(
    "Tokens máximos para contexto:",
    min_value=10000,
    max_value=MAX_TOKENS,
    value=100000,
    step=10000
)

# 🔊 Carregamento do modelo Whisper apenas sob demanda
@st.cache_resource(show_spinner="🔊 Carregando modelo Whisper...")
def carregar_modelo_whisper():
    return whisper.load_model("tiny")

def transcrever_audio_do_youtube(url):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'quiet': True
            }

            if os.path.exists("youtube_cookies.txt"):
                ydl_opts["cookies"] = "youtube_cookies.txt"

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = None
                for f in os.listdir(temp_dir):
                    if f.endswith(".mp3"):
                        filename = os.path.join(temp_dir, f)
                        break

            if filename:
                model = carregar_modelo_whisper()
                result = model.transcribe(filename)
                return result["text"]
            else:
                return "Erro ao processar o áudio."
    except Exception as e:
        return f"Erro: {e}"

# 📚 Coleta de documentos
docs = []

# 📊 Upload CSV
csv_file = st.file_uploader("📈 Envie um arquivo CSV com os dados", type=["csv"])
if csv_file:
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        try:
            csv_file_data = csv_file.getvalue()
            df = pd.read_csv(pd.io.common.BytesIO(csv_file_data), encoding='ISO-8859-1', delimiter=';')
        except Exception as e:
            st.error(f"Erro ao tentar ler o arquivo CSV: {e}")
            df = None

    if df is not None:
        st.success("CSV carregado com sucesso!")
        st.dataframe(df.head())
        
        # Limita o número de linhas processadas se for muito grande
        max_rows = 10000
        if len(df) > max_rows:
            st.warning(f"CSV muito grande ({len(df)} linhas). Processando apenas as primeiras {max_rows} linhas.")
            df = df.head(max_rows)
        
        # Processa em chunks menores para economizar memória
        chunk_size = 500  # Reduzido para ser mais leve
        processed_rows = 0
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_text = []
            
            for _, row in chunk.iterrows():
                # Cria entrada mais compacta
                row_data = []
                for col in chunk.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        row_data.append(f"{col}:{row[col]}")
                
                if row_data:
                    chunk_text.append(" | ".join(row_data))
                    processed_rows += 1
            
            # Combina as linhas do chunk em um documento
            if chunk_text:
                docs.append(f"[CSV Chunk {i//chunk_size + 1}]\n" + "\n".join(chunk_text))
        
        st.info(f"Processadas {processed_rows} linhas do CSV em {len(docs)} chunks")

# 📄 Upload de documentos
uploaded_docs = st.file_uploader("📄 Envie arquivos .txt ou .pdf", type=["txt", "pdf"], accept_multiple_files=True)
if uploaded_docs:
    for file in uploaded_docs:
        content = file.read().decode("utf-8", errors="ignore")
        
        # Limita o tamanho de cada documento
        max_doc_chars = 50000  # Aproximadamente 14k tokens
        
        for file in uploaded_docs:
            try:
                content = file.read().decode("utf-8", errors="ignore")
                
                # Estatísticas do documento
                stats = get_text_stats(content)
                
                if stats["chars"] > max_doc_chars:
                    st.warning(f"📄 {file.name}: {stats['chars']:,} chars ({stats['tokens']:,} tokens) - Truncando para caber no limite")
                    content = truncate_text(content, max_doc_chars // 4)  # Conversão aproximada
                else:
                    st.success(f"📄 {file.name}: {stats['chars']:,} chars ({stats['tokens']:,} tokens)")
                
                docs.append(f"[Arquivo: {file.name}]\n{content}")
                
            except Exception as e:
                st.error(f"Erro ao processar {file.name}: {e}")

# 🔐 Upload manual do arquivo de cookies do YouTube
cookie_file = st.file_uploader("🔑 Envie seu arquivo 'youtube_cookies.txt' (opcional)", type=["txt"])
if cookie_file:
    with open("youtube_cookies.txt", "wb") as f:
        f.write(cookie_file.read())
    st.success("Arquivo de cookies salvo com sucesso!")

# 📺 Link do YouTube
youtube_link = st.text_input("🎥 Cole um link de vídeo do YouTube para transcrição automática:")
if youtube_link:
    with st.spinner("Transcrevendo áudio do vídeo..."):
        try:
            transcricao = transcrever_audio_do_youtube(youtube_link)
            st.success("Transcrição concluída!")
            
            # Limita o tamanho da transcrição
            max_transcription_chars = 100000  # Aproximadamente 28k tokens
            transcription_stats = get_text_stats(transcricao)
            
            if transcription_stats["chars"] > max_transcription_chars:
                st.warning(f"🎥 Transcrição muito longa ({transcription_stats['chars']:,} chars, {transcription_stats['tokens']:,} tokens) - Usando resumo inteligente")
                transcricao = smart_text_summary(transcricao, max_transcription_chars // 4)
            else:
                st.success(f"🎥 Transcrição: {transcription_stats['chars']:,} chars ({transcription_stats['tokens']:,} tokens)")
            
            st.text_area("📝 Texto extraído do vídeo:", transcricao, height=200)
            docs.append(f"[YouTube: {youtube_link}]\n{transcricao}")
        except Exception as e:
            st.error(f"Erro ao transcrever vídeo: {e}")

# Campo de pergunta sempre visível
user_question = st.text_input("🧠 Faça uma pergunta técnica:")

# RAG e Pergunta
if user_question:
    if docs:
        # Processa o contexto baseado no modo selecionado
        if processing_mode == "Automático":
            contexto = process_large_context(docs, user_question)
        elif processing_mode == "Truncar":
            full_context = "\n".join(docs)
            contexto = truncate_text(full_context, max_context_tokens)
        elif processing_mode == "Resumir":
            full_context = "\n".join(docs)
            if count_tokens(full_context) > max_context_tokens:
                contexto = smart_text_summary(full_context, max_context_tokens)
            else:
                contexto = full_context
        elif processing_mode == "Chunks Relevantes":
            relevant_chunks = extract_relevant_chunks(docs, user_question)
            contexto = "\n".join(relevant_chunks)
        
        # 🔍 Detectar colunas no CSV
        colunas_relevantes = ['índice_taxa', 'ue_medio', 'site', 'bts']
        colunas_presentes = []
        if 'df' in locals() and df is not None:
            colunas_presentes = [col for col in colunas_relevantes if col in df.columns]

        if colunas_presentes:
            observacao = "OBSERVAÇÃO: Valores altos em colunas como " + ", ".join(colunas_presentes) + " representam pior desempenho da rede."
        else:
            observacao = ""

    else:
        contexto = ""
        observacao = ""

    # Prompt final com contagem de tokens
    prompt = f"""Você é um especialista técnico em redes móveis.

DADOS:
{contexto}

PERGUNTA:
{user_question}

{observacao}
"""

    # Mostra informações sobre o prompt final
    final_tokens = count_tokens(prompt)
    st.sidebar.markdown(f"📤 **Prompt Final:** {final_tokens:,} tokens")
    
    # Mostra estatísticas detalhadas
    if docs:
        total_docs = len(docs)
        total_context_tokens = count_tokens(contexto)
        st.sidebar.markdown(f"📚 **Documentos:** {total_docs}")
        st.sidebar.markdown(f"📊 **Contexto usado:** {total_context_tokens:,} tokens")
    
    if final_tokens > MAX_TOKENS:
        st.error(f"❌ Prompt ainda muito grande ({final_tokens:,} tokens). Limite: {MAX_TOKENS:,}")
        st.info("💡 Tente usar um modo de processamento diferente ou reduzir o número de documentos.")
        st.stop()
    
    # Indicador de saúde do prompt
    token_usage_pct = (final_tokens / MAX_TOKENS) * 100
    if token_usage_pct > 90:
        st.sidebar.error(f"⚠️ Uso alto: {token_usage_pct:.1f}%")
    elif token_usage_pct > 70:
        st.sidebar.warning(f"⚠️ Uso médio: {token_usage_pct:.1f}%")
    else:
        st.sidebar.success(f"✅ Uso baixo: {token_usage_pct:.1f}%")

    try:
        with st.spinner("Consultando IA..."):
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )

        if response and hasattr(response, "choices") and response.choices:
            answer = response.choices[0].message.content
            st.markdown("### ✅ Resposta da IA:")
            st.success(answer)
            
            # Estatísticas da resposta
            answer_stats = get_text_stats(answer)
            st.sidebar.markdown(f"📝 **Resposta:** {answer_stats['tokens']:,} tokens")
            
        else:
            st.warning("⚠️ A resposta da IA veio vazia ou malformada.")

    except Exception as e:
        st.error(f"Erro ao consultar a IA: {e}")
        st.info("💡 Tente reduzir o tamanho do contexto ou usar um modo de processamento diferente.")

# Rodapé com informações
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Como usar:")
st.sidebar.markdown("""
- **Automático**: Escolhe a melhor estratégia
- **Truncar**: Corta o texto no limite
- **Resumir**: Cria resumo inteligente
- **Chunks Relevantes**: Usa apenas partes relevantes
""")

st.sidebar.markdown("### 💡 Dicas:")
st.sidebar.markdown("""
- Arquivos menores = respostas mais rápidas
- Use palavras-chave específicas nas perguntas
- Modo 'Chunks Relevantes' é bom para documentos longos
""")