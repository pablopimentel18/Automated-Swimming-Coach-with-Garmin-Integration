import os
import json
from wsgiref import headers
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from garminconnect import Garmin
import garth
import re # Biblioteca para encontrar o JSON no meio do texto
# ==========================================
# CONFIGURAÇÕES DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(page_title="Técnico Virtual", page_icon="🏊‍♂️")
st.title("🏊‍♂️ Técnico Virtual de Natação")

# Coloque sua chave aqui
CHAVE_API = "Sua chave api" 
genai.configure(api_key=CHAVE_API)
os.environ["GOOGLE_API_KEY"] = CHAVE_API # Necessário para o LangChain

# ==========================================
# FUNÇÕES DE MEMÓRIA (JSON) E RAG
# ==========================================
ARQUIVO_MEMORIA = "historico_usuario.json"

def carregar_memoria():
    """Lê o histórico do disco. Se não existir, retorna lista vazia."""
    if os.path.exists(ARQUIVO_MEMORIA):
        with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_memoria(historico):
    """Salva o histórico atualizado no disco."""
    with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=4)

@st.cache_resource 
def configurar_base_conhecimento():
    conteudo_base = """
    Regras do Técnico de Natação:
    - O treino A1 foca em resistência aeróbica básica, nado contínuo e leve.
    - O nado com palmares aumenta a força de tração, mas deve ser limitado a 20% do volume total para evitar lesões no ombro.
    - Para treinos de velocidade pura, use tiros curtos de 15m a 25m com descanso longo (1 a 2 minutos).
    - Equipamentos como nadadeiras ajudam a corrigir a posição do quadril na água e fortalecer as pernas.
    - Ter séries de nado submerso (ondulações) é extremamente para ganho de velocidade
    - Exercícios como Palmateios aumentam a sensibilidade na água e melhoram a técnica de braçada.
    - Fazer exercícios de respiração bilateral ajuda a equilibrar a técnica e reduzir o risco de lesões.
    - Fazer exercícios em apneia auxiliam na condição pulmonar.
    - Aquecimento Inicial deve possuir 200m variando posições como palmateio e braçada submersa e nado livre.
    - Após aquecimento, trabalho de pernada focando em técnica é essencial para ativação.
    - Após trabalho de perna vem os educativos para refinar a técnica.
    - Após educativos vem a parte principal do treino, que pode variar entre resistência, velocidade e potência ou técnica.
    """
    with open("manual_natacao.txt", "w", encoding="utf-8") as f:
        f.write(conteudo_base)

    loader = TextLoader("manual_natacao.txt", encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(loader.load())

    # Usando o modelo Open-Source (Perfeito para português e agora instantâneo graças ao cache)
    modelo_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return Chroma.from_documents(chunks, modelo_embeddings, persist_directory="./chroma_db_streamlit")

# ==========================================
# INICIALIZAÇÃO E INTERFACE
# ==========================================
banco_vetorial = configurar_base_conhecimento()

# Inicializa o modelo LLM
modelo_llm = genai.GenerativeModel(
    model_name="gemini-3.5-flash-lite",
    system_instruction="""
        Você é um técnico de natação de elite. Seja motivador e use o contexto da base de dados.
        
        REGRA VITAL: 
        1. PRIMEIRO, escreva o treino de forma visual e amigável para o usuário ler na tela, usando bullet points e emojis.
        2. SÓ DEPOIS, no final da resposta, inclua o bloco JSON estruturado.
        
        O JSON suporta passos simples e blocos de repetição (séries).
        ESTILOS: "livre", "costas", "peito", "borboleta", "medley", "qualquer"
        EQUIPAMENTOS: "nadadeiras", "prancha", "palmares", "pullbuoy", "snorkel", "nenhum"
        TIPOS: "warmup", "active", "recovery", "cooldown", "rest"

        Use este formato:
        ESTILOS PERMITIDOS: "livre", "costas", "peito", "borboleta", "medley", "qualquer"
        EQUIPAMENTOS PERMITIDOS: "nadadeiras", "prancha", "palmares", "pullbuoy", "snorkel", "nenhum"
        TIPOS DE PASSO: "warmup", "active", "recovery", "cooldown", "rest"

        Use EXATAMENTE este formato (adapte conforme o treino):
        ```json
        [
          {
            "tipo": "warmup",
            "distancia": 400,
            "estilo": "livre",
            "equipamento": "nenhum",
            "descricao": "Aquecimento solto"
          },
          {
            "tipo": "repeat",
            "repeticoes": 6,
            "omitir_ultimo_descanso": true,
            "passos": [
              {
                "tipo": "active",
                "distancia": 50,
                "estilo": "borboleta",
                "equipamento": "nadadeiras",
                "descricao": "Tiro forte"
              },
              {
                "tipo": "rest",
                "tempo_descanso": 15,
                "descricao": "Pausa na borda"
              }
            ]
          },
          {
            "tipo": "cooldown",
            "distancia": 200,
            "estilo": "qualquer",
            "equipamento": "nenhum",
            "descricao": "Soltura final"
          }
        ] 
        ```
        Regras do JSON:
        - Se for descanso por tempo, use 'tempo_descanso' (em segundos) em vez de 'distancia'.
        - O bloco de repetição deve ter o tipo "repeat" contendo "repeticoes" e a lista de "passos".
        """
)

# Carrega a memória para a sessão atual do site
if "historico" not in st.session_state:
    st.session_state.historico = carregar_memoria()

# Desenha as mensagens antigas na tela
for msg in st.session_state.historico:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["parts"][0])

# ==========================================
# LÓGICA DO CHAT
# ==========================================
# Caixa de texto na parte inferior do site
pergunta_usuario = st.chat_input("Digite seu objetivo de treino ou dúvida...")

if pergunta_usuario:
    # 1. Mostra a pergunta do usuário na tela
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)
    
    # 2. Salva na memória oficial e no JSON
    st.session_state.historico.append({"role": "user", "parts": [pergunta_usuario]})
    salvar_memoria(st.session_state.historico)
    
    # 3. Faz o RAG
    resultados = banco_vetorial.similarity_search(pergunta_usuario, k=2)
    contexto_recuperado = "\n".join([doc.page_content for doc in resultados])
    
    prompt_aumentado = f"""
    Responda usando o contexto: {contexto_recuperado}
    Pergunta: {pergunta_usuario}
    """


    # 4. Prepara a lista para o Gemini
    MAX_MENSAGENS = 10
    historico_janela = st.session_state.historico[:-1][-MAX_MENSAGENS:]  
    if len(historico_janela) > 0 and historico_janela[0]["role"] == "model":
        historico_janela = historico_janela[1:]  # Remove a primeira mensagem se for do modelo

    
    mensagens_para_api = list(historico_janela)  # Copia o histórico recente
    mensagens_para_api.append({"role": "user", "parts": [prompt_aumentado]})
    
    # 5. Mostra o "Digitando..." e faz a chamada
    with st.chat_message("assistant"):
        with st.spinner("Analisando o treino..."):
            try:
                resposta = modelo_llm.generate_content(mensagens_para_api)
                texto_resposta = resposta.text
                st.markdown(texto_resposta)
                
                # Salva a resposta na memória e no JSON
                st.session_state.historico.append({"role": "model", "parts": [texto_resposta]})
                salvar_memoria(st.session_state.historico)
                
            except Exception as erro:
                st.error(f"Erro de conexão: {erro}")
                st.session_state.historico.pop() # Remove para não corromper


import os
import garth  # Usaremos apenas o motor base agora!

import requests # A arma definitiva, sem intermediários!

import requests

def enviar_para_garmin(cookie_garmin, csrf_garmin, nome_treino, passos_simples):
    """
    Clona perfeitamente o cURL interceptado, falsificando cabeçalhos de segurança 
    (sec-fetch) e utilizando a nova estrutura DTO (ID 4 para Natação).
    """
    try:
        # 1. CONSTRUÇÃO DO PAYLOAD (Atualizado com os IDs reais do seu cURL)
        tipos_garmin = {"warmup": 1, "cooldown": 2, "active": 8, "recovery": 4, "rest": 5}
        estilos_garmin = {"livre": 6, "costas": 2, "peito": 3, "borboleta": 4, "medley": 5, "qualquer": 1}
        equips_garmin = {"nadadeiras": 1, "prancha": 2, "palmares": 3, "pullbuoy": 4, "snorkel": 5}

        workout_steps = []
        step_id_counter = 1
        step_order_counter = 1

        def construir_passo_dto(passo_json, s_id, s_order, child_id=None):
            tipo_key = passo_json.get("tipo", "active")
            tipo_id = tipos_garmin.get(tipo_key, 8)
            
            dto = {
                "type": "ExecutableStepDTO",
                "stepId": s_id,
                "stepOrder": s_order,
                "stepType": {"stepTypeId": tipo_id, "stepTypeKey": tipo_key, "displayOrder": tipo_id},
                "targetType": {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target", "displayOrder": 1}
            }
            
            if child_id is not None:
                dto["childStepId"] = child_id
                
            if "descricao" in passo_json:
                dto["description"] = passo_json["descricao"]

            # Condição de Término (Distância vs Tempo de Descanso)
            # Condição de Término (Distância vs Apertar o Botão)
            if tipo_key == "rest":
                # Forçando o padrão 'lap.button' para todos os descansos
                dto["endCondition"] = {"conditionTypeId": 1, "conditionTypeKey": "lap.button", "displayOrder": 1, "displayable": True}
                dto["endConditionValue"] = 0
            else:
                dto["endCondition"] = {"conditionTypeId": 3, "conditionTypeKey": "distance", "displayOrder": 3, "displayable": True}
                dto["endConditionValue"] = passo_json.get("distancia", 50)
                dto["preferredEndConditionUnit"] = {"unitKey": "meter"}


            # Adiciona o Estilo da Braçada (Stroke Type)
            estilo = passo_json.get("estilo", "qualquer").lower()
            est_id = estilos_garmin.get(estilo, 1)
            dto["strokeType"] = {"strokeTypeId": est_id, "strokeTypeKey": estilo, "displayOrder": est_id}

            # Adiciona o Equipamento (Equipment Type)
            equip = passo_json.get("equipamento", "nenhum").lower()
            if equip in equips_garmin:
                eq_id = equips_garmin[equip]
                dto["equipmentType"] = {"equipmentTypeId": eq_id, "equipmentTypeKey": equip, "displayOrder": eq_id}

            return dto

        # Varre o JSON gerado pela IA
        for item in passos_simples:
            if item.get("tipo") == "repeat":
                # É UM LOOP! Cria o bloco RepeatGroupDTO
                child_id_master = step_id_counter
                
                repeat_dto = {
                    "type": "RepeatGroupDTO",
                    "stepId": step_id_counter,
                    "stepOrder": step_order_counter,
                    "stepType": {"stepTypeId": 6, "stepTypeKey": "repeat", "displayOrder": 6},
                    "numberOfIterations": item.get("repeticoes", 2),
                    "smartRepeat": False,
                    "childStepId": child_id_master,
                    "endCondition": {"conditionTypeId": 7, "conditionTypeKey": "iterations", "displayOrder": 7, "displayable": False},
                    "skipLastRestStep": item.get("omitir_ultimo_descanso", True),
                    "workoutSteps": []
                }
                
                step_id_counter += 1
                
                # Processa os passos dentro do loop
                for passo_filho in item.get("passos", []):
                    filho_dto = construir_passo_dto(passo_filho, step_id_counter, step_order_counter, child_id_master)
                    repeat_dto["workoutSteps"].append(filho_dto)
                    step_id_counter += 1
                    step_order_counter += 1
                    
                workout_steps.append(repeat_dto)
                
            else:
                # É um passo normal (aquecimento, soltura, etc)
                passo_dto = construir_passo_dto(item, step_id_counter, step_order_counter)
                workout_steps.append(passo_dto)
                step_id_counter += 1
                step_order_counter += 1

        # O payload_garmin final continua exatamente o mesmo:
        payload_garmin = {
            "sportType": {"sportTypeId": 4, "sportTypeKey": "swimming", "displayOrder": 3},
            "subSportType": None,
            "workoutName": nome_treino,
            "poolLength": 25,
            "poolLengthUnit": {"unitKey": "meter"},
            "workoutSegments": [{
                "segmentOrder": 1,
                "sportType": {"sportTypeId": 4, "sportTypeKey": "swimming", "displayOrder": 3},
                "workoutSteps": workout_steps
            }]
        }

        # 2. CABEÇALHOS DE SEGURANÇA (O disfarce perfeito para o Cloudflare)
        headers = {
            "accept": "*/*",
            "connect-csrf-token": csrf_garmin,
            "content-type": "application/json",
            "cookie": cookie_garmin,
            "origin": "https://connect.garmin.com",
            "referer": "https://connect.garmin.com/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 OPR/134.0.0.0"
        }

        # 3. O DISPARO
        url_api = "https://connect.garmin.com/gc-api/workout-service/workout"
        resposta = requests.post(url_api, headers=headers, json=payload_garmin, allow_redirects=False)
        # ==========================================
        # MODO DE DEPURAÇÃO (DEBUG NO TERMINAL)
        # ==========================================
        print("\n" + "="*50)
        print(f"STATUS CODE RETORNADO: {resposta.status_code}")
        print("CABEÇALHOS DA RESPOSTA:")
        for key, value in resposta.headers.items():
            print(f"  {key}: {value}")
        print("\nCORPO DA RESPOSTA (Primeiros 1000 caracteres):")
        print(resposta.text[:1000])
        print("="*50 + "\n")
        # 4. VALIDAÇÃO
        if resposta.status_code == 200:
            return True, "Upload concluído! Treino sincronizado via injeção HTTP."
        elif resposta.status_code == 302:
            return False, "Erro 302: Redirecionamento detectado."
        else:
            return False, f"Falha no servidor. Código {resposta.status_code}: {resposta.text}"

    except Exception as e:
        return False, f"Erro interno: {str(e)}"

with st.sidebar:
    st.header("⌚ Integração Direta Garmin")
    cookie_garmin = st.text_input("Valor do 'Cookie' (inteiro)", type="password")
    csrf_garmin = st.text_input("Valor do 'Connect-Csrf-Token'", type="password")
    st.caption("Volte no F12 > Request Headers e copie os valores desses dois campos.")

    st.divider()

    st.header("⚙️ Configurações do Chat")
    if st.button("🗑️ Limpar histórico"):
        st.session_state.historico = []
        salvar_memoria([])
        st.rerun()

# Na parte onde você exibe a resposta do LLM, você faz o seguinte:
if len(st.session_state.historico) > 0:
    ultima_mensagem = st.session_state.historico[-1]["parts"][0]
    
    # Procura se o Gemini anexou um bloco ```json na resposta
    padrao = r'```json\n(.*?)\n```'
    match = re.search(padrao, ultima_mensagem, re.DOTALL)
    
    if match:
        st.success("✨ Treino estruturado identificado!")
        json_treino = json.loads(match.group(1)) # Transforma o texto em lista Python
        
        # Cria o botão de upload
        if st.button("🚀 Injetar no meu Garmin 955"):
            if not cookie_garmin or not csrf_garmin:
                st.warning("⚠️ Preencha o Cookie e o CSRF Token no menu lateral.")
            else:
                with st.spinner("Clonando sessão e injetando payload..."):
                    sucesso, mensagem_retorno = enviar_para_garmin(
                        cookie_garmin,
                        csrf_garmin,
                        "Treino IA - Técnico Virtual", 
                        json_treino
                    )
                    if sucesso:
                        st.balloons()
                        st.success(mensagem_retorno)
                    else:
                        st.error(mensagem_retorno)