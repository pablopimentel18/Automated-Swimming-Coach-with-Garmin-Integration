# 🏊‍♂️ Técnico Virtual de Natação (Garmin Connect Sync)

Um agente de Inteligência Artificial construído com **Python, LangChain, ChromaDB e Gemini**. Este sistema atua como um técnico virtual que gera treinos de natação personalizados com base em manuais de fisiologia do esporte e os **sincroniza nativamente com o seu relógio Garmin**. De tal forma que um usuário faz uma requisição, o sistema faz uma busca em um banco de dados vetorial e envia os cruzamentos dessa busca a um LLM que vai estruturar o treino no formato desejado. Após isso, o script vai injetar o treino na plataforma Garmin Connect e este estará disponível para uso.

Devido aos bloqueios de segurança recentes da Garmin (Cloudflare WAF / TLS Fingerprinting), este projeto utiliza engenharia reversa e injeção direta de sessão HTTP, clonando a assinatura de um navegador real para realizar o upload nativo dos treinos.

---

## ✨ Funcionalidades

- **Geração Inteligente (RAG):** Consulta uma base vetorial local de manuais de natação para montar o treino ideal.
- **Memória de Sessão:** A IA lembra do que foi conversado ao longo do uso.
- **Integração Garmin Avançada:** Suporta estruturação de treino completa (DTOs), incluindo:
  - Repetições e Loops (Séries como "8x 50m").
  - Estilos de natação (Livre, Peito, Borboleta, Costas, Medley).
  - Equipamentos (Nadadeiras, Palmares, Prancha, Pullbuoy, Snorkel).
  - Tipos de descanso nativos (Lap Button).

---

## 🚀 Como Instalar

**1. Clone o repositório**
```bash
git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
cd SEU_REPOSITORIO
```

**2. Crie e ative um ambiente virtual**
```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Configure sua chave da API**
Abra o arquivo `app.py` e insira sua chave do Google Gemini (AI Studio) na variável `CHAVE_API` (lembre-se de nunca commitar sua chave publicamente!).

---

## 🔑 Como pegar os Tokens da Garmin (Bypass de Segurança)

Para que o script consiga enviar o treino para a sua conta, você precisa fornecer a sessão ativa do seu navegador. 

1. Acesse o [Garmin Connect Web](https://connect.garmin.com) pelo computador e faça login.
2. Pressione `F12` para abrir o Painel do Desenvolvedor.
3. Vá na aba **Network** (Rede) e clique no ícone de "🚫" (Limpar) para limpar a tela.
4. No site da Garmin, clique no botão de "Criar um exercício" (ou apenas atualize a página) para gerar tráfego de rede.
5. Na lista do *Network*, clique em qualquer requisição que aparecer (ex: `workouts?start=1...`).
6. Na aba lateral que abrir, vá em **Headers > Request Headers** (Cabeçalhos de Requisição).
7. Copie os valores EXATOS de dois campos:
   - `cookie` (Um texto gigante começando com `_pk_id...` ou similar).
   - `connect-csrf-token`
8. Cole esses dois valores no menu lateral da aplicação Streamlit.

---

## ▶️ Como Rodar a Aplicação

Com tudo instalado e configurado, rode o comando abaixo no terminal:

```bash
streamlit run app.py
```
A interface web será aberta automaticamente no seu navegador. Peça um treino, veja a mágica do JSON estruturado acontecer, e clique em "Injetar no meu Garmin"! 

---
*Aviso: Este projeto tem fins puramente educacionais e acadêmicos. Não possui vínculo oficial com a Garmin.*