import streamlit as st
import torch
import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM



st.set_page_config(page_title="Hybrid AI Chatbot", page_icon="🤖")
st.title("🤖 Hybrid Open Innovation Chatbot")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"



@st.cache_resource
def load_kb():
    index = faiss.read_index("innovation.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

index, chunks = load_kb()


@st.cache_resource
def load_embedder():
    return SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        device=DEVICE
    )

embedder = load_embedder()


@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    return tokenizer, model


tokenizer, llm = load_llm()



def retrieve_context(query, k=3):

    emb = embedder.encode([query], normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)

    D, I = index.search(emb, k)

    results = []

    for idx in I[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return "\n\n".join(results[:2])


def route_query(query):

    q = query.lower()

    # 1. Hard RAG triggers
    rag_keywords = [
        "pdf", "document", "research", "innovation",
        "patent", "report", "study", "paper"
    ]

    # 2. General chat triggers
    general_keywords = [
        "hi", "hello", "who are you", "what is", "explain",
        "idea", "startup", "career", "mbbs"
    ]

    use_rag = any(k in q for k in rag_keywords)

    use_general = True  # default always general AI

    if use_rag:
        return "rag"

    if any(k in q for k in general_keywords):
        return "general"

    return "general"


SYSTEM_PROMPT = """
You are a powerful AI assistant for Open Innovation.

You are:
- creative
- helpful
- professional
- idea generator
- knowledge explainer

You can answer anything from:
- MBBS careers
- startups
- innovation
- technology
- general knowledge

Be natural like ChatGPT.
"""

def generate_answer(question):

    mode = route_query(question)

    context = ""

    if mode == "rag":
        context = retrieve_context(question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # chat memory
    for msg in st.session_state.messages[-6:]:
        messages.append(msg)

    # add context only if needed
    if context:
        messages.append({
            "role": "system",
            "content": f"""
Knowledge Base Context (use only if relevant):
{context}
"""
        })

    messages.append({
        "role": "user",
        "content": question
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    inputs = {k: v.to(llm.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    result = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(result, skip_special_tokens=True).strip(), mode


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask anything about innovation...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, mode = generate_answer(question)

            st.markdown(answer)

            st.caption(f"Mode used: {mode}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
