"""
Utilidades de IA compartidas por los comandos y vistas del Taller 3.

Usa unicamente las dos APIs sugeridas por el profesor:
- OpenAI (texto e imagenes) con la key en openAI.env.
- HuggingFace Inference API (texto e imagenes) con el token en openAI.env,
  como alternativa cuando OpenAI no tiene creditos (ver 2b_huggingfaceapikey.md).

Si ninguna de las dos tiene cuota disponible, las funciones lanzan la
excepcion correspondiente (no hay respaldo con servicios no oficiales).

Para los embeddings (paso 6/7 del taller) se usa sentence-transformers de
forma local (modelo all-MiniLM-L6-v2): son embeddings semanticos reales,
generados por un modelo de IA, sin depender de creditos de ninguna API.
"""
import os

import numpy as np
import requests
from openai import OpenAI
from dotenv import load_dotenv

# openAI.env vive en la raiz del proyecto Taller (un nivel arriba de DjangoProjectBase)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', 'openAI.env'))

HF_TEXT_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
HF_CHAT_URL = 'https://router.huggingface.co/v1/chat/completions'

HF_IMAGE_MODEL = 'stabilityai/stable-diffusion-3-medium-diffusers'
HF_IMAGE_URL = f'https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}'

_openai_client = None
_embedding_model = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get('openai_apikey')
        _openai_client = OpenAI(api_key=api_key) if api_key else None
    return _openai_client


def get_completion(prompt, model='gpt-3.5-turbo'):
    """Genera texto con la API real de OpenAI; si falla (sin creditos, sin key,
    error de red) usa la API de HuggingFace como alternativa."""
    messages = [{'role': 'user', 'content': prompt}]

    client = _get_openai_client()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0,
            )
            return response.choices[0].message.content.strip(), 'openai'
        except Exception as e:
            print(f'⚠️  OpenAI no disponible ({e}); usando HuggingFace como alternativa')

    hf_token = os.environ.get('huggingface_token')
    if not hf_token:
        raise RuntimeError('No hay openai_apikey ni huggingface_token configurados en openAI.env')

    headers = {'Authorization': f'Bearer {hf_token}'}
    response = requests.post(
        HF_CHAT_URL, headers=headers,
        json={'model': HF_TEXT_MODEL, 'messages': messages}, timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['message']['content'].strip(), 'huggingface'


def generate_image_bytes(prompt, width=256, height=384):
    """Genera una imagen con IA y devuelve los bytes.

    Intenta primero la API de OpenAI (DALL-E); si falla (sin creditos, sin
    key) usa la API de HuggingFace como alternativa."""
    client = _get_openai_client()
    if client is not None:
        try:
            response = client.images.generate(
                model='dall-e-2', prompt=prompt, size='256x256', quality='standard', n=1,
            )
            image_url = response.data[0].url
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
            return image_response.content
        except Exception as e:
            print(f'⚠️  OpenAI no disponible ({e}); usando HuggingFace como alternativa')

    hf_token = os.environ.get('huggingface_token')
    if not hf_token:
        raise RuntimeError('No hay openai_apikey ni huggingface_token configurados en openAI.env')

    headers = {'Authorization': f'Bearer {hf_token}'}
    response = requests.post(HF_IMAGE_URL, headers=headers, json={'inputs': prompt}, timeout=90)
    response.raise_for_status()
    return response.content


def get_embedding_model():
    """Carga (una sola vez) el modelo local de embeddings semanticos."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def get_embedding(text):
    """Devuelve el embedding (vector numpy float32) de un texto."""
    model = get_embedding_model()
    vec = model.encode(text)
    return np.array(vec, dtype=np.float32)


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
