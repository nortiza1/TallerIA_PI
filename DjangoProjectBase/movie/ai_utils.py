"""
Utilidades de IA compartidas por los comandos y vistas del Taller 3.

Este proyecto intenta usar la API real de OpenAI (con la key en openAI.env).
Si la cuenta no tiene creditos (error 429 insufficient_quota) o no hay key,
se usa automaticamente Pollinations.ai como respaldo: un servicio gratuito,
sin llave, compatible con el SDK de openai para texto e imagenes. Esto sigue
la indicacion del profesor de usar una API alternativa cuando no hay creditos.

Para los embeddings (paso 6/7 del taller) se usa sentence-transformers de
forma local (modelo all-MiniLM-L6-v2): son embeddings semanticos reales,
generados por un modelo de IA, sin depender de creditos de ninguna API.
"""
import os
import io

import numpy as np
import requests
from openai import OpenAI
from dotenv import load_dotenv

# openAI.env vive en la raiz del proyecto Taller (un nivel arriba de DjangoProjectBase)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', 'openAI.env'))

POLLINATIONS_BASE_URL = 'https://text.pollinations.ai/openai'
POLLINATIONS_MODEL = 'openai'
IMAGE_BASE_URL = 'https://image.pollinations.ai/prompt'

_real_client = None
_fallback_client = None
_embedding_model = None


def _get_real_client():
    global _real_client
    if _real_client is None:
        api_key = os.environ.get('openai_apikey')
        _real_client = OpenAI(api_key=api_key) if api_key else None
    return _real_client


def _get_fallback_client():
    global _fallback_client
    if _fallback_client is None:
        # Pollinations no valida la key, pero el SDK exige que exista un valor
        _fallback_client = OpenAI(api_key='not-needed', base_url=POLLINATIONS_BASE_URL)
    return _fallback_client


def get_completion(prompt, model='gpt-3.5-turbo'):
    """Genera texto con la API real de OpenAI; si falla (sin creditos, sin key,
    error de red) usa Pollinations.ai como respaldo gratuito."""
    messages = [{'role': 'user', 'content': prompt}]

    client = _get_real_client()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0,
            )
            return response.choices[0].message.content.strip(), 'openai'
        except Exception as e:
            print(f'⚠️  OpenAI no disponible ({e}); usando Pollinations.ai como respaldo')

    response = _get_fallback_client().chat.completions.create(
        model=POLLINATIONS_MODEL, messages=messages,
    )
    return response.choices[0].message.content.strip(), 'pollinations'


def generate_image_bytes(prompt, width=256, height=384):
    """Genera una imagen con Pollinations.ai (gratis, sin llave) y devuelve los bytes."""
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    url = f'{IMAGE_BASE_URL}/{encoded_prompt}?width={width}&height={height}&nologo=true&model=flux'
    response = requests.get(url, timeout=60)
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
