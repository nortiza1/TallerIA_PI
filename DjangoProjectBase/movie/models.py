from django.db import models
import numpy as np

# create your models here

# Dimensión del modelo de embeddings usado (all-MiniLM-L6-v2 -> 384)
EMBEDDING_DIM = 384


def get_default_array():
    """Embedding por defecto (vector aleatorio) para peliculas que aun no tienen uno calculado."""
    default_arr = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    return default_arr.tobytes()


class Movie(models.Model):
    title = models.CharField(max_length=100)
    description = models.CharField(max_length=1500)
    image = models.ImageField(upload_to='movie/images/', default='movie/images/default.jpg')
    url = models.URLField(blank=True)
    genre = models.CharField(blank=True, max_length=250)
    year = models.IntegerField(blank=True, null=True)
    emb = models.BinaryField(default=get_default_array)

    def __str__(self):
        return self.title
