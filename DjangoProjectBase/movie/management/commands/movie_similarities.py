from django.core.management.base import BaseCommand
from movie.models import Movie
from movie.ai_utils import get_embedding, cosine_similarity


class Command(BaseCommand):
    help = "Compare two movies and a prompt using semantic embeddings"

    def handle(self, *args, **kwargs):
        # Cambia estos titulos por cualquier par de peliculas que quieras comparar
        try:
            movie1 = Movie.objects.get(title="A Trip to the Moon")
            movie2 = Movie.objects.get(title="The Great Train Robbery")
        except Movie.DoesNotExist:
            movies = list(Movie.objects.all()[:2])
            movie1, movie2 = movies[0], movies[1]

        # Generar embeddings de ambas peliculas
        emb1 = get_embedding(movie1.description)
        emb2 = get_embedding(movie2.description)

        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(self.style.SUCCESS(
            f"🎬 Similaridad entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}"
        ))

        # Comparar contra un prompt libre
        prompt = "pelicula de ciencia ficcion sobre un viaje espacial"
        prompt_emb = get_embedding(prompt)

        sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(self.style.SUCCESS(f"📝 Prompt: '{prompt}'"))
        self.stdout.write(self.style.SUCCESS(
            f"📝 Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"📝 Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}"
        ))
