from django.core.management.base import BaseCommand
from movie.models import Movie
from movie.ai_utils import get_embedding


class Command(BaseCommand):
    help = "Generate and store embeddings for all movies in the database"

    def handle(self, *args, **kwargs):
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in the database")

        for movie in movies:
            try:
                emb = get_embedding(movie.description)
                movie.emb = emb.tobytes()
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"👌 Embedding stored for: {movie.title}"))
            except Exception as e:
                self.stderr.write(f"❌ Failed to generate embedding for {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS("🎯 Finished generating embeddings for all movies"))
