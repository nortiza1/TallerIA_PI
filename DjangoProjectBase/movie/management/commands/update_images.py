import os
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from movie.models import Movie
from movie.ai_utils import generate_image_bytes


class Command(BaseCommand):
    help = "Generate an AI image (OpenAI/HuggingFace) and update movie image field"

    def handle(self, *args, **kwargs):
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        for movie in movies:
            try:
                image_bytes = generate_image_bytes(f"Movie poster of {movie.title}")

                image_filename = f"m_{movie.title}.png"
                movie.image.save(image_filename, ContentFile(image_bytes), save=True)

                self.stdout.write(self.style.SUCCESS(f"Saved and updated image for: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {e}")

            # Solo se procesa la primera pelicula, para demostracion
            break

        self.stdout.write(self.style.SUCCESS("Process finished (only first movie updated)."))
