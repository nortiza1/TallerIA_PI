from django.core.management.base import BaseCommand
from movie.models import Movie
from movie.ai_utils import get_completion


class Command(BaseCommand):
    help = "Update movie descriptions using the OpenAI API (fallback: Pollinations.ai)"

    def handle(self, *args, **kwargs):
        # Instruccion que guia la respuesta de la IA
        instruction = (
            "Vas a actuar como un aficionado del cine que sabe describir de forma clara, "
            "concisa y precisa cualquier pelicula en menos de 200 palabras. La descripcion "
            "debe incluir el genero de la pelicula y cualquier informacion adicional que sirva "
            "para crear un sistema de recomendacion."
        )

        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        for movie in movies:
            self.stdout.write(f"Processing: {movie.title}")
            try:
                prompt = (
                    f"{instruction} "
                    f"Vas a actualizar la descripcion '{movie.description}' de la pelicula '{movie.title}'."
                )

                print(f"Title: {movie.title}")
                print(f"Original Description: {movie.description}")

                updated_description, source = get_completion(prompt)

                print(f"Updated Description ({source}): {updated_description}")

                movie.description = updated_description
                movie.save()

                self.stdout.write(self.style.SUCCESS(f"Updated ({source}): {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {str(e)}")

            # No quitar el break: solo se actualiza la primera pelicula para ahorrar recursos
            break
