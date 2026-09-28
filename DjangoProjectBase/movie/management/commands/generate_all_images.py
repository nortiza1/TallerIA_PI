"""
Reemplazo del paso 'update_images_from_folder' del Taller 3.

El profesor indico que el enlace de Sharepoint con las imagenes ya generadas
se rompio, y autorizo generarlas de forma independiente. Este comando genera
una imagen (poster) con IA para cada pelicula de la base de datos usando
Pollinations.ai (gratis, sin llave) y actualiza el campo image.

Uso:
    python manage.py generate_all_images            # todas las peliculas sin imagen generada
    python manage.py generate_all_images --limit 10 # solo las primeras N (pruebas rapidas)
"""
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from movie.models import Movie
from movie.ai_utils import generate_image_bytes


class Command(BaseCommand):
    help = "Generate an AI-generated poster image for every movie (Pollinations.ai)"

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=None)

    def handle(self, *args, **options):
        movies = Movie.objects.all()
        if options['limit']:
            movies = movies[:options['limit']]

        total = len(movies)
        self.stdout.write(f"Found {total} movies")

        updated, failed = 0, 0
        for i, movie in enumerate(movies, start=1):
            try:
                image_bytes = generate_image_bytes(f"Movie poster of {movie.title}")
                image_filename = f"m_{movie.title}.png"
                movie.image.save(image_filename, ContentFile(image_bytes), save=True)
                updated += 1
                self.stdout.write(self.style.SUCCESS(f"[{i}/{total}] Imagen generada para: {movie.title}"))
            except Exception as e:
                failed += 1
                self.stderr.write(f"[{i}/{total}] Fallo con {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"Finished. {updated} imagenes generadas, {failed} fallidas."
        ))
