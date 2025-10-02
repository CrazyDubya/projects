import io
from shareconvo import create_app
from workers import WorkerEntrypoint, Response

class Entrypoint(WorkerEntrypoint):
    """
    This is the entrypoint for the Cloudflare Worker.
    It translates the Cloudflare request into a WSGI-compatible format
    that our Flask application can understand.
    """

    # Create the Flask app instance when the worker is initialized
    app = create_app()

    async def fetch(self, request, env, ctx):
        """
        This is the main method that handles incoming requests.
        """

        # NOTE TO USER: The primary challenge for deploying this application to
        # Cloudflare is the database. The Flask app uses Flask-SQLAlchemy, which
        # expects a standard database connection (like a file or a network URI).
        # Cloudflare Workers use a specific API to access the D1 database (via `env.DB`).
        #
        # To make this work, a custom SQLAlchemy dialect for D1 would be needed
        # to bridge the gap between SQLAlchemy's requirements and D1's API.
        # This script does NOT include such a dialect.
        #
        # The code below is a basic WSGI bridge that handles the HTTP request
        # and response translation, but it will fail when the app tries to
        # connect to the database.

        # A simplified WSGI environment. A real implementation would be more complex.
        environ = {
            'REQUEST_METHOD': request.method,
            'SCRIPT_NAME': '',
            'PATH_INFO': request.path,
            'QUERY_STRING': request.query,
            'CONTENT_TYPE': request.headers.get('Content-Type', ''),
            'CONTENT_LENGTH': request.headers.get('Content-Length', ''),
            'SERVER_NAME': 'shareconvo.com',
            'SERVER_PORT': '443',
            'SERVER_PROTOCOL': 'HTTP/1.1',
            'wsgi.version': (1, 0),
            'wsgi.url_scheme': 'https',
            'wsgi.input': io.BytesIO(await request.body()),
            'wsgi.errors': io.StringIO(),
            'wsgi.multithread': False,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
        }

        # A simple start_response function that captures the status and headers
        response_status = None
        response_headers = None
        def start_response(status, headers):
            nonlocal response_status, response_headers
            response_status = status
            response_headers = headers

        # Call the Flask app's WSGI interface
        response_body_iter = self.app.wsgi_app(environ, start_response)
        response_body = b''.join(response_body_iter)

        # Create the final Response object for Cloudflare Workers
        return Response(
            response_body,
            status=int(response_status.split(' ')[0]),
            headers=dict(response_headers)
        )
