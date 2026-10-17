import os
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect


def login(timeout_seconds=300):
    """Complete browser login and return an authenticated Kite client."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Set KITE_API_KEY and KITE_API_SECRET in your .env."
        )

    kite = KiteConnect(api_key=api_key)
    state = secrets.token_urlsafe(32)
    request_token = None
    login_failed = False

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Default HTTP logs would expose the request token in the URL.
            pass

        def do_GET(self):
            nonlocal request_token, login_failed

            url = urlsplit(self.path)
            params = parse_qs(url.query)

            if url.path != "/callback":
                self.send_error(404, "Not found")
                return

            returned_state = params.get("state", [""])[0]
            if not secrets.compare_digest(returned_state, state):
                self.send_error(403, "Invalid login state")
                return

            token = params.get("request_token", [""])[0]
            if params.get("status") != ["success"] or not token:
                login_failed = True
                self.send_error(400, "Login failed. Restart the script to retry.")
                return

            request_token = token
            body = (
                b"Login callback received. You can close this tab "
                b"and check your terminal."
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

    # Kite echoes redirect_params back as callback query parameters.
    login_url = kite.login_url() + "&" + urlencode({
        "redirect_params": urlencode({"state": state}),
    })

    # Bind before opening the browser so the callback is ready.
    with HTTPServer(("127.0.0.1", 8000), CallbackHandler) as server:
        server.timeout = 1
        deadline = time.monotonic() + timeout_seconds

        print("Sign in to Zerodha in your browser.")
        if not webbrowser.open(login_url):
            print("Could not open the browser. Open this URL manually:")
            print(login_url)

        while request_token is None and not login_failed:
            if time.monotonic() >= deadline:
                raise TimeoutError("Login timed out. Run the script again.")
            server.handle_request()

    if login_failed:
        raise RuntimeError("Zerodha login failed. Run the script again.")

    # Sets the client's access token automatically; API errors propagate.
    kite.generate_session(request_token, api_secret=api_secret)
    set_key(env_path, "KITE_ACCESS_TOKEN", kite.access_token)
    return kite


if __name__ == "__main__":
    kite = login()
    print("Authenticated successfully. The Kite client is ready.")

    # Call kite.historical_data(...) here.
    # The token is available as kite.access_token if needed.