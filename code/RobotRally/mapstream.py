"""Stream images to a web browser as MJPEG over HTTP.

Much lighter than SSH X forwarding: each frame is sent as a ~20-40 KB JPEG
instead of raw pixels. Open http://<robot>:<port>/ in a browser to watch.
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"frame"

PAGE = b"""<!doctype html>
<html><head><title>Arlo world view</title>
<style>body{margin:0;background:#222;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100vw;max-height:100vh;image-rendering:pixelated}</style></head>
<body><img src="/stream"></body></html>"""


class MapStreamer(object):
    """Serves the most recently published image as an MJPEG stream."""

    def __init__(self, port=8000, jpeg_quality=80):
        self.jpeg_quality = jpeg_quality
        self._jpeg = None
        self._frame_id = 0
        self._cond = threading.Condition()

        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    last_id = -1
                    try:
                        while True:
                            jpeg, last_id = streamer._wait_for_frame(last_id)
                            self.wfile.write(b"--" + BOUNDARY + b"\r\n"
                                             b"Content-Type: image/jpeg\r\n"
                                             b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                                             + jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # browser closed the page
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass  # keep the robot's console clean

        self._server = ThreadingHTTPServer(("", port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"mapstream.py: streaming on http://<this host>:{port}/")

    def publish(self, img):
        """Encode img (BGR numpy array) as JPEG and hand it to connected browsers."""
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._frame_id += 1
            self._cond.notify_all()

    def _wait_for_frame(self, last_id):
        """Block until a frame newer than last_id is available."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id and self._jpeg is not None)
            return self._jpeg, self._frame_id

    def close(self):
        self._server.shutdown()
        self._server.server_close()
