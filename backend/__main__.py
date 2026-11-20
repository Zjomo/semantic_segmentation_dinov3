import os

from .app import create_app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.getenv("CORESG_HOST", "127.0.0.1"),
        port=int(os.getenv("CORESG_PORT", "5000")),
        debug=os.getenv("CORESG_DEBUG", "0") == "1",
    )
