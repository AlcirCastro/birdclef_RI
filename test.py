from pathlib import Path

from bird_search.bootstrap import build_application


if __name__ == "__main__":
    app = build_application(Path(__file__).resolve().parent)
    app.run(debug=True, use_reloader=False)