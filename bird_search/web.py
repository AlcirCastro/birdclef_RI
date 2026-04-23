from flask import Flask, abort, render_template_string, request, send_from_directory

from bird_search.embedding import Embedder
from bird_search.evaluation import evaluate
from bird_search.models import Record
from bird_search.search import SearchIndex
from bird_search.settings import Settings

HTML = """<!doctype html>
<html lang=\"pt-BR\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Bird Audio Search</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Manrope:wght@400;600;700&display=swap');
  :root {
    --bg: #f2f0e8;
    --bg2: #dce6d5;
    --ink: #152019;
    --muted: #5f6f62;
    --card: #ffffffd9;
    --line: #d4ddd1;
    --brand: #1f4a30;
    --accent: #8c5f3e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
    min-height: 100vh;
    background:
      radial-gradient(1200px 400px at -10% -10%, #b8caa9 10%, transparent 70%),
      radial-gradient(700px 350px at 110% 20%, #dec7a3 8%, transparent 65%),
      linear-gradient(180deg, var(--bg), var(--bg2));
  }
  header { padding: 26px 18px 14px; text-align: center; }
  h1 { margin: 0; font-family: 'Fraunces', serif; font-size: clamp(1.35rem, 3.5vw, 2rem); font-weight: 500; }
  .meta { margin-top: 8px; color: var(--muted); font-size: .85rem; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 10px 16px 32px; }
  .card {
    backdrop-filter: blur(4px);
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px;
    margin-top: 12px;
  }
  .label { font-size: .92rem; font-weight: 700; margin-bottom: 8px; }
  .hint { font-size: .8rem; color: var(--muted); margin-bottom: 10px; }
  .drop {
    border: 2px dashed #b4c6b6;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    background: #edf3eb;
    cursor: pointer;
  }
  .drop:hover { border-color: #8ea991; }
  input[type=file] { display: none; }
  #filename { margin-top: 6px; font-size: .8rem; color: var(--muted); }
  button {
    margin-top: 10px;
    border: 0;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 700;
    cursor: pointer;
  }
  .primary { background: var(--brand); color: #fff; }
  .secondary { background: transparent; color: var(--brand); border: 2px solid var(--brand); }
  .error { color: #7c1f1f; background: #ffe7e7; border-left: 4px solid #bd4a4a; padding: 10px; border-radius: 8px; }
  .results { margin-top: 14px; display: grid; gap: 10px; }
  .result { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
  .row { display: flex; gap: 10px; align-items: center; }
  .rank { width: 30px; height: 30px; border-radius: 50%; background: var(--brand); color: #fff; display: flex; align-items: center; justify-content: center; font-size: .8rem; font-weight: 700; }
  .name { font-weight: 700; }
  .sub { font-size: .8rem; color: var(--muted); }
  .score { margin-top: 6px; font-size: .78rem; color: var(--accent); }
  audio { width: 100%; margin-top: 8px; }
</style>
</head>
<body>
<header>
  <h1>BirdCLEF Similarity Search</h1>
  <div class=\"meta\">{{ total }} arquivos indexados | embedding: {{ embedding_name }} | backend: FAISS</div>
</header>
<div class=\"wrap\">
  <div class="meta">{{ total }} arquivos indexados | embedding: {{ embedding_name }} | backend: Elasticsearch</div>
  <div class=\"card\">
    <div class=\"label\">Buscar audio similar</div>
    <div class=\"hint\">Embedding robusto com multi-janela e normalizacao L2.</div>
    <form method=\"post\" enctype=\"multipart/form-data\">
      <div class=\"drop\" onclick=\"document.getElementById('fi').click()\">
        Clique para enviar um arquivo de audio
        <input id=\"fi\" type=\"file\" name=\"audio\" accept=\"audio/*\" onchange=\"document.getElementById('filename').textContent=this.files[0]?.name||''\">
      </div>
      <div id=\"filename\"></div>
      <button class=\"primary\">Buscar</button>
    </form>
  </div>

  <div class=\"card\">
    <div class=\"label\">Avaliacao</div>
    <div class=\"hint\">MAP, MRR, P@5, Top-1, Top-5.</div>
    <form method=\"post\" action=\"/eval\"><button class=\"secondary\">Executar benchmark</button></form>
    {% if eval_result %}
      <div style=\"margin-top:10px;font-size:.88rem\">
        <div style="margin-bottom:6px;color:var(--muted)">{{ eval_result.get('eval_mode', 'standard') }}</div>
        MAP {{ '%.4f'|format(eval_result['MAP']) }} | MRR {{ '%.4f'|format(eval_result['MRR']) }} | P@5 {{ '%.4f'|format(eval_result['P@5']) }}<br>
        Top-1 {{ '%.4f'|format(eval_result['Top-1']) }} | Top-5 {{ '%.4f'|format(eval_result['Top-5']) }} | avg {{ '%.1f'|format(eval_result['avg_query_ms']) }} ms
      </div>
    {% endif %}
  </div>

  {% if results %}
  <div class=\"results\">
    {% for r in results %}
      <div class=\"result\">
        <div class=\"row\"><div class=\"rank\">{{ loop.index }}</div><div><div class=\"name\">{{ r.common_name or 'Unknown' }}</div><div class=\"sub\">{{ r.scientific_name }} | {{ r.primary_label }} | rating {{ '%.1f'|format(r.rating) }}</div></div></div>
        <div class=\"score\">score {{ '%.5f'|format(r.score) }}</div>
        <audio controls preload=\"none\"><source src=\"/audio/{{ r.item_id }}\"></audio>
      </div>
    {% endfor %}
  </div>
  {% endif %}
</div>
</body>
</html>
"""


def create_app(
    settings: Settings,
    embedder: Embedder,
    search_index: SearchIndex,
    train_records: list[Record],
    test_records: list[Record],
) -> Flask:
    app = Flask(__name__)
    by_id = {r.item_id: r for r in train_records}

    @app.route("/audio/<int:item_id>")
    def audio(item_id: int):
        rec = by_id.get(item_id)
        if rec is None:
            abort(404)
        return send_from_directory(rec.local_path.parent, rec.local_path.name)

    def render(error: str | None = None, results: list[dict] | None = None, eval_result: dict | None = None):
        return render_template_string(
            HTML,
            error=error,
            results=results or [],
            eval_result=eval_result,
            total=len(by_id),
          embedding_name=settings.embedding_name,
        )

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return render()

        file = request.files.get("audio")
        if file is None or not file.filename:
            return render(error="Nenhum arquivo enviado")

        vec = embedder.embed_bytes(file.read())
        if vec is None:
            return render(error="Nao foi possivel processar o audio")

        try:
            hits = search_index.knn_search(vec, k=settings.top_k)
        except Exception as exc:
            return render(error=f"Busca falhou: {exc}")

        results = [
            {
                "item_id": h["_source"]["item_id"],
                "primary_label": h["_source"]["primary_label"],
                "scientific_name": h["_source"].get("scientific_name", ""),
                "common_name": h["_source"].get("common_name", ""),
                "rating": float(h["_source"].get("rating", 0.0)),
                "score": float(h.get("_score", 0.0)),
            }
            for h in hits
        ]

        return render(results=results)

    @app.route("/eval", methods=["POST"])
    def run_eval():
        result = evaluate(test_records, embedder=embedder, search_index=search_index, k=settings.eval_top_k)
        return render(eval_result=result)

    return app
