"""ClipLab MVP - clips de YouTube VOD y directos Twitch/Kick/YouTube (localhost)."""
import asyncio, atexit, json, os, shutil, sqlite3, subprocess, threading, uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
for d in ("sources", "clips", "live", "tmp"):
    (DATA / d).mkdir(parents=True, exist_ok=True)

db = sqlite3.connect(DATA / "app.db", check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, url TEXT, platform TEXT, title TEXT,
  file TEXT, status TEXT, duration REAL, suggestions TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS clips(id TEXT PRIMARY KEY, source TEXT, url TEXT, start REAL, end REAL,
  score REAL, mode TEXT, file TEXT, vertical INT, created TEXT);
""")
lock = threading.Lock()
now = lambda: datetime.now().isoformat(timespec="seconds")


def q(sql, a=()):
    with lock:
        c = db.execute(sql, a)
        db.commit()
        return [dict(r) for r in c.fetchall()] if c.description else []


def platform(url):
    u = url.lower()
    return "twitch" if "twitch.tv" in u else "kick" if "kick.com" in u else "youtube"


# ---------- ffmpeg ----------
def pick_encoder():
    """Usa la GPU si ffmpeg puede (NVENC / AMF / QuickSync / VideoToolbox); si no, CPU."""
    for enc, extra in (("h264_nvenc", ["-preset", "p4"]), ("h264_amf", []), ("h264_qsv", []),
                       ("h264_videotoolbox", [])):
        try:
            r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.2",
                                "-c:v", enc, *extra, "-f", "null", "-"], capture_output=True, timeout=20)
            if r.returncode == 0:
                return ["-c:v", enc, *extra, "-b:v", "8M"]
        except Exception:
            pass
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]


ENC = pick_encoder()
print("Codificador de video:", ENC[1])


class Fmt(BaseModel):
    layout: str = "original"   # original | template | blur | split
    swap: bool = False         # split: derecha arriba
    cam_on: bool = True        # template: el streamer tiene cámara
    cam_pos: str = "top"       # template: top | bottom
    cam_h: float = 0.3         # template: alto de la cámara en el lienzo (0.2-0.4 de 1920)
    cam: list[float] = [0.75, 0, 0.22]   # cámara en el video: [x, y, ancho] normalizado
    gz: float = 1.0            # zoom del juego (1 = alto completo, menor = más cerca)
    gk: list[list[float]] = []  # movimiento del juego: [[t_seg, cx, cy], ...] (t relativo al clip)


def _c(v):
    return min(max(float(v), 0.0), 1.0)


def keyframes(f, dur):
    k = sorted(f.gk) or [[0.0, 0.5, 0.5]]

    def pos(t):
        if t <= k[0][0]:
            return k[0][1:3]
        for a, b in zip(k, k[1:]):
            if t <= b[0]:
                u = (t - a[0]) / max(b[0] - a[0], 1e-6)
                return [a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]
        return k[-1][1:3]
    ts = sorted({0.0, round(dur, 3), *[round(a[0], 3) for a in k if 0 < a[0] < dur]})
    return pos, ts


def cmd_text(f, dur):
    """Archivo para sendcmd: un tramo lineal por intervalo, mueve x/y del crop del juego."""
    pos, ts = keyframes(f, dur)
    out = []
    for a, b in zip(ts, ts[1:]):
        (xa, ya), (xb, yb) = pos(a), pos(b)
        u = f"(t-{a})/{b - a:.3f}"
        out.append(f"{a}-{b} crop@g x 'clip(lerp({xa:.4f},{xb:.4f},{u})*iw-ow/2,0,iw-ow)', "
                   f"crop@g y 'clip(lerp({ya:.4f},{yb:.4f},{u})*ih-oh/2,0,ih-oh)';")
    return "\n".join(out)


def vfilter(f):
    if f.layout == "template":  # lienzo 1080x1920: cámara a todo el ancho + juego en el resto (sin blur)
        ch = int(round(min(max(f.cam_h, 0.2), 0.4) * 1920 / 2)) * 2 if f.cam_on else 0
        gh, z = 1920 - ch, min(max(f.gz, 0.4), 1.0)
        game = f"scale=-2:{int(gh / z / 2) * 2},sendcmd=f=CMDFILE,crop@g=1080:{gh}:0:0"
        if not ch:
            return f"[0:v]{game}[v]"
        x, y, w = min(_c(f.cam[0]), 0.95), min(_c(f.cam[1]), 0.95), max(_c(f.cam[2]), 0.05)
        cam = f"crop=iw*{w}:iw*{w}/{1080 / ch:.4f}:iw*{x}:ih*{y},scale=1080:{ch}"
        order = "[cam][game]" if f.cam_pos == "top" else "[game][cam]"
        return f"[0:v]split=2[a][b];[a]{cam}[cam];[b]{game}[game];{order}vstack[v]"
    if f.layout == "blur":
        # fondo: se reduce a 54x96, se difumina y se amplía (mismo aspecto, ~3x más rápido)
        return ("[0:v]split[a][b];[a]scale=54:96:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                "crop=54:96,boxblur=2:2,scale=1080:1920:flags=bicubic[bg];[b]scale=1080:-2[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]")
    if f.layout == "split":
        half = lambda x: f"crop=iw/2:ih:{x}:0,scale=-2:960,pad=1080:960:(ow-iw)/2:0:black"
        top, bot = ("iw/2", "0") if f.swap else ("0", "iw/2")
        return f"[0:v]split[a][b];[a]{half(top)}[t];[b]{half(bot)}[u];[t][u]vstack[v]"
    return None


def encode(pre, out, f=None, copy=False, dur=0, trim=None):
    """trim=(inicio, duración) para entradas concat: hace que el tiempo del filtro arranque en 0."""
    c = ["ffmpeg", "-y", "-loglevel", "error", *pre]
    vf, cwd, cmdf = (vfilter(f) if f else None), None, None
    if vf:
        if f.layout == "template":
            cmdf = DATA / "tmp" / f"{uuid.uuid4().hex[:8]}.cmd"
            cmdf.write_text(cmd_text(f, dur))
            vf, cwd = vf.replace("CMDFILE", cmdf.name), cmdf.parent  # ruta relativa: evita problemas con "C:\\"
        if trim:
            vf = vf.replace("[0:v]", f"[0:v]trim=start={trim[0]}:duration={trim[1]},setpts=PTS-STARTPTS,", 1)
            c += ["-af", f"atrim=start={trim[0]}:duration={trim[1]},asetpts=PTS-STARTPTS"]
        c += ["-filter_complex", vf, "-map", "[v]", "-map", "0:a?"]
    if copy and not vf:
        c += ["-c", "copy", "-bsf:a", "aac_adtstoasc"]
    else:
        c += [*ENC, "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac"]
    r = subprocess.run([*c, "-movflags", "+faststart", str(out)], capture_output=True, text=True, cwd=cwd)
    if cmdf:
        cmdf.unlink(missing_ok=True)
    if r.returncode:
        raise HTTPException(500, "ffmpeg: " + r.stderr[-300:])


def save_clip(source, url, s, e, score, mode, fmt, pre, copy=False, trim=None):
    cid = uuid.uuid4().hex[:8]
    f = f"{cid}.mp4"
    encode(pre, DATA / "clips" / f, fmt, copy, dur=e - s, trim=trim)
    q("INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?,?)",  # columna "vertical" guarda el layout
      (cid, source, url, s, e, score, mode, f, fmt.layout, now()))
    return {"id": cid, "file": f}


# ---------- detección (audio RMS + mapa de calor de YouTube) ----------
def suggest(path, heat, before=20, after=10, k=8, gap=30):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    n = len(x) // 8000
    if n < 10:
        return []
    rms = np.sqrt((x[: n * 8000].reshape(n, 8000) ** 2).mean(axis=1))
    rms = np.convolve(rms / (rms.max() or 1), np.ones(3) / 3, mode="same")
    score = rms.copy()
    if heat:
        h = np.zeros(n)
        for p in heat:
            h[int(p["start_time"]): int(p["end_time"]) + 1] = p["value"]
        score = 0.5 * rms + 0.5 * h
    picks = []
    for t in np.argsort(score)[::-1]:
        if len(picks) >= k or score[t] < 0.05:
            break
        if all(abs(int(t) - o) >= gap for o in picks):
            picks.append(int(t))
    return [{"start": max(0, t - before), "end": min(n, t + after), "score": round(float(score[t]), 2)}
            for t in picks]


def ingest(sid, url):
    try:
        import yt_dlp
        opts = {"format": "bv*[height<=720]+ba/b[height<=720]", "merge_output_format": "mp4",
                "outtmpl": str(DATA / "sources" / f"{sid}.%(ext)s"), "quiet": True}
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=True)
        f = f"{sid}.mp4"
        sug = suggest(DATA / "sources" / f, info.get("heatmap"))
        q("UPDATE sources SET title=?,file=?,status='listo',duration=?,suggestions=? WHERE id=?",
          (info.get("title"), f, info.get("duration") or 0, json.dumps(sug), sid))
    except Exception as ex:
        q("UPDATE sources SET status=? WHERE id=?", (f"error: {str(ex)[:150]}", sid))


# ---------- directos: streamlink -> ffmpeg -> segmentos HLS de 2 s (buffer 20 min ≈ 1 GB a 1080p) ----------
SEG, BUF = 2, 1200
lives = {}


def kill_all():
    for l in lives.values():
        for p in (l["sl"], l["ff"]):
            p.kill()


atexit.register(kill_all)


def read_segs(lid):
    """Guarda la duración de cada segmento del playlist, por número de secuencia."""
    try:
        lines = (DATA / "live" / lid / "live.m3u8").read_text().splitlines()
    except OSError:
        return
    dur = SEG
    for ln in lines:
        if ln.startswith("#EXTINF:"):
            dur = float(ln[8:].split(",")[0])
        elif ln.endswith(".ts"):
            lives[lid]["dur"][int(ln[1:7])] = dur


def live_pin(lid, seq):
    """Al marcar el inicio, protege (hardlink) los segmentos desde ahí para que el buffer no los borre."""
    d = DATA / "live" / lid
    (d / "pin").mkdir(exist_ok=True)
    read_segs(lid)
    for p in d.glob("s*.ts"):
        dst = d / "pin" / p.name
        if int(p.name[1:7]) >= seq and not dst.exists():
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)


def live_clip(lid, a, b, fmt):
    L, d = lives[lid], DATA / "live" / lid
    if b.seq < a.seq:
        raise HTTPException(400, "El fin debe ir después del inicio")
    read_segs(lid)
    seqs = list(range(a.seq, b.seq + 1))
    paths = []
    for n in seqs:
        p = d / f"s{n:06d}.ts"
        p = p if p.exists() else d / "pin" / p.name
        if not p.exists():
            raise HTTPException(410, "Esa parte ya salió del buffer")
        paths.append(p)
    durs = [L["dur"].get(n, SEG) for n in seqs]
    total = sum(durs) - a.off - (durs[-1] - b.off)
    if total <= 0:
        raise HTTPException(400, "El fin debe ir después del inicio")
    lst = d / "list.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths))
    copy = fmt.layout == "original"  # original: instantáneo, a nivel de segmento (±2 s); vertical: corte exacto
    pre = ["-f", "concat", "-safe", "0", "-i", str(lst)]
    res = save_clip(lid, L["url"], 0, round(sum(durs) if copy else total, 1), 0, "live", fmt, pre, copy,
                    None if copy else (a.off, total))
    shutil.rmtree(d / "pin", ignore_errors=True)
    return res


# ---------- API ----------
app = FastAPI(title="ClipLab")


class UrlIn(BaseModel):
    url: str


class ClipIn(Fmt):
    source: str
    start: float
    end: float
    mode: str = "manual"
    score: float = 0


class Mark(BaseModel):
    seq: int          # número de segmento (sNNNNNN.ts)
    off: float = 0    # segundos dentro de ese segmento


class LiveClipIn(Fmt):
    start: Mark
    end: Mark


@app.post("/api/sources")
def add_source(b: UrlIn):
    sid = uuid.uuid4().hex[:8]
    q("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?)",
      (sid, b.url, platform(b.url), "", "", "descargando", 0, "[]", now()))
    threading.Thread(target=ingest, args=(sid, b.url), daemon=True).start()
    return {"id": sid}


@app.get("/api/sources")
def list_sources():
    rows = q("SELECT * FROM sources ORDER BY created DESC")
    for r in rows:
        r["suggestions"] = json.loads(r["suggestions"] or "[]")
    return rows


@app.post("/api/clips")
def make_clip(b: ClipIn):
    s = q("SELECT * FROM sources WHERE id=?", (b.source,))
    if not s or s[0]["status"] != "listo":
        raise HTTPException(404, "Fuente no lista")
    if b.end <= b.start:
        raise HTTPException(400, "El fin debe ser mayor que el inicio")
    pre = ["-ss", str(b.start), "-i", str(DATA / "sources" / s[0]["file"]), "-t", str(b.end - b.start)]
    return save_clip(b.source, s[0]["url"], b.start, b.end, b.score, b.mode, b, pre)


@app.get("/api/clips")
def list_clips():
    return q("SELECT * FROM clips ORDER BY created DESC")


@app.delete("/api/clips/{cid}")
def del_clip(cid: str):
    for r in q("SELECT file FROM clips WHERE id=?", (cid,)):
        (DATA / "clips" / r["file"]).unlink(missing_ok=True)
    q("DELETE FROM clips WHERE id=?", (cid,))
    return {"ok": True}


@app.post("/api/live")
def live_start(b: UrlIn):
    if not (shutil.which("streamlink") and shutil.which("ffmpeg")):
        raise HTTPException(500, "Faltan streamlink o ffmpeg en el PATH")
    lid = uuid.uuid4().hex[:8]
    d = DATA / "live" / lid
    d.mkdir(parents=True)
    sl = subprocess.Popen(["streamlink", "-O", b.url, "best"], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-c", "copy", "-f", "hls",
                           "-hls_time", str(SEG), "-hls_list_size", str(BUF // SEG),
                           "-hls_flags", "delete_segments",
                           "-hls_segment_filename", str(d / "s%06d.ts"), str(d / "live.m3u8")],
                          stdin=sl.stdout)
    sl.stdout.close()
    lives[lid] = {"url": b.url, "platform": platform(b.url), "sl": sl, "ff": ff, "dur": {}}
    return {"id": lid}


@app.get("/api/live")
def live_list():
    return [{"id": k, "url": v["url"], "platform": v["platform"], "alive": v["ff"].poll() is None}
            for k, v in lives.items()]


@app.post("/api/live/{lid}/pin")
def live_pin_ep(lid: str, m: Mark):
    if lid not in lives:
        raise HTTPException(404, "Directo no encontrado")
    live_pin(lid, m.seq)
    return {"ok": True}


@app.post("/api/live/{lid}/clip")
def live_clip_ep(lid: str, b: LiveClipIn):
    if lid not in lives:
        raise HTTPException(404, "Directo no encontrado")
    return live_clip(lid, b.start, b.end, b)


@app.delete("/api/live/{lid}")
def live_stop(lid: str):
    l = lives.pop(lid, None)
    if l:
        l["sl"].kill()
        l["ff"].kill()
        shutil.rmtree(DATA / "live" / lid, ignore_errors=True)
    return {"ok": True}


app.mount("/media", StaticFiles(directory=DATA), name="media")


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")
