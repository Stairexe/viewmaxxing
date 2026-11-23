"""Stand-in for vf.pipeline in tests: behaves like a finished job."""
import json, sys, time
from pathlib import Path

d = Path(sys.argv[1])
opts = json.loads((d / "job.json").read_text())
if opts.get("fail"):
    print("boom"); sys.exit(3)
(d / "output").mkdir(exist_ok=True)
(d / "output" / "scene.html").write_text("<html>splat</html>")
(d / "secret.txt").write_text("not published")
(d / "status.json").write_text(json.dumps({"state": "done", "stage": None, "done": ["prepare", "export"],
                                           "outputs": ["output/scene.html"], "message": "finished"}))
print("fake pipeline ok", len(list((d / "input").iterdir())), "photos")
