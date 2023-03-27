import json
from pathlib import Path
import sqlite3
import cloudpickle
import subprocess
import tempfile
import inspect
from contextlib import contextmanager
import shutil


def _main_stub():
    # this function is used to run the function
    # it is called by the temporary file
    import argparse
    import nvtx
    import cloudpickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--runfile", type=Path)
    args = parser.parse_args()

    # load the function data
    data = cloudpickle.load(open(args.runfile, "rb"))

    func, args, kwargs = data["func"], data["args"], data["kwargs"]

    # run the function
    nvtx.push_range(f"run")
    func(*args, **kwargs)
    nvtx.pop_range()

@contextmanager
def run(func, *args, **kwargs):
    # create a temporary file to store the function data
    f = tempfile.NamedTemporaryFile(delete=False)
    cloudpickle.dump({"func": func, "args": args, "kwargs": kwargs}, f)
    f.flush()
    f.close()

    # copy the stub function to a temporary file
    stub = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    source = inspect.getsource(_main_stub)
    stub.write(source)
    stub.write("_main_stub()")
    stub.flush()

    # create temporary directory for the output
    output_dir = Path(tempfile.mkdtemp())

    # run the stub file
    subprocess.run(
        [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt,cudnn",
            "--show-output=false",
            "--force-overwrite=true",
            "--sample=cpu",
            "--cuda-memory-usage=true",
            "--output=" + str(output_dir) + "/profile",
            "--export=sqlite",
            "python",
            stub.name,
            "--runfile",
            f.name,
        ],
    )

    sqlite_file = output_dir / "profile.sqlite"
    try:
        yield sqlite_file
    finally:
        shutil.rmtree(output_dir)

def query_results_to_json(cursor):
    columns = [column[0] for column in cursor.description]
    result_dicts = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return json.dumps(result_dicts, ensure_ascii=False)

class NsightAnalyzer():
    def __init__(self, sqlite_file: Path) -> None:
        self._sqlite_file = sqlite_file
        self._conn = sqlite3.connect(sqlite_file)
    
    def get_nvtx_events(self):
        c = self._conn.cursor()
        c.execute("SELECT * FROM NVTX_EVENTS")
        return query_results_to_json(c)

    def get_gpu_info(self):
        c = self._conn.cursor()
        c.execute("SELECT * FROM TARGET_INFO_GPU")
        return query_results_to_json(c)

if __name__ == "__main__":
    def foo(a, b):
        return a + b

    with run(foo, 1, 2) as s:
        analyzer = NsightAnalyzer(s)
        print(analyzer.get_nvtx_events())
