"""Convert all examples to python files"""

from pathlib import Path
import nbconvert

file_dir = Path(__file__).parent
for file in file_dir.glob("**/*.ipynb"):
    body, _ = nbconvert.PythonExporter().from_file(file)
    with open(file.with_suffix(".py"), "w") as f:
        f.write(body)
