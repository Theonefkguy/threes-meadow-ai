"""Build report.html (the V46 picture report) from report/template.html and features.json."""
from pathlib import Path
L = Path(__file__).resolve().parent
data = (L.parent / 'features.json').read_text().strip()
(L.parent / 'report.html').write_text((L / 'template.html').read_text().replace('__DATA__', data))
