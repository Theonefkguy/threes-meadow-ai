"""Build report.html (V47 human game review) from report/template.html and analysis.json."""
from pathlib import Path
L = Path(__file__).resolve().parent
data = (L.parent / 'analysis.json').read_text().strip()
(L.parent / 'report.html').write_text((L / 'template.html').read_text().replace('__DATA__', data))
