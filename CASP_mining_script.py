import requests
from pathlib import Path

target = "T0986s2"
group = "A7D"
model = 1

candidate_urls = [
    f"https://predictioncenter.org/download_area/CASP13/predictions/{target}{group}_{model}.pdb",
    f"https://predictioncenter.org/download_area/CASP13/predictions/{target}{group}{model}.pdb",
    f"https://predictioncenter.org/download_area/CASP13/predictions/{target}.{group}_{model}.pdb",
    f"https://predictioncenter.org/download_area/CASP13/predictions/{target}/{target}{group}_{model}.pdb",
    f"https://predictioncenter.org/download_area/CASP13/predictions/regular/{target}{group}_{model}.pdb",
    f"https://predictioncenter.org/download_area/CASP13/models/{target}{group}_{model}.pdb",
    f"https://predictioncenter.org/download_area/CASP13/models/{target}/{target}{group}_{model}.pdb",
]

out = Path(f"{target}_{group}_model{model}.pdb")

for url in candidate_urls:
    print("trying:", url)
    r = requests.get(url, timeout=30)
    if r.ok and ("ATOM" in r.text or "MODEL" in r.text or "PFRMAT TS" in r.text):
        out.write_text(r.text)
        print("SUCCESS:", url)
        print("saved:", out)
        break
else:
    print("No direct URL worked.")
    print("Next step: scrape CASP results.cgi links or contact predictioncenter.org.")