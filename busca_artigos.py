import os
import re
import requests
from bs4 import BeautifulSoup
import time
import sys

# Garante saída UTF-8 se o terminal permitir
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

PASTA_PDFS = "pdfs"

artigos = [
    {
        "titulo": "An empirical comparison of supervised learning algorithms",
        "autores": "Caruana, R.; Niculescu-Mizil, A."
    },
    {
        "titulo": "Statistical comparisons of classifiers over multiple data sets",
        "autores": "Demšar, J."
    },
    {
        "titulo": "UCI Machine Learning Repository",
        "autores": "Dua, D.; Graff, C."
    },
    {
        "titulo": "Statistical comparison of classifiers using multiple datasets",
        "autores": "Fernandes, J. L. et al."
    },
    {
        "titulo": "Matplotlib: A 2D graphics environment",
        "autores": "Hunter, J. D."
    },
    {
        "titulo": "Titanic - Machine Learning from Disaster",
        "autores": "Kaggle"
    },
    {
        "titulo": "pandas-dev/pandas: Pandas",
        "autores": "Pandas Development Team"
    },
    {
        "titulo": "Scikit-learn: Machine Learning in Python",
        "autores": "Pedregosa, F. et al."
    },
    {
        "titulo": "Machine learning in Python",
        "autores": "Scikit-learn"
    },
    {
        "titulo": "Seaborn: Statistical Data Visualization",
        "autores": "Waskom, M."
    }
]

def criar_pasta_pdf():
    if not os.path.exists(PASTA_PDFS):
        os.makedirs(PASTA_PDFS)

def baixar_pdf(url_pdf, nome_arquivo):
    print(f"-> Tentando baixar PDF de {url_pdf}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        resposta = requests.get(url_pdf, stream=True, timeout=15, headers=headers)
        resposta.raise_for_status()
        content_type = resposta.headers.get('Content-Type', '').lower()
        if "pdf" not in content_type:
            print("[ERRO] Conteúdo não é PDF.")
            return False
        with open(nome_arquivo, "wb") as f:
            for chunk in resposta.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[OK] Baixado com sucesso: {nome_arquivo}")
        return True
    except Exception as e:
        print("[ERRO] Erro ao baixar PDF: {}".format(str(e)), flush=True)
        return False

def buscar_pdf_semanticscholar(titulo, autores):
    print("[INFO] Buscando no Semantic Scholar...")
    query = f"{titulo} {autores}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&limit=1&fields=title,authors,openAccessPdf"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        papers = data.get("data", [])
        if papers and papers[0].get("openAccessPdf"):
            return papers[0]["openAccessPdf"]["url"], papers[0]["title"]
    except Exception as e:
        print(f"[ERRO] Semantic Scholar: {e}")
    return None, None

def buscar_pdf_arxiv(titulo):
    print("[INFO] Buscando no arXiv...")
    base = "http://export.arxiv.org/api/query?search_query=all:{}&max_results=1"
    query = requests.utils.quote(titulo)
    url = base.format(query)
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        if "<entry>" in r.text:
            match = re.search(r'<link title="pdf" href="([^"]+)"', r.text)
            if match:
                return match.group(1), titulo
    except Exception as e:
        print(f"[ERRO] arXiv: {e}")
    return None, None

def buscar_pdf_duckduckgo(titulo):
    print("[INFO] Buscando no DuckDuckGo (busca genérica)...")
    search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(titulo + ' filetype:pdf')}"
    try:
        r = requests.get(search_url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=True)
        for link in links:
            href = link['href']
            if href.lower().endswith(".pdf"):
                return href, titulo
    except Exception as e:
        print(f"[ERRO] DuckDuckGo: {e}")
    return None, None

def limpar_nome_arquivo(texto):
    texto = texto.strip()
    texto = re.sub(r'[\\/*?:"<>|]', "_", texto)
    if len(texto) > 50:
        texto = texto[:50]
    return texto

def main():
    criar_pasta_pdf()
    for i, artigo in enumerate(artigos, 1):
        print(f"\n=== Artigo {i}: {artigo['titulo']} ===")

        url_pdf, titulo_encontrado = buscar_pdf_semanticscholar(artigo['titulo'], artigo['autores'])
        if not url_pdf:
            url_pdf, titulo_encontrado = buscar_pdf_arxiv(artigo['titulo'])
        if not url_pdf:
            url_pdf, titulo_encontrado = buscar_pdf_duckduckgo(artigo['titulo'])

        if url_pdf:
            nome_arquivo = os.path.join(PASTA_PDFS, f"{i}_{limpar_nome_arquivo(titulo_encontrado)}.pdf")
            if not baixar_pdf(url_pdf, nome_arquivo):
                print("[ERRO] Falha ao baixar PDF.")
        else:
            print(f"[FALHA] Não foi possível encontrar PDF para: {artigo['titulo']}")

        time.sleep(3)

if __name__ == "__main__":
    main()
