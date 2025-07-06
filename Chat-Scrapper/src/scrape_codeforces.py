import requests
from bs4 import BeautifulSoup
import os

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Cookie": "YOUR_CF_COOKIES_HERE"
}

def save_problem_statement(contest_id, index, save_dir="data/problems"):
    url = f"https://codeforces.com/contest/{contest_id}/problem/{index}"
    print(f"Fetching {url}")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to fetch {url} (status {resp.status_code})")
        return
    soup = BeautifulSoup(resp.text, "html.parser")
    statement = soup.find("div", class_="problem-statement")
    if statement:
        text = statement.get_text(separator="\n")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{contest_id}_{index}.txt"
        with open(filename, "w") as f:
            f.write(text)
        print(f"Saved {filename}")
    else:
        print(f"Problem statement not found at {url}")

# Example usage: Download problem 1A
if __name__ == "__main__":
    save_problem_statement(1, "A")
    resp = requests.get("https://codeforces.com/api/problemset.problems")
    data = resp.json()

    for problem in data['result']['problems']:
        print(f"Contest {problem['contestId']}, Problem {problem['index']}, Name: {problem['name']}")