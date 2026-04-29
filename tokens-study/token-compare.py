import os

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from google import genai
from transformers import AutoTokenizer

load_dotenv()

# ---------- Init Clients ----------
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

qwen_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    use_fast=True
)

gpt_enc = tiktoken.get_encoding("o200k_base")


# ---------- Token Counters ----------
def count_gemini_tokens(text: str) -> int:
    try:
        response = client.models.count_tokens(
            model="gemini-2.5-flash",
            contents=text,
        )
        return response.total_tokens
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return -1


def count_qwen_tokens(text: str) -> int:
    return len(qwen_tokenizer.encode(text, add_special_tokens=False))


def count_gpt_tokens(text: str) -> int:
    return len(gpt_enc.encode(text))


# ---------- Single Comparison ----------
def compare_tokens(text: str):
    gemini_count = count_gemini_tokens(text)
    qwen_count = count_qwen_tokens(text)
    gpt_count = count_gpt_tokens(text)

    print("\n" + "=" * 30)
    print("      TOKEN COMPARISON")
    print("=" * 30)
    print(f"Input: \"{text[:50]}...\"")
    print(f"{'Model':<12} | {'Tokens':<6}")
    print("-" * 22)
    print(f"{'Gemini 2.5-flash':<12} | {gemini_count:<6}")
    print(f"{'Qwen 2.5-7B':<12} | {qwen_count:<6}")
    print(f"{'GPT-4o':<12} | {gpt_count:<6}")


# ---------- Dataset Study ----------
dataset = {
    "Legal (Standard)": "The contractor shall be liable for all third-party claims.",
    "Legal (Complex)": "Notwithstanding anything to the contrary herein, the Indemnified Party shall not be entitled to indemnification for any Loss arising from its own gross negligence or willful misconduct.",
    "Technical Catalogue": "High-efficiency HVAC unit with 5000BTU capacity, IoT-enabled thermal sensors, and R-32 refrigerant.",
    "Code (Go)": "func CalculateRisk(claims []Claim) float64 { return float64(len(claims)) * 0.15 }",
    "Creative Marketing": "Experience the ultimate in luxury with our bespoke summer collection, designed for the modern professional."
}


def run_study():
    results = []

    for domain, text in dataset.items():
        g = count_gemini_tokens(text)
        q = count_qwen_tokens(text)
        gpt = count_gpt_tokens(text)

        char_count = len(text)

        results.append({
            "Domain": domain,
            "Chars": char_count,
            "Gemini": g,
            "Qwen": q,
            "GPT-4o": gpt,
            "Gemini/100ch": round((g / char_count) * 100, 2) if g > 0 else -1,
            "Qwen/100ch": round((q / char_count) * 100, 2),
            "GPT/100ch": round((gpt / char_count) * 100, 2),
        })

    df = pd.DataFrame(results)

    print("\n### TOKEN EFFICIENCY DATASET\n")
    print(df.to_markdown(index=False))

    df.to_csv("token_study_results.csv", index=False)
    print("\nSaved to token_study_results.csv")


# ---------- Main ----------
if __name__ == "__main__":
    test_text = "The contractor shall be liable for all third-party claims."

    compare_tokens(test_text)
    run_study()
