import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load the data
df = pd.read_csv("token_study_results.csv")

# 2. Set up the plot
plt.style.use('ggplot') # Clean, academic look
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(df['Domain']))
width = 0.25

# 3. Create bars for 'Tokens per 100 Characters'
# This normalizes the data regardless of how long the input text was
rects1 = ax.bar(x - width, df['Gemini/100ch'], width, label='Gemini 2.5-flash', color='#4285F4')
rects2 = ax.bar(x, df['Qwen/100ch'], width, label='Qwen 2.5-7B (Local)', color='#34A853')
rects3 = ax.bar(x + width, df['GPT/100ch'], width, label='GPT-4o', color='#EA4335')

# 4. Add labels and styling
ax.set_ylabel('Tokens per 100 Characters (Lower is Better)')
ax.set_title('Tokenization Efficiency Across Domains', fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['Domain'], rotation=15)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()

# 5. Save the 'Proof'
plt.savefig('token_efficiency_comparison.png', dpi=300)
print("\n📊 Visualization saved as 'token_efficiency_comparison.png'")
