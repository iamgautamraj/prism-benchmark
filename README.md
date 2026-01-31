---
title: PRISM Benchmark
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.28.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 🔬 PRISM Benchmark

**P**rior-art **I**ntelligence **S**core for **M**odels

> Testing if AI models tell you the truth about existing startup ideas — or just agree with everything.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/spaces/your-space)

---

## 🎯 What is PRISM?

PRISM is a benchmark that measures **AI honesty in startup ideation**. 

When you ask an AI for a "unique startup idea," does it:
- ✅ Tell you when your idea already exists (like Klarna, Airbnb, or Flexport)?
- ❌ Or just validate your "brilliant" idea that's been done 100 times?

We tested **17 LLMs** with 30 prompts based on **existing billion-dollar companies**.

---

## 📊 Key Results

| Rank | Model | PRISM Score | Grade |
|------|-------|-------------|-------|
| 1 | Gemini 3 Flash | 97.0 | A+ |
| 2 | Gemini 3 Pro | 96.5 | A+ |
| 3 | Claude Sonnet 4.5 | 96.0 | A+ |
| 4 | Claude Opus 4.5 | 95.0 | A+ |
| 5 | Kimi-K2 | 94.5 | A |
| ... | ... | ... | ... |
| 17 | Llama-3.1-8B | 58.5 | D |

**Key Finding:** Top models achieved 0% sycophancy on stress tests, while the worst model failed 46% of the time.

---

## 📖 Understanding the Scores

| Score | Technical Name | In Simple Terms | Good or Bad? |
|-------|----------------|-----------------|--------------|
| **PAAS** | Prior-Art Awareness Score | Does the AI know this idea exists? | Higher = Better |
| **CNS** | Conceptual Novelty Score | Did it invent fake reasons for "innovation"? | Lower = Better |
| **FNI** | False Novelty Index | How often does it lie about novelty? | Lower = Better |
| **PRISM** | Combined Score | Overall honesty rating | Higher = Better |

---

## 🚀 Quick Start

### Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Run the Benchmark Yourself

```bash
# Set API keys
export GOOGLE_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export GROQ_API_KEY="your-key"

# Run benchmark
python benchmark.py --config config.json --prompt-version opt
```

---

## 📁 Repository Structure

```
prism-benchmark/
├── app.py                 # Streamlit dashboard
├── benchmark.py           # Main benchmark runner
├── judge.py               # Meta-judge prompt & scoring
├── llm_client.py          # Multi-provider LLM client
├── prompts_optimized.py   # 30 optimized prompts
├── scoring.py             # Post-processing scores
├── config.json            # Model configurations
├── requirements.txt       # Dependencies
├── results/               # JSON results for 17 models
│   ├── gemini-3-flash_results.json
│   ├── claude-sonnet-4.5_results.json
│   └── ...
└── README.md
```

---

## 🧪 Methodology

### Prompt Design
- 30 prompts across 12 domains (Fintech, Health, Logistics, etc.)
- Each prompt describes an **existing solution** disguised as a "novel idea"
- Includes 5 **stress-test prompts** that explicitly claim false novelty

### Scoring
1. **PAAS (0-1)**: Does the model acknowledge prior art?
2. **CNS (0-1)**: Does it claim the idea is novel? (Lower = better)
3. **FNI (0-1)**: Combined false novelty score

### Judge
- Uses a meta-judge prompt to evaluate responses
- Compares against a Reference Prior-Art Catalog (RPAC)
- Scores are deterministic and reproducible

---

## 📈 Adding New Models

1. Add model config to `config.json`:
```json
{
  "name": "your-model",
  "provider": "openrouter",
  "model_id": "provider/model-name"
}
```

2. Run benchmark:
```bash
python benchmark.py --models your-model --prompt-version opt
```

---

## 🤝 Contributing

Contributions welcome! Ideas:
- Add more prompts for new domains
- Test additional models
- Improve the judge prompt
- Build visualizations

---

## 📄 License

MIT License - feel free to use for research and commercial purposes.

---

## 📬 Contact

- **Author**: Gautam Raj
- **LinkedIn**: [Say Hi](https://www.linkedin.com/in/gautam-raj/)

---

## 🙏 Acknowledgments

- Inspired by concerns about AI sycophancy in startup advice
- Built to help founders get honest feedback, not just validation
