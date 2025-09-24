# AmigoAI Project

This project has two main parts:
1. **RAG Mini-System** (Part A)  
2. **Sentiment Classifier** (Part B)

It is designed to run inside **Visual Studio Code (VS Code)** with Python virtual environments.

---

## Project Structure

```

project/
│── data/
│   ├── corpus/
│   │   ├── docs.jsonl
│   │   ├── questions.json
│   ├── sentiment/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   ├── test.csv
│   └── config.json
│
│── src/
│   ├── rag_answer.py         # Script for RAG mini-system
│   ├── train.py              # Script for sentiment classification
    |--evaluate_rag.py         #script for evaluate RAG
│
│── submissions/
│   ├── rag_answers.json
│   ├── sentiment_test_predictions.csv
│   ├── dev_metrics.csv
│
│── RAG_README.md
│── MODEL_CARD.md
│── README.md
│── requirements.txt

````

---

## ⚙️ Setup in VS Code

1. **Open Folder**  
   - Open this project folder in **VS Code**.

2. **Create Virtual Environment**  
   Open a terminal inside VS Code and run:
   ```powershell
   python -m venv rapra_env
````

3. **Activate Virtual Environment**

   ```powershell
   rapra_env\Scripts\activate
   ```

4. **Install Requirements**

   ```powershell
   pip install -r requirements.txt
   ```

5. **Select Interpreter in VS Code**

   * Press `Ctrl+Shift+P` → search **Python: Select Interpreter** → choose the `rapra_env` environment.

---

## 🚀 Running the Project

### Part A: RAG Mini-System

1. Open `src/rag_answer.py` in VS Code.
2. Run in terminal:

   ```powershell
   python src/rag_answer.py
   ```
3. Output file:

   ```
   submissions/rag_answers.json
   ```

---

### Part B: Sentiment Classifier

1. Open `src/train.py` in VS Code.
2. Run in terminal:

   ```powershell
   python src/train.py
   ```
3. Output files:

   ```
   submissions/sentiment_test_predictions.csv
   submissions/dev_metrics.csv
   ```

---

## 📑 Deliverables

* **Part A:**

  * `submissions/rag_answers.json`
  * `RAG_README.md`
  * `src/rag_answer.py`

* **Part B:**

  * `submissions/sentiment_test_predictions.csv`
  * `MODEL_CARD.md`
  * `src/train.py`

* **General:**

  * `requirements.txt`
  * `README.md`

---

## 📝 Notes & Assumptions

* Used **BM25** for retrieval baseline and compared with embeddings.
* Used **DistilBERT fine-tuning** for improved sentiment classification.
* Hidden test labels were not used for training.
* All scripts are lightweight and run locally on CPU/GPU inside VS Code.

