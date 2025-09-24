# Model Card: DistilBERT Sentiment Classifier

## Overview
This model is a **DistilBERT-based sentiment classifier** fine-tuned on a labeled sentiment dataset.  
It predicts whether a given text has a **positive or negative** sentiment.  
We trained with **early stopping** to prevent overfitting and saved the best model based on dev set accuracy.

---

## Data
- **Train/Dev Split:**  
  - Train: 80% of available labeled data  
  - Dev: 20% (stratified split, random seed=42)  
  - Test: Provided unlabeled dataset  

- **Dataset Sources:**  
  Local dataset stored in `data/sentiment/train.csv` and `test.csv`.

- **Preprocessing:**  
  - Tokenized using `distilbert-base-uncased` tokenizer  
  - Max sequence length: 128 tokens  
  - Padded/truncated uniformly  

---

## Training
- **Model Architecture:** DistilBERT + classification head  
- **Hyperparameters:**  
  - Learning Rate: 2e-5  
  - Batch Size: 16  
  - Epochs: Up to 10 (early stopping after patience=2)  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** AdamW  
- **Scheduler:** Linear schedule with warmup  

---

## Performance
Evaluated on the dev set (20% split).  

| Model                | Dev Accuracy | Dev F1 Score |
|-----------------------|--------------|--------------|
| Baseline (Majority)   | ~50%         | ~0.50        |
| DistilBERT (ours)     | **95.56%**   | **0.9556**   |

---

## Error Analysis
We inspected several mistakes made on the dev set:  

1. **"I expected this product to be better 😞" → Predicted: Positive | Gold: Negative**  
   - Model struggles with subtle disappointment signals.  

2. **"Not bad at all!" → Predicted: Negative | Gold: Positive**  
   - Trouble handling negations ("not bad" = positive).  

3. **"It’s okay, nothing special." → Predicted: Positive | Gold: Neutral/Negative**  
   - Neutral expressions sometimes misclassified as positive.  

---

## Fairness & Robustness
- **Typos:** The model is somewhat robust, but heavy misspellings reduce accuracy.  
- **Emojis:** Handles common ones (😊, 😞), but struggles with rare or sarcastic emoji use.  
- **Code-switching (mix of languages):** Limited performance; mostly optimized for English.  
- **Lengthy reviews:** Performance stable up to 128 tokens; very long texts get truncated.  

---

## Usage
- Input: Plain text sentence/review  
- Output: Sentiment label (`0` = Negative, `1` = Positive)  
- Predictions saved to: `submissions/sentiment_test_predictions.csv`  

---

## Limitations
- Misclassifies subtle sarcasm and nuanced emotions  
- Limited performance on non-English text  
- Not tuned for domain-specific jargon (e.g., medical or legal reviews)  

---

## Future Improvements
- Add data augmentation (typos, back-translation)  
- Expand support for emojis and slang  
- Incorporate multilingual fine-tuning for better code-switching performance  
