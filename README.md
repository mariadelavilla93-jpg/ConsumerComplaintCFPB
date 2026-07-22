# Consumer Complaint Product Classification with SEC-BERT

This repository contains a Natural Language Processing prototype for automatically classifying financial consumer complaints into product categories based on their written narratives.

The project uses complaints published by the **Consumer Financial Protection Bureau (CFPB)** and fine-tunes **SEC-BERT**, a transformer model pretrained on financial text.

The objective is to predict the financial product associated with a complaint using only the `Consumer complaint narrative` field.

---

## 1. Business problem

Financial institutions and consumer-protection organizations receive large volumes of written complaints. Manually assigning each complaint to the correct product category is costly, time-consuming and potentially inconsistent.

An automated classifier could support:

- Complaint routing.
- Operational reporting.
- Identification of product-specific issues.
- Prioritization of cases for manual review.
- Monitoring changes in complaint volumes and topics.

This project develops a prototype that assigns each complaint to one of nine consolidated product families.

---

## 2. Dataset

The data comes from the public **CFPB Consumer Complaint Database**.

The dataset used in this project contains complaints:

- Received between **1 June 2023 and 31 May 2024**.
- With a non-empty consumer narrative.
- Exported in CSV format.

The dataset can be downloaded directly from:

[Download the CFPB complaints dataset](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_min=2023-06-01&date_received_max=2024-05-31&has_narrative=true&format=csv&no_aggs=true)

The downloaded file contains approximately **582,490 complaints** used in the analysis.

The two main variables are:

| Variable | Description |
|---|---|
| `Consumer complaint narrative` | Free-text description written by the consumer |
| `Product` | Product category assigned to the complaint |

The dataset is not stored in this repository because of its size.

---

## 3. Target categories

The original CFPB product labels are consolidated into nine broader product families:

1. `Checking or savings account`
2. `Credit or prepaid card`
3. `Credit reporting`
4. `Debt collection or management`
5. `Money transfer, virtual currency, or money service`
6. `Mortgage`
7. `Payday, title, personal or advance loan`
8. `Student loan`
9. `Vehicle loan or lease`

This consolidation reduces label fragmentation and creates categories with clearer business meaning.

---

## 4. Main modelling challenge

The dataset is strongly imbalanced.

`Credit reporting` represents approximately **72% of all complaints**, while several product families account for less than 2% individually.

As a result, accuracy alone is not sufficient to evaluate the model. The project also reports:

- Macro F1.
- Weighted F1.
- Precision and recall by product.
- Normalized confusion matrix.
- Coverage and accepted accuracy after applying a confidence threshold.

---

## 5. Approaches considered

Three modelling alternatives were considered.

### 5.1 Regex baseline

A simple and interpretable baseline was designed using keywords and n-grams associated with each product family.

The methodology consisted of:

- Extracting representative unigrams, bigrams and trigrams from the training data.
- Measuring their frequency within each product family.
- Considering their specificity relative to the remaining categories.
- Creating weighted regex rules.
- Assigning the category with the highest matching score.
- Leaving cases without sufficient evidence unclassified.

This approach is inexpensive and easy to explain, but it depends on explicit lexical matches and has difficulties with:

- Ambiguous narratives.
- Shared vocabulary across products.
- Complaints mentioning several financial products.
- Minority categories with limited examples.
- Cases where the relevant product is only implied by the context.

The regex approach was used as a conceptual baseline. The implementation included in this repository focuses on the selected transformer-based solution.

### 5.2 SEC-BERT fine-tuning

The selected solution fine-tunes:

`nlpaueb/sec-bert-base`

SEC-BERT is a BERT-based language model pretrained on financial documents.

Unlike regex rules, SEC-BERT can use the context and relationships between words in the complaint narrative. This makes it more suitable for complaints where the relevant product cannot be identified using isolated keywords.

This repository contains the complete implementation and evaluation of this approach.

### 5.3 LLM-based classification

A third alternative based on prompting a Large Language Model was also considered.

A prompt-based classifier could receive:

- The complaint narrative.
- The nine allowed product categories.
- A short definition of each category.
- One or more labelled examples (One-shot).
- A required structured output.

This alternative would provide flexibility when categories change and could potentially be used as a second-stage classifier for uncertain complaints.

However, it was not implemented within the scope of this prototype. A production decision would require evaluating:

- Classification quality.
- Inference latency.
- Token consumption and cost.
- Privacy requirements.
- Reproducibility.
- Dependence on an external provider.

---

## 6. Repository structure

```text
ConsumerComplaintCFPB/
├── .gitignore
├── README.md
└── SEC_BERT.ipynb
```

### Files

- `SEC_BERT.ipynb`: complete workflow, including data preparation, modelling, validation, test evaluation, error analysis and inference.
- `.gitignore`: excludes the original dataset, temporary notebook files and model checkpoints.
- `README.md`: project documentation and reproducibility instructions.

The repository contains the implementation of the **second approach: SEC-BERT fine-tuning**.

---

## 7. Methodology

### 7.1 Data preparation

The workflow:

1. Loads the CFPB CSV.
2. Retains the product label and consumer narrative.
3. Removes complaints without valid text.
4. Maps the original CFPB labels into nine product families.
5. Creates numerical labels for model training.

### 7.2 Data split

A stratified split is used to preserve the product distribution:

| Split | Percentage | Observations |
|---|---:|---:|
| Train | 80% | 465,992 |
| Validation | 10% | 58,249 |
| Test | 10% | 58,249 |

The three partitions have separate purposes:

- **Train:** model fitting.
- **Validation:** model evaluation and confidence-threshold selection.
- **Test:** final performance estimation.

The test set is not used for training or threshold selection.

### 7.3 Training balance

The original training set is dominated by `Credit reporting`.

To reduce this dominance, the number of training observations is capped at **20,000 complaints per category**.

Minority categories retain all their available observations.

This reduces the effective training sample from:

- **465,992 original training complaints**
- to **113,192 balanced training complaints**

Validation and test retain their original distribution so that evaluation remains representative of the real dataset.

### 7.4 Tokenization

Complaints are tokenized with the SEC-BERT tokenizer.

The maximum sequence length is:

```python
MAX_LENGTH = 256
```

Complaints longer than 256 tokens are truncated. Shorter complaints are padded dynamically within each batch.

The dataset contains relatively long complaint: approximately 225 tokens per complaint

A length of 256 tokens provides a compromise between retaining context and maintaining feasible GPU memory and training time.

Longer complaints may still lose information, which remains a limitation of the prototype.

---

## 8. Model configuration

The main training configuration is:

| Parameter | Value |
|---|---|
| Base model | `nlpaueb/sec-bert-base` |
| Maximum sequence length | 256 tokens |
| Number of categories | 9 |
| Epochs | 1 |
| Learning rate | `2e-5` |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Weight decay | `0.01` |
| Padding | Dynamic |
| Mixed precision | Enabled when GPU is available |
| Random seed | 42 |

A single epoch was used because of the computational limitations of the prototype environment.

Training took approximately **29 minutes** on the available GPU.

---

## 9. Validation results

The model obtains the following results on the validation set:

| Metric | Result |
|---|---:|
| Validation loss | 0.4156 |
| Accuracy | 86.89% |
| Weighted F1 | 87.71% |
| Macro F1 | 75.25% |

The difference between weighted F1 and macro F1 reflects the class imbalance.

Weighted F1 is strongly influenced by the majority category, while macro F1 assigns the same importance to every product family.

---

## 10. Confidence-based abstention

The model produces a softmax score for each category.

The maximum score is used as an operational confidence measure:

```text
confidence = maximum predicted softmax score
```

Different thresholds are tested on the validation set.

A threshold of:

```python
CONFIDENCE_THRESHOLD = 0.70
```

is selected as a compromise between coverage and prediction quality.

On validation, this threshold provides:

| Metric | Result |
|---|---:|
| Coverage | 86.51% |
| Accuracy on accepted predictions | 91.82% |
| Macro F1 on accepted predictions | 83.92% |
| Rejected proportion | 13.49% |

Predictions below the threshold are returned as `None`.

The softmax confidence is not calibrated and should not be interpreted as a true probability of correctness.

---

## 11. Final test results

### 11.1 Results without threshold

On the complete test set of 58,249 complaints:

| Metric | Result |
|---|---:|
| Accuracy | 87.00% |
| Weighted F1 | 87.80% |
| Macro F1 | 74.95% |
| Incorrect predictions | 7,575 |

The model correctly classifies approximately 87% of the test complaints.

Macro F1 is lower because the minority product families remain more difficult than the majority category.

### 11.2 Results with threshold

After applying the validation-selected threshold of 0.70:

| Metric | Result |
|---|---:|
| Coverage | 86.55% |
| Accepted accuracy | 91.91% |
| Accepted macro F1 | 83.56% |
| Automatically accepted complaints | 50,417 |
| Rejected complaints | 7,832 |
| Rejected proportion | 13.45% |

The threshold increases the quality of accepted predictions at the cost of leaving some complaints without an automatic classification.

This trade-off is useful in production environments where incorrect automatic assignments may be more costly than sending uncertain cases to review.

---

## 12. Product-level performance

The strongest performance is obtained for:

- `Credit reporting`
- `Mortgage`
- `Student loan`
- `Checking or savings account`

The most difficult categories are:

- `Payday, title, personal or advance loan`
- `Vehicle loan or lease`
- `Debt collection or management`
- `Money transfer, virtual currency, or money service`

Approximate F1 scores without threshold are:

| Product family | F1 |
|---|---:|
| Credit reporting | 92.86% |
| Mortgage | 85.67% |
| Student loan | 82.16% |
| Checking or savings account | 81.39% |
| Credit or prepaid card | 77.14% |
| Money transfer, virtual currency, or money service | 70.59% |
| Debt collection or management | 69.22% |
| Vehicle loan or lease | 62.80% |
| Payday, title, personal or advance loan | 52.73% |

After applying the confidence threshold, F1 improves across all product families among accepted predictions.

The largest relative improvements occur in the most difficult categories.

---

## 13. Error analysis

The normalized confusion matrix reveals several recurrent patterns.

### Money transfer vs. checking accounts

Approximately 25% of `Money transfer...` complaints are classified as `Checking or savings account`.

These complaints frequently mention:

- Bank accounts.
- Transfers.
- Deposits.
- Withdrawals.
- Blocked transactions.
- Missing funds.

The same vocabulary may describe either the payment service or the account from which the money was sent.

### Payday loans vs. debt collection

Approximately 13% of `Payday...` complaints are classified as `Debt collection or management`.

Many narratives focus on what happens after the original loan:

- Missed payments.
- Collection calls.
- Debt recovery.
- Collection agencies.
- Credit-reporting consequences.

The text may therefore contain stronger evidence about debt collection than about the original loan product.

### Debt collection vs. credit reporting

Some debt-collection complaints are assigned to `Credit reporting`.

These narratives often describe:

- A debt appearing on a credit report.
- Requests to remove an account.
- Disputes about inaccurate reporting.
- Collection accounts affecting the consumer's credit history.

### Vehicle loans

Vehicle-loan complaints are sometimes confused with:

- Debt collection.
- Credit reporting.
- General loan-related complaints.

This occurs when the narrative focuses on repossession, delinquency or reporting rather than explicitly mentioning the vehicle loan.

### Ambiguous and multi-product complaints

Some complaints genuinely contain evidence for more than one category.

For example, one narrative may involve:

1. A loan.
2. A missed payment.
3. A collection agency.
4. A negative credit-report entry.

A single-label classifier must choose only one category, even when several products or processes are relevant.

---

## 14. Limitations

The main limitations of the prototype are:

### Limited training

The model is trained for one epoch because of GPU and time constraints.

Additional epochs could improve performance but would require monitoring validation loss to prevent overfitting.

### Text truncation

Narratives longer than 256 tokens are truncated.

Important information appearing later in the complaint may therefore be lost.

### Class imbalance

The training cap reduces the dominance of the largest class but does not completely solve the imbalance.

The chosen strategy may also discard useful examples from the majority category.

### Overlapping labels

Several categories are semantically connected.

A complaint may describe a financial product, a collection process and a credit-reporting consequence in the same text.

### Confidence calibration

The maximum softmax score is used as confidence, but it has not been calibrated.

A score of 0.80 does not necessarily mean the prediction has an 80% probability of being correct.

### Potential label noise

Some apparent model errors may reflect ambiguous or imperfect original labels rather than a purely incorrect prediction.

---

## 15. Potential improvements

Possible next steps include:

- Training for additional epochs with early stopping.
- Comparing maximum lengths of 256 and 512 tokens.
- Using sliding windows for long complaints.
- Combining information from the beginning and end of long narratives.
- Testing class-weighted loss functions.
- Comparing undersampling, oversampling and alternative balancing strategies.
- Calibrating confidence using temperature scaling.
- Optimizing the confidence threshold for a specific business cost.
- Adding high-precision regex rules for clearly identifiable cases.
- Using an LLM as a second-stage classifier for uncertain complaints.
- Reviewing whether a hierarchical `Product` and `Sub-product` classifier would better reflect the data.
- Monitoring class distribution and performance drift after deployment.

---

## 16. Production proposal

A possible production workflow is:

```text
Complaint narrative
        │
        ▼
Text validation and preprocessing
        │
        ▼
SEC-BERT classifier
        │
        ▼
Maximum confidence score
        │
        ├── Confidence ≥ 0.70
        │          │
        │          ▼
        │   Automatic assignment
        │
        └── Confidence < 0.70
                   │
                   ▼
        Manual review or second-stage model
```

A more advanced architecture could introduce high-precision rules before SEC-BERT:

```text
Complaint narrative
        │
        ▼
High-precision regex rules
        │
        ├── Clear rule match ──► Automatic assignment
        │
        └── No clear match
                   │
                   ▼
             SEC-BERT
                   │
        ├── Confidence ≥ 0.70 ──► Automatic assignment
        │
        └── Confidence < 0.70 ──► Review or LLM
```

This architecture combines:

- Interpretability for highly explicit cases.
- Contextual classification for general complaints.
- Controlled review for uncertain cases.

---

## 17. Reproducibility

### 17.1 Clone the repository

```bash
git clone https://github.com/mariadelavilla93-jpg/ConsumerComplaintCFPB.git
cd ConsumerComplaintCFPB
```

### 17.2 Download the dataset

Download the filtered CFPB dataset from:

[CFPB Consumer Complaint Database — June 2023 to May 2024](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_min=2023-06-01&date_received_max=2024-05-31&has_narrative=true&format=csv&no_aggs=true)

The downloaded filename may vary.

### 17.3 Open the notebook

The notebook was developed for Google Colab.

Upload the CSV to the Colab runtime and update the path in the notebook when necessary:

```python
DATA_PATH = "/content/complaints-2026-07-15_02_28.csv"
```

For local execution, replace it with the path where the dataset is stored, for example:

```python
DATA_PATH = "data/complaints.csv"
```

### 17.4 Install dependencies

The main dependencies are:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers datasets accelerate
```

### 17.5 Run the notebook

Open:

```text
SEC_BERT.ipynb
```

Then execute the cells sequentially.

A CUDA-compatible GPU is strongly recommended for fine-tuning.

The expected runtime and memory consumption may vary depending on:

- GPU type.
- Available RAM.
- Batch size.
- Sequence length.
- Library versions.

---

## 18. Main conclusions

SEC-BERT provides a strong prototype for classifying financial consumer complaints from their narratives.

The final model achieves:

- **87.00% accuracy** over the complete test set.
- **87.80% weighted F1**.
- **74.95% macro F1**.
- **91.91% accuracy** over accepted predictions after applying a confidence threshold.
-  **92.25% weighted F1** over accepted predictions after applying a confidence threshold.
-  **83.56% macro F1** over accepted predictions after applying a confidence threshold.
- **86.55% automatic coverage**.

The results show that a transformer pretrained on financial text can classify most complaints reliably while identifying a smaller set of uncertain cases for additional review.

The main remaining challenges are:

- Minority-category performance.
- Semantically overlapping products.
- Long-text truncation.
- Confidence calibration.
- Production monitoring.

Within the scope of the prototype, SEC-BERT offers the best balance between contextual understanding, reproducibility, inference control and operational applicability.

# Alternative 3: LLM-based classification

As a third alternative, a general-purpose Large Language Model could be used to classify each complaint into one of the nine product categories.
Unlike the regex baseline and the SEC-BERT model, this approach would not require training and maintaining a dedicated classification model. Each complaint would be sent to the LLM together with a fixed prompt containing:

- A description of the classification task.
- The allowed product categories.
- Basic classification rules.
- The required JSON output format.
- A one-shot example.

A **one-shot example** consists of including one complete example of an input complaint and its expected classification inside the prompt. This helps the model understand both the task and the exact response format.
The model would receive the complaint identifier and narrative, and would return only the same identifier and one of the allowed categories.

Calls would be made with:

```python
temperature = 0
```

This configuration reduces output variability and encourages the same complaint to receive the same classification across repeated executions. However, absolute determinism cannot be assumed and should be tested empirically.

---

## Allowed categories

The nine allowed product categories are:

1. `Mortgage`
2. `Credit reporting`
3. `Debt collection or management`
4. `Checking or savings account`
5. `Credit or prepaid card`
6. `Payday, title, personal or advance loan`
7. `Vehicle loan or lease`
8. `Money transfer, virtual currency, or money service`
9. `Student loan`

---

## Proposed system prompt

```text
You are a financial complaint classification system.

Your task is to classify each consumer complaint into exactly one of the allowed product categories.

You will receive:
- A complaint ID.
- The text of a consumer complaint.

You must analyze the main issue described in the complaint and select the single category that best represents the financial product involved.

Allowed categories:

1. Mortgage
2. Credit reporting
3. Debt collection or management
4. Checking or savings account
5. Credit or prepaid card
6. Payday, title, personal or advance loan
7. Vehicle loan or lease
8. Money transfer, virtual currency, or money service
9. Student loan

Classification guidelines:

- Select only one category.
- Use the main subject of the complaint, even when other financial products are also mentioned.
- For complaints about incorrect, negative, fraudulent or outdated information appearing on a credit report, select "Credit reporting".
- For complaints mainly concerning collection attempts, debt collectors, debt settlement, debt relief or debt management services, select "Debt collection or management".
- Do not create new categories.
- Do not return explanations, reasoning, confidence scores, markdown or additional text.
- Preserve the complaint ID exactly as provided.
- Return only one valid JSON object using this exact structure:

{
  "complaint_id": "<complaint ID>",
  "category": "<one allowed category>"
}

One-shot example:

Input:
{
  "complaint_id": "7051632",
  "complaint": "The loan was refinanced with another company XX/XX/2022. Still appear negative on credit report."
}

Output:
{
  "complaint_id": "7051632",
  "category": "Credit reporting"
}

Now classify the complaint provided by the user following exactly the same rules and output format.
```

---

## Input format

Each request would include the complaint identifier and its narrative:

```json
{
  "complaint_id": "7051632",
  "complaint": "The loan was refinanced with another company XX/XX/2022. Still appear negative on credit report."
}
```

---

## Output format

The expected response would contain only a valid JSON object:

```json
{
  "complaint_id": "7051632",
  "category": "Credit reporting"
}
```

The prompt explicitly restricts the output to:

- One category.
- One of the nine permitted values.
- A valid JSON object.
- The exact fields `complaint_id` and `category`.
- No explanations, reasoning or additional text.

In a real implementation, these restrictions should also be enforced using structured output or a JSON Schema whenever supported by the selected API.

---

## Token consumption estimate

The estimated token consumption is based on the following averages:

| Component | Average tokens per complaint |
|---|---:|
| Complaint narrative | 225.04 |
| System prompt and one-shot example | 365.00 |
| **Total input tokens** | **590.04** |
| JSON output | **16.33** |

The expected processing volume is:

- Approximately **500,000 complaints per year**.
- Approximately **42,000 complaints per month**.

### Monthly input tokens

Each request would contain approximately:

```text
225.04 + 365 = 590.04 input tokens
```

For 42,000 complaints:

```text
590.04 × 42,000 = 24,781,680 input tokens
```

The estimated monthly input volume is therefore:

**24.78 million input tokens.**

### Monthly output tokens

Each response would contain approximately:

```text
16.33 output tokens
```

For 42,000 complaints:

```text
16.33 × 42,000 = 685,860 output tokens
```

The estimated monthly output volume is therefore:

**0.686 million output tokens.**

---

## Models considered

Two Flash models were considered because of their balance between cost, latency and classification capability.

| Model | Input price per 1M tokens | Output price per 1M tokens | Estimated latency |
|---|---:|---:|---:|
| Gemini 3 Flash | $0.50 | $3.00 | 1.19 s |
| Gemini 2.5 Flash | $0.30 | $2.50 | 0.50 s |

Latency values are indicative and may vary depending on region, concurrency, provider limits and infrastructure conditions.

The models would be used without additional reasoning capabilities because this is a closed classification task. This keeps both latency and token consumption lower.

---

## Estimated monthly cost

### Gemini 3 Flash

Input cost:

```text
24.78168 million × $0.50 = $12.39084
```

Output cost:

```text
0.68586 million × $3.00 = $2.05758
```

Total monthly cost:

```text
$12.39084 + $2.05758 = $14.44842
```

Estimated cost:

- **$14.45 per month**
- **$0.00034 per complaint**

### Gemini 2.5 Flash

Input cost:

```text
24.78168 million × $0.30 = $7.43450
```

Output cost:

```text
0.68586 million × $2.50 = $1.71465
```

Total monthly cost:

```text
$7.43450 + $1.71465 = $9.14915
```

Estimated cost:

- **$9.15 per month**
- **$0.00022 per complaint**

---

## Cost comparison

| Model | Monthly input cost | Monthly output cost | Total monthly cost | Cost per complaint |
|---|---:|---:|---:|---:|
| Gemini 3 Flash | $12.39 | $2.06 | **$14.45** | $0.00034 |
| Gemini 2.5 Flash | $7.43 | $1.71 | **$9.15** | $0.00022 |

Under these assumptions, Gemini 2.5 Flash would cost approximately:

- **$5.30 less per month**
- Around **37% less** than Gemini 3 Flash

---

## Operational considerations

The cost estimate assumes:

- Approximately 42,000 complaints per month.
- One independent request per complaint.
- The complete system prompt is included in each request.
- The model returns only the expected JSON.
- `temperature = 0` is used.
- No additional reasoning mode is enabled.
- No retries are required.
- No invalid responses are produced.
- Prompt caching or batch discounts are not considered.
- Prices remain unchanged.

The estimate includes only model inference. It does not include:

- Integration development.
- Storage.
- Monitoring.
- Data transfer.
- Logging.
- Retry mechanisms.
- Manual review.
- Additional infrastructure.

Before storing a response, the system should validate that:

1. The output is valid JSON.
2. The returned `complaint_id` matches the submitted identifier.
3. The category belongs to the nine allowed values.
4. No unexpected fields or additional text are returned.

Invalid responses could be sent to a controlled retry mechanism. If the second attempt also failed, the complaint could be routed to manual review.

---

## Recommended evaluation

Before adopting the LLM-based solution, both models should be tested on the same test set used for the regex and SEC-BERT approaches.

The comparison should include:

- Accuracy.
- Macro F1.
- Precision and recall by category.
- Performance on minority categories.
- Percentage of valid JSON responses.
- Percentage of categories outside the allowed catalogue.
- Prediction consistency across repeated runs.
- Average latency and latency percentiles.
- Actual token consumption.
- Actual cost per complaint.

This would allow the three approaches to be compared under equivalent conditions:

1. Regex top-k baseline.
2. SEC-BERT fine-tuning.
3. LLM-based classification.

## Conclusion

The LLM approach is flexible because it does not require training a dedicated model and allows categories or classification rules to be changed directly in the prompt.

Under the assumptions used:

- **Gemini 2.5 Flash** would have an estimated monthly inference cost of **$9.15**, with an indicative latency of approximately **0.50 seconds**.
- **Gemini 3 Flash** would have an estimated monthly inference cost of **$14.45**, with an indicative latency of approximately **1.19 seconds**.

The final decision should not be based only on price or latency. Both models would need to be evaluated on the same test set to determine whether their quality is comparable to or better than SEC-BERT.

If the Flash models did not achieve sufficient performance, especially for ambiguous complaints or minority categories, more capable models could be evaluated. Any improvement would need to be weighed against the expected increase in inference cost and response time.
