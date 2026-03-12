"""
Domain data loading for continual learning experiments.

Provides predefined domain datasets and custom data loading.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("molly_evolution")

# Predefined domain datasets (HuggingFace)
DOMAIN_CONFIGS = {
    "general": {
        "dataset": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text",
        "max_samples": 5000,
        "description": "General English text (WikiText-103)",
    },
    "code": {
        "dataset": "sahil2801/CodeAlpaca-20k",
        "config": None,
        "split": "train",
        "text_field": "output",
        "max_samples": 5000,
        "description": "Source code (CodeAlpaca)",
    },
    "legal": {
        "dataset": "lex_glue",
        "config": "unfair_tos",
        "split": "train",
        "text_field": "text",
        "max_samples": 5000,
        "description": "Legal terms of service (LexGLUE/UnfairToS)",
    },
    "medical": {
        "dataset": "medalpaca/medical_meadow_medqa",
        "config": None,
        "split": "train",
        "text_field": "input",
        "max_samples": 2000,
        "description": "Medical Q&A",
    },
    "science": {
        "dataset": "scientific_papers",
        "config": "arxiv",
        "split": "train",
        "text_field": "article",
        "max_samples": 500,
        "description": "Scientific papers (arXiv)",
    },
    "finance": {
        "dataset": "financial_phrasebank",
        "config": "sentences_allagree",
        "split": "train",
        "text_field": "sentence",
        "max_samples": 500,
        "description": "Financial sentiment text",
    },
}

# Built-in quick-test data (no download needed)
# Each domain needs ~80,000 chars to produce 200+ train samples at 256 tokens
QUICKTEST_DATA = {
    "general": (
        "The history of artificial intelligence began in antiquity, with myths and "
        "stories of artificial beings endowed with intelligence. The seeds of modern "
        "AI were planted by philosophers who attempted to describe the process of human "
        "thinking as the mechanical manipulation of symbols. This work culminated in "
        "the invention of the programmable digital computer in the 1940s. "
        "The field of AI research was founded at a workshop at Dartmouth College in 1956. "
        "Attendees Allen Newell, Herbert Simon, John McCarthy, Marvin Minsky, and Arthur "
        "Samuel became the founders and leaders of AI research. They and their students "
        "produced programs that the press described as astonishing. Computers were winning "
        "at checkers, solving word problems in algebra, proving logical theorems, and "
        "speaking English. By the middle of the 1960s, research in the US was heavily "
        "funded by the Department of Defense and laboratories had been established around "
        "the world. Researchers in the 1960s and 1970s were convinced that their methods "
        "would eventually succeed in creating a thinking machine and gave their patrons "
        "optimistic predictions. In 1965, Herbert Simon predicted that machines will be "
        "capable of doing any work a man can do within twenty years. Marvin Minsky agreed, "
        "writing that within a generation the problem of creating artificial intelligence "
        "will substantially be solved. They failed to recognize the difficulty of some of "
        "the remaining tasks. Progress slowed and in 1974, in response to the criticism "
        "of Sir James Lighthill and ongoing pressure from the US Congress to fund more "
        "productive projects, both the US and British governments cut off exploratory "
        "research in AI. The next few years would later be called an AI winter, a period "
        "when obtaining funding for AI projects was difficult. In the early 1980s, AI "
        "research was revived by the commercial success of expert systems, a form of AI "
        "program that simulated the knowledge and analytical skills of human experts. "
        "By 1985 the market for AI had reached over a billion dollars. At the same time, "
        "Japan's fifth generation computer project inspired the US and British governments "
        "to restore funding for academic research. However, beginning with the collapse "
        "of the Lisp Machine market in 1987, AI once again fell into disrepute, and a "
        "second, longer-lasting winter began. Many researchers began to doubt that the "
        "symbolic approach would be able to imitate all the processes of human cognition. "
        "A number of researchers began to look into sub-symbolic approaches to artificial "
        "intelligence problems. Robotics researchers such as Rodney Brooks rejected "
        "symbolic AI and focused on the basic engineering problems that would allow "
        "robots to move, survive, and learn their environment. Neural networks research "
        "had been abandoned by AI and computer science around the same time. This line "
        "of research was revived by David Rumelhart and others in the middle of the 1980s. "
        "These and other sub-symbolic approaches are now known collectively as computational "
        "intelligence. In the 1990s and early 21st century, AI achieved its greatest "
        "successes. AI is used for logistics, data mining, medical diagnosis, and many "
        "other areas throughout the technology industry. The success was due to several "
        "factors: the increasing computational power of computers, a greater emphasis on "
        "solving specific sub-problems, the creation of new ties between AI and other "
        "fields working on similar problems, and above all, a new commitment by researchers "
        "to solid mathematical methods and rigorous scientific standards. "
    ) * 200,
    "code": (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) "
        "+ fibonacci(n-2)\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n\n"
        "class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n"
        "        self.left = None\n        self.right = None\n\n"
        "    def insert(self, value):\n        if value < self.value:\n"
        "            if self.left is None:\n                self.left = BinarySearchTree(value)\n"
        "            else:\n                self.left.insert(value)\n"
        "        else:\n            if self.right is None:\n"
        "                self.right = BinarySearchTree(value)\n"
        "            else:\n                self.right.insert(value)\n\n"
        "    def search(self, value):\n        if value == self.value:\n            return True\n"
        "        elif value < self.value and self.left:\n            return self.left.search(value)\n"
        "        elif value > self.value and self.right:\n            return self.right.search(value)\n"
        "        return False\n\n"
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n"
        "    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\n"
        "def merge(left, right):\n    result = []\n    i = j = 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n"
        "        else:\n            result.append(right[j])\n            j += 1\n"
        "    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n\n"
        "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n"
        "    def append(self, data):\n        new_node = Node(data)\n"
        "        if not self.head:\n            self.head = new_node\n            return\n"
        "        current = self.head\n        while current.next:\n"
        "            current = current.next\n        current.next = new_node\n\n"
        "    def delete(self, data):\n        if not self.head:\n            return\n"
        "        if self.head.data == data:\n            self.head = self.head.next\n"
        "            return\n        current = self.head\n"
        "        while current.next:\n            if current.next.data == data:\n"
        "                current.next = current.next.next\n                return\n"
        "            current = current.next\n\n"
        "class HashMap:\n    def __init__(self, size=256):\n"
        "        self.size = size\n        self.buckets = [[] for _ in range(size)]\n\n"
        "    def _hash(self, key):\n        return hash(key) % self.size\n\n"
        "    def put(self, key, value):\n        idx = self._hash(key)\n"
        "        for i, (k, v) in enumerate(self.buckets[idx]):\n"
        "            if k == key:\n                self.buckets[idx][i] = (key, value)\n"
        "                return\n        self.buckets[idx].append((key, value))\n\n"
        "    def get(self, key):\n        idx = self._hash(key)\n"
        "        for k, v in self.buckets[idx]:\n            if k == key:\n"
        "                return v\n        raise KeyError(key)\n\n"
        "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n"
        "    distances[start] = 0\n    visited = set()\n"
        "    import heapq\n    pq = [(0, start)]\n"
        "    while pq:\n        dist, node = heapq.heappop(pq)\n"
        "        if node in visited:\n            continue\n"
        "        visited.add(node)\n"
        "        for neighbor, weight in graph[node].items():\n"
        "            new_dist = dist + weight\n"
        "            if new_dist < distances[neighbor]:\n"
        "                distances[neighbor] = new_dist\n"
        "                heapq.heappush(pq, (new_dist, neighbor))\n"
        "    return distances\n\n"
    ) * 200,
    "legal": (
        "The defendant is hereby charged with violation of Section 1983 of Title 42 "
        "of the United States Code. The plaintiff alleges that the defendant, acting "
        "under color of state law, deprived the plaintiff of rights secured by the "
        "Constitution. The court must determine whether qualified immunity shields "
        "the defendant from liability. Under the doctrine of stare decisis, the court "
        "is bound by the precedent established in prior rulings. "
        "REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of "
        "27 April 2016 on the protection of natural persons with regard to the processing "
        "of personal data and on the free movement of such data. Article 1: Subject-matter "
        "and objectives. This Regulation lays down rules relating to the protection of "
        "natural persons with regard to the processing of personal data and rules relating "
        "to the free movement of personal data. This Regulation protects fundamental rights "
        "and freedoms of natural persons and in particular their right to the protection "
        "of personal data. Article 5: Principles relating to processing of personal data. "
        "Personal data shall be processed lawfully, fairly and in a transparent manner in "
        "relation to the data subject. Personal data shall be collected for specified, "
        "explicit and legitimate purposes and not further processed in a manner that is "
        "incompatible with those purposes. Personal data shall be adequate, relevant and "
        "limited to what is necessary in relation to the purposes for which they are "
        "processed. Personal data shall be accurate and, where necessary, kept up to date. "
        "Every reasonable step must be taken to ensure that personal data that are inaccurate "
        "are erased or rectified without delay. Personal data shall be kept in a form which "
        "permits identification of data subjects for no longer than is necessary. "
        "The Supreme Court of the United States held in Marbury v. Madison that the "
        "Constitution is the supreme law of the land and that it is the duty of the "
        "judicial branch to say what the law is. The doctrine of judicial review "
        "established in this case remains a cornerstone of American constitutional law. "
        "The Commerce Clause grants Congress the power to regulate commerce among the "
        "several states. In Wickard v. Filburn, the Court held that even activity that "
        "is local in character may be regulated by Congress if it exerts a substantial "
        "economic effect on interstate commerce. The Due Process Clause of the Fourteenth "
        "Amendment prohibits states from depriving any person of life, liberty, or property "
        "without due process of law. The Equal Protection Clause requires that no state "
        "shall deny to any person within its jurisdiction the equal protection of the laws. "
        "In contract law, consideration is an essential element for the formation of a "
        "binding agreement. The doctrine of promissory estoppel may apply when one party "
        "makes a clear and definite promise upon which the other party reasonably relies "
        "to their detriment. Under the Uniform Commercial Code, a contract for the sale "
        "of goods for a price of five hundred dollars or more is not enforceable unless "
        "there is some writing sufficient to indicate that a contract for sale has been "
        "made between the parties. The parol evidence rule generally prevents parties from "
        "introducing extrinsic evidence to contradict or vary the terms of a written "
        "agreement that the parties intended as a final expression of their agreement. "
    ) * 200,
    "medical": (
        "The patient presents with acute myocardial infarction characterized by "
        "ST-segment elevation in leads V1-V4. Troponin I levels are elevated at "
        "15.2 ng/mL. The recommended treatment includes immediate percutaneous "
        "coronary intervention with dual antiplatelet therapy consisting of aspirin "
        "and a P2Y12 inhibitor. Echocardiography reveals reduced left ventricular "
        "ejection fraction of 35%. "
        "A 65-year-old male presents to the emergency department with sudden onset "
        "of left-sided weakness and slurred speech. CT scan of the head shows no "
        "hemorrhage. The National Institutes of Health Stroke Scale score is 12. "
        "Given presentation within the 4.5 hour window, tissue plasminogen activator "
        "is administered intravenously. MRI diffusion-weighted imaging confirms acute "
        "ischemic stroke in the right middle cerebral artery territory. "
        "Diabetes mellitus type 2 is characterized by insulin resistance and relative "
        "insulin deficiency. First-line pharmacotherapy includes metformin, which acts "
        "by decreasing hepatic glucose production and increasing insulin sensitivity. "
        "Hemoglobin A1c target is generally below 7.0 percent. Complications include "
        "diabetic nephropathy, retinopathy, neuropathy, and cardiovascular disease. "
        "Regular monitoring of renal function and lipid profiles is recommended. "
        "Pneumonia is an infection of the lung parenchyma. Community-acquired pneumonia "
        "is most commonly caused by Streptococcus pneumoniae. Empiric treatment for "
        "outpatient CAP includes amoxicillin or doxycycline for patients without "
        "comorbidities. For inpatients, a beta-lactam plus a macrolide or respiratory "
        "fluoroquinolone monotherapy is recommended. Chest radiograph typically shows "
        "consolidation or infiltrates. Complete blood count may reveal leukocytosis "
        "with left shift. Blood cultures should be obtained before antibiotic initiation. "
        "Chronic obstructive pulmonary disease is characterized by persistent respiratory "
        "symptoms and airflow limitation due to airway and alveolar abnormalities. "
        "Spirometry showing FEV1/FVC ratio less than 0.70 confirms the diagnosis. "
        "Inhaled bronchodilators are the mainstay of pharmacological management. "
        "Long-acting muscarinic antagonists and long-acting beta-agonists are used for "
        "maintenance therapy. Inhaled corticosteroids may be added for patients with "
        "frequent exacerbations. Pulmonary rehabilitation improves exercise tolerance "
        "and quality of life. Supplemental oxygen is indicated for severe resting "
        "hypoxemia with PaO2 less than 55 mmHg. "
    ) * 200,
}


def load_domain_data(domain: str, tokenizer, max_length: int = 256,
                     n_train: int = 100, n_eval: int = 50,
                     quicktest: bool = False) -> Tuple[dict, dict]:
    """
    Load train and eval data for a domain.

    Args:
        domain: Domain name (from DOMAIN_CONFIGS or QUICKTEST_DATA).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        n_train: Number of training examples.
        n_eval: Number of eval examples.
        quicktest: Use built-in data instead of downloading.

    Returns:
        (train_encodings, eval_encodings)
    """
    if quicktest or domain not in DOMAIN_CONFIGS:
        return _load_quicktest(domain, tokenizer, max_length, n_train, n_eval)

    return _load_hf_domain(domain, tokenizer, max_length, n_train, n_eval)


def _load_quicktest(domain, tokenizer, max_length, n_train, n_eval):
    """Load from built-in text data."""
    text = QUICKTEST_DATA.get(domain, QUICKTEST_DATA["general"])
    logger.info(f"  Loading '{domain}' (quicktest, built-in data)")

    # Replicate text to get enough tokens
    while len(text) < max_length * (n_train + n_eval):
        text = text + " " + text

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length * (n_train + n_eval),
                    padding=False)

    # Split into chunks
    all_ids = enc["input_ids"][0]
    chunks = []
    for i in range(0, len(all_ids) - max_length, max_length):
        chunks.append(all_ids[i:i+max_length])

    n_total = min(len(chunks), n_train + n_eval)
    n_train = min(n_train, n_total - 1)
    n_eval = min(n_eval, n_total - n_train)

    import torch
    train_ids = torch.stack(chunks[:n_train])
    eval_ids = torch.stack(chunks[n_train:n_train+n_eval])

    train_enc = {
        "input_ids": train_ids,
        "attention_mask": torch.ones_like(train_ids),
    }
    eval_enc = {
        "input_ids": eval_ids,
        "attention_mask": torch.ones_like(eval_ids),
    }

    logger.info(f"    train: {train_ids.shape[0]} x {train_ids.shape[1]} tokens")
    logger.info(f"    eval:  {eval_ids.shape[0]} x {eval_ids.shape[1]} tokens")
    return train_enc, eval_enc


def _load_hf_domain(domain, tokenizer, max_length, n_train, n_eval):
    """Load from HuggingFace datasets."""
    from datasets import load_dataset

    cfg = DOMAIN_CONFIGS[domain]
    logger.info(f"  Loading '{domain}' from {cfg['dataset']}...")

    ds_args = [cfg["dataset"]]
    if cfg.get("config"):
        ds_args.append(cfg["config"])

    max_samples = cfg.get("max_samples", n_train + n_eval + 50)
    try:
        ds = load_dataset(*ds_args, split=f"{cfg['split']}[:{max_samples}]")
    except Exception as e:
        logger.warning(f"  Failed to load '{domain}': {e}")
        logger.warning(f"  Falling back to quicktest data")
        return _load_quicktest(domain, tokenizer, max_length, n_train, n_eval)

    # Extract text
    text_field = cfg["text_field"]
    texts = [row[text_field] for row in ds if row.get(text_field)]
    texts = [t for t in texts if len(t) > 50]  # filter too-short

    if not texts:
        logger.warning(f"  No valid texts for '{domain}', falling back to quicktest")
        return _load_quicktest(domain, tokenizer, max_length, n_train, n_eval)

    # Tokenize and chunk
    import torch
    all_text = " ".join(texts)
    enc = tokenizer(all_text, return_tensors="pt", truncation=True,
                    max_length=max_length * (n_train + n_eval + 10),
                    padding=False)

    all_ids = enc["input_ids"][0]
    chunks = []
    for i in range(0, len(all_ids) - max_length, max_length):
        chunks.append(all_ids[i:i+max_length])

    n_total = min(len(chunks), n_train + n_eval)
    n_train = min(n_train, n_total - 1)
    n_eval = min(n_eval, n_total - n_train)

    train_ids = torch.stack(chunks[:n_train])
    eval_ids = torch.stack(chunks[n_train:n_train+n_eval])

    train_enc = {
        "input_ids": train_ids,
        "attention_mask": torch.ones_like(train_ids),
    }
    eval_enc = {
        "input_ids": eval_ids,
        "attention_mask": torch.ones_like(eval_ids),
    }

    logger.info(f"    train: {train_ids.shape[0]} x {train_ids.shape[1]} tokens")
    logger.info(f"    eval:  {eval_ids.shape[0]} x {eval_ids.shape[1]} tokens")
    return train_enc, eval_enc
