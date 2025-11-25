# Evolutionary Prompt Model

This project uses a Genetic Algorithm (GA) to automatically evolve and optimize LLM prompts. The goal is to find individuals (composed of `role`, `topic`, `prompt`, and `keywords`) that generate text with high `fitness` (semantic similarity) relative to a reference text.

## 1\. Environment Setup

### Prerequisites

  * Python 3.10+
  * An Ollama service running locally. The `LLMAgent` is configured to connect to `http://127.0.0.1:11434`. Ensure that Ollama is running and has the model you wish to use (e.g., `llama3`).

### Installation

1.  Clone the repository.
2.  (Optional but recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  The `nltk` library will automatically download the necessary models (`wordnet`, `averaged_perceptron_tagger_eng`) the first time the mutation script is executed.

## 2\. Corpus Preparation (One-time Step)

The algorithm requires a corpus of reference texts. The `prepare_corpus.py` script is designed to filter a massive CSV file.

1.  Place your massive corpus file in the root of the project and ensure it is named `corpus.csv`.
2.  Run the filtering script **only once**:
    ```bash
    python prepare_corpus.py
    ```
3.  This will generate a new file, `corpus_filtrado.csv`, which will be used by the algorithm.

## 3\. Running the Genetic Algorithm

The main script is `main.py`. It is executed from the terminal and accepts various arguments to configure the run.

### Execution Example

```bash
python main.py --n 50 --generaciones 10 --model llama3
```

### Key Arguments

You can view all arguments in `main.py`. The most important ones are:

  * `--n` (Required): Number of individuals in the population (e.g., `50` or `100`).
  * `--generaciones`: Number of generations the algorithm will evolve (e.g., `10`).
  * `--model`: The name of the Ollama model to use (e.g., `llama3`, `mistral`).
  * `--k`: Tournament size for selection (e.g., `3`).
  * `--prob-crossover`: Crossover probability (e.g., `0.8`).
  * `--prob-mutacion`: Mutation probability (e.g., `0.1`).
  * `--num-elitismo`: Number of elite individuals that pass to the next generation (e.g., `2`).
  * `--texto-referencia`: (Optional) Path to a specific `.txt` file if you do not want to use a random one from the filtered corpus.
  * `--bert-model`: The model to use for BERTScore (e.g., `bert-base-uncased`).

## 4\. Output and Results

All executions are saved in the `exec/` directory.

Each execution creates a unique folder with a *timestamp* (e.g., `exec/2025-11-09_01-08-32/`), which will contain:

  * `reference.txt`: The random reference text used for this execution.
  * `data_initial_population.json`: The individuals of Generation 0 (before evaluation).
  * `data_inicial_evaluada.json`: Generation 0 with its calculated `fitness`.
  * `data_final_evaluada.json`: The final population after all generations.
  * `metrics_gen.csv`: A CSV with fitness statistics (min, max, mean) for each generation.
  * `runtime.txt`: Detailed execution times.