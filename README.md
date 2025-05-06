# Auto-regressive-IE

Auto-regressive-IE is a transformer-based information extraction system for joint entity and relation extraction. It supports training on custom datasets, inference, and deployment via Docker for cloud use.

---

## Features

- **Joint Entity & Relation Extraction**: Extracts entities and their relations in a single autoregressive decoding process.
- **Transformer-based Architecture**: Supports BERT, DeBERTa, SpanBERT, SciBERT, and more.
- **Training & Evaluation**: Scripts for training, evaluation, and usage on custom datasets.
- **Docker Support**: Ready-to-use Docker image for cloud deployment.

---

## Workspace Structure

This project is organized as follows:

```
.
├── docker-compose.yaml
├── evaluate.py
├── generate.py
├── metric.py
├── model.py
├── preprocess.py
├── README.md
├── requirements.txt
├── save_load.py
├── train.py
├── trans_enc.py
├── usage.py
└── layers/
    ├── base.py
    ├── span_embedding.py
    ├── structure.py
    └── token_embedding.py
```

- **layers/**: Custom neural network layers and span representations.
- **docker-compose.yaml**: For running the project in cloud environments.
- **requirements.txt**: Python dependencies.

This structure supports both local development and cloud deployment on Windows or other operating systems.

---

## Model Architecture

The core model is implemented in [`IeGenerator`](model.py):

- **Token Representation**: Uses Flair's TransformerWordEmbeddings (BERT, DeBERTa, SciBERT, etc.).
- **BiLSTM Layer**: Adds sequential context on top of transformer embeddings.
- **Span Representation Layer**: Multiple strategies (see [`layers/span_embedding.py`](layers/span_embedding.py )).
- **Autoregressive Transformer Decoder**: Custom transformer decoder ([`TransDec`](trans_enc.py )) for sequential prediction of entities and relations.
- **Joint Decoding**: Decodes entities and relations in a single sequence, allowing for flexible graph extraction.

---

## Training

To train the model on the dataset:

1. **Prepare the dataset**: Place your preprocessed dataset in the appropriate directory.
2. **Configure training**: Edit the arguments in [`train.py`](train.py ) or use command-line options.
3. **Run training**:

    ```sh
    python train.py --data_file path/to/redfm.pkl --model_name deberta --n_epochs 10 --batch_size 8 --log_dir checkpoints/
    ```

- The script supports various transformer backbones (see [`MODELS`](train.py ) in [`train.py`](train.py )).
- Training logs and best checkpoints are saved in the specified [`log_dir`](train.py ).

---

## Inference and Usage

```bash
docker pull platonlel/atg_image::latest
docker run -p 8000:8000 platonlel/atg_image::latest
```
- The web interface will be available at [http://localhost:8000](http://localhost:8000).

You can also use [`docker-compose.yaml`](docker-compose.yaml ) for cloud deployment.


## Requirements

- Python 3.9
- See [`requirements.txt`](requirements.txt ) for Python dependencies.

---

## Example Usage

```python
from save_load import load_model
from usage import usage

model = load_model("checkpoints/best_checkpoint.pt")
result = usage("Wonder Woman is a superheroine appearing in American comic books published by DC Comics.", model)
print(result)
```

---

## References

- Model and code adapted from [PlatonLel/Auto-regressive-IE](https://github.com/PlatonLel/Auto-regressive-IE)
- Uses [Flair](https://github.com/flairNLP/flair), [Transformers](https://github.com/huggingface/transformers), [AllenNLP](https://github.com/allenai/allennlp)

---

## Docker_image


For more details, see the code in [model.py](model.py), [train.py](train.py), and [app.py](app.py).
