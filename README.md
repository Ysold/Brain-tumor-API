# Brain-tumor-API

## Team

- Tristan NIO
- Rayan MAMACHE
- Tony OSEI
- Mathis TALBI

## Routes

Take CSV file, train model, with randomForest:

```sh
"/training"
```

Take CSV file, train model, but with tensorFlow:

```sh
"/train_tensorflow"
```

Call model, if detect a tumor are detected:

```sh
"/predict"
```

Create a new model ...with many files :

```sh
"/training_classification"
```

Make call with Huggingface & Mistral library :

```sh
"/model"
```