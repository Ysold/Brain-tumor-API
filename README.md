# Brain-tumor-API

## Team

- Tristan NIO
- Rayan MAMACHE
- Tony OSEI
- Mathis TALBI 

## Routes

Take CSV file, train model, with RandomForest:

```sh
"/training"
```

Take CSV file, train model, but with tensorflow:

```sh
"/train_tensorflow"
```

Call model, and detect a tumor on a brain MRI:

```sh
"/predict"
```

Make call with OpenAI :

```sh
"/model"
```
