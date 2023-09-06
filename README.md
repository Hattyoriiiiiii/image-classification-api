# image-classification-api

- This repository is a part of the picmobi app.
- Backend API for image classification using FastAPI and PyTorch.
- For the frontend, please refer to [picmobi app](https://github.com/Hattyoriiiiiii/picmobi).

## How to run

1. Clone this repository

```bash
git clone https://github.com/Hattyoriiiiiii/image-classification-api.git
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```bash
brew install libheif

python3 -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]"
pip install numpy python-multipart pillow torch torchvision
pip install pyheif

# pip install -r requirements.txt
```

3. Run the application:

```bash
# uvicorn src.main:app --reload --port 8080
# uvicorn src.main:app --host 0.0.0.0 --port 8000
uvicorn src.main_flyer:app --host 0.0.0.0 --port 8000
```

The server will be available at http://localhost:8000.

4. Test the application:

Tests are located in the `tests/` directory. To run the tests, use the following command:

```bash
pytest
curl -X 'POST' \
  'http://[IP address]:8000/predict' \
  -H 'accept: application/json' \
  -F 'image=@/Users/hattori/Downloads/test_mapo.png'

```


## Training the Model

The model training script is located in `scripts/train_model.py`. To train a new model, use the following command:

```bash
python scripts/train_model.py
```

This will train a new model and save it as `src/models/mnist_model.pth`.

- `POST /predict`: Accepts an image file as input and returns the predicted digit.



## Create a flutter project

```bash
flutter create image_classifier

```

## License

Please see [LICENSE](./LICENSE) for details on how this project is licensed.