# Vietnamese Ocr Error Detector

This project implements the paper titled [Using Large N-gram for Vietnamese Spell Checking](https://link.springer.com/chapter/10.1007/978-3-319-11680-8_49). 

The paper's main contribution is to provide a systematic approach to approximate a 5-gram model using the 3-gram model to enable a trade-off between accuracy and efficiency. The project aims to detect potential errors in Vietnamese addresses.

## Installation Note
```
nltk==3.5 is required.
```

## Usage
To use the project, import the Runner class from runner.py and create an instance. Then call the inference_address method on a Vietnamese address string to check for potential errors.
```python
from runner import Runner
predictor = Runner(istrain=False, mode='bothward')
text = "Aa Điền X Cộng Hòa, H. Nam Sách, T. Hai Dương"
print(predictor.inference_address(text))
```