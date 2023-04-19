# Resnet50 MNIST, TensorRT tests

## Entrenar y evaluar Resnet50 MNIST:

Seguir el codigo: `mnist.ipynb`

Donde se guardaran los pesos del modelo en `weights/mnist.pth`

## Transformar en formato ONNX

Una vez guardado el modelo en `pth`, transformarlo a `onnx` usando el codigo en

``` 
onnx_test.py
```

## de ONNX a  TensorRT

Luego de tener el modelo en ONNX hay que reconstruirlo con TensorRT, usando el codigo en:

``` 
build_test.py
```

lo que da como resultado un archvo `.engine`


## evaluar modelo en TensorRT

Luego, seguir los passo en `mnist.ipynb` o bien en `evalRT.py` para evaluar el modelo generado por TensorRT sobre la data de prueba. 

# REFS
* mnist resnet50: `https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb`
* resnet50: `https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html`
* TensorRT functions: `https://github.com/triple-Mu/YOLOv8-TensorRT`
