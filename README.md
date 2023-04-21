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

Luego, seguir los paso en `mnist.ipynb` o bien en `evalRT.py` para evaluar el modelo generado por TensorRT sobre la data de prueba. 

# comparaciones

## sistema usado para hacer las evaluaciones:

* OS: Ubuntu 22.04.2 LTS
* CPU: 12th Gen Intel® Core™ i3-12100F × 8
* GPU: NVIDIA Corporation GA106 [GeForce RTX 3060 Lite Hash Rate]
* RAM: 32,0 GiB 3200 MHz

## resultados
se usa el programa en `compare.py` para evaluar los timepos de ejecucion entre ambos modelos

* 100 iteraciones

    Tiempo promedio de eval: 5.04 segundos

    Tiempo promedio del evalRT: 1.76 segundos (fp16 y seg)

# REFS
* mnist resnet50: `https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb`
* resnet50: `https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html`
* TensorRT functions: `https://github.com/triple-Mu/YOLOv8-TensorRT`
