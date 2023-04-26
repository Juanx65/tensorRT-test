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
## pendiente: implementar int8 usando el ejemplo en refs
## sistema usado para hacer las evaluaciones:

* OS: Ubuntu 22.04.2 LTS
* CPU: 12th Gen Intel® Core™ i3-12100F × 8
* GPU: NVIDIA Corporation GA106 [GeForce RTX 3060 Lite Hash Rate]
* RAM: 32,0 GiB 3200 MHz

## resultados
se usa el programa en `compare.py` ( realiza 1000 iteraciones) para evaluar los timepos de ejecucion entre los modelos


|             | size MB | Time avg s  |Time min s|Time max s| accuracy %|
|-------------|---------|-------------|----------|----------|-----------|
| no RT       | 94.4    | 5.04        |5.00      |5.39      |98.46      |
| fp32 (RT)   | 96.6    | 1.89        |1.86      |1.99      |98.53      |
| fp16        | 49.1    | 1.76        |1.72      |2.01      |98.53      |
| int8        | -       | -           |-         |-         | -         |

## Pendientes

* converit a int8 usando tensorRT para comparar
* implementar en fpga usando hls4ml ( no fue posible debido a que no se puede implementar una maxpool2d )

# REFS
* TensorRT python : `https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics`
* mnist resnet50: `https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb`
* resnet50: `https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html`
* TensorRT functions: `https://github.com/triple-Mu/YOLOv8-TensorRT`
* TensorRT INT8 Example: `https://github.com/srihari-humbarwadi/retinanet-tensorflow2.x/tree/dfff5019ad515dda5c4753343c970d97af494f60/retinanet/tensorrt`
* To solve load_state_dict `problem: https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet`
