import hls4ml

from tensorflow.keras.models import load_model

""" DEVICE = 'cuda:0'
NUM_CLASSES = 10

## Carga el modelo PyTorch
model_torch = resnet18(NUM_CLASSES)
model_torch.to(DEVICE)
state_dict = torch.load("weights/mnist18.pth")
model_torch.load_state_dict(state_dict, strict=False)
model_torch.eval()
#summary(model_torch,(1,28,28))

hls_model = hls4ml.converters.convert_from_pytorch_model(model=model_torch, input_shape=(1,28,28)) """

## en pausa, porque no es posible implementar el bloque residual ni el maxpooling2d

""" ## cargar modelo onnx
model_onnx = onnx.load("weights/mnist18.onnx")
hls_moel = hls4ml.converters.convert_from_onnx_model(model=model_onnx) """

""" # Convertir a modelo Keras
modelo_onnx = onnx.load('weights/mnist18.onnx')
modelo_keras = onnx_to_keras(onnx_model=modelo_onnx,input_names=['images'],name_policy='renumerate')
hls_model = hls4ml.converters.convert_from_keras_model(model=modelo_keras) """

## cargar modelo entrenado usando keras

keras_model = load_model("weights/resnet18_mnist.h5")
hls_model = hls4ml.converters.convert_from_keras_model(model=keras_model,output_dir='weights/hls_test',project_name='mnist18_test',backend='Vivado') 
hls_model.build()
#board='xc7a35ticsg324-1L'
# Print out the report if you want 
#hls4ml.report.read_vivado_report('my-hls-test')