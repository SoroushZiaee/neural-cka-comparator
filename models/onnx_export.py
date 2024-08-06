import torch
from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names

# Define a custom export function
def custom_export(model, dummy_input, custom_names):    
    
    # Prepare input and output names
    input_names = ['x']
    output_names = ['output']

    torch.onnx.export(model, 
                      dummy_input, 
                      "resnet_custom_names.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={'x': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}},
                      opset_version=11,
                      do_constant_folding=True,
                      custom_opsets={},
                      export_params=True,
                      keep_initializers_as_inputs=None,
                      verbose=False)