/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <webnn/webnn_cpp.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>

#include <fp16/fp16.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/optimize/sparsity/format_converter.h"

namespace tflite {
namespace webnn {
namespace {

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
  friend class Subgraph;

 public:
  explicit Delegate(const TfLiteWebNNDelegateOptions* options) {
    // TODO(nhu): support MLDevicePreference and MLPowerPreference
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite WebNN delegate for CPU.");
  }

  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);
  TfLiteDelegate* tflite_delegate() { return &delegate_; }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  // Unpacked data for quasi-static tensors, i.e. tensors produced by
  // dequantizing or unpacking static buffers.
  std::vector<char> static_unpacked_data_;
  // Mapping from a tensor index for a quasi-static tensor to the offset to
  // its unpacked data within static_unpacked_data_.
  std::unordered_map<int, size_t> static_unpacked_data_map_;
  // Set of indices of nodes which unpack static data, e.g. Dequantize
  // operators which convert FP16 static weights to FP32. These nodes are simply
  // ignored in the delegate implementation, because their outputs are
  // pre-unpacked in DelegatePrepare.
  std::unordered_set<int> static_unpack_nodes_;
  // Set of indices of tensors with unpacked static sparse weights.
  std::unordered_set<int> static_sparse_weights_;
};

class Subgraph {
 public:
  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          const Delegate* delegate) {
    // Convert subgraph inputs and outputs to hash sets for faster lookup.
    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    std::unordered_set<int> outputs;
    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      // Exclude quasi-static tensors which may have become subgraph outputs
      // after partitioning.
      if (delegate->static_unpacked_data_map_.count(output_tensor_idx) == 0) {
        outputs.insert(output_tensor_idx);
      }
    }

    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return nullptr;
    }

    // Create WebNN context and graph builder
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    ml::Context ml_context = ml::Context(webnn_native::CreateContext());
    if (!ml_context) {
      TF_LITE_KERNEL_LOG(context, "Failed to create WebNN context.");
      return nullptr;
    }
    ml::GraphBuilder ml_builder = ml::CreateGraphBuilder(ml_context);
    if (!ml_builder) {
      TF_LITE_KERNEL_LOG(context, "Failed to create WebNN graph builder.");
      return nullptr;
    }

    bool has_sparse_weights = false;
    // Detect which tensors are used as inputs or outputs of any subgraph nodes.
    // -1 denotes tensor not used in the subgraph. These indexes will be
    // filtered out and removed later.
    std::vector<int> tensors(context->tensors_size, -1);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      // Detect if any of the node's inputs are sparse weights.
      if (!has_sparse_weights) {
        for (int i = 0; i < node->inputs->size; i++) {
          if (delegate->static_sparse_weights_.count(node->inputs->data[i]) !=
              0) {
            has_sparse_weights = true;
          }
        }
      }

      if (delegate->static_unpack_nodes_.count(node_index) != 0) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      switch (registration->builtin_code) {
        case kTfLiteBuiltinReshape:
          // Ignore the second input (new shape),
          // because it is represented as parameters of the WebNN operator
          // rather than extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
          }
          break;
        default:
          // All other operators: process all inputs
          for (int k = 0; k < node->inputs->size; k++) {
            const int t = node->inputs->data[k];
            if (t >= 0) {
              tensors[t] = t;
            }
          }
      }
      for (int k = 0; k < node->outputs->size; k++) {
        const int t = node->outputs->data[k];
        if (t >= 0) {
          tensors[t] = t;
        }
      }
    }
    // Filter out and remove -1 (unused) indexes.
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](int i) { return i < 0; }),
                  tensors.end());
    std::sort(tensors.begin(), tensors.end());

    // WebNN operands for TFLite tensors
    std::vector<ml::Operand> webnn_operands(tensors.back() + 1);
    std::unordered_set<int> compute_inputs;
    for (int t : tensors) {
      ml::OperandType datatype;
      switch (context->tensors[t].type) {
        case kTfLiteFloat32:
          datatype = ml::OperandType::Float32;
          break;
        default:
          TF_LITE_KERNEL_LOG(
              context,
              "unsupported datatype (%s) of tensor %d in WebNN delegate",
              TfLiteTypeGetName(context->tensors[t].type), t);
          return nullptr;
      }

      const void* data = nullptr;
      if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
        data = context->tensors[t].data.raw_const;
      } else {
        // Check for quasi-static data.
        const auto it = delegate->static_unpacked_data_map_.find(t);
        if (it != delegate->static_unpacked_data_map_.end()) {
          data = delegate->static_unpacked_data_.data() + it->second;
        }
      }

      std::vector<int32_t> dims(
          &context->tensors[t].dims->data[0],
          &context->tensors[t].dims->data[context->tensors[t].dims->size]);

      if (inputs.count(t) != 0) {
        ml::OperandDescriptor desc;
        desc.dimensions = dims.data();
        desc.dimensionsCount = dims.size();
        desc.type = datatype;

        ml::Operand operand;
        if (data == nullptr) {
          compute_inputs.insert(t);
          std::string name = std::to_string(t);
          operand = ml_builder.Input(name.c_str(), &desc);
        } else {
          operand = ml_builder.Constant(&desc, data, context->tensors[t].bytes);
        }
        webnn_operands[t] = operand;
      }
    }

    // Create a set of quasi-static tensors for VisitNode function
    std::unordered_set<int> quasi_static_tensors;
    for (const std::pair<const int, size_t>& entry :
         delegate->static_unpacked_data_map_) {
      quasi_static_tensors.insert(entry.first);
    }

    // Create WebNN nodes for TFLite delegate nodes
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];
      if (delegate->static_unpack_nodes_.count(node_index)) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(ml_builder, context, registration, node, node_index,
                    quasi_static_tensors, webnn_operands) != kTfLiteOk) {
        return nullptr;
      }
    }

    ml::NamedOperands named_operands = ml::CreateNamedOperands();
    for (auto o : outputs) {
      std::string name = std::to_string(o);
      named_operands.Set(name.c_str(), webnn_operands[o]);
    }

    ml::Graph ml_graph = ml_builder.BuildSync(named_operands);
    if (!ml_graph) {
      TF_LITE_KERNEL_LOG(context, "failed to build WebNN graph");
      return nullptr;
    }

    return new Subgraph(ml_graph, std::move(compute_inputs), std::move(outputs));
  }

  TfLiteStatus Prepare(TfLiteContext* context) { return kTfLiteOk; }

  TfLiteStatus Invoke(TfLiteContext* context) {
    std::vector<ml::Input> ml_inputs;
    ml::NamedInputs named_inputs = ml::CreateNamedInputs();
    for (int t : inputs_) {
      ml::Input ml_input;
      ml_input.buffer = context->tensors[t].data.raw;
      ml_input.size = context->tensors[t].bytes;
      std::string name = std::to_string(t);
      ml_inputs.push_back(ml_input);
      named_inputs.Set(name.c_str(), &(ml_inputs.back()));
    }

    std::vector<ml::Output> ml_outputs;
    ml::NamedOutputs named_outputs = ml::CreateNamedOutputs();
    for (int t : outputs_) {
      ml::Output ml_output;
      ml_output.buffer = context->tensors[t].data.raw;
      ml_output.size = context->tensors[t].bytes;
      std::string name = std::to_string(t);
      ml_outputs.push_back(ml_output);
      named_outputs.Set(name.c_str(), &(ml_outputs.back()));
    }

    ml::ComputeGraphStatus status = ml_graph_.ComputeSync(named_inputs, named_outputs);
    if (status != ml::ComputeGraphStatus::Success) {
      TF_LITE_KERNEL_LOG(context, "failed to compute WebNN graph");
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus ConvertActivationToOutputRange(
      TfLiteContext* context, int node_index, TfLiteFusedActivation activation,
      float* output_min, float* output_max) {
    switch (activation) {
      case kTfLiteActNone:
        *output_min = -std::numeric_limits<float>::infinity();
        *output_max = +std::numeric_limits<float>::infinity();
        return kTfLiteOk;
      case kTfLiteActRelu:
        *output_min = 0.0f;
        *output_max = +std::numeric_limits<float>::infinity();
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
        *output_min = -1.0f;
        *output_max = +1.0f;
        return kTfLiteOk;
      case kTfLiteActRelu6:
        *output_min = 0.0f;
        *output_max = 6.0f;
        return kTfLiteOk;
      case kTfLiteActTanh:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Tanh) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSignBit:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sigmoid) in node #%d",
            node_index);
        return kTfLiteError;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid fused activation (%d) in node #%d",
                                 static_cast<int>(activation), node_index);
        return kTfLiteError;
    }
  }

  static TfLiteStatus CheckConvolutionParams(TfLiteContext* context,
                                             const TfLiteConvParams* params,
                                             int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckDepthwiseConvolutionParams(
      TfLiteContext* context, const TfLiteDepthwiseConvParams* params,
      int output_channels, int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->depth_multiplier <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid depth multiplier %d in node #%d",
                               params->depth_multiplier, node_index);
      return kTfLiteError;
    }
    if (output_channels % params->depth_multiplier != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "depth multiplier %d is incompatible with "
                               "number of output channels %d in node #%d",
                               params->depth_multiplier, output_channels,
                               node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckMediaPipeTransposedConvolutionParams(
      TfLiteContext* context, const TfLiteTransposeConvParams* params,
      int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckMediaPipePoolParams(TfLiteContext* context,
                                               const TfLitePoolParams* params,
                                               int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }
    if (params->filter_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter width %d in node #%d",
                               params->filter_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter height %d in node #%d",
                               params->filter_height, node_index);
      return kTfLiteError;
    }
    if (params->filter_width != params->stride_width) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "filter width %d does not match stride width %d in node #%d",
          params->filter_width, params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height != params->stride_height) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "filter height %d does not match stride height %d in node #%d",
          params->filter_height, params->stride_height, node_index);
      return kTfLiteError;
    }
    switch (params->activation) {
      case kTfLiteActNone:
        break;
      case kTfLiteActRelu:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Relu) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (ReluMinus1To1) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActRelu6:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Relu6) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActTanh:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Tanh) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSignBit:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sigmoid) in node #%d",
            node_index);
        return kTfLiteError;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "invalid fused activation (%d) in node #%d",
            static_cast<int>(params->activation), node_index);
        return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckFullyConnectedParams(
      TfLiteContext* context, const TfLiteFullyConnectedParams* params,
      int node_index) {
    if (params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported non-default weights format in node #%d",
          node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckPoolingParams(TfLiteContext* context,
                                         const TfLitePoolParams* params,
                                         int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->filter_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter width %d in node #%d",
                               params->filter_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter height %d in node #%d",
                               params->filter_height, node_index);
      return kTfLiteError;
    }

    if (params->filter_width == 1 && params->filter_height == 1 &&
        std::max(params->stride_width, params->stride_height) > 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unsupported pooling with 1x1 filter "
                               "and %dx%d stride in node #%d",
                               params->stride_width, params->stride_height,
                               node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputsAndOutputs(
      TfLiteContext* context, TfLiteNode* node, int min_num_inputs,
      int max_num_inputs, int expected_num_outputs, int node_index) {
    if (node->inputs->size < min_num_inputs ||
        node->inputs->size > max_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of inputs (%d) in node #%d",
                               node->inputs->size, node_index);
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d != %d) in node #%d",
          node->outputs->size, expected_num_outputs, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputsAndOutputs(TfLiteContext* context,
                                               TfLiteNode* node,
                                               int expected_num_inputs,
                                               int expected_num_outputs,
                                               int node_index) {
    if (node->inputs->size != expected_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of inputs (%d != %d) in node #%d",
          node->inputs->size, expected_num_inputs, node_index);
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d != %d) in node #%d",
          node->outputs->size, expected_num_outputs, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorType(TfLiteContext* context,
                                      const TfLiteTensor& tensor,
                                      TfLiteType expected_type,
                                      int tensor_index, int node_index) {
    if (tensor.type != expected_type) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported type %s in tensor #%d in node #%d",
          TfLiteTypeGetName(tensor.type), tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloat32Type(TfLiteContext* context,
                                             const TfLiteTensor& tensor,
                                             int tensor_index, int node_index) {
    return CheckTensorType(context, tensor, kTfLiteFloat32, tensor_index,
                           node_index);
  }

  static TfLiteStatus CheckTensorFloat32OrQInt8Type(TfLiteContext* context,
                                                    const TfLiteTensor& tensor,
                                                    int tensor_index,
                                                    int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        break;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
        return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloat32OrQInt32Type(TfLiteContext* context,
                                                     const TfLiteTensor& tensor,
                                                     int tensor_index,
                                                     int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        break;
#ifndef XNN_NO_QS8_OPERATORS
      case kTfLiteInt32:
        if (tensor.params.scale == 0.0f) {
          TF_LITE_MAYBE_KERNEL_LOG(
              context,
              "unsupported quantization scale %.7g in tensor #%d in node #%d",
              tensor.params.scale, tensor_index, node_index);
          return kTfLiteError;
        }
        break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
        return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int min_num_dims, int max_num_dims,
                                       int tensor_index) {
    if (min_num_dims == max_num_dims) {
      if (tensor.dims->size != min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "%d dimensions expected",
            tensor.dims->size, tensor_index, min_num_dims);
        return kTfLiteError;
      }
    } else {
      if (tensor.dims->size < min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "at least %d dimensions expected",
            tensor.dims->size, tensor_index, min_num_dims);
        return kTfLiteError;
      }
      if (tensor.dims->size > max_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "at most %d dimensions expected",
            tensor.dims->size, tensor_index, max_num_dims);
        return kTfLiteError;
      }
    }
    for (int i = 0; i < tensor.dims->size; i++) {
      if (tensor.dims->data[i] <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid num of elements (%d) in "
                                 "dimension #%d in tensor #%d",
                                 tensor.dims->data[i], i, tensor_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int expected_num_dims,
                                       int tensor_index) {
    return CheckTensorShape(context, tensor, expected_num_dims,
                            expected_num_dims, tensor_index);
  }

  static TfLiteStatus CheckSlopeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int tensor_index, int node_index) {
    if (tensor.dims->size < 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "tensor #%d in node #%d: "
                               "expected at least a 1D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    // Validate that all non-channel dimensions (if any) are exactly 1.
    for (int i = 0; i < tensor.dims->size - 1; i++) {
      if (tensor.dims->data[i] != 1) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unexpected value %d of shape dimension #%d in "
            "tensor #%d in node #%d: "
            "expected 1 for non-channel dimensions",
            tensor.dims[i], i, tensor_index, node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckPaddingsTensorShape(TfLiteContext* context,
                                               const TfLiteTensor& tensor,
                                               int expected_rows,
                                               int tensor_index,
                                               int node_index) {
    if (tensor.dims->size != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "padding tensor #%d in node #%d: "
                               "expected a 2D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    if (tensor.dims->data[0] != expected_rows) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of rows (%d) in "
                               "padding tensor #%d in node #%d: "
                               "%d rows expected",
                               tensor.dims->size, tensor_index, node_index,
                               expected_rows);
      return kTfLiteError;
    }
    if (tensor.dims->data[1] != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of columns (%d) in "
                               "padding tensor #%d in node #%d: "
                               "2 columns expected",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckAxesTensorShape(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    if (tensor.dims->size != 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "axes tensor #%d in node #%d: "
                               "expected a 1D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckShapeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int tensor_index, int node_index) {
    if (tensor.dims->size != 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "shape tensor #%d in node #%d: "
                               "expected a 1D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorNonDynamicAllocation(
      TfLiteContext* context, const TfLiteTensor& tensor, int tensor_index,
      int node_index) {
    // TODO: remove checks once dynamic tensors are supported
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected non-dynamic tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorStaticAllocation(TfLiteContext* context,
                                                  const TfLiteTensor& tensor,
                                                  int tensor_index,
                                                  int node_index) {
    if (tensor.allocation_type != kTfLiteMmapRo ||
        tensor.data.raw_const == nullptr) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected static read-only tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitNode(
      ml::GraphBuilder builder, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index,
      const std::unordered_set<int>& quasi_static_tensors,
      std::vector<ml::Operand>& webnn_operands) {
    // TFLite context used for logging purposes. When we create a new node
    // (subgraph is non-null), logging context is the same as context, and error
    // messages are passed to TFLite. When we detect supported operations
    // (subgraph is null), logging context is null, and error messages are
    // supressed.
    TfLiteContext* logging_context = builder == nullptr ? nullptr : context;
    switch (registration->builtin_code) {
      case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return VisitAddNode(builder, logging_context, node_index, node,
                            context->tensors, add_params, webnn_operands);
      }
      default:
        return kTfLiteError;
    }
  }

  static TfLiteStatus VisitAddNode(
      ml::GraphBuilder builder, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteAddParams* add_params,
      std::vector<ml::Operand>& webnn_operands) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input1_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input2_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (builder) {
      ml::Operand output =
        builder.Add(webnn_operands[node->inputs->data[0]], webnn_operands[node->inputs->data[1]]);

      float output_min = -std::numeric_limits<float>::infinity();
      float output_max = +std::numeric_limits<float>::infinity();
      if (add_params != nullptr) {
        TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
            logging_context, node_index, add_params->activation, &output_min,
            &output_max));
        ml::OperandDescriptor desc;
        std::vector<int32_t> dims = {1};
        desc.type = ml::OperandType::Float32;
        desc.dimensions = dims.data();
        desc.dimensionsCount = dims.size();
        ml::Operand minValue = builder.Constant(&desc, &output_min, sizeof(float));
        ml::Operand maxValue = builder.Constant(&desc, &output_max, sizeof(float));
        ml::ClampOptions options;
        options.maxValue = maxValue;
        options.minValue = minValue;
        output = builder.Clamp(output, &options);
      }
      webnn_operands[node->outputs->data[0]] = output;
    }

    return kTfLiteOk;
  }

 private:
  Subgraph(ml::Graph graph, std::unordered_set<int>&& inputs, std::unordered_set<int>&& outputs)
      : ml_graph_(graph), inputs_(inputs), outputs_(outputs) {}

  ml::Graph ml_graph_;
  // TFLite Tensor IDs == name of input/output tensors for the
  // delegated subgraph.
  std::unordered_set<int> inputs_;
  std::unordered_set<int> outputs_;
};

TfLiteIntArray* Delegate::PrepareOpsToDelegate(TfLiteContext* context) {
  // Clear previous data, in case the delegate is reused without re-creation.
  static_unpacked_data_map_.clear();
  static_unpacked_data_.clear();
  static_unpack_nodes_.clear();
  static_sparse_weights_.clear();

  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  // Mapping for quasi-static (unpacked from static) tensor index to the node
  // index that produced it.
  std::unordered_map<int, int> quasi_static_tensors_producers;
  // Set of all quasi-static tensors in the execution plan.
  std::unordered_set<int> quasi_static_tensors;
  // Set of quasi-static tensors consumed by the delegated nodes.
  std::unordered_set<int> quasi_static_tensors_to_unpack;

  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to WebNN
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         node_index);
      continue;  // Soft error (skip this node).
    }

    // Prepare to unpack FP16 tensors.
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];
      if ((input_tensor.allocation_type == kTfLiteMmapRo ||
           quasi_static_tensors.count(node->inputs->data[0]) != 0) &&
          input_tensor.type == kTfLiteFloat16 &&
          output_tensor.type == kTfLiteFloat32) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);

        if (input_tensor.allocation_type != kTfLiteMmapRo) {
          quasi_static_tensors_to_unpack.insert(node->inputs->data[0]);
        }

        // If dequantized input is sparse, so is its output
        if (static_sparse_weights_.count(node->inputs->data[0]) != 0) {
          static_sparse_weights_.insert(node->outputs->data[0]);
        }

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    // Prepare to unpack sparse tensors.
    // TODO(b/157729695): In the future, we also need to handle the case where a
    // sparse tensor is fed to a TFLite op directly, and no Densify() op is
    // inserted. For now this is not a problem because the Conv() op in tflite
    // can only consume dense tensors.
    if (registration->builtin_code == kTfLiteBuiltinDensify &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];
      if (input_tensor.allocation_type == kTfLiteMmapRo &&
          input_tensor.sparsity != nullptr &&
          (input_tensor.type == kTfLiteFloat16 ||
           input_tensor.type == kTfLiteFloat32) &&
          output_tensor.type == input_tensor.type) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);
        static_sparse_weights_.insert(node->outputs->data[0]);

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    ml::GraphBuilder null_builder;
    std::vector<ml::Operand> empty_webnn_operands;
    if (Subgraph::VisitNode(null_builder, context, registration, node,
                            node_index, quasi_static_tensors,
                            empty_webnn_operands) != kTfLiteOk) {
      // If a non-delegated node consumes output of a node that unpacks static
      // data, that node shouldn't be delegated.
      for (int j = 0; j < node->inputs->size; j++) {
        const auto it =
            quasi_static_tensors_producers.find(node->inputs->data[j]);
        if (it != quasi_static_tensors_producers.end()) {
          static_unpack_nodes_.erase(it->second);
        }
      }

      // Non-delegatable node is not an error.
      continue;
    }

    for (int j = 0; j < node->inputs->size; j++) {
      if (quasi_static_tensors.count(node->inputs->data[j]) != 0) {
        quasi_static_tensors_to_unpack.insert(node->inputs->data[j]);
      }
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }

  // Sort quasi-static tensors to be unpacked by the node index the produced
  // them. This ensures that in situations where quasi-static tensor is
  // produced from another quasi-static tensor, the tensors are unpacked in
  // the original execution plan order.
  std::vector<int> sorted_quasi_static_tensors_to_unpack(
      quasi_static_tensors_to_unpack.cbegin(),
      quasi_static_tensors_to_unpack.cend());
  std::sort(sorted_quasi_static_tensors_to_unpack.begin(),
            sorted_quasi_static_tensors_to_unpack.end(),
            [&quasi_static_tensors_producers](int t1, int t2) {
              return quasi_static_tensors_producers[t1] <
                     quasi_static_tensors_producers[t2];
            });

  // Unpack static data of all tensors
  for (int t : sorted_quasi_static_tensors_to_unpack) {
    const int producer_index = quasi_static_tensors_producers[t];
    // Check if TFLite nodes can be delegated to WebNN
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->inputs->size != 1) {
      TF_LITE_KERNEL_LOG(context, "unexpected number of inputs (%d) in node %d",
                         node->inputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(context,
                         "unexpected number of outputs (%d) in node %d",
                         node->outputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];

    // Consider the case when the input to unpacking node is quasi-static.
    const auto static_unpacked_input_it_ =
        static_unpacked_data_map_.find(node->inputs->data[0]);
    if (static_unpacked_input_it_ == static_unpacked_data_map_.end()) {
      if (input_tensor.allocation_type != kTfLiteMmapRo) {
        TF_LITE_KERNEL_LOG(
            context,
            "unexpected allocation type (%d) in tensor %d in node %d (%d)",
            input_tensor.allocation_type, node->inputs->data[0], producer_index,
            registration->builtin_code);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    const TfLiteTensor& output_tensor = context->tensors[t];
    size_t tensor_elements = output_tensor.bytes;
    switch (output_tensor.type) {
      case kTfLiteFloat32:
        tensor_elements /= sizeof(float);
        break;
      case kTfLiteFloat16:
        tensor_elements /= sizeof(uint16_t);
        break;
      default: {
        TF_LITE_KERNEL_LOG(context,
                           "unexpected datatype (%s) in tensor %d in node %d",
                           TfLiteTypeGetName(output_tensor.type),
                           node->outputs->data[0], producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    const size_t tensor_offset = static_unpacked_data_.size();
    static_unpacked_data_.resize(tensor_offset + context->tensors[t].bytes);

    char* unpacked_data = static_unpacked_data_.data() + tensor_offset;
    const char* packed_data =
        static_unpacked_input_it_ != static_unpacked_data_map_.end()
            ? static_unpacked_data_.data() + static_unpacked_input_it_->second
            : static_cast<const char*>(input_tensor.data.data);
    switch (registration->builtin_code) {
      case kTfLiteBuiltinDequantize: {
        if (input_tensor.type != kTfLiteFloat16) {
          TF_LITE_KERNEL_LOG(
              context, "unexpected tensor %d data type (%s) in node %d",
              node->inputs->data[0], TfLiteTypeGetName(input_tensor.type),
              producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        if (input_tensor.sparsity != nullptr) {
          TF_LITE_KERNEL_LOG(context,
                             "unexpected FP16 sparse tensor %d in node %d",
                             node->inputs->data[0], producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        // Actual data unpacking
        float* unpacked_fp32_data = reinterpret_cast<float*>(unpacked_data);
        const uint16_t* packed_fp16_data =
            reinterpret_cast<const uint16_t*>(packed_data);
        for (size_t i = 0; i < tensor_elements; i++) {
          unpacked_fp32_data[i] = fp16_ieee_to_fp32_value(packed_fp16_data[i]);
        }
        break;
      }
      case kTfLiteBuiltinDensify: {
        if (input_tensor.sparsity == nullptr) {
          TF_LITE_KERNEL_LOG(context, "unexpected dense tensor %d in node %d",
                             node->inputs->data[0], producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        const int dims_count = output_tensor.dims->size;
        std::vector<int> vector_shape(dims_count);
        for (int i = 0; i < dims_count; i++) {
          vector_shape[i] = output_tensor.dims->data[i];
        }

        switch (input_tensor.type) {
          case kTfLiteFloat32: {
            const size_t dense_size = context->tensors[t].bytes / sizeof(float);
            float* unpacked_fp32_data = reinterpret_cast<float*>(unpacked_data);
            tflite::optimize::sparsity::FormatConverter<float> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const float*>(input_tensor.data.data), dense_size,
                unpacked_fp32_data, context);
            break;
          }
          case kTfLiteFloat16: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(Eigen::half);
            Eigen::half* unpacked_fp16_data =
                reinterpret_cast<Eigen::half*>(unpacked_data);
            tflite::optimize::sparsity::FormatConverter<Eigen::half> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const Eigen::half*>(input_tensor.data.data),
                dense_size, unpacked_fp16_data, context);
            break;
          }
          default: {
            TF_LITE_KERNEL_LOG(
                context, "unexpected tensor %d data type (%s) in node %d",
                node->inputs->data[0], TfLiteTypeGetName(input_tensor.type),
                producer_index);
            TfLiteIntArrayFree(nodes_to_delegate);
            return nullptr;  // Hard error.
          }
        }
        break;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "unexpected op registration %d at node %d",
                           registration->builtin_code, producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
    }

    static_unpacked_data_map_[t] = tensor_offset;
  }

  // Add nodes that unpack static data consumed by delegated nodes.
  // Note: this is done purely to avoid the overhead of running these nodes
  // again in TFLite interpreter which would allocate memory for their outputs.
  // We mark them as delegated, but the delegate would simply ignore these nodes
  // as the static weights are already unpacked.
  for (int node_index : static_unpack_nodes_) {
    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }
  std::sort(&nodes_to_delegate->data[0],
            &nodes_to_delegate->data[nodes_to_delegate->size]);

#ifdef WEBNN_DELEGATE_TEST_MODE
  // In the test mode build (used by unit tests), WebNN delegate claims to
  // support all operators in the execution plan to disable fallback to the
  // default TensorFlow Lite kernels. Thus, if any of the ops in the model are
  // not supported by the delegate, they will cause a failure in
  // ::tflite::Interpreter::ModifyGraphWithDelegate, to be caught in the unit
  // tests.
  nodes_to_delegate->size = execution_plan->size;
  std::copy(&execution_plan->data[0],
            &execution_plan->data[execution_plan->size],
            &nodes_to_delegate->data[0]);
#endif

  return nodes_to_delegate;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  return static_cast<void*>(Subgraph::Create(
      context, params,
      static_cast<::tflite::webnn::Delegate*>(params->delegate->data_)));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Prepare(context);
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Invoke(context);
}

void SubgraphFree(TfLiteContext* context, void* buffer) {
  if (buffer != nullptr) {
    delete static_cast<Subgraph*>(buffer);
  }
}

const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteWebNNDelegate",
    /*.version=*/2,
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::webnn::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
  if (ops_to_replace == nullptr) {
    return kTfLiteError;
  }

  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace webnn
}  // namespace tflite

TfLiteWebNNDelegateOptions TfLiteWebNNDelegateOptionsDefault() {
  TfLiteWebNNDelegateOptions options = {0, 0};
  return options;
}

TfLiteDelegate* TfLiteWebNNDelegateCreate(
    const TfLiteWebNNDelegateOptions* options) {
  auto* webnn_delegate = new ::tflite::webnn::Delegate(options);
  return webnn_delegate ? webnn_delegate->tflite_delegate() : nullptr;
}

void TfLiteWebNNDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate != nullptr) {
    delete static_cast<::tflite::webnn::Delegate*>(delegate->data_);
  }
}
