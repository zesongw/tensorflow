/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/lite/delegates/webnn/unpack_tester.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace webnn {

template <class T>
void UnpackTester::Test(Interpreter* delegate_interpreter,
                        Interpreter* default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int32_t> input_distribution(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::generate(default_input_data,
                default_input_data + ComputeSize(InputShape()),
                std::ref(input_rng));

  T* webnn_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            webnn_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  int output_size = NumSplits();
  std::vector<T*> default_output(output_size);
  std::vector<T*> delegate_output(output_size);
  for (size_t i = 0; i < output_size; i++) {
    default_output[i] = default_interpreter->typed_output_tensor<T>(i);
    delegate_output[i] = delegate_interpreter->typed_output_tensor<T>(i);
  }

  for (size_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < ComputeSize(OutputShape()); j++) {
      ASSERT_EQ(default_output[i][j], delegate_output[i][j]);
    }
  }
}

template <>
void UnpackTester::Test<float>(Interpreter* delegate_interpreter,
                               Interpreter* default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input_distribution(-1.0f, 1.0f);
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate(default_input_data,
                default_input_data + ComputeSize(InputShape()),
                std::ref(input_rng));
  float* webnn_input_data = delegate_interpreter->typed_input_tensor<float>(0);

  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            webnn_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  int output_size = NumSplits();
  std::vector<float*> default_output(output_size);
  std::vector<float*> delegate_output(output_size);
  for (size_t i = 0; i < output_size; i++) {
    default_output[i] = default_interpreter->typed_output_tensor<float>(i);
    delegate_output[i] = delegate_interpreter->typed_output_tensor<float>(i);
  }

  for (size_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < ComputeSize(OutputShape()); j++) {
      ASSERT_EQ(default_output[i][j], delegate_output[i][j]);
    }
  }
}

void UnpackTester::Test(TensorType tensor_type,
                        TfLiteDelegate* delegate) const {
  std::vector<char> buffer = CreateTfLiteModel(tensor_type);
  const Model* model = GetModel(buffer.data());

  int32_t axis = UnpackAxis();
  axis += axis < 0 ? InputShape().size() : 0;
  ASSERT_EQ(0, InputShape()[axis] % NumSplits());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);
  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);
  ASSERT_EQ(delegate_interpreter->outputs().size(), NumSplits());
  ASSERT_EQ(default_interpreter->outputs().size(), NumSplits());
  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  switch (tensor_type) {
    case TensorType_FLOAT32:
      Test<float>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case TensorType_INT8:
      Test<int8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case TensorType_UINT8:
      Test<uint8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
    default:
      GTEST_FAIL();
  }
}

std::vector<char> UnpackTester::CreateTfLiteModel(
    TensorType tensor_type) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_UNPACK);

  std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  std::vector<flatbuffers::Offset<Tensor>> tensors{{CreateTensor(
      builder,
      builder.CreateVector<int32_t>(InputShape().data(), InputShape().size()),
      tensor_type,
      /*buffer=*/0, /*name=*/0)}};

  for (int i = 0; i < NumSplits(); i++) {
    tensors.push_back(
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(OutputShape().data(),
                                                   OutputShape().size()),
                     tensor_type,
                     /*buffer=*/0, /*name=*/0));
  }

  const std::array<int32_t, 1> op_inputs{{0}};
  std::vector<int32_t> op_outputs;
  op_outputs.reserve(NumSplits());
  for (int i = 0; i < NumSplits(); i++) {
    op_outputs.push_back(op_inputs.size() + i);
  }
  EXPECT_EQ(op_outputs.size(), NumSplits());
  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_UnpackOptions,
      CreateUnpackOptions(builder, NumSplits(), UnpackAxis()).Union());

  const std::array<int32_t, 1> subgraph_inputs = op_inputs;
  const std::vector<int32_t> subgraph_outputs = op_outputs;
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("Unpack model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t UnpackTester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace webnn
}  // namespace tflite