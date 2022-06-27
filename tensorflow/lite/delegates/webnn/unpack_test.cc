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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include "tensorflow/lite/delegates/webnn/unpack_tester.h"
#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

namespace tflite {
namespace webnn {

TEST(Split, 4Dto3D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));
  const std::vector<int32_t> shape(
      {shape_rng() * 2, shape_rng() * 2, shape_rng() * 2, shape_rng() * 2});

  for (int i = -4; i < 4; i++) {
    UnpackTester().InputShape(shape).UnpackAxis(i).NumSplits(2).Test(
        TensorType_FLOAT32, webnn_delegate.get());
  }
}

TEST(Split, 3Dto2D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));
  const std::vector<int32_t> shape(
      {shape_rng() * 2, shape_rng() * 2, shape_rng() * 2});

  for (int i = -3; i < 3; i++) {
    UnpackTester().InputShape(shape).UnpackAxis(i).NumSplits(2).Test(
        TensorType_FLOAT32, webnn_delegate.get());
  }
}

TEST(Split, 2Dto1D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng() * 2, shape_rng() * 2});

  for (int i = -2; i < 2; i++) {
    UnpackTester().InputShape(shape).UnpackAxis(i).NumSplits(2).Test(
        TensorType_FLOAT32, webnn_delegate.get());
  }
}

TEST(Split, 1Dto0D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(1, 5), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng() * 2});

  for (int i = -1; i < 1; i++) {
    UnpackTester().InputShape(shape).UnpackAxis(i).NumSplits(2).Test(
        TensorType_FLOAT32, webnn_delegate.get());
  }
}
}  // namespace webnn
}  // namespace tflite