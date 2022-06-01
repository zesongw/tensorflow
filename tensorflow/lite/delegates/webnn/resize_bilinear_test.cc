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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/webnn/resize_bilinear_tester.h"
#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

namespace tflite {
namespace webnn {

TEST(ResizeBilinear, AlignCenters) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  ResizeBilinearTester()
      .HalfPixelCenters(true)
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(webnn_delegate.get());
}

// Webnn does not support this option. Corresponding issue:https://github.com/webmachinelearning/webnn/issues/270
TEST(ResizeBilinear, DISABLED_AlignCentersTF1X) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  ResizeBilinearTester()
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(webnn_delegate.get());
}

// Webnn does not support this option. Corresponding issue:https://github.com/webmachinelearning/webnn/issues/270
TEST(ResizeBilinear, DISABLED_AlignCorners) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  ResizeBilinearTester()
      .AlignCorners(true)
      .InputHeight(size_rng())
      .InputWidth(size_rng())
      .OutputHeight(size_rng())
      .OutputWidth(size_rng())
      .Channels(channel_rng())
      .Test(webnn_delegate.get());
}

}  // namespace webnn
}  // namespace tflite
