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
#include <string>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace tools {

class WebNNDelegateProvider : public DelegateProvider {
 public:
  WebNNDelegateProvider() {
    default_params_.AddParam("use_webnn", ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;

  std::string GetName() const final { return "WebNN"; }
};
REGISTER_DELEGATE_PROVIDER(WebNNDelegateProvider);

std::vector<Flag> WebNNDelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_webnn", params, "use WebNN")};
  return flags;
}

void WebNNDelegateProvider::LogParams(const ToolParams& params,
                                        bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_webnn", "Use WebNN", verbose);
}

TfLiteDelegatePtr WebNNDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_xnnpack")) {
    return evaluation::CreateXNNPACKDelegate(
        params.Get<int32_t>("num_threads"));
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

}  // namespace tools
}  // namespace tflite
