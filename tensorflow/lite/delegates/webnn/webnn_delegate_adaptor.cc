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
#include <vector>

#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/delegates/webnn/webnn_delegate_adaptor.h"

namespace tflite {
namespace tools {

TfLiteDelegate* CreateTfLiteWebNNDelegateFromOptions(char** options_keys,
                                                     char** options_values,
                                                     size_t num_options) {
  TfLiteWebNNDelegateOptions options = TfLiteWebNNDelegateOptionsDefault();
  // Parse key-values options to TfLiteWebNNDelegateOptions by mimicking them as
  // command-line flags.
  std::vector<const char*> argv;
  argv.reserve(num_options + 1);
  constexpr char kWebNNDelegateParsing[] = "webnn_delegate_parsing";
  argv.push_back(kWebNNDelegateParsing);

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv.push_back(option_args.rbegin()->c_str());
  }

  constexpr char kWebNNDevicePreference[] = "webnn_device";

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kWebNNDevicePreference,
                               reinterpret_cast<int32_t*>(&options.devicePreference),
                               "WebNN device (0:default, 1:gpu, 2:cpu)."),
  };

  int argc = num_options + 1;
  if (!tflite::Flags::Parse(&argc, argv.data(), flag_list)) {
    return nullptr;
  }

  TFLITE_LOG(INFO) << "WebNN delegate: WebNN device set to "
                   << options.devicePreference << ".";

  return TfLiteWebNNDelegateCreate(&options);
}

}  // namespace tools
}  // namespace tflite

extern "C" {

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::tools::CreateTfLiteWebNNDelegateFromOptions(
      options_keys, options_values, num_options);
}

void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteWebNNDelegateDelete(delegate);
}

}  // extern "C"
