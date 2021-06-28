load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

EMSCRIPTEN_LINKOPTS = [
    "-s ASSERTIONS=2",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s DEMANGLE_SUPPORT=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s TOTAL_MEMORY=134217728",
]

exports_files([
    "webnn_delegate.h",
])

cc_library(
    name = "webnn_delegate",
    srcs = ["webnn_delegate.cc"],
    hdrs = ["webnn_delegate.h"],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize/sparsity:format_converter",
        "@FP16",
    ],
)

cc_library(
    name = "webnn_delegate_hdrs_only",
    hdrs = ["webnn_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)