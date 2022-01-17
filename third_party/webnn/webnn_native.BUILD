include_files = glob([
    "out/Release/gen/src/include/**/*.h",
])

lib_files = [
    "out/Release/libwebnn_native.so",
    "out/Release/libngraph_c_api.so",
    "out/Release/libwebnn_proc.so",
]

cc_library(
    name = "webnn-native",
    hdrs = include_files,
    includes = ["include"],
    strip_include_prefix = "out/Release/gen/src/include",
    visibility = ["//visibility:public"],
    srcs = lib_files,
)

cc_library(
    name = "webnn-native-1",
    hdrs = glob(["src/include/*/*.h"]),
    includes = ["include"],
    strip_include_prefix = "src/include",
    visibility = ["//visibility:public"],
    srcs = lib_files,
)

