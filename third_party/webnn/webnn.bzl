def _get_webnn_native_dir(repository_ctx):
    """Gets the Webnn-native path"""
    webnn_native_dir = repository_ctx.os.environ.get("WEBNN_NATIVE_DIR")
    if webnn_native_dir != None:
        return webnn_native_dir
    else:
        fail("Cannot find Webnn-native dir, please set 'WEBNN_NATIVE_DIR' environment variable.")

def _webnn_native_impl(repository_ctx):
    webnn_native_dir = _get_webnn_native_dir(repository_ctx)
    repository_ctx.symlink(webnn_native_dir, "webnn-native")
    repository_ctx.file("BUILD", """
cc_library(
    name = "webnn-native",
    hdrs = glob([
        "webnn-native/out/Release/gen/src/include/**/*.h",
        "webnn-native/src/include/*/*.h"
    ]),
    srcs = select({
        "@bazel_tools//src/conditions:windows": glob([
            "webnn-native/out/Release/webnn_native.dll",
            "webnn-native/out/Release/webnn_native.dll.lib",
            "webnn-native/out/Release/webnn_proc.dll",
            "webnn-native/out/Release/webnn_proc.dll.lib",
            "webnn-native/out/Release/gen/src/webnn/webnn_cpp.cpp"
        ]),
        "//conditions:default":glob([
            "webnn-native/out/Release/libngraph_c_api.so",
            "webnn-native/out/Release/libwebnn_native.so",
            "webnn-native/out/Release/libwebnn_proc.so",
            "webnn-native/out/Release/gen/src/webnn/webnn_cpp.cpp"
        ]),
    }),
    includes = [
        "webnn-native/out/Release/gen/src/include",
        "webnn-native/src/include"
    ],
    visibility = ["//visibility:public"],
)
    """)

webnn_configure = repository_rule(
    implementation = _webnn_native_impl,
    local = True,
    environ = [
        "WEBNN_NATIVE_DIR",
    ],
)