# Contributing

感谢参与 `ops-tilelang`。提交算子前请确认：

1. 公共 API 不暴露 TileLang 内部对象；
2. wrapper 对 device、dtype、shape、stride 和输出缓冲区做显式校验；
3. kernel 文件只包含 Ascend 调度及其必要的生成器；
4. 测试覆盖正确性、空输入、非法输入和预分配输出；
5. benchmark 使用稳定、可复现的参数；
6. 不在导入阶段初始化 NPU、编译 kernel 或申请设备内存；
7. `ruff check .`、`ruff format --check .` 和相关 pytest 均通过。

算子优化需要同时报告：目标芯片、CANN/PyTorch NPU/TileLang 版本、输入参数、精度误差、优化前后时延及测试命令。

