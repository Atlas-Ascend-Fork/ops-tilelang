# 安全声明

本文说明 `ops-tilelang` 在安装、开发、测试和运行过程中的安全注意事项。硬件和软件环境要求见 [README.md](README.md#环境要求)。

## 运行用户建议

使用普通用户完成日常构建、测试和算子调用，仅授予其访问所需 NPU 设备、输入数据和工作目录的权限。除驱动或系统组件安装等必要操作外，避免使用 `root` 或其他管理员账户运行项目脚本。

## 文件权限控制

- 在宿主机和容器中使用 `umask 0027` 或更严格的权限掩码；通常新建目录权限不超过 `750`，普通文件权限不超过 `640`。已有文件的权限需要单独检查。
- 对源码、模型输入、业务数据、构建产物和调试记录分别设置访问权限，避免无关用户读取敏感内容或修改运行所依赖的文件。建议值见[文件权限附录](#file-permissions)。
- TileLang 会在即时编译过程中生成并缓存内核代码和二进制文件。缓存位置由所安装的 TileLang 版本及配置决定，常见默认位置为 `~/.tilelang/cache`，可通过 `TILELANG_CACHE_DIR` 指定。缓存目录应由运行用户控制，不与不可信用户共用可写目录。
- 测试可能生成性能基线、NPU 显存画像、临时文件和日志。通过 `--benchmark-output`、`--npu-mem-output` 指定输出位置时，应使用可信目录，并管理输出文件的读取权限和保存期限。
- 启用 `OPS_TILELANG_PRINT_KERNEL_SOURCE=1` 会打印生成的内核源码。共享日志或上传测试结果前，应检查并移除业务数据、敏感路径及其他不宜公开的信息。

## 构建安全声明

从可信来源获取本项目及其依赖，并在受控的 Python 环境中安装。依赖清单见 [pyproject.toml](pyproject.toml) 和 [requirements.txt](requirements.txt)；PyTorch、`torch_npu`、支持 Ascend 的 TileLang、CANN Toolkit 和驱动需要按照目标环境配套使用。

构建过程和 TileLang 即时编译会执行构建工具并生成文件，应仅编译可信源码。对 `build/`、`dist/`、临时目录及编译缓存设置必要的访问权限；分发 wheel 前检查其中的内容，避免包含测试数据、凭据或调试产物。

## 运行安全声明

- 根据主机内存、NPU 显存和磁盘容量选择输入规模与测试并发。参考实现、性能测试和大规模参数组合也会消耗资源；资源不足可能导致执行失败或进程退出。
- 调用算子前核对设备、数据类型、形状、步长及输出缓冲区要求。对来自外部的输入先进行边界检查，并遵循各算子接口说明。
- 遇到运行异常时，结合错误信息、运行日志及依赖版本定位原因。需要排查异步执行错误时，按所用运行时的调试说明进行同步定位。
- PyTorch NPU 的安装和使用还应遵循其[安全声明](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。
- CI 中的访问密钥应由平台的 secrets 功能注入，不应写入源码、配置文件或日志。运行外部贡献的代码前，应使用具备适当隔离和凭据访问控制的执行环境。

## 公网地址声明

以下列出仓库配置及本文引用的主要公网地址。项目元数据中的地址按当前 `pyproject.toml` 记录；CI 变量、包索引和代理配置可能引入额外地址，部署时应一并核查。

| 类型 | 文件名 | 公网地址或域名 | 用途 |
| --- | --- | --- | --- |
| 项目元数据 | `pyproject.toml` | `https://gitcode.com/cann/ops-tilelang`、`https://gitcode.com/cann/ops-tilelang/issues` | 声明项目主页、代码仓库和问题反馈入口 |
| 开发依赖 | `.pre-commit-config.yaml` | `https://gitcode.com/gh_mirrors/ru/ruff-pre-commit` | 获取 Ruff 的 pre-commit hooks |
| CI 工作流 | `.gitcode/workflows/ops-tilelang_actions.yml`、`.gitcode/workflows/codecheck_action.yml` | `gitcode.com`，仓库路径 `cann/.gitcode` | 引用社区 CI actions 和可复用工作流 |
| CI 存储 | `.gitcode/workflows/ops-tilelang_actions.yml`、`.gitcode/workflows/codecheck_action.yml` | `https://obs.cn-north-4.myhuaweicloud.com` | 上传和下载 CI 文件及检查结果 |
| CI 镜像 | `.gitcode/workflows/ops-tilelang_actions.yml`、`.gitcode/workflows/codecheck_action.yml` | `swr.cn-north-4.myhuaweicloud.com` | 获取 CI 容器镜像 |
| 安全参考 | `SECURITY.md` | `https://gitee.com/ascend/pytorch` | 查阅 PyTorch NPU 安全说明 |
| 漏洞管理 | `SECURITY.md` | `https://gitcode.com/cann/community/blob/master/security/security.md` | 查阅 CANN 社区漏洞报告与处理流程 |

使用 pip 安装依赖时，下载地址取决于用户或 CI 配置的包索引；使用默认 PyPI 时，通常会访问 `pypi.org` 和 `files.pythonhosted.org`。仓库没有在依赖文件中固定包索引地址。

## 漏洞机制说明

发现疑似安全漏洞时，请按照 [CANN 社区漏洞管理流程](https://gitcode.com/cann/community/blob/master/security/security.md)提供的渠道报告，并遵循其披露要求。在问题完成评估和协调披露前，避免通过公开 issue、PR 或日志暴露漏洞利用细节。

报告中建议包含受影响的提交或版本、硬件及依赖版本、触发条件、最小复现步骤、影响范围和联系方式。提交复现数据时请移除真实业务数据及凭据。

## 附录

<a id="file-permissions"></a>

### 文件（夹）各场景权限建议

下表适用于 Linux 环境。根据业务需要进一步收紧权限；构建、缓存及日志目录仍需保留运行用户必要的写权限。

| 文件或目录用途 | 建议权限上限 |
| --- | --- |
| 用户主目录 | `750`（`rwxr-x---`） |
| 已部署的程序、脚本和库文件 | `550`（`r-xr-x---`） |
| 已部署的程序目录 | `550`（`r-xr-x---`） |
| 配置文件 | `640`（`rw-r-----`） |
| 配置目录 | `750`（`rwxr-x---`） |
| 已归档日志 | `440`（`r--r-----`） |
| 正在写入的日志 | `640`（`rw-r-----`） |
| 日志目录 | `750`（`rwxr-x---`） |
| 调试文件 | `640`（`rw-r-----`） |
| 调试、临时和编译缓存目录 | `750`（`rwxr-x---`） |
| 需要组内协作写入的维护目录 | `770`（`rwxrwx---`） |
| 业务数据文件 | `640`（`rw-r-----`） |
| 业务数据目录 | `750`（`rwxr-x---`） |
| 私钥、凭据和密文所在目录 | `700`（`rwx------`） |
| 私钥、凭据和密文文件 | `600`（`rw-------`） |
| 仅所有者执行的加解密脚本 | `500`（`r-x------`） |
