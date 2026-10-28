# 1 软件安装

| 软件名称     | 下载链接（官网）                                                                                                              | 安装教程                                                    |
|--------------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| CloudCompare | https://www.cloudcompare.org/release/index.html （选择 Stable 版本）                                                          | https://blog.csdn.net/m0_59079614/article/details/130422026 |
| Anaconda     | 官网：https://www.anaconda.com/download/success?reg=skipped<br/>镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/  | https://blog.csdn.net/Natsuago/article/details/143081283    |
| PyCharm      | https://www.jetbrains.com/zh-cn/pycharm/download/?section=windows                                                           | 软件安装：https://www.jetbrains.com/zh-cn/help/pycharm/installation-guide.html#silent<br/>Student Pack认证：https://www.jetbrains.com/zh-cn/academy/student-pack/  |
| LeWoS        | https://github.com/dwang520/LeWoS-py                                                                                                                              |https://github.com/dwang520/LeWoS-py                                                             |

注：已有 Anaconda/Miniconda 的同学，不用重新安装 Anaconda；已有 VSCode 等 IDE 的同学，不用额外安装 PyCharm。

# 2 Conda 环境配置

```shell
# 打开 cmd 终端或 Powershell，检查 conda 版本
conda --version

# 添加镜像源（已经配置过 conda 的跳过此步） 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

# 显示镜像链接
conda config --set show_channel_urls yes
conda config --show channels

# 通过 conda 创建新的虚拟环境，参数文件是同目录下的 env.yml
conda env create --file env.yml

# 激活环境
conda activate lidar

# 安装 ipykernel 和 ipywidgets，用于 Jupyter
pip install ipykernel
pip install ipywidgets

# 安装 LeWoS，提前将 LeWoS 仓库复制到项目根目录下
pip install .\LeWoS-py-main\
```

# 3 下载数据

数据链接（北大网盘）：https://disk.pku.edu.cn/link/AA5B0008D43A634BEEA73C12852CDF67E3

- **BLK 数据（裁剪后）**：BLK_small.laz
- **Faro 数据（裁剪后）**：Faro_DAG_clip.las
- **BLK、Faro 和手持扫描的完整数据** 在 Complete data 文件夹下，有需要的同学可以下载

课程演示只用到裁剪后的数据（BLK_small.laz、Faro_DAG_clip.las）。将数据下载后，放入项目根目录下的 Data 文件夹中即可。