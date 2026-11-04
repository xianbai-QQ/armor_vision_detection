# armor_vision_detect

传统视觉装甲板识别（C++17 + OpenCV 4）。当前是脚手架：能读图、能编译、能跑，检测流程还没写。

## 目录结构

```
armor_vision_detect/
├── CMakeLists.txt        # 构建配置（每一行都有注释，可以直接读）
├── include/              # 头文件：只放「接口声明」
│   └── armor_detector.h
├── src/                  # 源文件：放「具体实现」
│   ├── main.cpp
│   └── armor_detector.cpp
├── images/               # 测试图片
├── output/               # 程序保存 mask、候选图、结果图的位置
└── build/                # 编译产物，删掉重编不影响源码
```

## 编译

```bash
mkdir -p build
cd build
cmake ..
make
cd ..
```

或者一条命令（不进入 build 目录）：

```bash
cmake -S . -B build && cmake --build build
```

## 运行

```bash
./build/armor_vision_detect images/outpost_frame.png
```

不传参数会打印用法并退出；图片读不到会提示并退出。

## 各文件作用

| 文件 | 作用 |
| --- | --- |
| `src/main.cpp` | 读命令行参数、读图、串起 HSV 分割和形态学、保存结果 |
| `include/armor_detector.h` | 数据结构（`DetectorParams` / `LightBar` / `Armor`）和所有类的声明 |
| `src/hsv_processor.cpp` | `HsvProcessor`：BGR → HSV → `inRange` → mask |
| `src/morphology_processor.cpp` | `MorphologyProcessor`：腐蚀 / 膨胀 / 开 / 闭 |
| `src/utils.cpp` | `SaveImage`：建目录 + 存图 + 检查 `imwrite` 返回值 |
| `src/armor_detector.cpp` | 检测总流程，目前 `Detect()` 直接返回空列表 |

## 当前进度

- [x] 读图、检查读取失败、打印宽高和通道数
- [x] 单帧接口 `ArmorDetector::Detect(const cv::Mat&)` 定好了
- [x] 颜色分割出橙色 mask（`output/orange_mask.png`）
- [x] 形态学清理（`output/orange_cleaned.png`）
- [ ] 灯条候选提取（`findContours`）
- [ ] 灯条配对
- [ ] 四角点排序与结果绘制
