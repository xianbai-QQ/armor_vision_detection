# ---------------------------------------------------------------------------
# 1. 最低 CMake 版本。Ubuntu 22.04 自带 3.22，写 3.10 是为了兼容别人的机器。
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)

# 2. 项目名。构建系统、IDE 里都显示这个名字。
project(armor_vision_detect)

# 3. 用 C++17。REQUIRED ON 表示编译器不支持就直接报错，
#    而不是悄悄降级到老标准，然后在别处报出莫名其妙的错误。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. 找 OpenCV。找到后 CMake 会填好两个变量：
#    ${OpenCV_INCLUDE_DIRS} —— 头文件在哪个目录（/usr/include/opencv4）
#    ${OpenCV_LIBS}         —— 要链接哪些库
#    REQUIRED 表示找不到就停下并报错，不要继续。
find_package(OpenCV REQUIRED)

# 5. 生成一个叫 armor_vision_detect 的可执行文件，源码文件要列全。
#    以后新增 .cpp 要记得加到这里，否则链接时会报「找不到符号」。
add_executable(armor_vision_detect
    src/main.cpp
    src/armor_detector.cpp
    src/hsv_processor.cpp
    src/morphology_processor.cpp
    src/utils.cpp
)

# 6. 头文件搜索路径（等价于编译时的 -I 参数）。
#    ${CMAKE_CURRENT_SOURCE_DIR} 就是本 CMakeLists.txt 所在目录，
#    所以第一行等于把项目下的 include/ 目录加进搜索路径，
#    于是代码里写 #include "armor_detector.h" 就能找到。
#    PRIVATE 表示这条路径只给自己用，不会传播给别人（现在只有一个目标，无所谓）。
target_include_directories(armor_vision_detect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 7. 链接 OpenCV 的库。编译只是「语法检查」，链接这一步才真正把
#    imread / inRange / findContours 这些函数的实现拼进来。
target_link_libraries(armor_vision_detect PRIVATE ${OpenCV_LIBS})

# 8. 编译告警开关。让编译器把可疑写法提示出来：
#    -Wall   常用告警（未使用变量、可疑比较……）
#    -Wextra 更多告警（未使用的函数参数……）
#    不想要就删掉这行，不影响编译。
target_compile_options(armor_vision_detect PRIVATE -Wall -Wextra)
