# 自瞄项目架构设计

## Hsv类

- HsvRange
  1. H_min, max
  2. S_min, max
  3. V_min, max
- HsvProcessor
  - Public
    1. SetRange()
    2. SetBlurKernel()
    3. cv::Mat MakeMask()
  - Private
    1. range1, range2, bool use_two_range
    2. blur_kernel

## Morphology类

- Morphologyprocessor
  - public
    1. SetKernelSize()
    2. Erode, Dilate, Open, Close
  - private
    1. kernel_size

## 数据结构类

- color枚举类
    1. red, blue, orange, ...
- DetectorParams
    1. target_(range, color)
    2. (blur, morph)_kernel
    3. 灯条筛选
       1. area面积(max, min)
       2. aspect——l/w(max, min)
    4. 灯条配对
       1. length_ratio
       2. angle_diff
       3. spacing_ratio——d/w(max, min)
       4. baseline_diff(max)
       5. perp_error(max)
       6. armor_aspect(max, min)
    5. 灯条lightbar
       1. rect
       2. center
       3. length, width, angle
       4. color
    6. 装甲板Armor
       1. valid
       2. array<Point2f, 4> corners
       3. rect
       4. center
       5. height, width
       6. color
       7. bar_number(left, right)
    7. 输出结果Result
       1. aromors
       2. light_bars
       3. color_mask
       4. cleaned_mask

## 检测器

- FindLightBars
- MatchArmors
- ArmorDetector

## utilitis通用工具

- SaveImage
- ColorName
