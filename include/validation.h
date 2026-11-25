#ifndef VALIDATION_H
#define VALIDATION_H

#include "armor_detector.h"
#include "pose_solver.h"

#include <array>
#include <string>
#include <vector>

namespace armor_vision {

// 位姿验证用的选项：挑若干候选，做“粗略深度核对”和“单因素敏感性实验”。
struct ValidationOptions
{
    std::string video_path = "outpost.avi";
    std::string out_md = "output/pose/validation.md";

    CameraIntrinsics camera = CameraIntrinsics::FromTaskDoc();
    ArmorModel model;
    DetectorParams detector;

    // 挑选“接近正对相机”的候选：装甲板法线与光轴夹角余弦 >= 该值。
    // 0.85 约等于 31.8°，超过这个角度图像宽度被压缩得厉害，Z_rough 已经不适用。
    double frontal_cos_min = 0.85;
    // 挑选时要求重投影误差不超过该值，避免拿一个本身就解歪的样本做对比。
    double pick_max_reproj_px = 2.0;
    int want_samples = 3;

    // 单因素扰动量：两种扰动各自独立、对所有候选取同一数值。
    double corner_shift_px = 2.0;      // 把 0 号角点沿 +x 方向移动 2 像素
    double model_width_scale = 1.05;   // 模型宽度统一 +5%

    // “可疑结果”阈值（像素）：与视频流水线里用的是同一个数。
    double suspicious_reproj_px = 5.0;
};

struct ValidationStats
{
    bool ok = false;
    std::string error;
    int frames_scanned = 0;
    int candidates_seen = 0;
    int candidates_qualified = 0;
    int samples_used = 0;
};

// 扫描视频 -> 挑候选 -> Z_rough 与 tz 对比 -> 敏感性实验 -> 写 markdown 报告。
ValidationStats RunValidation(const ValidationOptions& options);

}  // namespace armor_vision

#endif
