#include "validation.h"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace armor_vision {

namespace {

struct Sample
{
    int frame_id = -1;
    int candidate_id = -1;
    std::array<cv::Point2f, 4> corners{};
    PoseResult pose;
    double frontal_cos = 0.0;   // |R(2,2)|，越接近 1 越正对相机
};

std::string F(double value, int precision)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

// 扰动一次角点：只改 0 号点的 x，再重新解算。
PoseResult SolveWithShiftedCorner(const Sample& sample, const ValidationOptions& options,
                                  double shift_px)
{
    std::array<cv::Point2f, 4> corners = sample.corners;
    corners[0].x += static_cast<float>(shift_px);
    return SolveArmorPose(corners, options.camera, options.model);
}

// 扰动一次模型尺寸：宽度统一乘 scale，再重新解算。
PoseResult SolveWithScaledModel(const Sample& sample, const ValidationOptions& options,
                                double scale)
{
    ArmorModel model = options.model;
    model.width *= scale;
    return SolveArmorPose(sample.corners, options.camera, model);
}

}  // namespace

ValidationStats RunValidation(const ValidationOptions& options)
{
    ValidationStats stats;

    cv::VideoCapture cap(options.video_path);
    if (!cap.isOpened())
    {
        stats.error = "打开视频失败：" + options.video_path;
        std::cerr << stats.error << "\n";
        return stats;
    }

    const ArmorDetector detector(options.detector);
    std::vector<Sample> qualified;
    cv::Mat frame;
    int frame_id = 0;
    while (cap.read(frame) && !frame.empty())
    {
        const DetectResult detected = detector.Detect(frame);
        ++stats.frames_scanned;

        for (std::size_t i = 0; i < detected.armors.size(); ++i)
        {
            const Armor& armor = detected.armors[i];
            if (!armor.valid)
            {
                continue;
            }
            ++stats.candidates_seen;
            PoseResult pose = SolveArmorPose(armor.corners, options.camera, options.model);
            if (!pose.valid)
            {
                continue;
            }
            // 正对程度：模型 z 轴（装甲板法线）在相机坐标系中的 z 分量。
            // R 的第 3 列就是“模型 z 轴转到相机坐标系”的结果，取 z 分量即夹角余弦。
            const double normal_z = pose.rotation.at<double>(2, 2);
            if (std::fabs(normal_z) < options.frontal_cos_min ||
                pose.reprojection_error_px > options.pick_max_reproj_px ||
                pose.tvec[2] <= 0.0)
            {
                continue;
            }
            Sample sample;
            sample.frame_id = frame_id;
            sample.candidate_id = static_cast<int>(i);
            sample.corners = armor.corners;
            sample.pose = pose;
            sample.frontal_cos = std::fabs(normal_z);
            qualified.push_back(sample);
        }
        ++frame_id;
    }
    stats.candidates_qualified = static_cast<int>(qualified.size());

    if (qualified.empty())
    {
        stats.error = "没有找到满足条件的候选（正对 + 低重投影误差），无法做验证实验";
        std::cerr << stats.error << "\n";
        return stats;
    }

    // 从合格候选里均匀挑几个，尽量覆盖视频前/中/后段。
    std::vector<Sample> chosen;
    const int want = std::min<int>(options.want_samples, static_cast<int>(qualified.size()));
    for (int k = 0; k < want; ++k)
    {
        const std::size_t idx = static_cast<std::size_t>(
            (static_cast<double>(k) + 0.5) * qualified.size() / want);
        chosen.push_back(qualified[std::min(idx, qualified.size() - 1)]);
    }
    stats.samples_used = static_cast<int>(chosen.size());

    // ---------------- 写报告 ----------------
    std::filesystem::create_directories(
        std::filesystem::path(options.out_md).parent_path());
    std::ofstream md(options.out_md);
    if (!md)
    {
        stats.error = "无法写入：" + options.out_md;
        std::cerr << stats.error << "\n";
        return stats;
    }

    md << "# 位姿验证结果（自动生成）\n\n";
    md << "- 视频：`" << options.video_path << "`\n";
    md << "- 相机参数：" << options.camera.name << "\n";
    md << "- 装甲板模型：w = " << options.model.width << " m，h = " << options.model.height << " m\n";
    md << "- 扫描帧数：" << stats.frames_scanned
       << "，检出候选 " << stats.candidates_seen
       << " 个，其中“正对 + 重投影 ≤ " << options.pick_max_reproj_px
       << " px” 的 " << stats.candidates_qualified << " 个\n\n";

    // 1. Z_rough 与 tz 对比
    md << "## 一、粗略深度 Z_rough 与 PnP 深度对比\n\n";
    md << "公式：`Z_rough = fx * W / p`，其中 fx = "
       << F(options.camera.K.at<double>(0, 0), 3)
       << " px，W = " << options.model.width
       << " m，p 为上/下边长度的平均值。\n\n";
    md << "| frame_id | 像素宽 p (px) | 正对程度 |R(2,2)| | Z_rough (m) | tz_PnP (m) | 直线距离 (m) | 相对差异 |\n";
    md << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const Sample& s : chosen)
    {
        const double p = ArmorPixelWidth(s.corners);
        const double z_rough = RoughDepthFromWidth(s.corners, options.camera.K.at<double>(0, 0),
                                                   options.model.width);
        const double tz = s.pose.tvec[2];
        const double rel = (tz != 0.0) ? (z_rough - tz) / tz * 100.0 : 0.0;
        md << "| " << s.frame_id << " | " << F(p, 2) << " | " << F(s.frontal_cos, 3)
           << " | " << F(z_rough, 3) << " | " << F(tz, 3)
           << " | " << F(s.pose.distance_m, 3)
           << " | " << (rel >= 0 ? "+" : "") << F(rel, 2) << "% |\n";
    }

    // 2. 单因素敏感性
    md << "\n## 二、单因素敏感性实验\n\n";
    md << "基准、扰动后都用同一组相机参数和同一块装甲板模型，只改一个因素：\n\n";
    md << "- **扰动 A**：0 号角点沿 +x 方向移动 " << F(options.corner_shift_px, 1) << " 像素；\n";
    md << "- **扰动 B**：模型宽度统一乘以 " << F(options.model_width_scale, 3)
       << "（+5%），高度不变。\n\n";
    md << "| frame_id | 条件 | tz (m) | Δtz | 直线距离 (m) | Δ距离 | 重投影误差 (px) | Δ误差 |\n";
    md << "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n";

    struct Row
    {
        double tz = 0.0;
        double dist = 0.0;
        double err = 0.0;
    };
    for (const Sample& s : chosen)
    {
        const Row base{s.pose.tvec[2], s.pose.distance_m, s.pose.reprojection_error_px};
        const PoseResult shifted = SolveWithShiftedCorner(s, options, options.corner_shift_px);
        const PoseResult scaled = SolveWithScaledModel(s, options, options.model_width_scale);

        auto emit = [&md](int frame, const char* tag, const Row& r, const Row& b) {
            md << "| " << frame << " | " << tag << " | " << F(r.tz, 3) << " | "
               << (r.tz - b.tz >= 0 ? "+" : "") << F(r.tz - b.tz, 3) << " | "
               << F(r.dist, 3) << " | " << (r.dist - b.dist >= 0 ? "+" : "") << F(r.dist - b.dist, 3)
               << " | " << F(r.err, 3) << " | "
               << (r.err - b.err >= 0 ? "+" : "") << F(r.err - b.err, 3) << " |\n";
        };
        emit(s.frame_id, "基准", base, base);
        if (shifted.valid)
        {
            emit(s.frame_id, "角点 +2px", Row{shifted.tvec[2], shifted.distance_m,
                                             shifted.reprojection_error_px}, base);
        }
        else
        {
            md << "| " << s.frame_id << " | 角点 +2px | 解算失败 | | | | | |\n";
        }
        if (scaled.valid)
        {
            emit(s.frame_id, "模型宽 +5%", Row{scaled.tvec[2], scaled.distance_m,
                                              scaled.reprojection_error_px}, base);
        }
        else
        {
            md << "| " << s.frame_id << " | 模型宽 +5% | 解算失败 | | | | | |\n";
        }
    }

    // 3. 规则
    md << "\n## 三、结果可信度判断规则\n\n";
    md << "一条位姿被记为**可记录**，必须同时满足：\n\n";
    md << "1. `pnp_valid = 1`（solvePnP 返回 true，且 rvec/tvec/R 全为有限值）；\n";
    md << "2. `tz > 0`（装甲板在相机前方，否则一定是点序/单位/参数错了）；\n";
    md << "3. `reprojection_error_px <= " << F(options.suspicious_reproj_px, 1)
       << " px`（四角点平均像素误差，见下）。\n\n";
    md << "阈值取 5 px 的理由：本视频里灯条宽只有 5~7 像素，四角点由灯条顶点拼出，"
          "单根灯条端点定位误差约 1~2 像素，正常帧的误差都在 1 px 以内；"
          "取 5 px 相当于给“角点被咬掉一角”“配对错一根灯条”留 3 倍余量，"
          "同时又能拒绝那些重投影点明显错位的解。\n\n";
    md << "这条规则会**漏报**：角点整体平移但保持形状（例如两根灯条同时被向右多认 2 像素）时，"
          "PnP 会把平移吸收进位姿，重投影误差仍然很小，结果看起来“可用”却整体偏了一点。\n\n";
    md << "这条规则会**误报**：装甲板严重倾斜或部分被遮挡时，四角点本身定位就不准，"
          "误差可能超过 5 px，但这些帧的位姿未必真的错得更多；反过来，"
          "`tz > 0` 在极少数“装甲板背面正对相机”的情况下也可能为真。\n";

    md.close();
    std::cout << "验证报告: " << options.out_md << "\n"
              << "  扫描 " << stats.frames_scanned << " 帧，合格候选 "
              << stats.candidates_qualified << " 个，取样 " << stats.samples_used << " 个\n";
    stats.ok = true;
    return stats;
}

}  // namespace armor_vision
