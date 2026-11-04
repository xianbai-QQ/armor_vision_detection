#include "armor_detector.h"

#include <algorithm>
#include <cmath>

namespace armor_vision {

namespace {

// 灯条长边相对「竖直方向」的夹角（度）：
// 完全竖直的灯条 = 0 度，越歪越大。后面配对时用它判断两根灯条是否平行。
float TiltDegrees(const cv::RotatedRect& rect)
{
    cv::Point2f pts[4];
    rect.points(pts);

    // 四条边里最长的那条，方向就是灯条的长边方向
    float best_len = 0.0f;
    cv::Point2f best_edge{0.0f, 0.0f};
    for (int i = 0; i < 4; ++i)
    {
        const cv::Point2f edge = pts[(i + 1) % 4] - pts[i];
        const float len = std::sqrt(edge.x * edge.x + edge.y * edge.y);
        if (len > best_len)
        {
            best_len = len;
            best_edge = edge;
        }
    }

    // atan2(|dx|, |dy|)：dy 占主导时角度接近 0（竖直）
    const float angle = std::atan2(std::fabs(best_edge.x), std::fabs(best_edge.y));
    return angle * 180.0f / static_cast<float>(CV_PI);
}

}  // namespace

std::vector<LightBar> FindLightBars(const cv::Mat& mask,
                                    const DetectorParams& params,
                                    ArmorColor color)
{
    std::vector<LightBar> bars;
    if (mask.empty())
    {
        return bars;
    }

    // 1. 在 mask 上找所有外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours)
    {
        // 2. 面积：太小的当噪点，太大的当背景结构
        const double area = cv::contourArea(contour);
        if (area < params.min_light_area || area > params.max_light_area)
        {
            continue;
        }

        // 3. 长宽比：灯条是细长的
        const cv::RotatedRect rect = cv::minAreaRect(contour);
        const float long_side = std::max(rect.size.width, rect.size.height);
        const float short_side = std::min(rect.size.width, rect.size.height);
        if (short_side < 1.0f)
        {
            continue;
        }

        const float aspect = long_side / short_side;
        if (aspect < params.min_light_aspect || aspect > params.max_light_aspect)
        {
            continue;
        }

        // 4. 方向：灯条在图像里是竖着的，所以外接框的高度必须大于宽度
        const cv::Rect bbox = cv::boundingRect(contour);
        if (bbox.height <= bbox.width)
        {
            continue;
        }

        LightBar bar;
        bar.rect = rect;
        bar.center = rect.center;
        bar.length = long_side;
        bar.width = short_side;
        bar.angle = TiltDegrees(rect);
        bar.color = color;
        bars.push_back(bar);
    }

    return bars;
}

std::vector<Armor> MatchArmors(const std::vector<LightBar>& bars,
                               const DetectorParams& params)
{
    std::vector<Armor> armors;

    // 两两组合。i 从 i+1 开始，保证同一对只被检查一次
    for (std::size_t i = 0; i < bars.size(); ++i)
    {
        for (std::size_t j = i + 1; j < bars.size(); ++j)
        {
            const LightBar& a = bars[i];
            const LightBar& b = bars[j];

            // 按 x 坐标分出左右，后面的条件都基于「左灯条 + 右灯条」
            const LightBar& left = (a.center.x < b.center.x) ? a : b;
            const LightBar& right = (a.center.x < b.center.x) ? b : a;

            // 条件 1：颜色必须一致
            if (left.color != right.color)
            {
                continue;
            }

            // 条件 2：长度要接近
            const float min_len = std::min(left.length, right.length);
            if (min_len < 1.0f)
            {
                continue;
            }
            if (std::max(left.length, right.length) / min_len > params.max_length_ratio)
            {
                continue;
            }

            // 条件 3：倾角要接近（两根灯条大致平行）
            if (std::fabs(left.angle - right.angle) > params.max_angle_diff)
            {
                continue;
            }

            const float avg_len = (left.length + right.length) * 0.5f;

            // 条件 4：上下不能错开太多
            if (std::fabs(left.center.y - right.center.y) / avg_len > params.max_center_dy_ratio)
            {
                continue;
            }

            // 条件 5：水平间距 / 灯条长度，要像一块装甲板
            const float spacing_ratio = (right.center.x - left.center.x) / avg_len;
            if (spacing_ratio < params.min_spacing_ratio ||
                spacing_ratio > params.max_spacing_ratio)
            {
                continue;
            }

            // 全部通过，组装一个装甲板候选
            Armor armor;
            armor.color = left.color;
            armor.center = (left.center + right.center) * 0.5f;
            armor.valid = true;

            // 外接框：把两根灯条的 8 个顶点合起来求包围盒
            std::vector<cv::Point2f> all_points;
            cv::Point2f pts[4];
            left.rect.points(pts);
            all_points.insert(all_points.end(), pts, pts + 4);
            right.rect.points(pts);
            all_points.insert(all_points.end(), pts, pts + 4);
            armor.bound_rect = cv::boundingRect(all_points);

            armors.push_back(armor);
        }
    }

    // 去重：如果两块候选的外接框重叠过半，说明是同一块装甲板被反复配对
    std::vector<Armor> unique;
    for (const Armor& armor : armors)
    {
        bool duplicated = false;
        for (const Armor& kept : unique)
        {
            const cv::Rect inter = armor.bound_rect & kept.bound_rect;
            const int smaller_area = std::min(armor.bound_rect.area(), kept.bound_rect.area());
            if (smaller_area > 0 && inter.area() > smaller_area * 0.5)
            {
                duplicated = true;
                break;
            }
        }
        if (!duplicated)
        {
            unique.push_back(armor);
        }
    }

    return unique;
}

ArmorDetector::ArmorDetector(DetectorParams params) : params_(params)
{
}

std::vector<Armor> ArmorDetector::Detect(const cv::Mat& img) const
{
    // 还没开始写检测流程，先返回空列表，保证程序能跑通
    (void)img;
    (void)params_;
    return {};
}

}  // namespace armor_vision
