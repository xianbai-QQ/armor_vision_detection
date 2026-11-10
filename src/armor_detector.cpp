#include "armor_detector.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace armor_vision {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// 单位向量。长度为 0 时给一个默认方向，避免后面除以 0。
cv::Point2f UnitVector(const cv::Point2f& v)
{
    const float len = std::sqrt(v.x * v.x + v.y * v.y);
    if (len < 1e-6f)
    {
        return {1.0f, 0.0f};
    }
    return {v.x / len, v.y / len};
}

// 灯条长边的方向，统一指向图像下方（y 变大）。
// 只记「歪了多少度」不够：两根灯条一根往左歪、一根往右歪时，
// 角度大小可能一样，但方向并不平行。判断平行、判断和基线垂直都要用这个向量。
cv::Point2f BarDirection(const cv::RotatedRect& rect)
{
    cv::Point2f pts[4];
    rect.points(pts);

    // 四条边里最长的那条，方向就是灯条的长边方向
    cv::Point2f best_edge{0.0f, 0.0f};
    float best_len = 0.0f;
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

    if (best_edge.y < 0.0f)
    {
        best_edge = -best_edge;   // 统一朝下，两条灯条的方向才能直接比较
    }
    return UnitVector(best_edge);
}

// 灯条长边相对「竖直方向」的夹角（度）：完全竖直 = 0 度，越歪越大。
float TiltDegrees(const cv::RotatedRect& rect)
{
    const cv::Point2f direction = BarDirection(rect);
    return std::atan2(std::fabs(direction.x), direction.y) * 180.0f / kPi;
}

// 在 8 个顶点里，找沿 dir 方向投影最大的那个点。
// dir 不用归一化：比较的是投影大小的顺序，长度不影响谁最大。
cv::Point2f FarthestPoint(const std::array<cv::Point2f, 8>& points, const cv::Point2f& dir)
{
    cv::Point2f best = points[0];
    float best_val = best.dot(dir);
    for (std::size_t i = 1; i < points.size(); ++i)
    {
        const float val = points[i].dot(dir);
        if (val > best_val)
        {
            best_val = val;
            best = points[i];
        }
    }
    return best;
}

// 有向面积（鞋带公式）。图像坐标 y 向下时，结果 > 0 表示顶点是顺时针排的。
float SignedArea(const std::array<cv::Point2f, 4>& quad)
{
    float twice_area = 0.0f;
    for (int i = 0; i < 4; ++i)
    {
        const cv::Point2f& a = quad[i];
        const cv::Point2f& b = quad[(i + 1) % 4];
        twice_area += a.x * b.y - b.x * a.y;
    }
    return twice_area * 0.5f;
}

// 四角点必须构成一个凸四边形：相邻两条边的叉积符号要一致。
// 如果两根灯条方向反了、或者配对本身是错的，这里会算出「凹」甚至自交的四边形，
// 与其把错误结果画到图上，不如直接丢掉这个候选。
bool IsConvexQuad(const std::array<cv::Point2f, 4>& quad)
{
    float sign = 0.0f;
    for (int i = 0; i < 4; ++i)
    {
        const cv::Point2f e1 = quad[(i + 1) % 4] - quad[i];
        const cv::Point2f e2 = quad[(i + 2) % 4] - quad[(i + 1) % 4];
        const float cross = e1.x * e2.y - e1.y * e2.x;
        if (std::fabs(cross) < 1e-4f)
        {
            return false;   // 三点共线，四边形退化了
        }
        if (sign == 0.0f)
        {
            sign = cross;
        }
        else if ((cross > 0.0f) != (sign > 0.0f))
        {
            return false;
        }
    }
    return true;
}

// 由左右两根灯条的旋转矩形顶点，拼出装甲板的四个角点。
// 顺序固定为：左上 -> 右上 -> 右下 -> 左下。
std::array<cv::Point2f, 4> ComputeCorners(const LightBar& left, const LightBar& right)
{
    // 装甲板自己的两根轴：axis_x 指向右，axis_down 指向下（图像坐标 y 向下）
    const cv::Point2f axis_x = UnitVector(right.center - left.center);
    const cv::Point2f axis_down(-axis_x.y, axis_x.x);

    // 把两根灯条的 4+4 个顶点放在一起，从里面挑角的候选
    std::array<cv::Point2f, 8> points;
    left.rect.points(points.data());
    right.rect.points(points.data() + 4);

    // 四个角分别是整个点集在装甲板「左上 / 右上 / 右下 / 左下」方向上最远的点。
    // 用对角线方向取极值，装甲板往哪边倾斜都不会取到灯条内侧的顶点，
    // 这是「倾斜时四角点顺序不能乱」的关键。
    std::array<cv::Point2f, 4> quad;
    quad[0] = FarthestPoint(points, -axis_x - axis_down);   // 左上
    quad[1] = FarthestPoint(points, axis_x - axis_down);    // 右上
    quad[2] = FarthestPoint(points, axis_x + axis_down);    // 右下
    quad[3] = FarthestPoint(points, -axis_x + axis_down);   // 左下

    // 兜底：如果算出来的绕向是逆时针（负面积），交换右上和左下两个点修正。
    // 正常情况下这里不会触发，留着是为了让「0-1-2-3」的绕向永远一致。
    if (SignedArea(quad) < 0.0f)
    {
        std::swap(quad[1], quad[3]);
    }
    return quad;
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

    // 两两组合。j 从 i+1 开始，保证同一对只被检查一次
    for (std::size_t i = 0; i < bars.size(); ++i)
    {
        for (std::size_t j = i + 1; j < bars.size(); ++j)
        {
            // 按 x 坐标分出左右，后面的条件都基于「左灯条 + 右灯条」。
            // 同时记下它们在 bars 里的编号，方便结果图和终端输出核对配对是否正确。
            const bool a_is_left = bars[i].center.x < bars[j].center.x;
            const LightBar& left = a_is_left ? bars[i] : bars[j];
            const LightBar& right = a_is_left ? bars[j] : bars[i];
            const int left_index = a_is_left ? static_cast<int>(i) : static_cast<int>(j);
            const int right_index = a_is_left ? static_cast<int>(j) : static_cast<int>(i);

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

            // 条件 3：两根灯条要大致平行。
            // 用方向向量夹角的锐角来判断，一根往左歪、一根往右歪不会被误判成平行。
            const cv::Point2f left_direction = BarDirection(left.rect);
            const cv::Point2f right_direction = BarDirection(right.rect);
            const float cos_parallel =
                std::min(1.0f, std::fabs(left_direction.dot(right_direction)));
            const float parallel_angle = std::acos(cos_parallel) * 180.0f / kPi;
            if (parallel_angle > params.max_angle_diff)
            {
                continue;
            }

            const float avg_len = (left.length + right.length) * 0.5f;
            const float dx = right.center.x - left.center.x;   // 一定 > 0
            const float dy = right.center.y - left.center.y;

            // 条件 4：基线倾角不能大得离谱（非常粗略的一道保险）
            const float baseline_tilt = std::atan2(std::fabs(dy), dx) * 180.0f / kPi;
            if (baseline_tilt > params.max_baseline_tilt)
            {
                continue;
            }

            // 条件 4.1：基线方向和灯条方向要接近垂直（90°）。
            // 这是对「两根灯条上下偏差是否合理」的比例化表达：
            // 上下偏差大 → 基线歪 → 和灯条方向的夹角偏离 90°。
            // 两条灯条方向都朝下，直接取平均就是装甲板的「竖直方向」。
            const cv::Point2f bar_direction = UnitVector(left_direction + right_direction);
            const float cos_angle = std::clamp(
                UnitVector({dx, dy}).dot(bar_direction), -1.0f, 1.0f);
            const float baseline_bar_angle = std::acos(cos_angle) * 180.0f / kPi;
            if (std::fabs(baseline_bar_angle - 90.0f) > params.max_perp_error)
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

            // 条件 6：用两根灯条的顶点拼出四个角点，并检查它们是不是一个凸四边形
            Armor armor;
            armor.corners = ComputeCorners(left, right);
            if (!IsConvexQuad(armor.corners))
            {
                continue;
            }

            // 条件 6.1：0、3 号角点必须来自左灯条，1、2 号来自右灯条。
            // 这是对「左上 / 右上 / 右下 / 左下」语义的检查，防止倾斜时左右接反。
            const bool corners_belong_to_right_bars =
                cv::norm(armor.corners[0] - left.center) < cv::norm(armor.corners[0] - right.center) &&
                cv::norm(armor.corners[3] - left.center) < cv::norm(armor.corners[3] - right.center) &&
                cv::norm(armor.corners[1] - right.center) < cv::norm(armor.corners[1] - left.center) &&
                cv::norm(armor.corners[2] - right.center) < cv::norm(armor.corners[2] - left.center);
            if (!corners_belong_to_right_bars)
            {
                continue;
            }

            // 条件 7：整体比例检查。width 是上/下边的平均长度（两灯条间距），
            // height 是左/右边（灯条自身）的平均长度，比值应该像一块长条形的装甲板。
            armor.width = (cv::norm(armor.corners[1] - armor.corners[0]) +
                           cv::norm(armor.corners[2] - armor.corners[3])) * 0.5f;
            armor.height = (cv::norm(armor.corners[3] - armor.corners[0]) +
                            cv::norm(armor.corners[2] - armor.corners[1])) * 0.5f;
            if (armor.height < 1.0f)
            {
                continue;
            }

            const float aspect = armor.width / armor.height;
            if (aspect < params.min_armor_aspect || aspect > params.max_armor_aspect)
            {
                continue;
            }

            // 全部通过，组装一个装甲板候选
            armor.color = left.color;
            armor.left_bar = left_index;
            armor.right_bar = right_index;
            armor.center = (armor.corners[0] + armor.corners[1] +
                            armor.corners[2] + armor.corners[3]) * 0.25f;
            armor.valid = true;

            // 外接框：四个角点合起来求包围盒（轴对齐，画图和输出都用它）
            const std::vector<cv::Point2f> points(armor.corners.begin(), armor.corners.end());
            armor.bound_rect = cv::boundingRect(points);

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

DetectResult ArmorDetector::Detect(const cv::Mat& img) const
{
    DetectResult result;

    // 0. 输入检查：空图、单通道图、4 通道图都先处理掉，不让 OpenCV 在内部抛异常
    if (img.empty())
    {
        return result;
    }

    cv::Mat bgr;
    if (img.channels() == 4)
    {
        cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
    }
    else if (img.channels() == 3)
    {
        bgr = img;   // 只复制一个头，不复制像素
    }
    else
    {
        return result;
    }

    // 1. 颜色分割：BGR -> HSV -> inRange，得到目标颜色 mask
    HsvProcessor hsv(params_.target_range);
    hsv.SetBlurKernel(params_.blur_kernel);
    result.color_mask = hsv.MakeMask(bgr);
    if (result.color_mask.empty())
    {
        return result;
    }

    // 2. 形态学清理：先开运算去孤立噪点，再闭运算补灯条上的小断口
    MorphologyProcessor morph(params_.morph_kernel);
    const cv::Mat opened = morph.Open(result.color_mask);
    result.cleaned_mask = morph.Close(opened);
    if (result.cleaned_mask.empty())
    {
        return result;
    }

    // 3. 轮廓筛选出灯条候选
    result.light_bars = FindLightBars(result.cleaned_mask, params_, params_.target_color);

    // 4. 灯条两两配对，得到装甲板候选（含四角点）
    result.armors = MatchArmors(result.light_bars, params_);

    return result;
}

}  // namespace armor_vision
