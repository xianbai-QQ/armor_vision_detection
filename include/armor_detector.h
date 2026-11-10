#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

#include <opencv2/opencv.hpp>

#include <array>
#include <string>
#include <vector>

namespace armor_vision {

// ===========================================================================
//  一、颜色分割：BGR 图 -> 单通道 mask（目标颜色 = 255，其它 = 0）
// ===========================================================================

// 一段 HSV 范围。红色在 OpenCV 里色相会跨越 0/180 两端，所以常常要两段。
struct HsvRange
{
    int h_min = 0;
    int s_min = 0;
    int v_min = 0;
    int h_max = 180;
    int s_max = 255;
    int v_max = 255;
};

class HsvProcessor
{
public:
    HsvProcessor() = default;                       // 用默认范围（等于不筛选，用前必须 SetRange）
    explicit HsvProcessor(const HsvRange& range);   // 构造时直接指定一段范围

    void SetRange(const HsvRange& range);                            // 单段范围
    void SetRange(const HsvRange& range1, const HsvRange& range2);   // 两段范围，结果取并集
    void SetBlurKernel(int kernel_size);                             // 滤波核大小，0 = 不滤波

    // 输入 BGR 三通道图，输出 CV_8UC1 的 mask（255 = 目标颜色，0 = 其它）
    cv::Mat MakeMask(const cv::Mat& bgr) const;

private:
    HsvRange range1_;               // 第一段范围
    HsvRange range2_;               // 第二段范围（只有 use_two_ranges_ 为 true 时才用）
    bool use_two_ranges_ = false;
    int blur_kernel_ = 3;           // 必须是奇数（3、5、7…），0 表示不滤波
};

// ===========================================================================
//  二、形态学：对 mask 做腐蚀/膨胀/开/闭，只处理，不负责存图
// ===========================================================================

class MorphologyProcessor
{
public:
    MorphologyProcessor() = default;
    explicit MorphologyProcessor(int kernel_size);   // 指定核大小，常用 3 或 5

    void SetKernelSize(int kernel_size);             // 必须是奇数

    cv::Mat Erode(const cv::Mat& mask) const;        // 腐蚀：白区变瘦，去小噪点
    cv::Mat Dilate(const cv::Mat& mask) const;       // 膨胀：白区变胖，补断裂
    cv::Mat Open(const cv::Mat& mask) const;         // 开 = 先腐蚀后膨胀，去噪点
    cv::Mat Close(const cv::Mat& mask) const;        // 闭 = 先膨胀后腐蚀，补小洞

private:
    int kernel_size_ = 3;
};

// ===========================================================================
//  三、检测结果的数据结构
// ===========================================================================

// 颜色分类，输出结果时要带上
enum class ArmorColor
{
    Unknown,
    Red,
    Blue,
    Orange
};

// 可调参数全部放这里，以后调阈值只改这一处。
// 下面的默认值不是猜的，是按 images/outpost_frame.png 实测出来的尺寸定的。
struct DetectorParams
{
    // ---- 颜色分割 ----
    // 目标灯条是橙色的。H 8~40 是实测 25~33 两边留余量，S、V 下限用来排除
    // 画面下方那两块「很亮但几乎没颜色」的白色灯板。
    HsvRange target_range{8, 100, 150, 40, 255, 255};
    ArmorColor target_color = ArmorColor::Orange;

    int blur_kernel = 3;      // 中值滤波核，0 = 不滤波
    int morph_kernel = 3;     // 形态学核大小，越大清理越狠，也越容易把灯条吃掉

    // ---- 灯条筛选 ----
    int min_light_area = 100;       // 清理后 mask 上的灯条面积实测 120~159
    int max_light_area = 400;
    float min_light_aspect = 2.5f;  // 长边/短边，实测 30/7≈4.3、27/5≈5.4
    float max_light_aspect = 8.0f;

    // ---- 灯条配对 ----
    float max_length_ratio = 1.6f;     // 两灯条长度比上限，实测 30/27≈1.11
    float max_angle_diff = 15.0f;      // 两根灯条方向向量夹角上限（度）
    float min_spacing_ratio = 1.2f;    // 水平间距 / 平均长度，实测 77/28.5≈2.7
    float max_spacing_ratio = 5.0f;

    // 「两根灯条中心的上下偏差」不用 dy/灯条长度 来判断，而是看两个角度：
    //   max_baseline_tilt  左右灯条中心连线（基线）最多允许倾斜多少度；
    //   max_perp_error     基线方向和灯条方向必须接近垂直，允许的偏差角度。
    // 原来的 dy/长度 在整块装甲板倾斜时会被放大（同一块装甲板旋转 20° 就判不出来），
    // 换成角度以后只和几何形状有关，和图像尺寸、装甲板是否倾斜都无关。
    // 主图实测：基线倾角 6.3°，基线与灯条方向夹角 96.3°（偏差 6.3°）。
    float max_baseline_tilt = 40.0f;
    float max_perp_error = 20.0f;

    // ---- 装甲板整体比例 ----
    // 四角点算出来的「宽 / 高」。实测 83/28.5≈2.9，太扁（两根灯条离太远）
    // 或太方（不是一块装甲板）的组合在这里被拒掉。
    float min_armor_aspect = 1.2f;
    float max_armor_aspect = 6.0f;
};

// 一根灯条的候选（第三步配对要用）
struct LightBar
{
    cv::RotatedRect rect;       // 旋转矩形，能同时给出中心、长宽、倾斜角
    cv::Point2f center;         // 中心点
    float length = 0.0f;        // 长边
    float width = 0.0f;         // 短边
    float angle = 0.0f;         // 倾斜角（度）：0 = 完全竖直
    ArmorColor color = ArmorColor::Unknown;   // 这根灯条属于哪种颜色
};

// 一块装甲板的结果
struct Armor
{
    // 四角点顺序固定：左上 -> 右上 -> 右下 -> 左下（图像坐标，y 向下，是顺时针绕向）。
    // 这个顺序是下一阶段 solvePnP 的输入约定，不能随便换。
    std::array<cv::Point2f, 4> corners{};
    cv::Rect bound_rect;                   // 外接框
    cv::Point2f center;                    // 中心点（四个角点的平均）
    float width = 0.0f;                    // 上/下两条边的平均长度
    float height = 0.0f;                   // 左/右两条边的平均长度
    int number = -1;                       // 装甲板数字，还没识别就是 -1
    bool valid = false;                    // 这个结果是否有效
    ArmorColor color = ArmorColor::Unknown;
    int left_bar = -1;                     // 组成它的左灯条在 LightBar 列表里的编号
    int right_bar = -1;                    // 右灯条编号
};

// 单帧检测的完整输出：装甲板 + 中间结果（方便存 mask、画候选图、排错）。
// 下一阶段接视频时，每帧调用一次 Detect，只取 armors 用即可。
struct DetectResult
{
    std::vector<Armor> armors;          // 最终装甲板候选，可能为空
    std::vector<LightBar> light_bars;   // 通过筛选的灯条候选
    cv::Mat color_mask;                 // 颜色分割后的原始 mask
    cv::Mat cleaned_mask;               // 形态学清理后的 mask
};

// ===========================================================================
//  四、检测器：把上面几步串起来
// ===========================================================================

// 从清理后的 mask 里找灯条候选。
// 返回的每个 LightBar 都已经过了面积、长宽比、竖直方向的筛选。
std::vector<LightBar> FindLightBars(const cv::Mat& mask,
                                    const DetectorParams& params,
                                    ArmorColor color = ArmorColor::Unknown);

// 把灯条两两配对成装甲板候选。
// 只有同时满足颜色一致、长度相近、倾角相近、上下对齐、间距合理、
// 整体宽高比合理的组合才会保留；同一块装甲板被重复配对时只留一个。
std::vector<Armor> MatchArmors(const std::vector<LightBar>& bars,
                               const DetectorParams& params);

class ArmorDetector
{
public:
    explicit ArmorDetector(DetectorParams params = {});

    // 输入一张彩色图，返回全部检测结果，包括中间 mask。
    // 传空图、非 3/4 通道图时返回空结果，不会抛异常。
    DetectResult Detect(const cv::Mat& img) const;

private:
    DetectorParams params_;
};

// ===========================================================================
//  五、通用工具
// ===========================================================================

// 保存图片，路径不存在会先创建目录；保存失败返回 false
bool SaveImage(const cv::Mat& img, const std::string& path);

// 颜色的中文名，打印结果时用
const char* ColorName(ArmorColor color);

}  // namespace armor_vision

#endif
