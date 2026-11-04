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
    // ---- 灯条筛选 ----
    int min_light_area = 100;       // 清理后 mask 上的灯条面积约 106~156
    int max_light_area = 400;
    float min_light_aspect = 2.5f;  // 长边/短边，实测 29/8≈3.6、23/6≈3.8
    float max_light_aspect = 8.0f;

    // ---- 灯条配对 ----
    float max_length_ratio = 1.6f;     // 两灯条长度比上限，实测 29/23≈1.26
    float max_angle_diff = 15.0f;      // 两根灯条的倾角差上限（度）
    float max_center_dy_ratio = 0.6f;  // 上下偏差 / 平均长度，实测 8/26≈0.31
    float min_spacing_ratio = 1.2f;    // 水平间距 / 平均长度，实测 78/26≈3.0
    float max_spacing_ratio = 5.0f;
};

// 一根灯条的候选（第三步配对要用）
struct LightBar
{
    cv::RotatedRect rect;       // 旋转矩形，能同时给出中心、长宽、倾斜角
    cv::Point2f center;         // 中心点
    float length = 0.0f;        // 长边
    float width = 0.0f;         // 短边
    float angle = 0.0f;         // 倾斜角（度）
    ArmorColor color = ArmorColor::Unknown;   // 这根灯条属于哪种颜色
};

// 一块装甲板的结果
struct Armor
{
    std::array<cv::Point2f, 4> corners{};  // 顺序固定：左上 -> 右上 -> 右下 -> 左下
    cv::Rect bound_rect;                   // 外接框
    cv::Point2f center;                    // 中心点
    int number = -1;                       // 装甲板数字，还没识别就是 -1
    bool valid = false;                    // 这个结果是否有效
    ArmorColor color = ArmorColor::Unknown;
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
// 只有同时满足颜色一致、长度相近、倾角相近、上下对齐、间距合理的一对才会保留。
std::vector<Armor> MatchArmors(const std::vector<LightBar>& bars,
                               const DetectorParams& params);

class ArmorDetector
{
public:
    explicit ArmorDetector(DetectorParams params = {});

    // 输入一张彩色图，返回检测到的装甲板（可能 0 块、1 块、多块）
    std::vector<Armor> Detect(const cv::Mat& img) const;

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
