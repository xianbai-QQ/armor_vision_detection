#include "armor_detector.h"

namespace armor_vision {

HsvProcessor::HsvProcessor(const HsvRange& range) : range1_(range)
{
}

void HsvProcessor::SetRange(const HsvRange& range)
{
    range1_ = range;
    use_two_ranges_ = false;
}

void HsvProcessor::SetRange(const HsvRange& range1, const HsvRange& range2)
{
    range1_ = range1;
    range2_ = range2;
    use_two_ranges_ = true;
}

void HsvProcessor::SetBlurKernel(int kernel_size)
{
    if (kernel_size < 0)
    {
        kernel_size = 0;
    }
    if (kernel_size > 0 && kernel_size % 2 == 0)
    {
        kernel_size += 1;  // OpenCV 的滤波核大小必须是奇数，偶数会直接抛异常
    }
    blur_kernel_ = kernel_size;
}

cv::Mat HsvProcessor::MakeMask(const cv::Mat& bgr) const
{
    // 输入不对就别往下走，否则 OpenCV 会在函数内部直接抛异常
    if (bgr.empty() || bgr.channels() != 3)
    {
        return cv::Mat();
    }

    // 1. 滤波。medianBlur（中值滤波）对付灯条边上的零星噪点最有效，
    //    而且不会像高斯模糊那样把灯条边缘糊开。
    cv::Mat src;
    if (blur_kernel_ >= 3)
    {
        // 输出到一个「空的」Mat，OpenCV 才会新分配内存。
        // 如果写成 cv::Mat src = bgr; 再传进 medianBlur，
        // src 和 bgr 共用同一块内存，等于把调用者的原图改了。
        cv::medianBlur(bgr, src, blur_kernel_);
    }
    else
    {
        src = bgr;  // 不滤波：只是多一个指针指向同一张图，不复制像素
    }

    // 2. BGR -> HSV。后面所有颜色判断都在 HSV 上做，
    //    因为 H（色相）不受亮度影响，比 BGR 三通道分别设阈值稳得多。
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // 3. 第一段范围。inRange 会输出单通道 CV_8UC1：
    //    范围内 = 255，范围外 = 0，正是后面 findContours 要的前景。
    cv::Mat mask1;
    cv::inRange(hsv,
                cv::Scalar(range1_.h_min, range1_.s_min, range1_.v_min),
                cv::Scalar(range1_.h_max, range1_.s_max, range1_.v_max),
                mask1);

    if (!use_two_ranges_)
    {
        return mask1;
    }

    // 4. 第二段范围。红色在 OpenCV 的 H 里是 0~10 和 170~180 两段，
    //    分别 inRange 之后按位或，把两段合成一张 mask。
    cv::Mat mask2;
    cv::inRange(hsv,
                cv::Scalar(range2_.h_min, range2_.s_min, range2_.v_min),
                cv::Scalar(range2_.h_max, range2_.s_max, range2_.v_max),
                mask2);

    cv::Mat mask;
    cv::bitwise_or(mask1, mask2, mask);
    return mask;
}

}  // namespace armor_vision
