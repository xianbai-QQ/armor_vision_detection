#include "armor_detector.h"

namespace armor_vision {

namespace {

// 只在本文件用的小工具：按边长生成方形结构元素。
// 放在匿名命名空间里，别的文件看不到它，也不会和别人的同名函数冲突。
cv::Mat MakeKernel(int size)
{
    return cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
}

}  // namespace

MorphologyProcessor::MorphologyProcessor(int kernel_size)
{
    SetKernelSize(kernel_size);
}

void MorphologyProcessor::SetKernelSize(int kernel_size)
{
    if (kernel_size < 1)
    {
        kernel_size = 3;
    }
    if (kernel_size % 2 == 0)
    {
        kernel_size += 1;  // 核大小统一用奇数，和滤波保持一致
    }
    kernel_size_ = kernel_size;
}

cv::Mat MorphologyProcessor::Erode(const cv::Mat& mask) const
{
    if (mask.empty())
    {
        return cv::Mat();
    }

    // 腐蚀：白色区域向内收缩一圈，细小的白点会直接消失，
    // 灯条也会变瘦，所以核别开太大。
    cv::Mat result;
    cv::erode(mask, result, MakeKernel(kernel_size_));
    return result;
}

cv::Mat MorphologyProcessor::Dilate(const cv::Mat& mask) const
{
    if (mask.empty())
    {
        return cv::Mat();
    }

    // 膨胀：白色区域向外扩张一圈，能补上灯条中间的断口，
    // 但也会让相邻的两个白块粘成一片。
    cv::Mat result;
    cv::dilate(mask, result, MakeKernel(kernel_size_));
    return result;
}

cv::Mat MorphologyProcessor::Open(const cv::Mat& mask) const
{
    if (mask.empty())
    {
        return cv::Mat();
    }

    // 开运算 = 先腐蚀后膨胀。整体大小基本不变，但孤立的小白点会被去掉。
    // 用来清理背景噪点。
    cv::Mat result;
    cv::morphologyEx(mask, result, cv::MORPH_OPEN, MakeKernel(kernel_size_));
    return result;
}

cv::Mat MorphologyProcessor::Close(const cv::Mat& mask) const
{
    if (mask.empty())
    {
        return cv::Mat();
    }

    // 闭运算 = 先膨胀后腐蚀。整体大小基本不变，但白块内部的小缺口会被填上。
    // 用来把断裂的灯条连起来。
    cv::Mat result;
    cv::morphologyEx(mask, result, cv::MORPH_CLOSE, MakeKernel(kernel_size_));
    return result;
}

}  // namespace armor_vision
