#include "armor_detector.h"

#include <filesystem>
#include <iostream>

namespace armor_vision {

bool SaveImage(const cv::Mat& img, const std::string& path)
{
    // 空图没东西可存，先拦住，别让 imwrite 去处理一个空 Mat
    if (img.empty())
    {
        std::cerr << "保存失败：图片为空 -> " << path << "\n";
        return false;
    }

    // 1. 先确保父目录存在。
    //    create_directories 会一层层建（比如 output/red/mask.png 会把 output/red 一起建好），
    //    目录已存在时它什么也不做，所以可以放心每次调用。
    //    这里用 error_code 版本：失败不抛异常，而是把错误码填进 ec，由我们决定怎么处理。
    const std::filesystem::path file_path(path);
    const std::filesystem::path dir = file_path.parent_path();
    if (!dir.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec)
        {
            std::cerr << "创建目录失败：" << dir << " (" << ec.message() << ")\n";
            return false;
        }
    }

    // 2. imwrite 会返回 bool，必须检查：路径不可写、扩展名不认识、
    //    磁盘满，这几种情况都不会抛异常，只会返回 false。
    if (!cv::imwrite(path, img))
    {
        std::cerr << "保存失败：imwrite 返回 false -> " << path << "\n";
        return false;
    }

    std::cout << "已保存：" << path << "\n";
    return true;
}

const char* ColorName(ArmorColor color)
{
    switch (color)
    {
        case ArmorColor::Red:
            return "红";
        case ArmorColor::Blue:
            return "蓝";
        case ArmorColor::Orange:
            return "橙";
        default:
            return "未知";
    }
}

}  // namespace armor_vision
