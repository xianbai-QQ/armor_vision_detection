#include "armor_detector.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// 点坐标转字符串，统一保留一位小数，打印和写报告都用它
std::string PointToString(const cv::Point2f& point)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << "(" << point.x << ", " << point.y << ")";
    return oss.str();
}

cv::Point ToPixel(const cv::Point2f& point)
{
    return {static_cast<int>(std::lround(point.x)), static_cast<int>(std::lround(point.y))};
}

// 画在图片上的文字必须是 ASCII：cv::putText 不支持中文，
// 直接写 ColorName() 的「橙」会在图上变成一串问号。终端和 txt 报告里仍然用中文。
const char* ColorTag(armor_vision::ArmorColor color)
{
    switch (color)
    {
        case armor_vision::ArmorColor::Red:
            return "RED";
        case armor_vision::ArmorColor::Blue:
            return "BLUE";
        case armor_vision::ArmorColor::Orange:
            return "ORANGE";
        default:
            return "UNKNOWN";
    }
}

// 灯条候选图：黄色旋转矩形 + 蓝色中心点 + L0/L1 编号
void DrawLightBars(cv::Mat& canvas, const std::vector<armor_vision::LightBar>& bars)
{
    for (std::size_t i = 0; i < bars.size(); ++i)
    {
        cv::Point2f pts[4];
        bars[i].rect.points(pts);
        for (int k = 0; k < 4; ++k)
        {
            cv::line(canvas, pts[k], pts[(k + 1) % 4],
                     cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }

        const cv::Point center = ToPixel(bars[i].center);
        cv::circle(canvas, center, 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        cv::putText(canvas, "L" + std::to_string(i), center + cv::Point(6, -6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}

// 最终结果图：绿色四边形 + 黄色外接框 + 红色中心点 + 0~3 号角点
void DrawArmors(cv::Mat& canvas, const std::vector<armor_vision::Armor>& armors)
{
    // 四个角点各用一种颜色，方便在图上分辨 0 1 2 3 的顺序
    const cv::Scalar corner_colors[4] = {
        cv::Scalar(255, 128, 0),    // 0 左上：蓝
        cv::Scalar(0, 200, 0),      // 1 右上：绿
        cv::Scalar(0, 0, 255),      // 2 右下：红
        cv::Scalar(255, 0, 255),    // 3 左下：紫
    };

    for (std::size_t i = 0; i < armors.size(); ++i)
    {
        const armor_vision::Armor& armor = armors[i];

        // 1. 四个角点连成四边形（绿色粗线），一眼能看出配对的是哪两根灯条
        for (int k = 0; k < 4; ++k)
        {
            cv::line(canvas, armor.corners[k], armor.corners[(k + 1) % 4],
                     cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        // 2. 轴对齐外接框（黄色细线）
        cv::rectangle(canvas, armor.bound_rect, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);

        // 3. 中心点（红点）
        cv::circle(canvas, ToPixel(armor.center), 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

        // 4. 四角点 + 编号
        for (int k = 0; k < 4; ++k)
        {
            const cv::Point p = ToPixel(armor.corners[k]);
            cv::circle(canvas, p, 4, corner_colors[k], -1, cv::LINE_AA);
            cv::putText(canvas, std::to_string(k), p + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, corner_colors[k], 2, cv::LINE_AA);
        }

        // 5. 文字标签：编号 + 颜色 + 用的是哪两根灯条
        const std::string label = "A" + std::to_string(i) + " " +
                                  ColorTag(armor.color) +
                                  " (L" + std::to_string(armor.left_bar) +
                                  ",R" + std::to_string(armor.right_bar) + ")";
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        int text_x = armor.bound_rect.x;
        int text_y = armor.bound_rect.y - 10;
        if (text_y < text_size.height + 4)
        {
            // 贴到画面上边就改画在下面，免得出画
            text_y = armor.bound_rect.y + armor.bound_rect.height + text_size.height + 8;
        }
        text_x = std::max(0, std::min(text_x, canvas.cols - text_size.width - 2));
        cv::putText(canvas, label, cv::Point(text_x, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
}

// 把装甲板附近裁出来放大，方便肉眼看四角点顺序（原图里目标只有几十像素）
void SaveArmorZoom(const cv::Mat& result, const armor_vision::Armor& armor,
                   const std::string& path)
{
    // 左右留宽一点，让上方那行 "A0 ORANGE (L0,R1)" 标签也能完整裁进来
    const int margin_x = 90;
    const int margin_y = 50;
    cv::Rect roi(armor.bound_rect.x - margin_x,
                 armor.bound_rect.y - margin_y,
                 armor.bound_rect.width + margin_x * 2,
                 armor.bound_rect.height + margin_y * 2);
    roi &= cv::Rect(0, 0, result.cols, result.rows);   // 超出画面就裁掉
    if (roi.width < 2 || roi.height < 2)
    {
        return;
    }

    cv::Mat zoom;
    cv::resize(result(roi), zoom, cv::Size(), 4.0, 4.0, cv::INTER_LINEAR);
    armor_vision::SaveImage(zoom, path);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "用法: " << argv[0] << " images/outpost_frame.png\n";
        return -1;
    }

    // ---------------------------------------------------------------------
    //  读图 + 检查
    //  imread 读不到文件时不会抛异常，只会返回空的 Mat，所以必须自己检查。
    // ---------------------------------------------------------------------
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "读取图片失败：" << argv[1] << "\n";
        return -1;
    }

    // 终端和 output/report.txt 输出同一份内容，这里先把报告攒起来
    std::ostringstream report;
    report << "装甲板检测报告\n"
           << "图片: " << argv[1] << "\n"
           << "尺寸: " << img.cols << " x " << img.rows
           << "  通道数: " << img.channels() << "\n";

    // ---------------------------------------------------------------------
    //  单帧检测：一张图进去，装甲板列表 + 中间 mask 出来。
    //  想调阈值只改 DetectorParams 里的默认值（见 include/armor_detector.h）。
    // ---------------------------------------------------------------------
    const armor_vision::DetectorParams params;
    const armor_vision::ArmorDetector detector(params);

    const auto start = std::chrono::steady_clock::now();
    const armor_vision::DetectResult result = detector.Detect(img);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    if (result.color_mask.empty())
    {
        std::cerr << "检测失败：mask 生成失败，输入图可能不是彩色图\n";
        return -1;
    }

    // ---------------------------------------------------------------------
    //  保存中间结果
    // ---------------------------------------------------------------------
    bool saved = true;
    saved = armor_vision::SaveImage(result.color_mask, "output/orange_mask.png") && saved;
    saved = armor_vision::SaveImage(result.cleaned_mask, "output/orange_cleaned.png") && saved;

    // 把 mask 盖回原图，方便肉眼确认「选中的到底是哪些像素」
    cv::Mat overlay = img.clone();
    overlay.setTo(cv::Scalar(0, 0, 255), result.cleaned_mask);
    saved = armor_vision::SaveImage(overlay, "output/orange_overlay.png") && saved;

    // 灯条候选图
    cv::Mat candidates = img.clone();
    DrawLightBars(candidates, result.light_bars);
    saved = armor_vision::SaveImage(candidates, "output/candidates.png") && saved;

    // 最终结果图
    cv::Mat result_image = img.clone();
    DrawArmors(result_image, result.armors);
    saved = armor_vision::SaveImage(result_image, "output/result.png") && saved;

    if (!result.armors.empty())
    {
        SaveArmorZoom(result_image, result.armors.front(), "output/armor_zoom.png");
    }

    // ---------------------------------------------------------------------
    //  结果统计
    // ---------------------------------------------------------------------
    report << "颜色分割白色像素: " << cv::countNonZero(result.color_mask) << "\n"
           << "形态学清理后白色像素: " << cv::countNonZero(result.cleaned_mask) << "\n"
           << "灯条候选数: " << result.light_bars.size() << "\n";
    for (std::size_t i = 0; i < result.light_bars.size(); ++i)
    {
        const armor_vision::LightBar& bar = result.light_bars[i];
        report << "  灯条#" << i
               << "  中心=" << PointToString(bar.center)
               << "  长度=" << bar.length
               << "  宽度=" << bar.width
               << "  倾角=" << bar.angle << "\n";
    }

    report << "装甲板候选数: " << result.armors.size() << "\n";
    for (std::size_t i = 0; i < result.armors.size(); ++i)
    {
        const armor_vision::Armor& armor = result.armors[i];
        report << "  装甲板#" << i
               << "  颜色=" << armor_vision::ColorName(armor.color)
               << "  有效=" << (armor.valid ? "是" : "否") << "\n"
               << "    外接框=(" << armor.bound_rect.x << "," << armor.bound_rect.y
               << " " << armor.bound_rect.width << "x" << armor.bound_rect.height << ")"
               << "  中心=" << PointToString(armor.center) << "\n"
               << "    尺寸=" << armor.width << " x " << armor.height << "\n"
               << "    灯条=左#" << armor.left_bar << " 右#" << armor.right_bar << "\n"
               << "    装甲板数字=" << (armor.number < 0 ? "未识别" : std::to_string(armor.number))
               << "\n"
               << "    四角点（左上 -> 右上 -> 右下 -> 左下）:\n";
        for (int k = 0; k < 4; ++k)
        {
            report << "      " << k << ": " << PointToString(armor.corners[k]) << "\n";
        }
    }

    if (result.armors.empty())
    {
        report << "  没有检测到装甲板（这是正常返回，不是崩溃）\n";
    }

    report << "耗时: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";

    // 终端和文件输出同一份报告
    std::cout << report.str();

    std::ofstream file("output/report.txt");
    if (!file)
    {
        std::cerr << "保存文本报告失败：output/report.txt\n";
    }
    else
    {
        file << report.str();
        if (!file)
        {
            std::cerr << "写入文本报告失败：output/report.txt\n";
        }
        else
        {
            std::cout << "已保存：output/report.txt\n";
        }
    }

    return saved ? 0 : -1;
}
