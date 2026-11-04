#include "armor_detector.h"

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "用法: " << argv[0] << " images/outpost_frame.png\n";
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "读取图片失败：" << argv[1] << "\n";
        return -1;
    }

    std::cout << "图片: " << argv[1]
              << "  宽: " << img.cols
              << "  高: " << img.rows
              << "  通道数: " << img.channels() << "\n";

    // =====================================================================
    //  HSV 颜色分割：目标色 = 橙色
    //  HsvRange 六个数的顺序: h_min, s_min, v_min, h_max, s_max, v_max
    //  OpenCV 里 8 位图像的 H 范围是 0~180，S 和 V 是 0~255。
    //
    //  这两个数不是猜的：把图上「很亮且有颜色」的像素统计了一遍，
    //  发现画面中部那两根竖直灯条的 H 在 25~33 之间、S≈162、V≈239。
    //  所以范围取 H 8~40（两边留余量）、S≥100、V≥150。
    // =====================================================================
    armor_vision::HsvRange orange{8, 100, 150, 40, 255, 255};
    armor_vision::HsvProcessor orange_processor(orange);
    // orange_processor.SetBlurKernel(5);   // 噪点多的时候可以加大滤波核

    cv::Mat orange_mask = orange_processor.MakeMask(img);
    if (orange_mask.empty())
    {
        std::cerr << "生成 mask 失败：输入图可能不是 3 通道\n";
        return -1;
    }

    armor_vision::SaveImage(orange_mask, "output/orange_mask.png");

    // ---------------------------------------------------------------------
    //  形态学清理
    //  开运算 = 先腐蚀后膨胀：去掉孤立噪点，灯条整体大小基本不变
    //  闭运算 = 先膨胀后腐蚀：把灯条中间的断口补上
    //  顺序是先开后闭：先把噪点清掉，再补灯条的断口，避免把噪点也补成一团
    // ---------------------------------------------------------------------
    // 核大小实测（这张图）：
    //   3x3 → 还剩两个小噪点；5x5 → 刚好清干净，灯条完整；
    //   7x7 → 灯条被吃掉（只剩 55 像素）；9x9 以上 → 全黑。
    // 所以 5 是这张图的上限，再大就把目标也一起删了。
    armor_vision::MorphologyProcessor morph(3);

    cv::Mat opened_mask = morph.Open(orange_mask);
    cv::Mat cleaned_mask = morph.Close(opened_mask);

    if (cleaned_mask.empty())
    {
        std::cerr << "形态学处理失败：mask 为空\n";
        return -1;
    }

    armor_vision::SaveImage(cleaned_mask, "output/orange_cleaned.png");

    // 把 mask 盖回原图，方便肉眼确认「选中的到底是哪些像素」
    cv::Mat orange_overlay = img.clone();
    orange_overlay.setTo(cv::Scalar(0, 0, 255), cleaned_mask);  // 选中的像素涂纯红，醒目
    armor_vision::SaveImage(orange_overlay, "output/orange_overlay.png");

    std::cout << "橙色 mask 白色像素: " << cv::countNonZero(orange_mask) << "\n";
    std::cout << "形态学清理后白色像素: " << cv::countNonZero(cleaned_mask) << "\n";

    // ---------------------------------------------------------------------
    //  轮廓检测：从清理后的 mask 里找灯条候选
    //  面积、长宽比、竖直方向这三个筛选条件都在 FindLightBars 里，
    //  具体阈值集中在 DetectorParams，改参数只改头文件那一处。
    // ---------------------------------------------------------------------
    const armor_vision::DetectorParams params;
    const std::vector<armor_vision::LightBar> bars =
        armor_vision::FindLightBars(cleaned_mask, params, armor_vision::ArmorColor::Orange);

    std::cout << "灯条候选数: " << bars.size() << "\n";
    for (std::size_t i = 0; i < bars.size(); ++i)
    {
        std::cout << "  灯条#" << i
                  << "  中心=(" << bars[i].center.x << "," << bars[i].center.y << ")"
                  << "  长度=" << bars[i].length
                  << "  宽度=" << bars[i].width
                  << "  倾角=" << bars[i].angle << "\n";
    }

    // 灯条候选图：旋转矩形 + 中心点 + 编号
    cv::Mat candidates = img.clone();
    for (std::size_t i = 0; i < bars.size(); ++i)
    {
        cv::Point2f pts[4];
        bars[i].rect.points(pts);
        for (int k = 0; k < 4; ++k)
        {
            cv::line(candidates, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 255, 255), 2);
        }

        const cv::Point center(static_cast<int>(bars[i].center.x),
                               static_cast<int>(bars[i].center.y));
        cv::circle(candidates, center, 4, cv::Scalar(255, 0, 0), -1);
        cv::putText(candidates, "L" + std::to_string(i),
                    cv::Point(center.x + 6, center.y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
    armor_vision::SaveImage(candidates, "output/candidates.png");

    // ---------------------------------------------------------------------
    //  灯条配对：把上面的候选两两组合，用几何关系判断哪些像一块装甲板
    // ---------------------------------------------------------------------
    const std::vector<armor_vision::Armor> armors = armor_vision::MatchArmors(bars, params);

    std::cout << "装甲板候选数: " << armors.size() << "\n";
    for (std::size_t i = 0; i < armors.size(); ++i)
    {
        const armor_vision::Armor& armor = armors[i];
        std::cout << "  装甲板#" << i
                  << "  颜色=" << armor_vision::ColorName(armor.color)
                  << "  外接框=(" << armor.bound_rect.x << "," << armor.bound_rect.y
                  << " " << armor.bound_rect.width << "x" << armor.bound_rect.height << ")"
                  << "  中心=(" << armor.center.x << "," << armor.center.y << ")"
                  << "  有效=" << (armor.valid ? "是" : "否") << "\n";
    }
    if (armors.empty())
    {
        std::cout << "  没有配对出装甲板（这是正常返回，不是崩溃）\n";
    }

    // 最终结果图：外接框 + 中心点 + 编号
    cv::Mat result = img.clone();
    for (std::size_t i = 0; i < armors.size(); ++i)
    {
        cv::rectangle(result, armors[i].bound_rect, cv::Scalar(0, 255, 0), 2);

        const cv::Point center(static_cast<int>(armors[i].center.x),
                               static_cast<int>(armors[i].center.y));
        cv::circle(result, center, 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(result, "A" + std::to_string(i),
                    cv::Point(armors[i].bound_rect.x, armors[i].bound_rect.y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    armor_vision::SaveImage(result, "output/result.png");

    // TODO: 下一步从两根灯条的旋转矩形算出四个角点（左上→右上→右下→左下），
    //       并把这整条流程搬进 ArmorDetector::Detect(img) 里。
    return 0;
}
