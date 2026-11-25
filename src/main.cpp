#include "armor_detector.h"
#include "pose_solver.h"
#include "validation.h"
#include "video_pipeline.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ===========================================================================
//  命令行
// ===========================================================================

struct CommandLine
{
    enum class Mode
    {
        Image,      // 第一阶段：静态图片回归入口
        Video,      // 第二阶段：逐帧检测 + 位姿解算 + 结果视频
        Validate,   // 第二阶段：Z_rough 核对 + 单因素敏感性分析
        Help
    };

    Mode mode = Mode::Image;
    std::string input = "images/outpost_frame.png";
    bool mode_explicit = false;

    // 视频模式的输出路径，留空则用默认值
    std::string out_video;
    std::string csv;
    std::string snapshot_dir;
    std::string validation_md;

    std::string calib = "doc";   // doc | params
    std::string detector = "video";  // video | default
    int max_frames = 0;
    std::string codec;           // 覆盖默认编码候选
};

void PrintUsage(const char* program)
{
    std::cout
        << "用法:\n"
        << "  " << program << " <图片路径>                  # 第一阶段静态图检测（默认）\n"
        << "  " << program << " --image <图片路径>          # 同上\n"
        << "  " << program << " --video <视频路径>          # 逐帧检测 + 位姿解算 + 结果视频\n"
        << "  " << program << " --validate <视频路径>       # 粗略深度核对 + 敏感性分析\n"
        << "\n"
        << "通用选项:\n"
        << "  --calib doc|params  相机参数来源（默认 doc = 姿态解算.md 表格）\n"
        << "  --detector video|default  视频用哪套检测阈值（默认 video = 暗光放宽版）\n"
        << "  --out <路径>        结果视频路径（默认 output/pose/outpost_pose.avi）\n"
        << "  --csv <路径>        逐帧数据路径（默认 output/pose/frames.csv）\n"
        << "  --snapshots <目录>  验证图目录（默认 output/pose/validation）\n"
        << "  --md <路径>         验证报告路径（默认 output/pose/validation.md）\n"
        << "  --codec <FOURCC>    结果视频编码（默认依次尝试 H264/XVID/MJPG）\n"
        << "  --max-frames N      只处理前 N 帧（0 = 全部，用于冒烟测试）\n"
        << "\n"
        << "示例:\n"
        << "  " << program << " images/outpost_frame.png\n"
        << "  " << program << " --video outpost.avi\n"
        << "  " << program << " --video outpost.avi --calib params --max-frames 300\n";
}

// 判断字符串是不是“像视频”的扩展名，用于单参数时的模式推断。
bool LooksLikeVideo(const std::string& path)
{
    const std::size_t pos = path.find_last_of('.');
    if (pos == std::string::npos)
    {
        return false;
    }
    std::string ext = path.substr(pos);
    for (char& c : ext)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".avi" || ext == ".mp4" || ext == ".mkv" || ext == ".mov";
}

bool ParseArgs(int argc, char** argv, CommandLine* cli)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&](std::string* out) {
            if (i + 1 >= argc)
            {
                std::cerr << "选项 " << arg << " 缺少参数\n";
                return false;
            }
            *out = argv[++i];
            return true;
        };

        if (arg == "--help" || arg == "-h")
        {
            cli->mode = CommandLine::Mode::Help;
            return true;
        }
        else if (arg == "--image")
        {
            cli->mode = CommandLine::Mode::Image;
            cli->mode_explicit = true;
            if (!next(&cli->input)) return false;
        }
        else if (arg == "--video")
        {
            cli->mode = CommandLine::Mode::Video;
            cli->mode_explicit = true;
            if (!next(&cli->input)) return false;
        }
        else if (arg == "--validate")
        {
            cli->mode = CommandLine::Mode::Validate;
            cli->mode_explicit = true;
            if (!next(&cli->input)) return false;
        }
        else if (arg == "--calib")
        {
            if (!next(&cli->calib)) return false;
        }
        else if (arg == "--detector")
        {
            if (!next(&cli->detector)) return false;
        }
        else if (arg == "--out")
        {
            if (!next(&cli->out_video)) return false;
        }
        else if (arg == "--csv")
        {
            if (!next(&cli->csv)) return false;
        }
        else if (arg == "--snapshots")
        {
            if (!next(&cli->snapshot_dir)) return false;
        }
        else if (arg == "--md")
        {
            if (!next(&cli->validation_md)) return false;
        }
        else if (arg == "--codec")
        {
            if (!next(&cli->codec)) return false;
        }
        else if (arg == "--max-frames")
        {
            std::string value;
            if (!next(&value)) return false;
            cli->max_frames = std::max(0, std::atoi(value.c_str()));
        }
        else if (!arg.empty() && arg[0] == '-')
        {
            std::cerr << "未知选项：" << arg << "\n";
            return false;
        }
        else
        {
            cli->input = arg;
            if (!cli->mode_explicit)
            {
                cli->mode = LooksLikeVideo(arg) ? CommandLine::Mode::Video
                                                : CommandLine::Mode::Image;
            }
        }
    }
    return true;
}

// ===========================================================================
//  第一阶段：静态图片
// ===========================================================================

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

int RunImageMode(const std::string& path)
{
    // ---------------------------------------------------------------------
    //  读图 + 检查
    //  imread 读不到文件时不会抛异常，只会返回空的 Mat，所以必须自己检查。
    // ---------------------------------------------------------------------
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "读取图片失败：" << path << "\n";
        return -1;
    }

    // 终端和 output/report.txt 输出同一份内容，这里先把报告攒起来
    std::ostringstream report;
    report << "装甲板检测报告\n"
           << "图片: " << path << "\n"
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

// ===========================================================================
//  第二阶段：视频 / 验证
// ===========================================================================

// 视频模式下选哪一套检测阈值：默认暗光放宽版；`default` 用第一阶段原阈值，
// 只用于对比“放宽前后”的检出差异。
armor_vision::DetectorParams PickDetector(const std::string& name)
{
    if (name == "default")
    {
        return armor_vision::DetectorParams{};
    }
    if (name != "video")
    {
        std::cerr << "未知的 --detector 取值：" << name << "（可用 video / default）\n";
    }
    return armor_vision::VideoDetectorParams();
}

armor_vision::CameraIntrinsics PickCamera(const std::string& name)
{
    if (name == "params")
    {
        return armor_vision::CameraIntrinsics::FromParamsMd();
    }
    if (name != "doc")
    {
        std::cerr << "未知的 --calib 取值：" << name << "（可用 doc / params）\n";
    }
    return armor_vision::CameraIntrinsics::FromTaskDoc();
}

int RunVideoMode(const CommandLine& cli)
{
    armor_vision::PoseRunOptions options;
    options.video_path = cli.input;
    options.camera = PickCamera(cli.calib);
    options.detector = PickDetector(cli.detector);
    options.max_frames = cli.max_frames;
    if (!cli.out_video.empty()) options.out_video_path = cli.out_video;
    if (!cli.csv.empty()) options.csv_path = cli.csv;
    if (!cli.snapshot_dir.empty()) options.snapshot_dir = cli.snapshot_dir;
    if (!cli.codec.empty())
    {
        options.codecs = {cli.codec};
    }

    const armor_vision::PoseRunStats stats = armor_vision::RunPoseVideo(options);
    if (!stats.ok)
    {
        std::cerr << "视频流水线失败：" << stats.error << "\n";
        return -1;
    }
    return 0;
}

int RunValidateMode(const CommandLine& cli)
{
    armor_vision::ValidationOptions options;
    options.video_path = cli.input;
    options.camera = PickCamera(cli.calib);
    options.detector = PickDetector(cli.detector);
    if (!cli.validation_md.empty()) options.out_md = cli.validation_md;

    const armor_vision::ValidationStats stats = armor_vision::RunValidation(options);
    if (!stats.ok)
    {
        std::cerr << "验证失败：" << stats.error << "\n";
        return -1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    CommandLine cli;
    if (argc == 1)
    {
        // 保持最朴素的行为不变：不带参数就按第一阶段的方式跑默认图片。
        std::cout << "未提供参数，使用默认静态图片：" << cli.input << "\n\n";
    }
    if (!ParseArgs(argc, argv, &cli))
    {
        PrintUsage(argv[0]);
        return -1;
    }

    switch (cli.mode)
    {
        case CommandLine::Mode::Help:
            PrintUsage(argv[0]);
            return 0;
        case CommandLine::Mode::Image:
            return RunImageMode(cli.input);
        case CommandLine::Mode::Video:
            return RunVideoMode(cli);
        case CommandLine::Mode::Validate:
            return RunValidateMode(cli);
    }
    return -1;
}
