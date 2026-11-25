#include "video_pipeline.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace armor_vision {

namespace {

// ---------------------------------------------------------------------------
//  小工具
// ---------------------------------------------------------------------------

cv::Point ToPixel(const cv::Point2f& p)
{
    return {static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y))};
}

bool MakeDirs(const std::string& file_path)
{
    const std::filesystem::path path(file_path);
    const std::filesystem::path dir = path.parent_path();
    if (dir.empty())
    {
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return !ec;
}

// 画在视频上的文字必须是 ASCII：cv::putText 不支持中文，
// 写中文会变成一串问号。终端和 CSV 里仍然用中文说明。
const char* StatusTag(bool pnp_valid, bool suspicious)
{
    if (!pnp_valid)
    {
        return "PNP-FAIL";
    }
    return suspicious ? "SUSPECT" : "OK";
}

// 候选按中心点排序：先 x 从小到大，x 相同时再按 y 从小到大。
// candidate_id 只用于区分同一帧内的候选，不代表跨帧跟踪身份。
std::vector<std::size_t> SortedCandidateOrder(const std::vector<Armor>& armors)
{
    std::vector<std::size_t> order(armors.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&armors](std::size_t a, std::size_t b) {
        if (armors[a].center.x != armors[b].center.x)
        {
            return armors[a].center.x < armors[b].center.x;
        }
        return armors[a].center.y < armors[b].center.y;
    });
    return order;
}

// ---------------------------------------------------------------------------
//  标注绘制
// ---------------------------------------------------------------------------

const cv::Scalar kCornerColors[4] = {
    cv::Scalar(255, 128, 0),    // 0 左上
    cv::Scalar(0, 200, 0),      // 1 右上
    cv::Scalar(0, 0, 255),      // 2 右下
    cv::Scalar(255, 0, 255),    // 3 左下
};

void DrawCandidate(cv::Mat& canvas, const Armor& armor, int candidate_id,
                   const PoseResult& pose, bool suspicious)
{
    // 1. 候选框（黄色轴对齐外接框）
    cv::rectangle(canvas, armor.bound_rect, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);

    // 2. 原始四角点连成的四边形（绿色）
    for (int k = 0; k < 4; ++k)
    {
        cv::line(canvas, armor.corners[k], armor.corners[(k + 1) % 4],
                 cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    // 3. 四角点 + 编号 0 1 2 3
    for (int k = 0; k < 4; ++k)
    {
        const cv::Point p = ToPixel(armor.corners[k]);
        cv::circle(canvas, p, 3, kCornerColors[k], -1, cv::LINE_AA);
        cv::putText(canvas, std::to_string(k), p + cv::Point(4, -4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, kCornerColors[k], 1, cv::LINE_AA);
    }

    // 4. 重投影点（青色小圈）+ 模型四条边，
    //    再画一条「原始角点 -> 重投影点」的细线，误差大的时候一眼能看出来。
    if (pose.valid)
    {
        std::array<cv::Point2f, 4> proj{};
        for (int k = 0; k < 4; ++k)
        {
            proj[k] = cv::Point2f(static_cast<float>(pose.reprojected[k].x),
                                  static_cast<float>(pose.reprojected[k].y));
        }
        for (int k = 0; k < 4; ++k)
        {
            cv::line(canvas, armor.corners[k], proj[k], cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
            cv::circle(canvas, ToPixel(proj[k]), 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
        }
        for (int k = 0; k < 4; ++k)
        {
            cv::line(canvas, proj[k], proj[(k + 1) % 4], cv::Scalar(255, 255, 0), 1, cv::LINE_4);
        }
    }

    // 5. 文字：候选编号 + 状态 + 距离 + 重投影误差
    std::ostringstream label;
    label << "C" << candidate_id << " " << StatusTag(pose.valid, suspicious);
    if (pose.valid)
    {
        label << " d=" << std::fixed << std::setprecision(2) << pose.distance_m << "m"
              << " e=" << std::setprecision(1) << pose.reprojection_error_px << "px";
    }
    const cv::Scalar color = !pose.valid ? cv::Scalar(0, 0, 255)
                            : (suspicious ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0));
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int text_x = std::clamp(armor.bound_rect.x, 0, std::max(0, canvas.cols - text_size.width - 2));
    int text_y = armor.bound_rect.y - 8;
    if (text_y < text_size.height + 4)
    {
        text_y = armor.bound_rect.y + armor.bound_rect.height + text_size.height + 6;
    }
    cv::putText(canvas, label.str(), cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

void DrawHud(cv::Mat& canvas, int frame_id, double time_s, double fps,
             std::size_t candidates, int valid_poses, const std::string& note)
{
    std::ostringstream hud;
    hud << "frame " << frame_id << "  t=" << std::fixed << std::setprecision(3) << time_s
        << "s  fps=" << std::setprecision(2) << fps
        << "  cand=" << candidates << "  pose=" << valid_poses;
    cv::putText(canvas, hud.str(), cv::Point(8, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(canvas, hud.str(), cv::Point(8, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    if (!note.empty())
    {
        cv::putText(canvas, note, cv::Point(8, 48),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
}

// 验证图：装甲板附近裁出来放大，保留四角点编号和重投影点，方便肉眼比较。
void SaveSnapshot(const cv::Mat& annotated, const Armor& armor,
                  int frame_id, const PoseResult& pose, const std::string& dir)
{
    const std::string stem = dir + "/frame_" + cv::format("%05d", frame_id);
    // 整幅标注图用 JPG（PNG 单张近 1 MB，10 张就快 10 MB），放大图用 PNG 保细节。
    cv::imwrite(stem + ".jpg", annotated, {cv::IMWRITE_JPEG_QUALITY, 92});

    const int margin_x = 110;
    const int margin_y = 80;
    cv::Rect roi(armor.bound_rect.x - margin_x, armor.bound_rect.y - margin_y,
                 armor.bound_rect.width + margin_x * 2, armor.bound_rect.height + margin_y * 2);
    roi &= cv::Rect(0, 0, annotated.cols, annotated.rows);
    if (roi.width < 4 || roi.height < 4)
    {
        return;
    }

    cv::Mat zoom;
    cv::resize(annotated(roi), zoom, cv::Size(), 3.0, 3.0, cv::INTER_LINEAR);

    // 放大图上写一行位姿摘要，数字与 CSV 一致
    std::ostringstream text;
    text << "frame " << frame_id;
    if (pose.valid)
    {
        text << "  tz=" << std::fixed << std::setprecision(3) << pose.tvec[2] << "m"
             << "  dist=" << std::setprecision(3) << pose.distance_m << "m"
             << "  err=" << std::setprecision(2) << pose.reprojection_error_px << "px";
    }
    else
    {
        text << "  pnp failed: " << pose.failure_reason;
    }
    cv::putText(zoom, text.str(), cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(zoom, text.str(), cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::imwrite(stem + "_zoom.png", zoom);
}

// CSV：一帧里每个候选一行；没有候选的帧也保留一行。
std::string CsvNumber(double value, int precision)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void WriteCsvRow(std::ostream& out, int frame_id, double time_s, int candidate_id,
                 bool detect_valid, const PoseResult& pose)
{
    out << frame_id << ',' << CsvNumber(time_s, 3) << ',';
    if (candidate_id >= 0)
    {
        out << candidate_id;
    }
    out << ',' << (detect_valid ? 1 : 0) << ',' << (pose.valid ? 1 : 0);

    if (pose.valid)
    {
        // 解算失败时这些字段必须留空，不能填 0：0 是一个合法数值，
        // 会让下游误以为“位姿在原点”。
        out << ',' << CsvNumber(pose.tvec[0], 6)
            << ',' << CsvNumber(pose.tvec[1], 6)
            << ',' << CsvNumber(pose.tvec[2], 6)
            << ',' << CsvNumber(pose.distance_m, 6)
            << ',' << CsvNumber(pose.rvec[0], 6)
            << ',' << CsvNumber(pose.rvec[1], 6)
            << ',' << CsvNumber(pose.rvec[2], 6)
            << ',' << CsvNumber(pose.reprojection_error_px, 4);
    }
    else
    {
        out << ",,,,,,,,";
    }
    out << '\n';
}

// 结果视频写完后再打开数一遍帧，确认“能正常播放”不是靠猜的。
int CountVideoFrames(const std::string& path, cv::Size* size, double* fps)
{
    cv::VideoCapture cap(path);
    if (!cap.isOpened())
    {
        return -1;
    }
    int count = 0;
    cv::Mat frame;
    while (cap.read(frame) && !frame.empty())
    {
        if (count == 0)
        {
            if (size != nullptr)
            {
                *size = frame.size();
            }
            if (fps != nullptr)
            {
                *fps = cap.get(cv::CAP_PROP_FPS);
            }
        }
        ++count;
    }
    return count;
}

}  // namespace

bool PrintVideoInfo(const std::string& path)
{
    cv::VideoCapture cap(path);
    if (!cap.isOpened())
    {
        std::cerr << "打开视频失败：" << path << "\n";
        return false;
    }
    const double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "视频: " << path << "\n"
              << "  宽 x 高 : " << static_cast<int>(width) << " x " << static_cast<int>(height) << "\n"
              << "  帧率    : " << fps << "\n"
              << "  总帧数  : " << static_cast<int>(frames) << "\n";
    return true;
}

PoseRunStats RunPoseVideo(const PoseRunOptions& options)
{
    PoseRunStats stats;

    cv::VideoCapture cap(options.video_path);
    if (!cap.isOpened())
    {
        stats.error = "打开视频失败：" + options.video_path;
        std::cerr << stats.error << "\n";
        return stats;
    }
    stats.video_opened = true;

    // ---- 1. 读属性，并对“明显无效”的属性做回退，但要提示 ----
    int width = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FRAME_WIDTH)));
    int height = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    double fps = cap.get(cv::CAP_PROP_FPS);
    const double reported_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    if (width <= 0 || height <= 0)
    {
        std::cerr << "警告：OpenCV 读到的画面尺寸无效（" << width << "x" << height
                  << "），回退为 1280x1024\n";
        width = 1280;
        height = 1024;
    }
    if (fps <= 0.0 || fps > 1000.0)
    {
        std::cerr << "警告：OpenCV 读到的帧率无效（" << fps << "），回退为 120.0\n";
        fps = 120.0;
    }

    std::cout << "视频: " << options.video_path << "\n"
              << "  宽 x 高 : " << width << " x " << height << "\n"
              << "  帧率    : " << std::fixed << std::setprecision(6) << fps << "\n"
              << "  总帧数  : " << static_cast<int>(reported_frames)
              << "（只作为参考，循环以实际读到的帧为准）\n";

    // 画面尺寸必须和标定一致，否则 fx/fy 的比例尺是错的，位姿没有意义。
    if (options.camera.width > 0 && options.camera.height > 0 &&
        (width != options.camera.width || height != options.camera.height))
    {
        stats.error = "画面尺寸与相机标定不一致，先停止位姿解算";
        std::cerr << "错误：" << stats.error << "（视频 " << width << "x" << height
                  << "，标定 " << options.camera.width << "x" << options.camera.height << "）\n";
        return stats;
    }

    std::cout << "相机参数: " << options.camera.name << "\n"
              << "  图像尺寸: " << options.camera.width << " x " << options.camera.height << "\n"
              << "  装甲板模型: " << options.model.width << " m x " << options.model.height
              << " m（角点顺序 左上 -> 右上 -> 右下 -> 左下）\n";

    if (!MakeDirs(options.out_video_path) || !MakeDirs(options.csv_path) ||
        !MakeDirs(options.snapshot_dir + "/placeholder"))
    {
        stats.error = "创建输出目录失败";
        std::cerr << stats.error << "\n";
        return stats;
    }

    // ---- 2. 打开视频写入器：依次尝试候选编码，用第一个能用的 ----
    cv::VideoWriter writer;
    for (const std::string& codec : options.codecs)
    {
        const int fourcc = cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
        writer.open(options.out_video_path, fourcc, fps, cv::Size(width, height));
        if (writer.isOpened())
        {
            stats.codec_used = codec;
            break;
        }
        writer.release();
    }
    if (!writer.isOpened())
    {
        stats.error = "无法创建输出视频：" + options.out_video_path;
        std::cerr << stats.error << "\n";
        return stats;
    }
    stats.writer_opened = true;
    std::cout << "输出视频编码: " << stats.codec_used << "  -> " << options.out_video_path << "\n";

    // ---- 3. CSV ----
    std::ofstream csv(options.csv_path);
    if (!csv)
    {
        stats.error = "无法写入 CSV：" + options.csv_path;
        std::cerr << stats.error << "\n";
        return stats;
    }
    csv << "frame_id,time_s,candidate_id,detect_valid,pnp_valid,"
        << "tx_m,ty_m,tz_m,distance_m,rx_rad,ry_rad,rz_rad,reprojection_error_px\n";

    // ---- 4. 检测器 ----
    const ArmorDetector detector(options.detector);

    int frame_id = 0;
    int last_snapshot_frame = -100000;
    cv::Mat frame;
    while (cap.read(frame))
    {
        // 读到空帧当正常结束处理（部分 AVI 会在末尾返回一次空图）。
        if (frame.empty())
        {
            break;
        }
        if (options.max_frames > 0 && frame_id >= options.max_frames)
        {
            break;
        }

        const double time_s = static_cast<double>(frame_id) / fps;
        const DetectResult detected = detector.Detect(frame);

        cv::Mat annotated = frame.clone();
        const std::vector<std::size_t> order = SortedCandidateOrder(detected.armors);

        int valid_poses = 0;
        for (std::size_t rank = 0; rank < order.size(); ++rank)
        {
            const Armor& armor = detected.armors[order[rank]];
            const int candidate_id = static_cast<int>(rank);

            PoseResult pose;
            if (armor.valid)
            {
                pose = SolveArmorPose(armor.corners, options.camera, options.model);
            }
            else
            {
                pose.failure_reason = "检测结果未标记为有效";
            }

            const bool suspicious = pose.valid &&
                (pose.reprojection_error_px > options.suspicious_reproj_px ||
                 pose.tvec[2] <= 0.0);
            if (pose.valid)
            {
                ++valid_poses;
                ++stats.poses_valid;
                if (suspicious)
                {
                    ++stats.poses_suspicious;
                }
            }

            DrawCandidate(annotated, armor, candidate_id, pose, suspicious);
            WriteCsvRow(csv, frame_id, time_s, candidate_id, armor.valid, pose);

            // 验证图：位姿有效、且离上一张够远，就存一组（数量有上限）
            if (pose.valid && stats.snapshots_saved < options.max_snapshots &&
                frame_id - last_snapshot_frame >= options.snapshot_min_gap)
            {
                SaveSnapshot(annotated, armor, frame_id, pose, options.snapshot_dir);
                last_snapshot_frame = frame_id;
                ++stats.snapshots_saved;
            }
        }

        if (order.empty())
        {
            // 没有候选的帧也要留一行，否则逐帧数据和视频帧数对不上。
            WriteCsvRow(csv, frame_id, time_s, -1, false, PoseResult{});
        }

        DrawHud(annotated, frame_id, time_s, fps, order.size(), valid_poses,
                order.empty() ? "no candidate" : "");

        writer.write(annotated);
        ++stats.frames_written;
        if (!order.empty())
        {
            ++stats.frames_with_candidate;
        }
        stats.last_frame_id = frame_id;
        ++frame_id;
        ++stats.frames_read;
    }

    // 循环是“读到空帧/末尾”正常退出的；这里区分一下正常 EOF 和提前失败。
    const bool reached_end = cap.get(cv::CAP_PROP_POS_FRAMES) >=
                             (reported_frames > 0 ? reported_frames - 0.5 : 0.0);
    if (!reached_end && stats.frames_read > 0 && options.max_frames == 0)
    {
        stats.error = "视频提前结束：最后成功处理的帧号 " +
                      std::to_string(stats.last_frame_id) + "（共处理 " +
                      std::to_string(stats.frames_read) + " 帧）";
        std::cerr << "警告：" << stats.error << "\n";
    }

    csv.flush();
    writer.release();
    cap.release();

    // ---- 5. 写完再读回来数一遍帧，确认文件真的能解码 ----
    cv::Size out_size;
    double out_fps = 0.0;
    const int readback = CountVideoFrames(options.out_video_path, &out_size, &out_fps);
    std::cout << "结果视频回读: " << readback << " 帧, " << out_size.width << " x "
              << out_size.height << ", fps=" << out_fps << "\n";
    if (readback != stats.frames_written)
    {
        std::cerr << "警告：回读帧数（" << readback << "）与写入帧数（"
                  << stats.frames_written << "）不一致\n";
    }

    std::cout << "处理帧数: " << stats.frames_read
              << "，其中有候选的帧: " << stats.frames_with_candidate
              << "，有效位姿: " << stats.poses_valid
              << "，可疑位姿: " << stats.poses_suspicious
              << "，验证图: " << stats.snapshots_saved << " 组\n";
    std::cout << "逐帧数据: " << options.csv_path << "\n";

    stats.ok = readback == stats.frames_written && stats.frames_written > 0;
    return stats;
}

}  // namespace armor_vision
