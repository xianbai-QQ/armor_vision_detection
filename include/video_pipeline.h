#ifndef VIDEO_PIPELINE_H
#define VIDEO_PIPELINE_H

#include "armor_detector.h"
#include "pose_solver.h"

#include <string>
#include <vector>

namespace armor_vision {

// 视频逐帧检测 + 位姿解算的全部可调项。
// 集中放在一个结构体里，main 负责从命令行填，测试时可以整体替换。
struct PoseRunOptions
{
    std::string video_path = "outpost.avi";
    std::string out_video_path = "output/pose/outpost_pose.avi";
    std::string csv_path = "output/pose/frames.csv";
    std::string snapshot_dir = "output/pose/validation";

    CameraIntrinsics camera = CameraIntrinsics::FromTaskDoc();
    ArmorModel model;
    DetectorParams detector;

    // 结果视频编码候选。依次尝试，用本机 OpenCV 支持的第一个。
    std::vector<std::string> codecs{"H264", "XVID", "MJPG"};

    // 每隔多少帧保存一张验证图（只在该帧位姿有效时保存），以及最多保存几张。
    int snapshot_min_gap = 350;
    int max_snapshots = 12;

    // “可疑结果”阈值：重投影误差超过它就标记为可疑（详见 POSE.md 的选择依据）。
    double suspicious_reproj_px = 5.0;

    // 只处理前 N 帧（0 = 全部），用于冒烟测试。
    int max_frames = 0;
};

struct PoseRunStats
{
    bool video_opened = false;
    bool writer_opened = false;
    std::string codec_used;

    int frames_read = 0;
    int frames_written = 0;
    int frames_with_candidate = 0;
    int poses_valid = 0;
    int poses_suspicious = 0;
    int snapshots_saved = 0;
    int last_frame_id = -1;

    // 视频提前读取失败时的说明，以及最终的返回码依据。
    std::string error;
    bool ok = false;
};

// 打开视频 -> 逐帧检测 -> solvePnP -> 画标注并写视频 / 写 CSV / 存验证图。
PoseRunStats RunPoseVideo(const PoseRunOptions& options);

// 读取视频属性并打印（宽、高、帧率、总帧数），返回是否成功打开。
// 单独暴露出来，方便 main 在做重活之前先把“视频读对没有”确认一遍。
bool PrintVideoInfo(const std::string& path);

}  // namespace armor_vision

#endif
