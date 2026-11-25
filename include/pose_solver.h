#ifndef POSE_SOLVER_H
#define POSE_SOLVER_H

#include <opencv2/core.hpp>

#include <array>
#include <string>

namespace armor_vision {

// ===========================================================================
//  相机内参 + 畸变系数
// ===========================================================================
//
// 两组参数都来自本项目随任务提供的资料，用来做同一件事：把相机坐标系
// 中的三维点映射到 1280x1024 的像素坐标。
//
//   K = [[fx, 0, cx],
//        [ 0, fy, cy],
//        [ 0,  0,  1]]        fx/fy 单位是像素，cx/cy 是主点像素坐标
//   D = [k1, k2, p1, p2, k3]   OpenCV 五参数顺序，无量纲
//
// 为什么不能换成别的相机/别的分辨率：fx 是把“米”换算成“像素”的比例尺，
// cx/cy 决定光轴落在图像的哪一列/哪一行。这几个数一旦和实际画面不匹配，
// solvePnP 求出的平移会整体偏掉（尤其是 tx/ty），所以把它写成一个结构体，
// 只在这一处定义，程序运行时把实际用到的 K、D 打印出来，方便和标定文件核对。
struct CameraIntrinsics
{
    cv::Mat K;              // 3x3 CV_64F
    cv::Mat D;              // 1x5 CV_64F
    int width = 0;          // 标定对应的图像宽（像素）
    int height = 0;         // 标定对应的图像高（像素）
    std::string name;       // 便于在日志里说明当前用的是哪一组

    bool Valid() const;

    // 姿态解算.md 第二节表格给出的那一组（本任务的默认值）
    static CameraIntrinsics FromTaskDoc();

    // 工程里 参数.md 文件给出的那一组（另一台/另一次标定）
    static CameraIntrinsics FromParamsMd();
};

// ===========================================================================
//  装甲板模型（单块小装甲板）
// ===========================================================================
//
// 单位统一为米。模型坐标系原点在装甲板中心：
//   x 轴沿宽度向右，y 轴沿高度向下，四个点都在 z = 0 平面上。
// 点序固定为 左上 -> 右上 -> 右下 -> 左下，必须和 LightBar/Armor 的角点顺序一致。
struct ArmorModel
{
    double width = 0.135;   // 米
    double height = 0.056;  // 米

    std::array<cv::Point3d, 4> ObjectPoints() const;
};

constexpr int kArmorCornerCount = 4;

// ===========================================================================
//  位姿结果
// ===========================================================================
//
// 坐标约定（OpenCV 相机坐标系）：
//   原点在相机光心；x 指向图像右；y 指向图像下；z 指向场景前方。
//   camera_point = R * model_point + tvec
//
//   tvec  ：装甲板中心在相机坐标系中的位置，米
//   rvec  ：Rodrigues 旋转向量，方向是旋转轴，长度是绕轴角度（弧度）
//   R     ：把模型坐标系中的方向旋转到相机坐标系的 3x3 矩阵
//   tz    ：tvec 的 z 分量，是“沿光轴的深度”，不是到光心的直线距离
//   distance_m：sqrt(tx^2 + ty^2 + tz^2)，装甲板中心到光心的直线距离，米
struct PoseResult
{
    bool valid = false;                          // solvePnP 成功且结果全为有限值
    cv::Vec3d rvec{0.0, 0.0, 0.0};               // 弧度
    cv::Vec3d tvec{0.0, 0.0, 0.0};               // 米
    cv::Mat rotation;                            // 3x3 CV_64F，由 rvec 转出
    double distance_m = 0.0;                     // 米
    double reprojection_error_px = 0.0;          // 四角点平均像素误差
    std::array<cv::Point2d, kArmorCornerCount> reprojected{};   // 重投影点，像素
    std::string failure_reason;                  // 失败时的一句话说明
};

// 对一块装甲板的四个图像角点（左上/右上/右下/左下，原图像素坐标）调用 solvePnP。
// 失败（点序异常、solvePnP 返回 false、结果非有限值）时返回 valid=false，
// 其它字段保持默认值，调用方不得读取。
PoseResult SolveArmorPose(const std::array<cv::Point2f, kArmorCornerCount>& corners,
                          const CameraIntrinsics& camera,
                          const ArmorModel& model);

// 四角点的平均重投影误差（像素）：
//   e_i = || p_i - p_hat_i ||，error = (e_0 + e_1 + e_2 + e_3) / 4
double MeanCornerError(const std::array<cv::Point2f, kArmorCornerCount>& corners,
                       const std::array<cv::Point2d, kArmorCornerCount>& projected);

// 装甲板在图像里的水平像素宽度 p：取上边和下边长度的平均值。
double ArmorPixelWidth(const std::array<cv::Point2f, kArmorCornerCount>& corners);

// 针孔成像的粗略深度核对：Z_rough = fx * W / p
// 只对“接近正对相机、画面中部、畸变影响小”的样例有意义，用来核对量级。
double RoughDepthFromWidth(const std::array<cv::Point2f, kArmorCornerCount>& corners,
                           double fx, double model_width);

}  // namespace armor_vision

#endif
