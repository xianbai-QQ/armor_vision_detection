#include "pose_solver.h"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace armor_vision {

namespace {

// 判断一个 Mat 里的数是否全部是有限值（不是 NaN / inf）。
// solvePnP 失败时偶尔会留下未写入或半写入的结果，读之前必须查一遍。
bool AllFinite(const cv::Mat& m)
{
    if (m.empty())
    {
        return false;
    }
    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(m.reshape(1), &min_val, &max_val);
    return std::isfinite(min_val) && std::isfinite(max_val);
}

}  // namespace

bool CameraIntrinsics::Valid() const
{
    if (K.rows != 3 || K.cols != 3 || K.type() != CV_64F)
    {
        return false;
    }
    if (D.empty() || D.total() < 4 || D.type() != CV_64F)
    {
        return false;
    }
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    return fx > 0.0 && fy > 0.0 && AllFinite(K) && AllFinite(D);
}

CameraIntrinsics CameraIntrinsics::FromTaskDoc()
{
    // 姿态解算.md 第二节「本阶段使用的相机参数」表格。
    // 这组 D 的 k3 很大（约 17.4），属于标定在边缘过拟合的径向畸变模型，
    // 只在画面最外圈明显，本任务的目标都在画面中部，影响很小。
    CameraIntrinsics cam;
    cam.name = "任务文档表格 (fx=1769.32, cx=762.99)";
    cam.width = 1280;
    cam.height = 1024;
    cam.K = (cv::Mat_<double>(3, 3) <<
             1769.321404, 0.0,          762.985097,
             0.0,         1768.461805,  572.3718881,
             0.0,         0.0,          1.0);
    cam.D = (cv::Mat_<double>(1, 5) <<
             0.1301456438, -3.237695144,
             -0.0007849919824, 0.001116547072, 17.43296795);
    return cam;
}

CameraIntrinsics CameraIntrinsics::FromParamsMd()
{
    // 工程内 参数.md 第一节给出的那一组标定结果。
    // 两组参数的 fx 几乎相同（1769 与 1781），但 cx 相差约 252 像素，
    // 说明它们来自两台不同装配的相机；用错一组会让 tx/ty 整体平移。
    CameraIntrinsics cam;
    cam.name = "参数.md (fx=1781.05, cx=510.58)";
    cam.width = 1280;
    cam.height = 1024;
    cam.K = (cv::Mat_<double>(3, 3) <<
             1781.053454, 0.0,         510.5751561,
             0.0,         1781.686873, 472.9735763,
             0.0,         0.0,         1.0);
    cam.D = (cv::Mat_<double>(1, 5) <<
             -0.04420740074, -0.1081537096,
             -0.0006506219039, 0.0002357380215, 0.466737247);
    return cam;
}

std::array<cv::Point3d, 4> ArmorModel::ObjectPoints() const
{
    const double hw = width * 0.5;
    const double hh = height * 0.5;
    // 0 左上(-x,-y) 1 右上(+x,-y) 2 右下(+x,+y) 3 左下(-x,+y)
    // 模型坐标 y 轴向下，所以「上」是 -y、「下」是 +y。
    return {
        cv::Point3d(-hw, -hh, 0.0),
        cv::Point3d( hw, -hh, 0.0),
        cv::Point3d( hw,  hh, 0.0),
        cv::Point3d(-hw,  hh, 0.0),
    };
}

double ArmorPixelWidth(const std::array<cv::Point2f, 4>& corners)
{
    const double top = cv::norm(corners[1] - corners[0]);
    const double bottom = cv::norm(corners[2] - corners[3]);
    return (top + bottom) * 0.5;
}

double MeanCornerError(const std::array<cv::Point2f, 4>& corners,
                       const std::array<cv::Point2d, 4>& projected)
{
    double sum = 0.0;
    for (int i = 0; i < kArmorCornerCount; ++i)
    {
        const double dx = static_cast<double>(corners[i].x) - projected[i].x;
        const double dy = static_cast<double>(corners[i].y) - projected[i].y;
        sum += std::sqrt(dx * dx + dy * dy);
    }
    return sum / kArmorCornerCount;
}

double RoughDepthFromWidth(const std::array<cv::Point2f, 4>& corners,
                           double fx, double model_width)
{
    // 针孔模型：像素宽度 p 与深度 Z 近似成反比，p/Z ≈ fx/W  =>  Z ≈ fx*W/p
    // 也就是“实际尺寸 / 成像尺寸 = 距离 / 焦距”。
    const double p = ArmorPixelWidth(corners);
    if (p < 1e-6 || fx <= 0.0)
    {
        return 0.0;
    }
    return fx * model_width / p;
}

PoseResult SolveArmorPose(const std::array<cv::Point2f, 4>& corners,
                          const CameraIntrinsics& camera,
                          const ArmorModel& model)
{
    PoseResult result;

    if (!camera.Valid())
    {
        result.failure_reason = "相机内参或畸变系数不合法";
        return result;
    }
    if (model.width <= 0.0 || model.height <= 0.0)
    {
        result.failure_reason = "装甲板模型尺寸不合法";
        return result;
    }

    const std::array<cv::Point3d, 4> object_points = model.ObjectPoints();
    std::vector<cv::Point3d> object;
    std::vector<cv::Point2d> image;
    object.reserve(kArmorCornerCount);
    image.reserve(kArmorCornerCount);
    for (int i = 0; i < kArmorCornerCount; ++i)
    {
        // 逐个检查角点，NaN/inf 直接拒绝：把坏点喂给 solvePnP 只会得到一个
        // 看起来“成功”但毫无意义的位姿。
        if (!std::isfinite(corners[i].x) || !std::isfinite(corners[i].y))
        {
            result.failure_reason = "角点包含非有限数值";
            return result;
        }
        object.push_back(object_points[i]);
        image.emplace_back(static_cast<double>(corners[i].x),
                           static_cast<double>(corners[i].y));
    }

    // 四边形面积退化（三点共线/四点重合）时 PnP 无唯一解，提前拦掉。
    double twice_area = 0.0;
    for (int i = 0; i < kArmorCornerCount; ++i)
    {
        const cv::Point2d& a = image[i];
        const cv::Point2d& b = image[(i + 1) % kArmorCornerCount];
        twice_area += a.x * b.y - b.x * a.y;
    }
    if (std::fabs(twice_area) * 0.5 < 10.0)
    {
        result.failure_reason = "角点围成的面积过小，PnP 退化";
        return result;
    }

    cv::Vec3d rvec;
    cv::Vec3d tvec;
    // 原图角点 + 原畸变系数一起传：畸变由 OpenCV 在解算时处理，
    // 不要再先 undistortPoints 一遍，否则等于把畸变算两次。
    const bool solved = cv::solvePnP(object, image, camera.K, camera.D,
                                     rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    if (!solved)
    {
        result.failure_reason = "solvePnP 返回 false";
        return result;
    }

    // solvePnP 返回 true 也可能给出 NaN（数值病态），必须再查一次。
    cv::Mat rvec_mat(rvec);
    cv::Mat tvec_mat(tvec);
    if (!AllFinite(rvec_mat) || !AllFinite(tvec_mat))
    {
        result.failure_reason = "solvePnP 结果包含非有限数值";
        return result;
    }

    cv::Mat rotation;
    cv::Rodrigues(rvec_mat, rotation);

    std::vector<cv::Point2d> projected;
    cv::projectPoints(object, rvec, tvec, camera.K, camera.D, projected);
    if (projected.size() != static_cast<std::size_t>(kArmorCornerCount))
    {
        result.failure_reason = "projectPoints 输出点数异常";
        return result;
    }

    result.valid = true;
    result.rvec = rvec;
    result.tvec = tvec;
    result.rotation = rotation;
    result.distance_m = std::sqrt(tvec[0] * tvec[0] + tvec[1] * tvec[1] + tvec[2] * tvec[2]);
    for (int i = 0; i < kArmorCornerCount; ++i)
    {
        result.reprojected[i] = projected[i];
    }
    result.reprojection_error_px = MeanCornerError(corners, result.reprojected);
    return result;
}

}  // namespace armor_vision
