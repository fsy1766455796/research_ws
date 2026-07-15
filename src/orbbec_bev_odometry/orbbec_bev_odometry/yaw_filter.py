import math


def gate_yaw_motion(
    yaw_step_rad,
    pending_yaw_rad,
    translation_mm,
    translation_deadband_mm,
    moving_yaw_deadband_deg,
    stationary_yaw_input_deadband_deg,
    stationary_yaw_release_deg,
):
    """过滤单帧 yaw，并保留近静止状态下连续同向的微小转角。"""
    yaw_step_rad = float(yaw_step_rad)
    if float(translation_mm) >= float(translation_deadband_mm):
        accepted = (
            yaw_step_rad
            if abs(math.degrees(yaw_step_rad)) >= float(moving_yaw_deadband_deg)
            else 0.0
        )
        return accepted, 0.0

    # ECC 静止噪声可能带轻微偏置；先去掉实测噪声底，再进行有符号累积。
    if abs(math.degrees(yaw_step_rad)) < float(stationary_yaw_input_deadband_deg):
        yaw_step_rad = 0.0

    # 有符号累积让随机正负噪声抵消，持续同向转动则会越过释放阈值。
    pending_yaw_rad = float(pending_yaw_rad) + yaw_step_rad
    release_rad = math.radians(float(stationary_yaw_release_deg))
    if abs(pending_yaw_rad) >= release_rad:
        return pending_yaw_rad, 0.0
    return 0.0, pending_yaw_rad
