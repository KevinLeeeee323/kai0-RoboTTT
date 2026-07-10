#!/bin/bash
# Activate USB-CAN interfaces with names expected by the older ALOHA collection launch.
#
# Old launch files under /home/agilex/aloha/Piper_ros_private-ros-noetic use:
#   can_left, can_right
# New DAgger launch files use:
#   can_left_slave, can_right_slave, can_left_mas, can_right_mas
#
# Run this script before old collection. Run activate_can_arms.sh before DAgger.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

BITRATE="${BITRATE:-1000000}"

# Defaults match activate_can_arms.sh. Override these environment variables if
# find_all_can_port.sh reports different USB bus-info values on the IPC.
LEFT_SLAVE_USB="${LEFT_SLAVE_USB:-1-13:1.0}"
RIGHT_SLAVE_USB="${RIGHT_SLAVE_USB:-1-12:1.0}"
LEFT_MASTER_USB="${LEFT_MASTER_USB:-1-6:1.0}"
RIGHT_MASTER_USB="${RIGHT_MASTER_USB:-1-5:1.0}"

# Legacy names. The older dual-arm collection launch reads can_left/can_right.
LEFT_SLAVE_NAME="${LEFT_SLAVE_NAME:-can_left}"
RIGHT_SLAVE_NAME="${RIGHT_SLAVE_NAME:-can_right}"

# Keep master arms available under the older long-form names without colliding
# with can_left/can_right.
LEFT_MASTER_NAME="${LEFT_MASTER_NAME:-can_left_master}"
RIGHT_MASTER_NAME="${RIGHT_MASTER_NAME:-can_right_master}"

activate_can() {
    local can_name="$1"
    local usb_bus_info="$2"

    echo "Activating $can_name at $BITRATE on USB $usb_bus_info"
    bash ./can_activate.sh "$can_name" "$BITRATE" "$usb_bus_info"
}

activate_can "$LEFT_SLAVE_NAME" "$LEFT_SLAVE_USB"
activate_can "$RIGHT_SLAVE_NAME" "$RIGHT_SLAVE_USB"
activate_can "$LEFT_MASTER_NAME" "$LEFT_MASTER_USB"
activate_can "$RIGHT_MASTER_NAME" "$RIGHT_MASTER_USB"

echo "Current CAN interfaces:"
ip -br link show type can
