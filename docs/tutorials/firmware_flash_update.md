![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Firmware Update Guide

- [Firmware Update Guide](#firmware-update-guide)
  - [Prerequisites](#prerequisites)
  - [Update Process Overview](#update-process-overview)
  - [Single Metis Device Update](#single-metis-device-update)
  - [Multiple Metis Devices Update](#multiple-metis-devices-update)
  - [Safety and Recovery](#safety-and-recovery)
  - [Troubleshooting](#troubleshooting)

> [!TIP]
> **Shorter Guide if you've already flashed your current board before**: If you've previously enabled firmware updates on this board and just need to update to a newer version, see the [Quick Firmware Update Guide](/docs/tutorials/quick_firmware_update.md) for simplified instructions.

> [!WARNING]
> **Before attempting any firmware updates, you must be sure that your board is enabled for updates.**
> Updating firmware without first enabling updates can permanently brick your board. Enable updates once per board before flashing firmware. If you are unsure whether this has been done, repeat the enablement procedure to ensure your board is safe to update.
> 
> If you wish to enable firmware updates on your board, please carefully follow the steps in the [Enable Card Firmware Update Guide](/docs/tutorials/enable_updates.md) **before proceeding with any update attempts.**

> [!NOTE]
> The firmware update procedure is currently only supported on Linux systems. If you are using Windows, you will need to temporarily connect your board to a Linux host to perform the firmware update. After the update is complete, you can reconnect your board to your Windows system. This limitation will be addressed in a future release of the Voyager SDK.

## Prerequisites

To take full advantage of the latest features, improvements, and bug fixes in the current Voyager SDK release, it is strongly recommended to update your board's firmware. Keeping your board's firmware up to date ensures optimal compatibility and performance with the Voyager SDK.

Before you start the firmware update, make sure to:

1. Install the Voyager SDK on your system if you have not done so already. You can follow the steps in the [Voyager SDK install guide](/docs/tutorials/install.md).
2. Activate the Voyager SDK Python virtual environment by running:
```bash
source venv/bin/activate
```

> [!NOTE]
> **All of the following steps must be performed with the Voyager SDK Python virtual environment (`venv`) activated.**
> Ensure you have run `source venv/bin/activate` before starting, and keep the virtual environment active throughout all update steps. If you perform power off and back on, remember to re-activate the environment before continuing.

## Update Process Overview

The interactive flash update script automatically handles all firmware update stages. Simply run the script as instructed below and follow the on-screen instructions - the process is fully guided and will prompt you when power cycling is needed.

## Single Metis Device Update

For systems with a single Metis device connected, follow these steps:

### Run the Interactive Firmware Update Tool

> [!WARNING]
> **Do Not Modify the Flash Update Script**
>
> **Never make local changes** to the `$AXELERA_DEVICE_DIR/firmware/interactive_flash_update.sh` script. Changing the flashing procedure without consulting Axelera can cause system malfunction and may render your board unresponsive. This script contains essential safety checks and proper sequencing that must not be altered under any circumstances.

```bash
$AXELERA_DEVICE_DIR/firmware/interactive_flash_update.sh
```

### Verify the Update

After the final power off and back on again, verify that the firmware update was successful:

```bash
axdevice
```

You should see output similar to:
```
Device 0: metis-0:1:0 4GiB pcie flver=1.4.0 bcver=7.0 clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

## Multiple Metis Devices Update

For systems with multiple Metis cards or Axelera® AI's PCIe card with 4 Metis® AIPU cores, each Metis must be programmed individually. Follow these specific steps:

### Step 1: Identify All Device IDs

First, identify the device IDs of all Metis® AIPU cores:

```bash
axdevice
```

This command will list all detected devices with their IDs:
```
Device 0: metis-0:6c:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
Device 1: metis-0:6d:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
Device 2: metis-0:6e:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
Device 3: metis-0:6f:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

### Step 2: Update Each Device Individually

Update the firmware for each device ID individually using the `--device` option:

```bash
./interactive_flash_update.sh --device <device-id>
```

For example, using the device IDs from step 1:

```bash
./interactive_flash_update.sh --device metis-0:6c:0
```
Go through all rounds. Then continue with each one by one:
```bash
./interactive_flash_update.sh --device metis-0:6d:0
```
```bash
./interactive_flash_update.sh --device metis-0:6e:0
```
```bash
./interactive_flash_update.sh --device metis-0:6f:0
```

> [!IMPORTANT]
> You must execute the firmware update command for each of the 4 device IDs to ensure all Metis® AIPU cores on the PCIe card are properly updated.

### Step 3: Verify All Updates

After updating all devices, verify that all firmware updates were successful:

```bash
axdevice
```

## Safety and Recovery

> [!WARNING]
> **Never interrupt power during firmware updates**

Firmware updates carry significant risks. Understanding the safety mechanisms and recovery procedures is crucial:

### Stage 1 Update - Metis Firmware Update

- **Failsafe Mechanism**: Not available for this release. The update uses dual-boot regions in flash memory and updates the entire safety mechanism
- **Recovery**: If power is lost during update, the board may become unresponsive. There is no mechanism for recovery. Contact Axelera AI for assistance
- **Bricking Risk**: **HIGH** - Ensure uninterrupted power supply throughout the process

### Stage 2 Update - Board Controller Firmware Update

- **Failsafe Mechanism**: No - Single region update
- **Recovery**: There is no mechanism for recovery. Contact Axelera AI for assistance
- **Bricking Risk**: **HIGH** - Ensure uninterrupted power supply throughout the process

### Safety Recommendations

1. **Use a UPS (Uninterruptible Power Supply)** to protect against power interruptions
2. **Avoid system updates or reboots** during the firmware update process
3. **Close unnecessary applications** to reduce system load
4. **Monitor system temperature** to prevent thermal shutdowns
5. **Follow all prompts carefully** and do not interrupt the process

## Troubleshooting

### Common Issues

If you encounter issues during the firmware update process:

1. **Check System Requirements**: Ensure your system meets all prerequisites
2. **Verify Environment**: Confirm the Voyager SDK environment is properly activated
3. **Review Logs**: Check for any error messages in the terminal output

### Getting Help

- **Community Support**: Visit the [Axelera AI Community](https://community.axelera.ai/)
- **Technical Support**: Contact your FAE or Axelera AI support team
- **Documentation**: Review related documentation in the [tutorials section](/docs/tutorials/)
