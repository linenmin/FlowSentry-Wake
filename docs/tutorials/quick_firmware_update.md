![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Quick Firmware Update Guide

> [!WARNING]
> **This guide is ONLY for boards that have already been enabled for firmware updates.**
> 
> If this is a **new board** or you're unsure whether firmware updates have been enabled, you **MUST** first go through the [Enable Card Firmware Update Guide](/docs/tutorials/enable_updates.md) before proceeding.

## Prerequisites

- Your board has already been enabled for firmware updates
- Voyager SDK is installed on your system
- You have the Voyager SDK Python virtual environment available

## Quick Update Steps

### 1. Activate the Voyager SDK Environment

```bash
source venv/bin/activate
```

### 2. Run the Interactive Firmware Update Tool

```bash
$AXELERA_DEVICE_DIR/firmware/interactive_flash_update.sh
```

Follow the on-screen instructions. The script will guide you through the entire process, including when power cycling is needed.

### 3. Verify the Update

After the final power off and back on, verify that the firmware update was successful:

```bash
axdevice
```

You should see output showing the updated firmware version.

## Important Notes

### Multi-Card Systems

If you have a system with **multiple Metis cards** or an **Axelera® AI PCIe card with 4 Metis® AIPU cores**, you **MUST** still follow the detailed steps in the [Multiple Metis Devices Update](/docs/tutorials/firmware_flash_update.md#multiple-metis-devices-update) section of the main firmware update guide.

Each Metis core must be updated individually using the `--device` option.

### New Boards

If you're working with a **new board** that hasn't been flashed before, you **MUST** first:
1. Go through the [Enable Card Firmware Update Guide](/docs/tutorials/enable_updates.md)
2. Then follow the [complete firmware update guide](/docs/tutorials/firmware_flash_update.md)

### Safety Reminder

- Never interrupt power during firmware updates
- Use a UPS if possible to prevent power interruptions
- Follow all prompts carefully and do not interrupt the process

## Getting Help

- **Community Support**: Visit the [Axelera AI Community](https://community.axelera.ai/)
- **Technical Support**: Contact your FAE or Axelera AI support team
- **Complete Guide**: For detailed information, see the [full firmware update guide](/docs/tutorials/firmware_flash_update.md)
