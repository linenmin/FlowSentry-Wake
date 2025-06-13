![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Deactivating BitLocker in 3 steps

- [Deactivating BitLocker in 3 steps](#deactivating-bitlocker-in-3-steps)
  - [Step 1: Open BitLocker](#step-1-open-bitlocker)
  - [Step 2: Deactivate BitLocker](#step-2-deactivate-bitlocker)
    - [Scenario 1: BitLocker is Not Active](#scenario-1-bitlocker-is-not-active)
    - [Scenario 2: BitLocker is Active](#scenario-2-bitlocker-is-active)
  - [Step 3: Confirm Deactivation](#step-3-confirm-deactivation)
  - [Next Steps](#next-steps)

This guide explains how to temporarily deactivate BitLocker **for the duration of one reboot** on a Windows system. BitLocker needs to be deactivated during the driver installation process to ensure proper system access.

## Step 1: Open BitLocker
1. Open the Windows search bar
2. Type "BitLocker"
3. Select "Manage BitLocker" from the search results

![Open BitLocker](/docs/images/windows/start_bitlocker.png)

## Step 2: Deactivate BitLocker
Look at the screen. One of two scenarios will be visible:

### Scenario 1: BitLocker is Not Active
If this screen is visible, BitLocker is already deactivated:
![BitLocker Deactivated](/docs/images/windows/bitlocker_deactivated.png)

In this case, proceed with the installation steps from the [Considerations](/docs/tutorials/windows/installing_driver.md#considerations) section.

### Scenario 2: BitLocker is Active
If BitLocker is active, this screen will be visible:
![Suspend Protection](/docs/images/windows/suspend_protection.png)

Click "Suspend protection" to continue.

## Step 3: Confirm Deactivation
A confirmation prompt will appear. Click "Yes" to confirm the BitLocker deactivation:

![Confirm Deactivation](/docs/images/windows/bitlocker_disable_prompt.png)

> [!WARNING]  
> BitLocker will automatically reactivate after the next system reboot. If additional changes using `bcdedit` are needed in the future, follow these steps again to temporarily deactivate BitLocker.

## Next Steps
After completing these steps, continue with the installation process from the [Considerations](/docs/tutorials/windows/installing_driver.md#considerations) section in the driver installation guide.
