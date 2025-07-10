![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Installing the Windows Driver

- [Installing the Windows Driver](#installing-the-windows-driver)
  - [Summary](#summary)
  - [Considerations](#considerations)
  - [Installation steps](#installation-steps)
    - [Step 1 - Getting the Driver](#step-1---getting-the-driver)
    - [Step 2 - Adding the Driver to Windows](#step-2---adding-the-driver-to-windows)
    - [Step 3 - Loading the driver for Metis](#step-3---loading-the-driver-for-metis)

## Summary

This guide covers the installation of the Windows driver for Metis devices:

## Considerations
As of May 2025, the Windows driver is not yet offered as a Microsoft-certified driver while the certification process is ongoing. Therefore, the driver is not installed automatically by Windows and needs to be installed manually.  To enable manual installation, Windows needs to be set up in testsign:
 - Open a Windows Command Prompt in Administrator Mode
 - Set up testsigning like:
```bash
bcdedit /set testsigning on
```
  - Temporarily disable BitLocker - [See how here](/docs/tutorials/windows/deactivate_bitlocker.md)
  - Reboot the PC
The PC desktop should display "Test Mode" in the lower right corner.

> [!WARNING]  
> BitLocker will automatically reactivate after the next system reboot. If additional changes using `bcdedit` are needed in the future, for example, to run:
> ```bash
> bcdedit /set testsigning off
> ```
> follow [these steps](/docs/tutorials/windows/deactivate_bitlocker.md) again to temporarily deactivate BitLocker.
> **If BitLocker is not disabled, the system may go into Recovery mode and require contacting your network administrator for the harddrive recovery key on boot. It is very important to disable BitLocker when making bcdedit changes.**

## Installation steps
> [!NOTE]  
> These steps require access to a Windows Administrator account.
### Step 1 - Getting the Driver
The driver archive can be downloaded from: [Metis-v1.3.3.zip](https://media.axelera.ai/releases/v1.3.3/build-Release-Windows-2204-amd64/package_repos/Metis-v1.3.3.zip)

Then extract the driver to a local folder. The extracted folder should contain three files: .cat, .inf and .sys.

### Step 2 - Adding the Driver to Windows
Open the Device Manager by right clicking on the Windows start button and selecting Device Manager.

![Open Device Manager](/docs/images/windows/open_device_manager.png)

Click on the unknown PCI Device and select Add Driver:

![Add Driver](/docs/images/windows/add_driver.png)

Set the path to the folder where the 3 driver files were extracted:

![Set Path](/docs/images/windows/set_path.png)

Select to install the driver anyway:

![Install Anyway](/docs/images/windows/install_anyway.png)

### Step 3 - Loading the driver for Metis

> [!NOTE]  
> If multiple "PCI Device" entries exist on the system, the Metis device can be identified by checking the device properties and hardware IDs details. All Metis devices on the system will have 1F9D vendor ID and 1100 device ID. The driver should be loaded for each one:

![Identifying Metis](/docs/images/windows/identifying_metis.png)

Right click on the PCI device and select Update Driver:

![Update Driver](/docs/images/windows/update_driver.png)

Select "Browse my computer":

![Browse Computer](/docs/images/windows/browse_computer.png)

Fill in the path to the location where the 3 driver files were extracted or built and select "Let me pick from a list":

![Let Me Pick](/docs/images/windows/let_me_pick.png)

Select Neural Processors (on Windows 10 this may be "Compute processors"):

![Neural Processors](/docs/images/windows/neuralp.png)

Select Metis:

![Metis Selection](/docs/images/windows/metis_selection.png)

Confirm device installation:

![Confirm Installation](/docs/images/windows/confirm.png)
