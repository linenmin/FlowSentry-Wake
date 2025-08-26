![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# axmonitor - Axelera AI System Monitoring Tool

> [!NOTE]  
> The Axelera AI System Monitoring Tool is currently supported only for Linux users. Future Voyager SDK releases will add Windows support.

`axmonitor` is a system monitoring tool for Axelera AI devices which runs on the host and provides real-time visibility into device metrics, much like CPU/GPU monitoring tools.

## Overview

The `axmonitor` application enables users to monitor key metrics from Axelera AI devices. It visualizes the current operational status of devices, including:

- Core Utilization
- Core & Board Temperatures
- Kernels Per Second (KPS)

Kernels on Axelera accelerators are always a number of fused compute operations together. They can refer to an inference or inference and a few mathematical functions combined. This parameter shows the number of these fused kernels executed per second on the hardware. For vision pipelines it is often the same as frames per second.
- Power Usage

These metrics are retrieved periodically (every 1 second) over a TCP/IP network connection to the Axelera System Service (`axsystemservice`), a host system service for device monitoring and management.

> [!NOTE]  
> Power measurements are supported only on [4-Metis AIPU PCIe cards](https://store.axelera.ai/products/pcie-ai-accelerator-card-powered-by-4-metis-aipu).

## Usage

Before running `axmonitor`, make sure the Axelera System Service is active. This service exposes a server that streams real-time device metrics, which `axmonitor` connects to as a subscriber.

### Configuring the Axelera System Service

By default, the Axelera System Service listens on `*:5555`. If you need to change this IP address or port, you can do so by modifying the `/lib/systemd/system/axsystemserver.service` file. Locate and update the following line:
```
ExecStart=/bin/bash -c 'exec axsystemserver --bind "<IP>:<Port>"'
```

Replace `<IP>:<Port>` with your desired address and port.

### 1. Start the System Monitoring Backend Service

To start the Axelera System Service: `sudo systemctl start axsystemserver.service`

This launches the backend service, opening the configured TCP address/port for device metrics streaming.

### 2. Run axmonitor

Once the service is running, start axmonitor by specifying the subscription address:

`axmonitor --server-address "127.0.0.1:5555"`

`axmonitor` will connect to the service’s TCP endpoint, receive device measurements every 1 second, and display them in a structured monitoring interface.

### axmonitor Command-Line Options

`axmonitor` provides a set of command-line options to customize its behavior. To view available options:
```
axmonitor --help
```
Example Help Output:
```
Axelera system monitoring tool for real-time hardware accelerator metrics.

Usage: axmonitor [options]
Allowed options:

Global options:
  -h [ --help ]                         Produce help message
  -l [ --log-level ] arg (=error)       Logging level (trace, debug, info, 
                                        warning, error, fatal)
  --log-file arg                        Redirect log messages to a file
  --server-address arg (=127.0.0.1:5555)
                                        Server address to connect to for 
                                        receiving metrics, in the format 
                                        IP:Port (default: 127.0.0.1:5555)
  --ui arg (=auto)                      ui mode: auto, console, gui
  -c [ --command ] arg                  Commands are read from string
  -t [ --topics ] arg                   List of topics to subscribe to
```

## User Interface modes

### GUI

The axmonitor interface is structured into multiple pages:

#### SYSTEM page

Displays top-level system information and general status of connected Axelera AI devices.

#### MONITOR page

Provides additional data for all metrics, offering deeper insight into the device behavior.

### Console mode

In addition to its graphical user interface, `axmonitor` also supports a console mode designed for terminal-based interaction. This mode is ideal for users who prefer lightweight or scriptable environments.

To start axmonitor in console mode:

```bash
axmonitor --server-address "127.0.0.1:5555" --ui "console"
```

Upon launch, you will see a prompt similar to:

```
axmonitor> Welcome to axmonitor. Type 'help' for available commands.
```

#### Interactive Commands

Once in console mode, you can enter commands directly at the prompt. Currently supported commands include:

```
axmonitor> help
Commands available:
 - help
        This help message
 - exit
        Quit the session
 - print
        Print last measurements
```

### One-shot Command Execution (-c option)

Console mode also supports non-interactive execution using the `-c` option. The provided command string must be a valid console command (e.g., `print`, etc.). This behaves similarly to a typical shell command - the specified command is executed immediately, and the tool exits. 

Example:

```bash
axmonitor --server-address "127.0.0.1:5555" --ui "console" -c print
```

This is especially useful for scripting or automation scenarios where only a single metric query or action is needed.

### Topics

In axmonitor, topics are used to filter the device metrics being subscribed to. 

#### Device Topics

These topics correspond to connected devices, allowing users to choose which specific device's metrics they want to monitor.

The available topics are:

- `DEV0`, `DEV1`, ..., `DEVN` where `N` is the number of connected devices. By default, the first device (`DEV0`) is chosen if no topic is specified.

To subscribe to a specific device, use the `--topics` option followed by a space-separated list of topics:

```bash
axmonitor --server-address "127.0.0.1:5555" --topics DEV0 DEV1
```

This will restrict the monitoring to only the devices `DEV0` and `DEV1`.

## Communication Architecture

axmonitor acts as a TCP client, subscribing to a specific address/port opened by axsystemservice. This connection is used to fetch live data every second, ensuring an up-to-date monitoring experience.

**Service-Tool Interaction**
```
+----------------+         TCP         +---------------+
| axsystemservice | <----------------> |   axmonitor    |
+----------------+                     +---------------+
```

- axsystemservice: Backend service collecting and broadcasting metrics
- axmonitor: Frontend CLI/GUI displaying metrics

## Notes

- axsystemservice must be running in the background before starting axmonitor.
- Future releases may expand monitored metrics.
