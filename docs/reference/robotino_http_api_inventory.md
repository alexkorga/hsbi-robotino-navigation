# Robotino HTTP API Inventory

- **Target:** `http://172.21.20.90`
- **Initial discovery:** 2026-07-28
- **Live lab confirmation:** 2026-08-18
- **Robotino controller:** hardware `3.0.0`, software `1.2.3`
- **Robotino OS image:** `4.20.9`
- **Implementation review:** 2026-08-29

## 1. Scope and Safety

This inventory combines:

1. Read-only requests made directly to this Robotino.
2. Routes extracted from the Robotino web application's bundled JavaScript.
3. The official Robotino REST API reference:
   <https://wiki.openrobotino.org/Rest_api.html>

Only `GET`, `HEAD`, and `OPTIONS` requests were used during discovery. No
motion, output, service-management, network-configuration, program-control, or
firmware-update request was sent.

The discovery was intentionally limited to the Robotino host and the HTTP
services explicitly referenced by its own web application. It was not an
intrusive scan of every TCP port. Routes may vary with the Robotino OS image
and installed packages.

Common Swagger/OpenAPI locations were checked but returned `404`; this image
does not expose machine-readable API documentation there.

## 2. Current Positioning Findings

The Robotino exposes working odometry:

```http
GET /data/odometry
```

Response format:

```text
[x, y, rot, vx, vy, omega, seq]
```

| Field | Meaning | Unit |
|---|---|---|
| `x` | Odometry-frame x position | m |
| `y` | Odometry-frame y position | m |
| `rot` | Odometry-frame heading | rad |
| `vx` | x velocity | m/s |
| `vy` | y velocity | m/s |
| `omega` | angular velocity | rad/s |
| `seq` | sequence number | integer |

The observed response was a seven-element array and the sequence number was
advancing.

This is raw odometry, not automatically an absolute pose in the factory map.
It is useful for measured velocity, relative motion, and consistency checks,
but it cannot initialize the factory position by itself.

The advanced `/data/pose` endpoint described below provides a separate pose
estimate. Laboratory testing showed that this estimate can start with a large
error and improve as the Robotino moves. It therefore must not be described as
guaranteed indoor GPS or accepted as localized merely because the HTTP request
succeeded. The runtime polls pose and odometry independently, applies the
calibrated pose-to-map transform, and keeps autonomous physical motion stopped
while that transform is unavailable. Automatic detection of the initially
inaccurate pose state remains an open laboratory task and must be resolved
before unsupervised live operation.

The installed SmartSoft localization and navigation components are present
but were inactive at discovery time:

| Service | Observed state |
|---|---|
| `SmartAmcl` | inactive |
| `SmartMapperGridMap` | inactive |
| `SmartPurePursuitNavigation` | inactive |
| `ComponentRobotinoBaseServer` | inactive |

Consequently, no active absolute map-localization source was confirmed during
the initial discovery. The later `/data/pose` observation does not change the
need to validate localization readiness.

## 3. Official Core Read Endpoints

These are documented by Robotino and were checked against this device.

| Endpoint | Status | Response |
|---|---:|---|
| `GET /cam0` | `200` | `image/jpeg`; current camera frame |
| `GET /sensorimage` | `200` | `image/png`; distance-sensor and bumper visualization |
| `GET /data/festoolcharger` | `200` | Empty on this configuration |
| `GET /data/powermanagement` | `200` | Object containing battery, external-power, charger-count, and voltage fields |
| `GET /data/charger0` | `200` | Charger 0 state, temperatures, voltage, current, time, and version |
| `GET /data/charger1` | `200` | Charger 1 state, temperatures, voltage, current, time, and version |
| `GET /data/controllerinfo` | `200` | Hardware and controller-software versions |
| `GET /data/services` | `200` | Hierarchical array of installed Robotino services |
| `GET /data/servicestatus/{name}` | `200` | Array of service-status text records |
| `GET /data/analoginputarray` | `200` | Eight numeric analog-input values |
| `GET /data/digitalinputarray` | `200` | Eight Boolean digital-input values |
| `GET /data/digitaloutputstatus` | `200` | Eight Boolean digital-output values |
| `GET /data/relaystatus` | `200` | Two Boolean relay values |
| `GET /data/bumper` | `200` | `{"value": boolean}` |
| `GET /data/distancesensorarray` | `200` | Nine distance-sensor values |
| `GET /data/scan0` | `200` | Active laser scan object; live payload confirmed 2026-08-18 |
| `GET /data/odometry` | `200` | `[x, y, rot, vx, vy, omega, seq]` |
| `GET /data/imageversion` | `200` | Robotino OS image version |
| `GET /data/poweroutputcurrent` | `200` | `{"current": value}` in amperes |

### Laser scan schema

When a first laser rangefinder is active, `/data/scan0` is documented to
return:

```json
{
  "seq": 0,
  "stamp": 0,
  "angle_min": 0,
  "angle_max": 0,
  "angle_increment": 0,
  "time_increment": 0,
  "scan_time": 0,
  "range_min": 0,
  "range_max": 0,
  "ranges": [],
  "intensities": []
}
```

During the initial 2026-07-28 discovery the route returned an empty body. On
2026-08-18 the same port-80 endpoint returned an active scan without any API
change. The live payload contained 441 samples over approximately 220 degrees:

```text
angle_min       -1.9198628664 rad  (about -110 degrees)
angle_max       +1.9285876751 rad  (about +110.5 degrees)
angle_increment  0.0087266453 rad  (0.5 degrees)
range_min        0 m
range_max       20 m
```

Range values of `0` occur in the live payload and must be treated as invalid
or absent returns rather than obstacles at the scanner origin. A later direct
read produced 380 valid returns out of 441 samples, ranging from about 0.71 m
to 14.68 m. The `intensities` array has the same length as `ranges` and in the
observed data contains mostly `0` with occasional `1` values.

The service states below therefore describe only the earlier discovery and
must not be used to infer current scan availability:

| Laser service | Observed state |
|---|---|
| `laserd3_ethernet` | inactive |
| `laserd3_serial` | inactive |
| `ComponentRobotinoLaserServer` | inactive |
| `SmartLaserNanoScan` | inactive |

### Current positioning fallback status

The confirmed core API remains available on port 80 even when the optional
advanced positioning service is unavailable. On 2026-08-18:

```http
GET http://172.21.20.90/data/odometry
```

returned:

```json
[0, 0, 0, 0, 0, 0, 27501]
```

and `/data/scan0` returned the active scan described above. In contrast,
`http://172.21.20.90:8153/data/pose` could not be reached. The fleet manager
can therefore continue acquiring odometry and LiDAR independently. These
measurements do not establish a factory-map pose, so the current runtime keeps
physical autonomous movement stopped until the port-8154 pose source is
available and a validated pose-to-map calibration has been recorded.

### Advanced pose endpoint confirmed on port 8154

Later on 2026-08-18, the additional pose service became reachable at:

```http
GET http://172.21.20.90:8154/data/pose
```

An observed response was:

```json
{
  "errphi": 0,
  "errx": 0,
  "erry": 0,
  "phi": 0.7389413007497438,
  "seq": 24976,
  "x": -16.112501853243806,
  "y": 1.1768848054199823
}
```

The fleet adapter maps `x` and `y` as metres, `phi` as the heading in radians,
and `seq` as the observation sequence. The semantics and units of `errx`,
`erry`, and `errphi` still need confirmation; they must not be interpreted as
Boolean validity flags merely because zero values were observed.

## 4. Official Core Write Endpoints

These routes were identified from the official API and frontend. They were
not invoked during discovery.

| Endpoint | Expected body | Effect |
|---|---|---|
| `PUT` or `POST /data/omnidrive` | `[vx, vy, omega]` | Sets translational and angular velocity |
| `PUT` or `POST /data/digitaloutput` | `{"num": 0..7, "val": boolean}` | Sets one digital output |
| `PUT` or `POST /data/digitaloutputarray` | Eight Booleans | Sets all digital outputs |
| `PUT` or `POST /data/relay` | `{"num": 0..1, "val": boolean}` | Sets one relay |
| `PUT` or `POST /data/relayarray` | Two Booleans | Sets both relays |
| `PUT` or `POST /data/poweroutput` | `{"value": -100..100}` | Sets power output |
| `PUT` or `POST /data/uploadProgram` | Name, raw size, and base64 data | Uploads a Robotino View program |

`/data/omnidrive` must receive commands at least every 200 ms while moving;
otherwise Robotino stops. The existing manual controller uses a conservative
10 Hz command rate and sends a zero command on exit.

HTTP `OPTIONS` returned a generic
`Allow: DELETE, GET, POST, PUT` header for tested `/data` routes. This appears
to be server-wide behavior and should not be interpreted as permission to use
every method on every route. Use the methods documented above.

## 5. Additional Port-80 Routes Used by the Web Application

The installed web frontend uses additional management endpoints that are not
part of the small public sensor/actuator API.

### Read routes

| Endpoint | Status | Observed response |
|---|---:|---|
| `GET /data/config` | `200` | Ten-node configuration tree |
| `GET /data/program` | `200` | Program list |
| `GET /data/globalvars` | `200` | Empty array at discovery time |
| `GET /data/robviewout` | `200` | Empty array at discovery time |
| `GET /data/wlanconfig` | `200` | WLAN configuration tree |
| `GET /data/wlanhwfound` | `200` | Object containing `found` |
| `GET /data/inputOutput` | `404` | Frontend route not provided by this image/configuration |
| `GET /data/gantt` | `404` | Frontend route not provided by this image/configuration |
| `GET /data/checkfirmware` | Not called | Avoided because it may initiate an external update check |
| `GET /logbundle` | Not called | Avoided because it may generate and download a large log archive |

The camera frontend adds a cache-busting query such as
`GET /cam0?stamp={value}`; it is the same camera resource.

### State-changing management routes

The frontend contains the following calls. They were not invoked:

| Endpoint | Frontend method | Purpose inferred from frontend |
|---|---|---|
| `/data/config` | `PUT` | Change Robotino configuration |
| `/data/restartdaemons` | `PUT` | Restart configured daemons |
| `/data/startService` | `PUT` | Start a named service |
| `/data/stopService` | `PUT` | Stop a named service |
| `/data/startStopProgram` | `PUT` | Start or stop a program |
| `/data/remove` | `PUT` | Remove a program |
| `/data/renameprogram` | `PUT` | Rename a program |
| `/data/startfirmwareupdate` | `PUT` | Start firmware update |
| `/data/wlanconfig` | `PUT` | Change WLAN configuration |

Payload schemas for these administrative routes should be derived and tested
individually before use. They should not be included in the navigation control
loop.

## 6. Installed Service Catalogue

`GET /data/services` reported four service groups:

| Group | Entries |
|---|---:|
| `robotino-daemons` | 15 |
| `robotino-smartsoft-master` | 20 |
| `robotino-smartsoft-slave` | 74 |
| `robotino-cobotta` | 2 |

Core daemon status during discovery:

| Service | Observed state |
|---|---|
| `restapid` | active |
| `controld3` | active |
| `camd2` | active |
| `controld2` | inactive |
| `gyrod` | inactive |
| `realsensed` | inactive |

Status is queried with:

```text
GET /data/servicestatus/{service-name}
```

Starting or stopping services is state-changing and was deliberately not
attempted.

## 7. Additional Services Referenced by the Frontend

### Ethernet discovery service on port 1880

| Endpoint | Status | Role |
|---|---:|---|
| `GET http://172.21.20.90:1880/eth-devices` | `200` | Reads Ethernet-device configuration |
| `PUT http://172.21.20.90:1880/eth-devices` | Not called | Changes Ethernet-device configuration |

### Optional COBOTTA service on port 8280

The frontend and official COBOTTA API reference expose:

| Endpoint | Method | Discovery result or role |
|---|---|---|
| `/cobotta/Status` | `GET` | `200`; status object |
| `/cobotta/Io/{id}` | `GET` | Reads one I/O value; parameterized route not called |
| `/cobotta/Io/{id}` | `PUT` | Writes one I/O value; not called |
| `/cobotta/Program/GetStatus` | `GET` | `200` |
| `/cobotta/Program/Start` | `PUT` | Not called |
| `/cobotta/Program/Stop` | `PUT` | Not called |
| `/cobotta/Program/Resume` | `PUT` | Not called |
| `/cobotta/Program/Hold` | `PUT` | Not called |
| `/cobotta/Reference` | `PUT` | Not called |
| `/cobotta/Speed` | `GET` | `200` |
| `/cobotta/Speed` | `PUT` | Not called |
| `/cobotta/Error` | `GET` | `403` in the current COBOTTA state |
| `/cobotta/Log` | `GET` | `200` |
| `/cobotta/Versions` | `GET` | `200` |
| `/ChangeConnectionParameters` | `PUT` | Not called |
| `/cobottaConfiguration` | `GET` | `404` on this installation |
| `/cobottaConnect` | `GET` | Action route; deliberately not called |
| `/cobottaDisconnect` | `GET` | Action route; deliberately not called |

Although the final two routes use `GET`, they are state-changing actions and
must not be treated as safe reads.

Official COBOTTA reference:
<https://wiki.openrobotino.org/COBOTTA_Rest_API.html>

## 8. Endpoints Used by the Current Fleet Runtime

The active HTTP adapter tries port `80` first for core telemetry and control,
then the configured advanced port (`8154` by default). `/data/pose` uses the
reverse order because it is supplied by the advanced service. Each endpoint is
polled independently so one unavailable sensor does not suppress the other
telemetry.

| Endpoint | Runtime use |
|---|---|
| `GET /data/pose` | Candidate physical pose; transformed into map coordinates only after coordinate calibration |
| `GET /data/odometry` | Measured relative pose and velocity |
| `GET /data/scan0` | LiDAR observation and dashboard overlay |
| `GET /data/distancesensorarray` | Nine clockwise proximity measurements |
| `GET /data/bumper` | Contact stop condition |
| `GET /data/powermanagement` | `batteryLow` status only |
| `PUT /data/omnidrive` | Already limited `[vx, vy, omega]` command in live mode |

The shadow command sink has no `/data/omnidrive` transmission path. Live and
shadow poses remain untrusted while `config/localization.yaml` has no calibrated
transform. Camera and administrative/service-management routes are not used by
the fleet runtime.

## 9. Safe Read Examples

```powershell
curl.exe http://172.21.20.90/data/odometry
curl.exe http://172.21.20.90/data/bumper
curl.exe http://172.21.20.90/data/powermanagement
curl.exe --output robotino-camera.jpg http://172.21.20.90/cam0
```

Do not turn exploratory commands into `PUT`, `POST`, or `DELETE` requests
without reviewing the endpoint payload and physical effect first.
