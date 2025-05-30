# CountywideSignDetectionServer
This is server side code that detects signs using yoloe segmentation finding tight bounding boxes and uses a custom algorithm to track and determine the location of the signs while calculating there dimensions. 


# C# Integration Changes for Python Sign-Detection Server

This document summarizes and organizes all the modifications you made to the C# client-side code. You can save it as `PythonIntegration.md`.

---

## 1. **Constants.cs**

### Added imports

```csharp
using System;
using System.IO;
using System.Diagnostics;  // required for starting Python process via ProcessStartInfo
```

### FindServerPath helper

```csharp
private static string FindServerPath()
{
    string? dir = AppContext.BaseDirectory;

    while (dir != null)
    {
        if (Path.GetFileName(dir)
                .Equals("SCDPW Countywide Sign app", StringComparison.OrdinalIgnoreCase))
        {
            var rootParent = Path.GetDirectoryName(dir)!;
            var candidate = Path.Combine(
                rootParent,
                "CountywideSignDetectionServer",
                "server.py"
            );
            if (File.Exists(candidate))
                return Path.GetFullPath(candidate);
            break;
        }
        dir = Path.GetDirectoryName(dir);
    }

    // fallback
    return Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory,
            "..", "CountywideSignDetectionServer", "server.py"
        )
    );
}
```

### New constants

```csharp
public static readonly string PYTHON_SERVER_PATH = FindServerPath();
public static readonly string PYTHON_EXE         = "python";
```

---

## 2. **Model.Sign**

Added properties to carry capture and size estimates:

```csharp
public string captureID { get; set; } = "";
public double EstimatedWidthInches  { get; set; } = 0.0;
public double EstimatedHeightInches { get; set; } = 0.0;
```

---

## 3. **DockWindow\.xaml.cs**

#### Note you might need to install

```bash
dotnet add package Python.Runtime
```

### Field & constructor

```csharp
// at top of class
private PythonServerInterface PythonServer = new();

// in constructor:
PythonServer.StartServer();
```

### On Exit

```csharp
// in Exit_Clicked:
PythonServer.Dispose();
```

---

## 4. **TakeSnapshot**

Replaced synchronous YOLO call with async Python call, generated capture ID & sign-map:

```csharp
// 1) hash full-frame into captureId
fullImage.Position = 0;
using var md5 = MD5.Create();
string captureId = BitConverter.ToString(md5.ComputeHash(fullImage))
                    .Replace("-", "").Substring(0, 8);

// 2) call Python for detection
var sw = Stopwatch.StartNew();
var detections = await PythonServer.DetectObjectsAsync(skImage);
sw.Stop();
Debug.WriteLine($"SendImageToServer: {sw.Elapsed.TotalMilliseconds:F2} ms");

// 3) build sign→bbox mapping & thumbnails
var mapping = new Dictionary<int, float[]>();
foreach (var detection in detections)
{
    var detectedSign = Sign.Clone(sign);
    detectedSign.captureID = captureId;

    // thumbnail
    var thumbImage = new MemoryStream(((MemoryStream)detectedSign.CroppedImage).ToArray());
    detectedSign.Thumbnail = ImageHelper.CreateThumbnail(thumbImage, detection);
    detectedSign.FrameLocation = sign.FrameLocation;

    int sid = detectedSign.GetHashCode();
    mapping[sid] = new float[] {
        detection.BoundingBox.Left,
        detection.BoundingBox.Top,
        detection.BoundingBox.Right,
        detection.BoundingBox.Bottom
    };

    AddSignToList(detectedSign);
}

// 4) send map to Python
PythonServer.SendSignMapping(captureId, mapping);
```

---

## 5. **AddSignAsFeature**

Fully async Track→ProcessDetections flow, storing Location & size:

```csharp
try
{
    await PythonServer.TrackFramesAsync(
        skImage,
        sign.captureID,
        sign.GetHashCode(),
        $"C:\vscode\dashCam\frames\{sign.ID}"
    );

    var dims = await PythonServer.ProcessDetectionsAsync(
        sign.captureID,
        sign.GetHashCode()
    );

    if (dims != null)
    {
        sign.Location = new MapPoint(
            dims.Longitude,
            dims.Latitude,
            SpatialReferences.Wgs84
        );
        sign.EstimatedWidthInches  = dims.EstimatedWidthInches;
        sign.EstimatedHeightInches = dims.EstimatedHeightInches;
    }

    SignsCaptured.Add(sign);
}
catch (Exception ex)
{
    MessageBox.Show(ex.Message, "Error");
}
```

---

## 6. **Helpers/PythonServerInterface.cs**

**New file**: exposes async RPC for detect, track, mapping, and final dimensions:

* `Task<List<ObjectDetection>> DetectObjectsAsync(...)`
* `Task<TrackerResult> TrackFramesAsync(...)`
* `void SendSignMapping(...)`
* `Task<SignDimensions?> ProcessDetectionsAsync(...)`
* `StartServer()` / `Dispose()` for lifecycle management

Key excerpt for `ProcessDetectionsAsync`:

```csharp
public async Task<SignDimensions?> ProcessDetectionsAsync(
    string captureId,
    int signHashCode,
    CancellationToken ct = default
)
{
    // build header
    var meta = new Dictionary<string,string> {
        ["capture_id"] = captureId,
        ["sign_id"]    = signHashCode.ToString()
    };
    var hdrBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(meta));

    using var client = new TcpClient();
    await client.ConnectAsync(_host, _port, ct);
    using var ns = client.GetStream();

    // send PROCESS_DETECTIONS (5)
    await ns.WriteAsync(new[]{ MSG_PROCESS_DETECTIONS }, ct);
    await ns.WriteAsync(UInt32BE(hdrBytes.Length), ct);
    await ns.WriteAsync(hdrBytes, 0, hdrBytes.Length, ct);

    // read response length
    var lenBuf = new byte[4];
    await ReadExactAsync(ns, lenBuf, ct);
    int n = IPAddress.NetworkToHostOrder(BitConverter.ToInt32(lenBuf,0));
    if (n == 0) return null;

    var payload = new byte[n];
    await ReadExactAsync(ns, payload, ct);
    var doc = JsonDocument.Parse(payload);

    var prop = doc.RootElement.EnumerateObject().First();
    var obj  = prop.Value;
    return new SignDimensions(
        obj["latitude"].GetDouble(),
        obj["longitude"].GetDouble(),
        obj["estimated_height_inches"].GetDouble(),
        obj["estimated_width_inches"].GetDouble()
    );
}
```

---

**Save this file** as `PythonIntegration.md` for reference. Let me know if you need any further tweaks!
