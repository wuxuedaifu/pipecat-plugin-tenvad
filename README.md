# LiveKit / Pipecat Plugin – TEN VAD

> ⚠️ **License Notice**
>
> This project is a **derivative integration** of  
> **TEN VAD** – https://github.com/TEN-framework/ten-vad  
> and is **NOT open source**.
>
> Usage, redistribution, and deployment of this plugin are governed by
> the original TEN VAD license.  
> You may **not** use this project in a way that competes with Agora / TEN offerings.

---

## Overview

**`pipecat-plugins-tenvad`** provides an integration layer that allows  
**TEN Voice Activity Detection (VAD)** to be used inside **Pipecat** audio pipelines.

The underlying VAD model and logic are provided by the upstream TEN VAD project.

For Local test purpose only.

---

## Usage

```python
from ten_vadAnalyzer import TenVADAnalyzer

FastAPIWebSocketParams(
    vad_analyzer = TenVADAnalyzer(sample_rate = 16000, threshold = 0.5)
)
```

This VAD instance can be plugged directly into  Pipecat audio pipelines.

---

## Legal

All rights to TEN VAD remain with their respective owners.
This repository provides **integration code only** and does not grant
any additional rights beyond the upstream license.
