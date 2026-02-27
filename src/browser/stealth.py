"""Browser stealth configuration adapted from visionbot's 8-layer stealth.

Layers:
1. WebDriver detection removal
2. Chrome runtime emulation
3. Plugins & MIME type spoofing
4. Permissions API faking
5. Hardware spoofing (CPU, memory, screen)
6. Canvas/WebGL fingerprint noise
7. Headless countermeasures
8. Behavioral simulation hooks
"""

from __future__ import annotations


class StealthConfig:
    """Generate stealth JavaScript injection scripts and Chrome launch args."""

    @staticmethod
    def get_stealth_init_script(
        enable_canvas_noise: bool = True,
        enable_webgl_spoof: bool = True,
    ) -> str:
        """Return combined JavaScript to inject via page.add_init_script()."""
        return "\n".join([
            _webdriver_removal(),
            _chrome_runtime(),
            _plugins_mime(),
            _permissions_api(),
            _hardware_spoof(),
            _canvas_noise() if enable_canvas_noise else "",
            _webgl_spoof() if enable_webgl_spoof else "",
            _headless_patches(),
        ])

    @staticmethod
    def get_launch_args() -> list[str]:
        """Chrome launch arguments for stealth."""
        return [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--disable-infobars",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-popup-blocking",
            "--disable-extensions",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-translate",
            "--disable-sync",
            "--metrics-recording-only",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--window-size=1920,1080",
        ]


def _webdriver_removal() -> str:
    """Layer 1: Remove webdriver detection flags."""
    return """
    // Layer 1: WebDriver removal
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    delete navigator.__proto__.webdriver;
    """


def _chrome_runtime() -> str:
    """Layer 2: Emulate Chrome runtime object."""
    return """
    // Layer 2: Chrome runtime
    if (!window.chrome) { window.chrome = {}; }
    if (!window.chrome.runtime) {
        window.chrome.runtime = {
            connect: function() {},
            sendMessage: function() {},
            onMessage: { addListener: function() {} },
            id: 'internal-extension-id'
        };
    }
    """


def _plugins_mime() -> str:
    """Layer 3: Spoof plugins and MIME types."""
    return """
    // Layer 3: Plugins & MIME
    Object.defineProperty(navigator, 'plugins', {
        get: () => {
            const plugins = [
                { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
                { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
                { name: 'Native Client', filename: 'internal-nacl-plugin', description: '' }
            ];
            plugins.length = 3;
            return plugins;
        }
    });
    Object.defineProperty(navigator, 'mimeTypes', {
        get: () => {
            const mimes = [
                { type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format' }
            ];
            mimes.length = 1;
            return mimes;
        }
    });
    """


def _permissions_api() -> str:
    """Layer 4: Fake Permissions API to return realistic values."""
    return """
    // Layer 4: Permissions API
    if (navigator.permissions) {
        const originalQuery = navigator.permissions.query.bind(navigator.permissions);
        navigator.permissions.query = (params) => {
            if (params.name === 'notifications') {
                return Promise.resolve({ state: 'prompt', onchange: null });
            }
            return originalQuery(params);
        };
    }
    """


def _hardware_spoof() -> str:
    """Layer 5: Spoof hardware info."""
    return """
    // Layer 5: Hardware spoofing
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
    Object.defineProperty(screen, 'colorDepth', { get: () => 24 });
    Object.defineProperty(navigator, 'maxTouchPoints', { get: () => 0 });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    Object.defineProperty(navigator, 'language', { get: () => 'en-US' });
    """


def _canvas_noise() -> str:
    """Layer 6: Add noise to canvas fingerprinting."""
    return """
    // Layer 6: Canvas fingerprint noise
    const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function(type) {
        if (type === 'image/png' || type === undefined) {
            const ctx = this.getContext('2d');
            if (ctx) {
                const imageData = ctx.getImageData(0, 0, this.width, this.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    imageData.data[i] = imageData.data[i] ^ (Math.random() > 0.5 ? 1 : 0);
                }
                ctx.putImageData(imageData, 0, 0);
            }
        }
        return origToDataURL.apply(this, arguments);
    };
    """


def _webgl_spoof() -> str:
    """Layer 6b: Spoof WebGL renderer info."""
    return """
    // Layer 6b: WebGL spoof
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(param) {
        if (param === 37445) return 'Google Inc. (NVIDIA)';
        if (param === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1650 Direct3D11 vs_5_0 ps_5_0, D3D11)';
        return getParameter.call(this, param);
    };
    """


def _headless_patches() -> str:
    """Layer 7: Patch headless detection vectors."""
    return """
    // Layer 7: Headless patches
    Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
    Object.defineProperty(navigator, 'vendor', { get: () => 'Google Inc.' });
    Object.defineProperty(navigator, 'appVersion', {
        get: () => '5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
    });
    """
