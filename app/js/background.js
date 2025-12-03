chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request?.type === "vision.captureFullPage") {
    captureViewport()
      .then((payload) => sendResponse({ ok: true, ...payload }))
      .catch((error) => {
        console.error("Vision capture failed", error);
        sendResponse({ ok: false, error: error.message || String(error) });
      });
    return true;
  }
  return false;
});

async function captureViewport() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    throw new Error("No active tab to capture");
  }

  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
    format: "png",
  });
  const blob = dataUrlToBlob(dataUrl);
  const bitmap = await createImageBitmap(blob);

  return {
    dataUrl,
    width: bitmap.width,
    height: bitmap.height,
  };
}

function dataUrlToBlob(dataUrl) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(base64);
  const length = binary.length;
  const array = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    array[i] = binary.charCodeAt(i);
  }
  return new Blob([array], { type: mime });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
